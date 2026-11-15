"""
Spatial frequency clean/adversarial feature alignment for CNN backbones.

FRA-Attack uses a DCT over ViT patch-token embeddings. ResNet and WideResNet
do not expose patch tokens; their natural analogue is an intermediate feature
map Z in R^(B,C,H,W). This module applies a 2-D DCT over H,W independently for
each channel, selects high-frequency coordinates from the detached clean
feature map, and aligns adversarial feature vectors at the same coordinates.
"""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn.functional as F

_DCT_CACHE: Dict[Tuple[int, str, torch.dtype], torch.Tensor] = {}


def dct_matrix(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return an orthonormal Type-II DCT matrix."""
    key = (int(length), str(device), dtype)
    if key in _DCT_CACHE:
        return _DCT_CACHE[key]
    k = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    n = torch.arange(length, device=device, dtype=dtype).unsqueeze(0)
    matrix = torch.cos(torch.pi / length * (n + 0.5) * k)
    matrix[0] *= (1.0 / length) ** 0.5
    if length > 1:
        matrix[1:] *= (2.0 / length) ** 0.5
    _DCT_CACHE[key] = matrix
    return matrix


def spatial_dct_2d(features: torch.Tensor) -> torch.Tensor:
    """
    Apply orthonormal 2-D DCT to each channel of a feature map.

    Args:
        features: [B, C, H, W]
    Returns:
        coefficients: [B, C, H, W]
    """
    if features.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(features.shape)}.")
    height, width = features.shape[-2:]
    dh = dct_matrix(height, features.device, features.dtype)
    dw = dct_matrix(width, features.device, features.dtype)
    return torch.einsum("uh,bchw,vw->bcuv", dh, features, dw)


def high_frequency_candidate_mask(
    height: int,
    width: int,
    min_radius: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Radial high-frequency mask, with DC at (0,0).

    Radius is normalized to [0,1] using the maximum available frequency
    coordinate. min_radius=0.5 means only the outer half of the radial
    spectrum is eligible for selection.
    """
    if not 0.0 <= min_radius <= 1.0:
        raise ValueError("min_radius must be in [0,1].")
    u = torch.arange(height, device=device).float().view(height, 1)
    v = torch.arange(width, device=device).float().view(1, width)
    denominator = max(((height - 1) ** 2 + (width - 1) ** 2) ** 0.5, 1.0)
    radius = (u.square() + v.square()).sqrt() / denominator
    return radius >= min_radius


def select_clean_frequency_coordinates(
    clean_coefficients: torch.Tensor,
    *,
    min_radius: float = 0.5,
    top_k: int = 10,
) -> torch.Tensor:
    """
    Select high-energy spatial frequency coordinates from CLEAN coefficients.

    Returns flattened (u,v) indices with shape [B,K].
    Energy at each frequency is the L2 norm across channels.
    """
    batch, _, height, width = clean_coefficients.shape
    mask = high_frequency_candidate_mask(height, width, min_radius, clean_coefficients.device)
    candidates = mask.flatten().nonzero(as_tuple=False).squeeze(1)
    if candidates.numel() == 0:
        raise ValueError("No DCT frequency coordinates satisfy min_radius.")
    number = min(int(top_k), int(candidates.numel()))
    energy = clean_coefficients.pow(2).sum(dim=1).sqrt().flatten(1)
    candidate_energy = energy[:, candidates]
    local_indices = candidate_energy.topk(number, dim=1, largest=True, sorted=True).indices
    return candidates[local_indices].reshape(batch, number)


def gather_frequency_vectors(coefficients: torch.Tensor, flat_indices: torch.Tensor) -> torch.Tensor:
    """Gather channel vectors at flattened DCT frequency positions: [B,K,C]."""
    flat = coefficients.flatten(2).transpose(1, 2)  # [B,HW,C]
    return flat.gather(
        1, flat_indices.unsqueeze(-1).expand(-1, -1, coefficients.shape[1])
    )


def global_cosine_alignment(adv_global: torch.Tensor, clean_global: torch.Tensor) -> torch.Tensor:
    """Stop-gradient clean anchor for global pooled representations."""
    return (1.0 - F.cosine_similarity(adv_global, clean_global.detach(), dim=1)).mean()


def indexed_spatial_frequency_alignment(
    adversarial_map: torch.Tensor,
    clean_map: torch.Tensor,
    *,
    min_radius: float = 0.5,
    top_k: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recommended CNN alignment: paired frequency-coordinate cosine loss.

    Coordinates are selected from detached clean features and reused for the
    corresponding adversarial image.
    """
    clean_coefficients = spatial_dct_2d(clean_map)
    adversarial_coefficients = spatial_dct_2d(adversarial_map)
    coordinates = select_clean_frequency_coordinates(
        clean_coefficients, min_radius=min_radius, top_k=top_k
    )
    clean_vectors = gather_frequency_vectors(clean_coefficients, coordinates)
    adversarial_vectors = gather_frequency_vectors(adversarial_coefficients, coordinates)
    loss = (1.0 - F.cosine_similarity(adversarial_vectors, clean_vectors.detach(), dim=-1)).mean()
    # loss = (1.0 - F.cosine_similarity(adversarial_vectors, clean_vectors.detach(), dim=-1)).mean()
    # loss =  F.mse_loss(adversarial_vectors, clean_vectors.detach(), reduction='mean')
    return loss, coordinates

######################################################################
def low_frequency_candidate_mask(
    height: int,
    width: int,
    max_radius: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Radial low-frequency mask, with DC at (0,0).

    Radius is normalized to [0,1] using the maximum available frequency
    coordinate. max_radius=0.5 means only the inner half of the radial
    spectrum is eligible for selection.
    """
    if not 0.0 <= max_radius <= 1.0:
        raise ValueError("max_radius must be in [0,1].")
    u = torch.arange(height, device=device).float().view(height, 1)
    v = torch.arange(width, device=device).float().view(1, width)
    denominator = max(((height - 1) ** 2 + (width - 1) ** 2) ** 0.5, 1.0)
    radius = (u.square() + v.square()).sqrt() / denominator
    return radius <= max_radius


def select_clean_low_frequency_coordinates(
    clean_coefficients: torch.Tensor,
    *,
    max_radius: float = 0.5,
    top_k: int = 10,
) -> torch.Tensor:
    """
    Select low-frequency spatial frequency coordinates from CLEAN coefficients.

    Returns flattened (u,v) indices with shape [B,K].
    Energy at each frequency is the L2 norm across channels.
    """
    batch, _, height, width = clean_coefficients.shape
    mask = low_frequency_candidate_mask(height, width, max_radius, clean_coefficients.device)
    candidates = mask.flatten().nonzero(as_tuple=False).squeeze(1)
    if candidates.numel() == 0:
        raise ValueError("No DCT frequency coordinates satisfy max_radius.")
    number = min(int(top_k), int(candidates.numel()))
    energy = clean_coefficients.pow(2).sum(dim=1).sqrt().flatten(1)
    candidate_energy = energy[:, candidates]
    local_indices = candidate_energy.topk(number, dim=1, largest=True, sorted=True).indices
    return candidates[local_indices].reshape(batch, number)


def indexed_low_frequency_alignment(
    adversarial_map: torch.Tensor,
    clean_map: torch.Tensor,
    *,
    max_radius: float = 0.5,
    top_k: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Low-frequency CNN alignment: paired frequency-coordinate cosine loss.

    Coordinates are selected from detached clean features and reused for the
    corresponding adversarial image.
    """
    clean_coefficients = spatial_dct_2d(clean_map)
    adversarial_coefficients = spatial_dct_2d(adversarial_map)
    coordinates = select_clean_low_frequency_coordinates(
        clean_coefficients, max_radius=max_radius, top_k=top_k
    )
    clean_vectors = gather_frequency_vectors(clean_coefficients, coordinates)
    adversarial_vectors = gather_frequency_vectors(adversarial_coefficients, coordinates)
    loss = (
        1.0
        - F.cosine_similarity(
            adversarial_vectors,
            clean_vectors.detach(),
            dim=-1,
        )
    ).mean()
    return loss, coordinates
#####################################################################

def sinkhorn_cosine_loss(
    adversarial_vectors: torch.Tensor,
    clean_vectors: torch.Tensor,
    *,
    entropic_regularization: float = 0.1,
    iterations: int = 30,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Entropy-regularized set alignment; included as a FRA-like ablation."""
    if entropic_regularization <= 0:
        raise ValueError("entropic_regularization must be positive.")
    adv = F.normalize(adversarial_vectors, dim=-1, eps=eps)
    clean = F.normalize(clean_vectors.detach(), dim=-1, eps=eps)
    cost = 1.0 - torch.einsum("bkc,bjc->bkj", adv, clean)
    batch, k_adv, k_clean = cost.shape
    row_mass = cost.new_full((batch, k_adv), 1.0 / k_adv)
    col_mass = cost.new_full((batch, k_clean), 1.0 / k_clean)
    kernel = torch.exp(-cost / entropic_regularization).clamp_min(eps)
    left = torch.ones_like(row_mass)
    right = torch.ones_like(col_mass)
    for _ in range(iterations):
        left = row_mass / (torch.einsum("bkj,bj->bk", kernel, right) + eps)
        right = col_mass / (torch.einsum("bkj,bk->bj", kernel, left) + eps)
    plan = left.unsqueeze(-1) * kernel * right.unsqueeze(-2)
    return (plan * cost).sum(dim=(1, 2)).mean()


def ot_spatial_frequency_alignment(
    adversarial_map: torch.Tensor,
    clean_map: torch.Tensor,
    *,
    min_radius: float = 0.5,
    top_k: int = 10,
    independent_selection: bool = True, #better to set True if using ot
    entropic_regularization: float = 0.1,
    iterations: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    OT ablation. When independent_selection=True, clean and adversarial maps
    independently choose their salient frequency coordinates, resembling the
    set-matching behavior of FRA-Attack more closely.
    """
    clean_coefficients = spatial_dct_2d(clean_map.detach())
    adversarial_coefficients = spatial_dct_2d(adversarial_map)
    clean_coordinates = select_clean_frequency_coordinates(
        clean_coefficients, min_radius=min_radius, top_k=top_k
    )
    if independent_selection:
        adv_coordinates = select_clean_frequency_coordinates(
            adversarial_coefficients, min_radius=min_radius, top_k=top_k
        )
    else:
        adv_coordinates = clean_coordinates
    clean_vectors = gather_frequency_vectors(clean_coefficients, clean_coordinates)
    adv_vectors = gather_frequency_vectors(adversarial_coefficients, adv_coordinates)
    return sinkhorn_cosine_loss(
        adv_vectors, clean_vectors,
        entropic_regularization=entropic_regularization,
        iterations=iterations,
    ), clean_coordinates


def ot_low_frequency_alignment(
    adversarial_map: torch.Tensor,
    clean_map: torch.Tensor,
    *,
    max_radius: float = 0.5,
    top_k: int = 10,
    independent_selection: bool = True,
    entropic_regularization: float = 0.1,
    iterations: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Low-frequency OT ablation.

    This mirrors ot_spatial_frequency_alignment, but selects low-frequency
    coordinates from the DCT spectrum instead of high-frequency ones.
    """
    clean_coefficients = spatial_dct_2d(clean_map.detach())
    adversarial_coefficients = spatial_dct_2d(adversarial_map)
    clean_coordinates = select_clean_low_frequency_coordinates(
        clean_coefficients, max_radius=max_radius, top_k=top_k
    )
    if independent_selection:
        adv_coordinates = select_clean_low_frequency_coordinates(
            adversarial_coefficients, max_radius=max_radius, top_k=top_k
        )
    else:
        adv_coordinates = clean_coordinates
    clean_vectors = gather_frequency_vectors(clean_coefficients, clean_coordinates)
    adv_vectors = gather_frequency_vectors(adversarial_coefficients, adv_coordinates)
    return sinkhorn_cosine_loss(
        adv_vectors, clean_vectors,
        entropic_regularization=entropic_regularization,
        iterations=iterations,
    ), clean_coordinates

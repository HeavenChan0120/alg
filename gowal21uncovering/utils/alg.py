import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.attacks import create_attack
from core.metrics import accuracy
from core.utils.context import ctx_noparamgrad_and_eval
from core.utils import SmoothCrossEntropyLoss
from core.utils import track_bn_stats
from core.utils.cnn_frequency_alignment import *

Alignment_FUN = {
    'indexed': indexed_low_frequency_alignment,
    'ot': ot_low_frequency_alignment
}

def _forward_with_features(model, x):
    module = model.module if isinstance(model, nn.DataParallel) else model
    if isinstance(module, nn.Sequential):
        layers = list(module.children())
        for layer in layers[:-1]:
            x = layer(x)
        module = layers[-1]
    return module(x, return_feat=True)


# def kl_alg_loss(model, teacher_model, x, y, optimizer, step_size=0.007, epsilon=0.031, perturb_steps=10,
#               beta=1.0, gamma=0.5, attack='linf-pgd', label_smoothing=0.1, top_k=10, radius=0.5, alg_fun='indexed'):
#     """
#     KL with feature alignment
#     """
#     criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
#     criterion_kl = nn.KLDivLoss(reduction='sum')
#     model.train()
#     track_bn_stats(model, False)
    
#     x_adv = x.detach() +  torch.FloatTensor(x.shape).uniform_(-epsilon, epsilon).cuda().detach()
#     x_adv = torch.clamp(x_adv, 0.0, 1.0)
#     logits_natural = model(x)
#     p_natural = F.softmax(logits_natural, dim=1).detach()
    
#     if attack == 'linf-pgd':
#         for _ in range(perturb_steps):
#             x_adv.requires_grad_()
#             with torch.enable_grad():
#                 loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
#             grad = torch.autograd.grad(loss_kl, [x_adv])[0]
#             x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
#             x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
#             x_adv = torch.clamp(x_adv, 0.0, 1.0)
#     elif attack == 'l2-pgd':
#         delta = torch.FloatTensor(x.shape).normal_(mean=0, std=1.0).cuda().detach()
#         delta.data = delta.data * np.random.uniform(0.0, epsilon) / (delta.data**2).sum([1, 2, 3], keepdim=True)**0.5
#         delta = Variable(delta.data, requires_grad=True).cuda()
        
#         batch_size = len(x)
#         optimizer_delta = torch.optim.SGD([delta], lr=step_size)
#         for _ in range(perturb_steps):
#             adv = x + delta
#             optimizer_delta.zero_grad()
#             with torch.enable_grad():
#                 loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
#             loss.backward(retain_graph=True)
          
#             grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
#             delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
#             if (grad_norms == 0).any():
#                 delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
#             optimizer_delta.step()

#             delta.data.add_(x)
#             delta.data.clamp_(0, 1).sub_(x)
#             delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)

#     model.train()
#     track_bn_stats(model, True)

#     x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
#     optimizer.zero_grad()

#     teacher_model.eval()
#     with torch.no_grad():
#         teacher_logit, teacher_feats = _forward_with_features(teacher_model, x)

#     logit, feats = _forward_with_features(model, x_adv)
#     # loss_adv = (1/len(x)) * criterion_kl(F.log_softmax(logit, dim=1), F.softmax(teacher_logit, dim=1))
#     loss_adv = criterion_ce(logit, y)
#     loss_natural = criterion_ce(logits_natural, y)


#     feature_alignment_loss = 0
#     for key in feats.keys():

#         if key != "penultimate":
#             frequency_alignment_loss, _ = Alignment_FUN[alg_fun](
#                 feats[key], teacher_feats[key], top_k=top_k, max_radius=radius
#             )
#             feature_alignment_loss += frequency_alignment_loss

#         else:
#             f_adv = feats[key]
#             f_nat = teacher_feats[key]
#             feature_alignment_loss += (1 - F.cosine_similarity(
#                 F.normalize(f_adv, dim=1),
#                 F.normalize(f_nat, dim=1),
#                 dim=1
#             )).mean()
#     loss = beta * loss_adv + gamma * feature_alignment_loss

#     batch_metrics = {'loss': loss.item(), 'loss_adv': loss_adv.item(), 'feature_alignment_loss': feature_alignment_loss.item()}
#     batch_metrics.update({'adversarial_acc': accuracy(y, logit.detach())}) 
#     batch_metrics.update({'clean_acc': accuracy(y, logits_natural.detach())})
#     return loss, batch_metrics

# def at_alg_loss(model, teacher_model, x, y, optimizer, step_size=0.007, epsilon=0.031, perturb_steps=10, beta=1.0, gamma=0.5, 
#                 attack='linf-pgd', label_smoothing=0.1, top_k=10, radius=0.5, alg_fun='indexed'):
#     """
#     Adversarial training with feature alignment
#     """
#     criterion_ce_smooth = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
#     criterion_ce = nn.CrossEntropyLoss()
#     model.train()
#     track_bn_stats(model, False)
    
#     attack = create_attack(model, criterion_ce, attack, epsilon, perturb_steps, step_size)
#     # Attackers = AttackerPolymer(epsilon, perturb_steps, step_size, num_classes=10, device=x.device)
#     with ctx_noparamgrad_and_eval(model):
#         x_adv, _ = attack.perturb(x, y)
#         # x_adv = Attackers.run_specified('PGD_10', model, x, y, return_acc=False)
        
#     model.train()
    
#     optimizer.zero_grad()
#     track_bn_stats(model, True)
    
#     out_clean = model(x)
#     # loss_clean = criterion_ce_smooth(out_clean, y)

#     teacher_model.eval()
#     with torch.no_grad():
#         teacher_logit, teacher_feats = _forward_with_features(teacher_model, x)

#     out_adv, feats = _forward_with_features(model, x_adv)
#     loss_adv = criterion_ce(out_adv, y)
#     # loss_adv = (1/len(x)) * nn.KLDivLoss(reduction='sum')(F.log_softmax(out_adv, dim=1), F.softmax(teacher_logit, dim=1))

#     feature_alignment_loss = 0
#     for key in feats.keys():

#         if key != "penultimate":
#             frequency_alignment_loss, _ = Alignment_FUN[alg_fun](
#                 feats[key], teacher_feats[key], top_k=top_k, max_radius=radius
#             )
#             feature_alignment_loss += frequency_alignment_loss

#         else:
#             f_adv = feats[key]
#             f_nat = teacher_feats[key]
#             feature_alignment_loss += (1 - F.cosine_similarity(
#                 F.normalize(f_adv, dim=1),
#                 F.normalize(f_nat, dim=1),
#                 dim=1
#             )).mean()

#     loss = beta * loss_adv + gamma * feature_alignment_loss
    
#     batch_metrics = {'loss': loss.item(), 'loss_adv': loss_adv.item(), 'feature_alignment_loss': feature_alignment_loss.item()}
#     batch_metrics.update({'adversarial_acc': accuracy(y, out_adv.detach())}) 
#     batch_metrics.update({'clean_acc': accuracy(y, out_clean.detach())})
#     return loss, batch_metrics


def trades_alg_loss(model, teacher_model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, 
                attack='linf-pgd', label_smoothing=0.1, top_k=10, radius=0.5, gamma=0.5, alg_fun='indexed'):
    """
    TRADES training (Zhang et al, 2019).
    """
  
    criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)
    # epsilon = 9/255
    x_adv = x_natural.detach() +  torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    with torch.no_grad():
        p_natural = F.softmax(model(x_natural), dim=1).detach()
    
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    elif attack == 'l2-pgd':
        delta = torch.FloatTensor(x_natural.shape).normal_(mean=0, std=1.0).cuda().detach()
        delta.data = delta.data * np.random.uniform(0.0, epsilon) / (delta.data**2).sum([1, 2, 3], keepdim=True)**0.5
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = torch.optim.SGD([delta], lr=step_size)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
            loss.backward(retain_graph=True)
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    
    model.train()
    track_bn_stats(model, True)
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    optimizer.zero_grad()
    logits_natural = model(x_natural)
    logits_adv, feats_adv = _forward_with_features(model, x_adv)
    loss_natural = criterion_ce(logits_natural, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits_natural, dim=1))

    teacher_model.eval()
    with torch.no_grad():
        teacher_logit, teacher_feats = _forward_with_features(teacher_model, x_natural)

    feature_alignment_loss = 0
    for key in feats_adv.keys():

        if key != "penultimate":
            frequency_alignment_loss, _ = Alignment_FUN[alg_fun](
                feats_adv[key], teacher_feats[key], top_k=top_k, max_radius=radius
            )
            feature_alignment_loss += frequency_alignment_loss

        else:
            f_adv = feats_adv[key]
            f_nat = teacher_feats[key]
            feature_alignment_loss += (1 - F.cosine_similarity(
                F.normalize(f_adv, dim=1),
                F.normalize(f_nat, dim=1),
                dim=1
            )).mean()


    loss = loss_natural + beta * loss_robust + gamma * feature_alignment_loss
    
    batch_metrics = {'loss': loss.item(), 'loss_adv': loss_robust.item(), 'feature_alignment_loss': feature_alignment_loss.item()}
    batch_metrics.update({'clean_acc': accuracy(y, logits_natural.detach()), 
                          'adversarial_acc': accuracy(y, logits_adv.detach())})    
    return loss, batch_metrics
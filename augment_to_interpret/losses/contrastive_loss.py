"""
Contain contrastive losses
"""

import torch

EPS = 1e-15


def simclr(z,
           z_aug,
           labels=None,
           temperature=0.07,
           base_temperature=0.07):
    features = torch.cat([
        z.view(z.shape[0], 1, z.shape[-1]),
        z_aug.view(z_aug.shape[0], 1, z_aug.shape[-1])
    ],
        dim=1)

    batch_size = features.shape[0]
    if labels is None:
        mask = torch.eye(batch_size, dtype=torch.float32, device=features.device)
    else:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float()

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T), temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask), 1,
        torch.arange(batch_size * anchor_count, device=mask.device).view(-1, 1), 0)
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    return loss.mean()


def barlow_twins(h1: torch.Tensor,
                 h2: torch.Tensor,
                 lambda_=None,
                 batch_norm=True,
                 eps=1e-5,
                 *args,
                 **kwargs):
    batch_size = h1.size(0)
    feature_dim = h1.size(1)

    if lambda_ is None:
        lambda_ = 1. / feature_dim

    if batch_norm:
        z1_norm = (h1 - h1.mean(dim=0)) / (h1.std(dim=0) + eps)
        z2_norm = (h2 - h2.mean(dim=0)) / (h2.std(dim=0) + eps)
        c = (z1_norm.T @ z2_norm) / batch_size
    else:
        c = h1.T @ h2 / batch_size

    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (1 - c.diagonal()).pow(2).sum()
    loss += lambda_ * c[off_diagonal_mask].pow(2).sum()

    return loss


def rationale_loss(x, x_aug, x_cp):
    # Taken from https://github.com/lsh0520/RGCL/blob/main/unsupervised_TU/rgcl.py

    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    x_cp_abs = x_cp.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / (torch.einsum(
        'i,j->ij', x_abs, x_aug_abs) + EPS)
    sim_matrix = torch.exp(sim_matrix / T)

    sim_matrix_cp = torch.einsum('ik,jk->ij', x, x_cp) / (torch.einsum(
        'i,j->ij', x_abs, x_cp_abs) + EPS)
    sim_matrix_cp = torch.exp(sim_matrix_cp / T)

    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss1 = pos_sim / (sim_matrix.sum(dim=1) + EPS)
    loss2 = pos_sim / (sim_matrix_cp.sum(dim=1) + pos_sim + EPS)

    loss = loss1 + 0.1 * loss2

    sufficiency_loss = -torch.log(loss1).mean()
    independence_loss = -torch.log(loss2).mean()
    loss = -torch.log(loss).mean()
    return loss, sufficiency_loss, independence_loss

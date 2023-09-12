"""
Code taken from https://github.com/PyGCL/PyGCL
"""

import torch
from abc import ABC, abstractmethod
from torch_scatter import scatter
import torch.nn.functional as F
import numpy as np


class Sampler(ABC):
    def __init__(self, intraview_negs=False):
        self.intraview_negs = intraview_negs

    def __call__(self, anchor, sample, *args, **kwargs):
        ret = self.sample(anchor, sample, *args, **kwargs)
        if self.intraview_negs:
            ret = self.add_intraview_negs(*ret)
        return ret

    @abstractmethod
    def sample(self, anchor, sample, *args, **kwargs):
        pass

    @staticmethod
    def add_intraview_negs(anchor, sample, pos_mask, neg_mask):
        num_nodes = anchor.size(0)
        device = anchor.device
        intraview_pos_mask = torch.zeros_like(pos_mask, device=device)
        intraview_neg_mask = torch.ones_like(pos_mask, device=device) - torch.eye(num_nodes, device=device)
        new_sample = torch.cat([sample, anchor], dim=0)  # (M+N) * K
        new_pos_mask = torch.cat([pos_mask, intraview_pos_mask], dim=1)  # M * (M+N)
        new_neg_mask = torch.cat([neg_mask, intraview_neg_mask], dim=1)  # M * (M+N)
        return anchor, new_sample, new_pos_mask, new_neg_mask


class SameScaleSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super(SameScaleSampler, self).__init__(*args, **kwargs)

    def sample(self, anchor, sample, *args, **kwargs):
        assert anchor.size(0) == sample.size(0)
        num_nodes = anchor.size(0)
        device = anchor.device
        pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
        neg_mask = 1. - pos_mask
        return anchor, sample, pos_mask, neg_mask


class CrossScaleSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super(CrossScaleSampler, self).__init__(*args, **kwargs)

    def sample(self, anchor, sample, batch=None, neg_sample=None, use_gpu=True, *args, **kwargs):
        num_graphs = anchor.shape[0]  # M
        num_nodes = sample.shape[0]  # N
        device = sample.device

        if neg_sample is not None:
            assert num_graphs == 1  # only one graph, explicit negative samples are needed
            assert sample.shape == neg_sample.shape
            pos_mask1 = torch.ones((num_graphs, num_nodes), dtype=torch.float32, device=device)
            pos_mask0 = torch.zeros((num_graphs, num_nodes), dtype=torch.float32, device=device)
            pos_mask = torch.cat([pos_mask1, pos_mask0], dim=1)  # M * 2N
            sample = torch.cat([sample, neg_sample], dim=0)  # 2N * K
        else:
            assert batch is not None
            if use_gpu:
                ones = torch.eye(num_nodes, dtype=torch.float32, device=device)  # N * N
                pos_mask = scatter(ones, batch, dim=0, reduce='sum')  # M * N
            else:
                pos_mask = torch.zeros((num_graphs, num_nodes), dtype=torch.float32).to(device)
                for node_idx, graph_idx in enumerate(batch):
                    pos_mask[graph_idx][node_idx] = 1.  # M * N

        neg_mask = 1. - pos_mask
        return anchor, sample, pos_mask, neg_mask


def get_sampler(mode: str, intraview_negs: bool) -> Sampler:
    if mode in {'L2L', 'G2G'}:
        return SameScaleSampler(intraview_negs=intraview_negs)
    elif mode == 'G2L':
        return CrossScaleSampler(intraview_negs=intraview_negs)
    else:
        raise RuntimeError(f'unsupported mode: {mode}')


def add_extra_mask(pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask.bool(), extra_pos_mask.bool()).float()
    if extra_neg_mask is not None:
        neg_mask = torch.bitwise_and(neg_mask.bool(), extra_neg_mask.bool()).float()
    else:
        neg_mask = 1. - pos_mask
    return pos_mask, neg_mask


class Loss(ABC):
    @abstractmethod
    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
        pass

    def __call__(self, anchor, sample, pos_mask=None, neg_mask=None, *args, **kwargs) -> torch.FloatTensor:
        loss = self.compute(anchor, sample, pos_mask, neg_mask, *args, **kwargs)
        return loss


class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=h1, sample=h2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=h2, sample=h1)
        elif self.mode == 'G2G':
            assert g1 is not None and g2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=g2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=g1)
        else:  # global-to-local
            if batch is None or batch.max().item() + 1 <= 1:  # single graph
                assert all(v is not None for v in [h1, h2, g1, g2, h3, h4])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, neg_sample=h4)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, neg_sample=h3)
            else:  # multiple graphs
                assert all(v is not None for v in [h1, h2, g1, g2, batch])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=batch)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=batch)

        pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


class InfoNCESP(Loss):
    """
    InfoNCE loss for single positive.
    """

    def __init__(self, tau):
        super(InfoNCESP, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        f = lambda x: torch.exp(x / self.tau)
        sim = f(_similarity(anchor, sample))  # anchor x sample
        assert sim.size() == pos_mask.size()  # sanity check

        neg_mask = 1 - pos_mask
        pos = (sim * pos_mask).sum(dim=1)
        neg = (sim * neg_mask).sum(dim=1)

        loss = pos / (pos + neg)
        loss = -torch.log(loss)

        return loss.mean()


class InfoNCE(Loss):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()


class DebiasedInfoNCE(Loss):
    def __init__(self, tau, tau_plus=0.1):
        super(DebiasedInfoNCE, self).__init__()
        self.tau = tau
        self.tau_plus = tau_plus

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        num_neg = neg_mask.int().sum()
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim)

        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        pos = pos_sum / pos_mask.int().sum(dim=1)
        neg_sum = (exp_sim * neg_mask).sum(dim=1)
        ng = (-num_neg * self.tau_plus * pos + neg_sum) / (1 - self.tau_plus)
        ng = torch.clamp(ng, min=num_neg * np.e ** (-1. / self.tau))

        log_prob = sim - torch.log((pos + ng).sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return loss.mean()


class HardnessInfoNCE(Loss):
    def __init__(self, tau, tau_plus=0.1, beta=1.0):
        super(HardnessInfoNCE, self).__init__()
        self.tau = tau
        self.tau_plus = tau_plus
        self.beta = beta

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        num_neg = neg_mask.int().sum()
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim)

        pos = (exp_sim * pos_mask).sum(dim=1) / pos_mask.int().sum(dim=1)
        imp = torch.exp(self.beta * (sim * neg_mask))
        reweight_neg = (imp * (exp_sim * neg_mask)).sum(dim=1) / imp.mean(dim=1)
        ng = (-num_neg * self.tau_plus * pos + reweight_neg) / (1 - self.tau_plus)
        ng = torch.clamp(ng, min=num_neg * np.e ** (-1. / self.tau))

        #         log_prob = sim - torch.log((pos + ng).sum(dim=1, keepdim=True)) # Throws exception
        log_prob = sim - torch.log((pos + ng).sum())
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return loss.mean()


class HardMixingLoss(torch.nn.Module):
    def __init__(self, projection):
        super(HardMixingLoss, self).__init__()
        self.projection = projection

    @staticmethod
    def tensor_similarity(z1, z2):
        z1 = F.normalize(z1, dim=-1)  # [N, d]
        z2 = F.normalize(z2, dim=-1)  # [N, s, d]
        return torch.bmm(z2, z1.unsqueeze(dim=-1)).squeeze()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, threshold=0.1, s=150, mixup=0.2, *args, **kwargs):
        f = lambda x: torch.exp(x / self.tau)
        num_samples = z1.shape[0]
        device = z1.device

        threshold = int(num_samples * threshold)

        refl1 = _similarity(z1, z1).diag()
        refl2 = _similarity(z2, z2).diag()
        pos_similarity = f(_similarity(z1, z2))
        neg_similarity1 = torch.cat([_similarity(z1, z1), _similarity(z1, z2)], dim=1)  # [n, 2n]
        neg_similarity2 = torch.cat([_similarity(z2, z1), _similarity(z2, z2)], dim=1)
        neg_similarity1, indices1 = torch.sort(neg_similarity1, descending=True)
        neg_similarity2, indices2 = torch.sort(neg_similarity2, descending=True)
        neg_similarity1 = f(neg_similarity1)
        neg_similarity2 = f(neg_similarity2)
        z_pool = torch.cat([z1, z2], dim=0)
        hard_samples1 = z_pool[indices1[:, :threshold]]  # [N, k, d]
        hard_samples2 = z_pool[indices2[:, :threshold]]
        hard_sample_idx1 = torch.randint(hard_samples1.shape[1], size=[num_samples, 2 * s]).to(device)  # [N, 2 * s]
        hard_sample_idx2 = torch.randint(hard_samples2.shape[1], size=[num_samples, 2 * s]).to(device)
        hard_sample_draw1 = hard_samples1[
            torch.arange(num_samples).unsqueeze(-1), hard_sample_idx1]  # [N, 2 * s, d]
        hard_sample_draw2 = hard_samples2[torch.arange(num_samples).unsqueeze(-1), hard_sample_idx2]
        hard_sample_mixing1 = mixup * hard_sample_draw1[:, :s, :] + (1 - mixup) * hard_sample_draw1[:, s:, :]
        hard_sample_mixing2 = mixup * hard_sample_draw2[:, :s, :] + (1 - mixup) * hard_sample_draw2[:, s:, :]

        h_m1 = self.projection(hard_sample_mixing1)
        h_m2 = self.projection(hard_sample_mixing2)

        neg_m1 = f(self.tensor_similarity(z1, h_m1)).sum(dim=1)
        neg_m2 = f(self.tensor_similarity(z2, h_m2)).sum(dim=1)
        pos = pos_similarity.diag()
        neg1 = neg_similarity1.sum(dim=1)
        neg2 = neg_similarity2.sum(dim=1)
        loss1 = -torch.log(pos / (neg1 + neg_m1 - refl1))
        loss2 = -torch.log(pos / (neg2 + neg_m2 - refl2))
        loss = (loss1 + loss2) * 0.5
        loss = loss.mean()
        return loss


class RingLoss(torch.nn.Module):
    def __init__(self):
        super(RingLoss, self).__init__()

    def forward(self, h1: torch.Tensor, h2: torch.Tensor, y: torch.Tensor, tau, threshold=0.1, *args, **kwargs):
        f = lambda x: torch.exp(x / tau)
        num_samples = h1.shape[0]
        device = h1.device
        threshold = int(num_samples * threshold)

        false_neg_mask = torch.zeros((num_samples, 2 * num_samples), dtype=torch.int).to(device)
        for i in range(num_samples):
            false_neg_mask[i] = (y == y[i]).repeat(2)

        pos_sim = f(_similarity(h1, h2))
        neg_sim1 = torch.cat([_similarity(h1, h1), _similarity(h1, h2)], dim=1)  # [n, 2n]
        neg_sim2 = torch.cat([_similarity(h2, h1), _similarity(h2, h2)], dim=1)
        neg_sim1, indices1 = torch.sort(neg_sim1, descending=True)
        neg_sim2, indices2 = torch.sort(neg_sim2, descending=True)

        y_repeated = y.repeat(2)
        false_neg_cnt = torch.zeros((num_samples)).to(device)
        for i in range(num_samples):
            false_neg_cnt[i] = (y_repeated[indices1[i, threshold:-threshold]] == y[i]).sum()

        neg_sim1 = f(neg_sim1[:, threshold:-threshold])
        neg_sim2 = f(neg_sim2[:, threshold:-threshold])

        pos = pos_sim.diag()
        neg1 = neg_sim1.sum(dim=1)
        neg2 = neg_sim2.sum(dim=1)

        loss1 = -torch.log(pos / neg1)
        loss2 = -torch.log(pos / neg2)

        loss = (loss1 + loss2) * 0.5
        loss = loss.mean()

        return loss

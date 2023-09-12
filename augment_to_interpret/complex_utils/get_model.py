"""
Contain functions to get the downstream model needed for the experiments and use it.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import InstanceNorm

from ..models import GIN, PNA, SPMotifNet, GINNode, ZincEncoder, MoleculeEncoder


def get_model(x_dim, edge_attr_dim, num_class, multi_label, model_config, device):
    if model_config['model_name'] == 'GIN':
        model = GIN(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    elif model_config['model_name'] == 'GIN_node':
        model = GINNode(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    elif model_config['model_name'] == 'PNA':
        logging.warning("PNA has not been adapted to the batchnorm switch.")
        model = PNA(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    elif model_config['model_name'] == 'SPMotifNet':
        logging.warning("SPMotifNet has not been adapted to the batchnorm switch.")
        model = SPMotifNet(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    elif model_config['model_name'] == "ZincEncoder":
        model = ZincEncoder(model_config['num_atom_type'],
                            model_config['num_bond_type'],
                            emb_dim=model_config['hidden_size'],
                            num_gc_layers=model_config['num_gc_layers'],
                            drop_ratio=model_config['drop_ratio'],
                            pooling_type=model_config['pooling_type'])
    elif model_config['model_name'] == "MoleculeEncoder":
        model = MoleculeEncoder(emb_dim=model_config['hidden_size'],
                                num_gc_layers=model_config['num_gc_layers'],
                                drop_ratio=model_config['drop_ratio'],
                                pooling_type=model_config['pooling_type'])
    else:
        raise ValueError('[ERROR] Unknown model name!')
    return model.to(device)


class Criterion(nn.Module):
    def __init__(self, num_class, multi_label):
        super(Criterion, self).__init__()
        self.num_class = num_class
        self.multi_label = multi_label
        print(f'[INFO] Using multi_label: {self.multi_label}')

    def forward(self, logits, targets):

        # print("TS", targets.shape)
        # print("LS", logits.shape)

        if self.num_class is None:  # Regression task
            return F.mse_loss(logits, targets)
        if self.num_class <= 1:
            raise ValueError("The task should have at least two classes or be "
                             "a regression task (with num_class set to None).")
        if self.multi_label:  # Multilabel classification
            is_labeled = targets == targets  # mask for labeled data
            return F.binary_cross_entropy_with_logits(logits[is_labeled], targets[is_labeled].float())
        if self.num_class == 2:  # Binary classificaty
            return F.binary_cross_entropy_with_logits(logits, targets.float().view(logits.shape))
        if self.num_class > 2:  # N-label classification
            return F.cross_entropy(logits, targets.long())


def get_preds(logits, multi_label):
    if multi_label:
        preds = (logits.sigmoid() > 0.5).float()
    elif logits.shape[1] > 1:  # multi-class
        preds = logits.argmax(dim=1).float()
    else:  # binary
        preds = (logits.sigmoid() > 0.5).float()
    return preds


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)


def init_layer(layer: torch.nn.Linear, w_scale=1.0) -> torch.nn.Linear:
    torch.nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)  # type: ignore
    torch.nn.init.constant_(layer.bias.data, 0)
    return layer

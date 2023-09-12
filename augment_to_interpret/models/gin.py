# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
import numpy as np

from .batchnorm_switch import BatchNormSwitch
from .conv_layers import GINConv, GINEConv


class GIN(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config, use_fc_out=True):
        super().__init__()

        self.n_layers = model_config['n_layers']
        hidden_size = model_config['hidden_size']
        self.edge_attr_dim = edge_attr_dim
        self.dropout_p = model_config['dropout_p']
        self.use_edge_attr = model_config.get('use_edge_attr', True)
        self.out_graph_dim = hidden_size
        self.out_node_dim = hidden_size
        self.use_fc_out = use_fc_out
        if model_config.get('atom_encoder', False):
            from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder  # Imported here as it may cause problems depending on the versions (not entering this "if" is safe)
            self.node_encoder = AtomEncoder(emb_dim=hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = BondEncoder(emb_dim=hidden_size)
        else:
            self.node_encoder = Linear(x_dim, hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = Linear(edge_attr_dim, hidden_size)

        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_add_pool

        for _ in range(self.n_layers):
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.convs.append(GINEConv(GIN.MLP(hidden_size, hidden_size), edge_dim=hidden_size))
            else:
                self.convs.append(GINConv(GIN.MLP(hidden_size, hidden_size)))

        n_out = 1 if num_class is None or (num_class == 2 and not multi_label) else num_class
        self.fc_out = nn.Sequential(nn.Linear(hidden_size, n_out))

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None, return_emb=True, return_x=False):
        x = self.node_encoder(x.float())
        if edge_attr is not None:
            assert self.use_edge_attr, ("It shouldn't happen that edge_attr is not None if self.use_edge_attr "
                                        "is False at this point in the code.")
            edge_attr = self.edge_encoder(edge_attr.float())

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        emb = self.pool(x, batch)
        logits = self.fc_out(emb)
        if return_x:
            return x, logits, emb
        if return_emb:
            return logits, emb
        return logits

    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            Linear(in_channels, out_channels),
            BatchNormSwitch(out_channels),
            nn.ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None, return_pool=False):
        x = self.node_encoder(x)
        if edge_attr is not None:
            assert self.use_edge_attr, ("It shouldn't happen that edge_attr is not None if self.use_edge_attr "
                                        "is False at this point in the code.")
            edge_attr = self.edge_encoder(edge_attr.float())

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        if return_pool:
            emb = self.pool(x, batch)
            return x, emb
        return x

    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))

    def get_embeddings(self, loader, device, is_rand_label=False):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                batch, x, edge_index = data.batch, data.x, data.edge_index
                edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                _, x = self.forward(x, edge_index, batch, edge_atten=edge_weight)

                ret.append(x.cpu().numpy())
                if is_rand_label:
                    y.append(data.rand_label.cpu().numpy())
                else:
                    y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

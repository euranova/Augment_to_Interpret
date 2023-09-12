import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool

from . import BatchNormSwitch
from .adgcl_conv_layers import GINEConv


class AdgclMoleculeEncoder(torch.nn.Module):
    """ Code highly copy-pasted from AD-GCL
    https://github.com/susheels/adgcl/blob/2605ef8f980934c28d545f2556af5cc6ff48ed18/unsupervised/encoder/molecule_encoder.py
    """

    def __init__(self, emb_dim=300, num_gc_layers=5, drop_ratio=0.0, pooling_type="standard", is_infograph=False):
        # Imported here as it may cause problems depending on the versions.
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

        super(AdgclMoleculeEncoder, self).__init__()

        self.pooling_type = pooling_type
        self.emb_dim = emb_dim
        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio
        self.is_infograph = is_infograph

        self.out_node_dim = self.emb_dim
        if self.pooling_type == "standard":
            self.out_graph_dim = self.emb_dim
        elif self.pooling_type == "layerwise":
            self.out_graph_dim = self.emb_dim * self.num_gc_layers
        else:
            raise NotImplementedError

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)

        for i in range(num_gc_layers):
            nn = Sequential(Linear(emb_dim, 2 * emb_dim), BatchNormSwitch(2 * emb_dim), ReLU(),
                            Linear(2 * emb_dim, emb_dim))
            conv = GINEConv(nn)
            bn = BatchNormSwitch(emb_dim)
            self.convs.append(conv)
            self.bns.append(bn)

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)

        # compute node embeddings using GNN
        xs = []
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index, edge_attr, edge_weight)
            x = self.bns[i](x)
            if i == self.num_gc_layers - 1:
                # remove relu for the last layer
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
            xs.append(x)

        # compute graph embedding using pooling
        if self.pooling_type == "standard":
            xpool = global_add_pool(x, batch)
            return xpool, x

        elif self.pooling_type == "layerwise":
            xpool = [global_add_pool(x, batch) for x in xs]
            xpool = torch.cat(xpool, 1)
            if self.is_infograph:
                return xpool, torch.cat(xs, 1)
            else:
                return xpool, x
        else:
            raise NotImplementedError

    def get_embeddings(self, loader, device, is_rand_label=False):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                batch, x, edge_index, edge_attr = data.batch, data.x, data.edge_index, data.edge_attr
                edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = AdgclMoleculeEncoder.forward(batch, x, edge_index, edge_attr, edge_weight)

                ret.append(x.cpu().numpy())
                if is_rand_label:
                    y.append(data.rand_label.cpu().numpy())
                else:
                    y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


class MoleculeEncoder(AdgclMoleculeEncoder):
    """ Wrapper to make it compatible with our pipeline. """

    def __init__(self, emb_dim=300, num_gc_layers=5, drop_ratio=0.0, pooling_type="standard", is_infograph=False):
        super().__init__(emb_dim=emb_dim, num_gc_layers=num_gc_layers, drop_ratio=drop_ratio, pooling_type=pooling_type,
                         is_infograph=is_infograph)
        self.fc_out = torch.nn.Sequential(torch.nn.Linear(emb_dim, 1))  # taken from GIN

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None, return_emb=True, return_x=False):
        xpool, x = super().forward(batch, x, edge_index, edge_attr.squeeze(dim=1), edge_weight=edge_atten)
        logits = self.fc_out(xpool)
        if return_x:
            return x, logits, xpool
        if return_emb:
            return logits, xpool
        return logits

    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None, return_pool=False):
        """ Needed for pipeline compatibility. """
        xpool, x = super().forward(batch, x, edge_index, edge_attr.squeeze(dim=1), edge_weight=edge_atten)
        if return_pool:
            return x, xpool
        return x

    def get_embeddings(self, loader, device, is_rand_label=False):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                batch, x, edge_index, edge_attr = data.batch, data.x, data.edge_index, data.edge_attr
                edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                _, emb = self.get_emb(x, edge_index, batch, edge_attr, edge_atten=edge_weight, return_pool=True)

                ret.append(emb.cpu().numpy())
                if is_rand_label:
                    y.append(data.rand_label.cpu().numpy())
                else:
                    y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

"""
Wrap GIN to make it usable by ADGCL.
"""

import numpy as np
import torch

from .gin import GIN


class GINWrapper(GIN):
    def __init__(
        self,
        num_dataset_features,
        emb_dim=64,
        num_gc_layers=5,
        drop_ratio=0.0,
        pooling_type="standard",
        is_infograph=False,
        num_class=2,
        use_edge_attr=None,
        edge_attr_dim=None,
    ):
        if use_edge_attr is None:
            raise NotImplementedError("The signature of GINWrapper has changed, please correct your code accordingly.")
        if use_edge_attr and edge_attr_dim is None:
            raise RuntimeError("edge_attr_dim has to be specified if use_edge_attr is True.")
        model_config = {
            "n_layers": num_gc_layers,
            "hidden_size": emb_dim,
            "dropout_p": drop_ratio,
            "use_edge_attr": use_edge_attr,
        }
        super(GINWrapper, self).__init__(
            x_dim=num_dataset_features,
            edge_attr_dim=edge_attr_dim,
            num_class=num_class,
            multi_label=False,
            model_config=model_config,
        )

        self.pooling_type = pooling_type
        self.emb_dim = emb_dim
        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio
        self.is_infograph = is_infograph

        self.out_node_dim = self.emb_dim
        if self.pooling_type == "standard":
            self.out_graph_dim = self.emb_dim
        elif self.pooling_type == "layerwise":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, batch, x, edge_index, edge_attr=None, edge_weight=None):
        assert self.pooling_type == "standard"
        if edge_weight is not None:
            assert len(edge_weight.shape) == 1
            edge_weight = edge_weight.unsqueeze(dim=1)

        x, logits, xpool = super().forward(
            x,
            edge_index,
            batch,
            edge_attr=edge_attr,
            edge_atten=edge_weight,
            return_emb=True,
            return_x=True,
        )
        return xpool, x

    def get_embeddings(self, loader, device, is_rand_label=False):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                batch, x, edge_index = data.batch, data.x, data.edge_index
                edge_weight = data.edge_weight if hasattr(data, "edge_weight") else None

                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(batch, x, edge_index, edge_weight)

                ret.append(x.cpu().numpy())
                if is_rand_label:
                    y.append(data.rand_label.cpu().numpy())
                else:
                    y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

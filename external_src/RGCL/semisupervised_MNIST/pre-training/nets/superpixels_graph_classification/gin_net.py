import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

from layers.gin_layer import GINLayer, ApplyNodeFunc, MLP
from layers.projection_head import projection_head
import pdb

class GINNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        n_mlp_layers = net_params['n_mlp_GIN']               # GIN
        learn_eps = net_params['learn_eps_GIN']              # GIN
        neighbor_aggr_type = net_params['neighbor_aggr_GIN'] # GIN
        readout = net_params['readout']                      # this is graph_pooling_type
        graph_norm = net_params['graph_norm']      
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']     
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        
        for layer in range(self.n_layers):
            mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, graph_norm, batch_norm, residual, 0, learn_eps))

        # Linear function for graph poolings (readout) of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        

        for layer in range(self.n_layers+1):
            self.linears_prediction.append(nn.Linear(hidden_dim, n_classes))

        self.projection_head = projection_head(hidden_dim, hidden_dim) 
        
        if readout == 'sum':
            self.pool = SumPooling()
        elif readout == 'mean':
            self.pool = AvgPooling()
        elif readout == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError
        
    def forward(self, g, h, e, snorm_n, snorm_e, mlp=True, head=False, return_graph=False):
        
        h = self.embedding_h(h)
        hidden_rep = [h]

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, snorm_n)
            hidden_rep.append(h)

        if return_graph:
            node_features = 0
            for feat in hidden_rep:
                node_features += feat
            g.ndata['h'] = node_features
            return g

        score_over_layer = 0
        vector_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            vector_over_layer += pooled_h
            score_over_layer += self.linears_prediction[i](pooled_h)
            
        if mlp:
            return score_over_layer
        else:
            if head:
                return self.projection_head(vector_over_layer)
            else:
                return vector_over_layer

    def forward_imp(self, g, h, snorm_n, node_imp):

        # mapping node_imp to [1,1.1]
        node_imp = node_imp/torch.max(node_imp)/10.0
        node_imp += 1.0

        h = self.embedding_h(h)
        hidden_rep = [h]

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, snorm_n)
            hidden_rep.append(h)

        vector_over_layer = 0

        for i, h in enumerate(hidden_rep):
            if i == len(hidden_rep) - 1:
                imp_h = torch.mul(h, node_imp)
            else:
                imp_h = h
            pooled_h = self.pool(g, imp_h)
            vector_over_layer += pooled_h

        return self.projection_head(vector_over_layer)

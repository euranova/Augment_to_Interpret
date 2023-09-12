"""
Wrap AD-GCL approach in modules that can be used in our pipeline.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter

from ..models.gin_wrapper import GINWrapper


class _GInfoMinMax(torch.nn.Module):
    def __init__(self, encoder, proj_hidden_dim=300):
        super(_GInfoMinMax, self).__init__()

        self.encoder = encoder
        self.input_proj_dim = self.encoder.out_graph_dim

        self.proj_head = Sequential(
            Linear(self.input_proj_dim, proj_hidden_dim), ReLU(inplace=True),
            Linear(proj_hidden_dim, proj_hidden_dim))

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, x, edge_index, edge_attr, edge_weight=None):

        z, node_emb = self.encoder(batch, x, edge_index, edge_attr,
                                   edge_weight)

        z = self.proj_head(z)
        # z shape -> Batch x proj_hidden_dim
        return z, node_emb

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True):
        # x and x_aug shape -> Batch x proj_hidden_dim

        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum(
            'i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:

            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = -torch.log(loss_0).mean()
            loss_1 = -torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = -torch.log(loss_1).mean()
            return loss_1

        return loss


class _ViewLearner(torch.nn.Module):
    def __init__(self, encoder, mlp_edge_model_dim=64):
        super(_ViewLearner, self).__init__()

        self.encoder = encoder
        self.input_dim = self.encoder.out_node_dim

        self.mlp_edge_model = Sequential(
            Linear(self.input_dim * 2, mlp_edge_model_dim), ReLU(),
            Linear(mlp_edge_model_dim, 1))
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, x, edge_index, edge_attr):

        _, node_emb = self.encoder(batch, x, edge_index, edge_attr)

        src, dst = edge_index[0], edge_index[1]
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]

        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb)

        return edge_logits


def run_one_epoch(data_loader, model, model_optimizer, view_learner, batch_size, view_optimizer,
                  reg_lambda, device, train=True):
    embed_l = []
    view_loss_all = 0
    model_loss_all = 0
    reg_all = 0
    all_clf_labels_l = []
    edge_att_l = []
    gt_edges_label_l = []
    for batch in data_loader:  # TODO: write a function which will run for whole epoch on train test and valid datasets
        # train view to maximize contrastive loss
        batch = batch.to(device)
        view_learner.train()
        view_learner.zero_grad()
        model.eval()

        x, _ = model(batch.batch, batch.x, batch.edge_index, edge_attr=batch.edge_attr, edge_weight=None)

        edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, edge_attr=batch.edge_attr)
        if train:
            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias -
                   (1 - bias)) * torch.rand_like(edge_logits) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + edge_logits) / temperature
        else:
            gate_inputs = edge_logits

        batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

        x_aug, _ = model(batch.batch, batch.x, batch.edge_index, edge_attr=batch.edge_attr,
                         edge_weight=batch_aug_edge_weight)
        embed_l.append(x)
        all_clf_labels_l.append(batch.y)
        edge_att_l.append(batch_aug_edge_weight)
        if getattr(batch, "edge_label", None) is not None:
            gt_edges_label_l.append(batch.edge_label.data.cpu())

        if train:

            # regularization

            row, col = batch.edge_index
            edge_batch = batch.batch[row]
            edge_drop_out_prob = 1 - batch_aug_edge_weight

            uni, edge_batch_num = edge_batch.unique(return_counts=True)
            sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")

            reg = []
            for b_id in range(batch_size):
                if b_id in uni:
                    num_edges = edge_batch_num[uni.tolist().index(b_id)]
                    reg.append(sum_pe[b_id] / num_edges)
                else:
                    # means no edges in that graph. So don't include.
                    pass
            num_graph_with_edges = len(reg)
            reg = torch.stack(reg)
            reg = reg.mean()

            view_loss = model.calc_loss(x, x_aug) - (reg_lambda * reg)
            view_loss_all += view_loss.item() * batch.num_graphs
            reg_all += reg.item()
            # gradient ascent formulation
            (-view_loss).backward()
            view_optimizer.step()

            # train (model) to minimize contrastive loss
            model.train()
            view_learner.eval()
            model.zero_grad()

            x, _ = model(batch.batch, batch.x, batch.edge_index, edge_attr=batch.edge_attr, edge_weight=None)
            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, edge_attr=batch.edge_attr)
            if train:
                temperature = 1.0
                bias = 0.0 + 0.0001  # If bias is 0, we run into problems
                eps = (bias -
                       (1 - bias)) * torch.rand_like(edge_logits) + (1 - bias)
                gate_inputs = torch.log(eps) - torch.log(1 - eps)
                gate_inputs = (gate_inputs + edge_logits) / temperature

            else:
                gate_inputs = edge_logits
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()
            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, edge_attr=batch.edge_attr,
                             edge_weight=batch_aug_edge_weight)

            model_loss = model.calc_loss(x, x_aug)
            model_loss_all += model_loss.item() * batch.num_graphs
            # standard gradient descent formulation
            model_loss.backward()
            model_optimizer.step()
    embs = torch.cat(embed_l, dim=0)
    all_clf_labels = torch.cat(all_clf_labels_l)
    edge_att = torch.cat(edge_att_l)
    gt_edges_label = torch.cat(gt_edges_label_l).detach() if len(gt_edges_label_l) > 0 else None
    return embs.detach(), all_clf_labels, edge_att.detach(), gt_edges_label


def _eval_one_batch(data, model, view_learner):
    model.eval()
    view_learner.eval()

    x, _ = model(data.batch, data.x, data.edge_index, edge_attr=data.edge_attr, edge_weight=None)
    edge_logits = view_learner(data.batch, data.x, data.edge_index, edge_attr=data.edge_attr)

    batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()

    edge_labels = data.edge_label.data.cpu().detach() if hasattr(data, "edge_label") else None
    return x.detach(), data.y, batch_aug_edge_weight.detach(), edge_labels


def save_adcgl_models(model, view_learner, saving_path, final_dic, model_name="best downstream model"):
    print(f"Saving {model_name}")
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path)
    torch.save(
        model.state_dict(), os.path.join(saving_path, "GInfoMinMax_model"))
    torch.save(
        view_learner.state_dict(), os.path.join(saving_path, "ViewLearner_model"))
    with open(os.path.join(saving_path, "final_res_dictionary.pkl"), 'wb') as f:
        pickle.dump(final_dic, f)


class ADGCLClusteringEmbeddings():
    def __init__(self, model, view_learner):
        self.clf = model
        self.view_learner = view_learner
        self.pool = global_add_pool

    def eval(self):
        self.clf.eval()
        self.view_learner.eval()

    def forward_pass(self, data, epoch, training=False, current_mask=None):
        if training or current_mask is not None:
            raise NotImplementedError

        embs, all_clf_labels, edge_att, gt_edges_label = _eval_one_batch(data, self.clf,
                                                                         self.view_learner)

        return (edge_att.detach().unsqueeze(dim=1), None, gt_edges_label.detach(),
                all_clf_labels.detach(), embs.detach())

    def get_embeddings(self, loader, device, is_rand_label=False):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                edge_att, _, gt_edges_label, all_clf_labels, embs = self.forward_pass(data, epoch=0, training=False)

                ret.append(embs.cpu().numpy())
                y.append(all_clf_labels.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y


def get_adgcl_models(args, x_dim, edge_attr_dim, device):
    model = _GInfoMinMax(GINWrapper(
            num_dataset_features=x_dim,
            emb_dim=args["emb_dim"],
            num_gc_layers=args["num_gc_layers"],
            drop_ratio=args["drop_ratio"],
            pooling_type=args["pooling_type"],
            use_edge_attr=args["use_edge_attr"],
            edge_attr_dim=edge_attr_dim,
        ), args["emb_dim"]).to(device)
    view_learner = _ViewLearner(GINWrapper(
            num_dataset_features=x_dim,
            emb_dim=args["emb_dim"],
            num_gc_layers=args["num_gc_layers"],
            drop_ratio=args["drop_ratio"],
            pooling_type=args["pooling_type"],
            use_edge_attr=args["use_edge_attr"],
            edge_attr_dim=edge_attr_dim,
        ), mlp_edge_model_dim=args["mlp_edge_model_dim"]).to(device)

    return model, view_learner


def load_adgcl_models_inplace(path, model, view_learner, device):
    loading_path_view_learner = Path(path, "ViewLearner_model")
    loading_path_mmodel = Path(path, "GInfoMinMax_model")

    model_statedict = torch.load(loading_path_mmodel, map_location=device)
    model.load_state_dict(model_statedict)
    model.eval()

    vw_statedict = torch.load(loading_path_view_learner, map_location=device)
    view_learner.load_state_dict(vw_statedict)
    view_learner.eval()

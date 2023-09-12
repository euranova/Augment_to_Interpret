"""
Wrap MEGA approach in modules that can be used in our pipeline.
"""

import os
import pickle
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter

from ..basic_utils import C, get_device

if str(C.PATH_EXTERNAL_SRC) not in sys.path:
    sys.path.append(str(C.PATH_EXTERNAL_SRC))
    print(f"Added {C.PATH_EXTERNAL_SRC} to the python_path.")

from MEGA.LGA_Lib.encoder import TUEncoder, TUEncoder_sd
from MEGA.LGA_Lib.LGA_learner import LGALearner
from MEGA.LGA_Lib.learning import MModel, MModel_sd


class _TUEncoderSdWrapper(TUEncoder_sd):
    def __init__(self, *args, device=None, **kwargs):
        super().__init__(*args, **kwargs)
        if device is None:
            device, C.CUDA_ID = get_device(C.CUDA_ID)
            warnings.warn(f"It is safer to explicitly set device to initialise TUEncoderSdWrapper. "
                          f"Using {device} by default.")
        self.device = device


class _TUEncoderWrapper(TUEncoder):
    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None, return_pool=True):
        xpool, x = self.forward(batch, x, edge_index, edge_attr=edge_attr, edge_weight=edge_atten)
        return xpool


class MEGAClusteringEmbeddings():
    def __init__(self, mmodel, view_learner):
        self.clf = mmodel
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

        return edge_att.detach().unsqueeze(dim=1), None, gt_edges_label.detach(), all_clf_labels.detach(), embs.detach()

    def get_embeddings(self, loader, device, is_rand_label=False):
        del is_rand_label  # Unused by mega
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

    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None, return_pool=False):
        x, _ = self.clf(batch, x, edge_index, edge_attr=edge_attr, edge_weight=edge_atten)
        if return_pool:
            emb = self.pool(x, batch)
            return x, emb
        return x


def run_one_epoch(data_loader, model, model_sd, model_optimizer, LGA_learner,
                  batch_size, LGA_lr, reg_expect, LGA_optimizer, device, epoch=0, train=True):
    eval_interval = 5
    embed_l = []
    all_clf_labels_l = []
    edge_att_l = []
    gt_edges_label_l = []

    model_losses = []
    LGA_losses = []
    LGA_regs = []

    model_loss_all = 0
    LGA_loss_all = 0
    reg_all = 0

    for batch in data_loader:  # TODO: write a function which will run for whole epoch on train test and valid datasets
        # train view to maximize contrastive loss
        batch = batch.to(device)
        model.train()
        LGA_learner.eval()
        model.zero_grad()
        x, x_pooled = model(batch.batch, batch.x, batch.edge_index, edge_attr=batch.edge_attr, edge_weight=None)
        edge_logits = LGA_learner(batch.batch, batch.x, batch.edge_index, edge_attr=batch.edge_attr)

        if train:
            bias = 0.0001
            eps = (bias - (1 - bias)) * torch.rand_like(edge_logits) + (1 - bias)
            edge_score = torch.log(eps) - torch.log(1 - eps)
            edge_score = (edge_score + edge_logits)
            batch_aug_edge_weight = torch.sigmoid(edge_score).squeeze().detach()
        else:
            batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze().detach()

        x_aug, _ = model(batch.batch, batch.x, batch.edge_index, edge_attr=batch.edge_attr,
                         edge_weight=batch_aug_edge_weight)

        embed_l.append(x)
        all_clf_labels_l.append(batch.y)
        edge_att_l.append(batch_aug_edge_weight)
        if getattr(batch, "edge_label", None) is not None:
            gt_edges_label_l.append(batch.edge_label.data.cpu())

        if train:

            model_loss = model.calc_loss(x, x_aug)
            model_loss_all += model_loss.item() * batch.num_graphs
            model_loss.backward()
            model_optimizer.step()

            # ========================train LGA======================== #
            LGA_learner.train()
            LGA_learner.zero_grad()
            model.eval()

            x, _ = model(batch.batch, batch.x, batch.edge_index, edge_attr=batch.edge_attr, edge_weight=None)

            edge_logits = LGA_learner(batch.batch, batch.x, batch.edge_index, edge_attr=batch.edge_attr)

            bias = 0.0001
            eps = (bias - (1 - bias)) * torch.rand_like(edge_logits) + (1 - bias)
            edge_score = torch.log(eps) - torch.log(1 - eps)
            edge_score = (edge_score + edge_logits)
            batch_aug_edge_weight = torch.sigmoid(edge_score).squeeze()

            row, col = batch.edge_index
            edge_batch = batch.batch[row]

            uni, edge_batch_num = edge_batch.unique(return_counts=True)
            sum_pe = scatter((1 - batch_aug_edge_weight), edge_batch, reduce="sum")

            reg = []
            for b_id in range(batch_size):
                if b_id in uni:
                    num_edges = edge_batch_num[uni.tolist().index(b_id)]
                    reg.append(sum_pe[b_id] / num_edges)
                else:
                    pass
            reg = torch.stack(reg)
            reg = reg.mean()
            ratio = reg / reg_expect

            batch_aug_edge_weight = batch_aug_edge_weight / ratio  # edge weight generalization

            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, edge_attr=batch.edge_attr,
                             edge_weight=batch_aug_edge_weight)

            model_loss = model.calc_loss(x, x_aug)

            # current parameter
            fast_weights = OrderedDict((name, param) for (name, param) in model.named_parameters())

            # create_graph flag for computing second-derivative
            grads = torch.autograd.grad(model_loss, model.parameters(), create_graph=True, allow_unused=True)
            data = [p.data for p in list(model.parameters())]
            # compute parameter' by applying sgd on multi-task loss
            # compute parameter' by applying sgd on multi-task loss
            fast_weights = OrderedDict(
                (name, param - LGA_lr * grad)
                for ((name, param), grad,
                     data) in zip(fast_weights.items(), grads, data)
                if not name.startswith('encoder.fc_out'))

            # compute primary loss with the updated parameter'
            x, _ = model_sd.forward(batch.batch, batch.x, batch.edge_index, edge_attr=batch.edge_attr, edge_weight=None,
                                    weights=fast_weights)
            x_aug, _ = model_sd.forward(batch.batch, batch.x, batch.edge_index, edge_attr=batch.edge_attr,
                                        edge_weight=batch_aug_edge_weight.detach(),
                                        weights=fast_weights)
            LGA_loss = 0.1 * model.calc_feature_loss(x, x_aug) + model.calc_instance_loss(x, x_aug)

            LGA_loss_all += LGA_loss.item() * batch.num_graphs
            reg_all += reg.item()
            LGA_loss.backward()
            LGA_optimizer.step()

            fin_model_loss = model_loss_all / len(data_loader)
            fin_LGA_loss = LGA_loss_all / len(data_loader)
            fin_reg = reg_all / len(data_loader)

            print('Epoch {}, Model Loss {}, LGA Loss {}'.format(epoch, fin_model_loss, fin_LGA_loss))
            model_losses.append(fin_model_loss)
            LGA_losses.append(fin_LGA_loss)
            LGA_regs.append(fin_reg)
            if epoch % eval_interval == 0:
                model.eval()

    embs = torch.cat(embed_l, dim=0)
    all_clf_labels = torch.cat(all_clf_labels_l)
    edge_att = torch.cat(edge_att_l)
    gt_edges_label = torch.cat(gt_edges_label_l).detach() if len(gt_edges_label_l) > 0 else None
    return embs.detach(), all_clf_labels, edge_att.detach(), gt_edges_label


def save_mega_models(model, model_sd, LGA_learner, saving_path, final_dic, model_name="best downstream model"):
    print(f"Saving {model_name}")
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path)
    torch.save(model.state_dict(), Path(saving_path, "MEGA_model"))
    torch.save(LGA_learner.state_dict(), Path(saving_path, "LGALearner_model"))
    with open(Path(saving_path, 'final_res_dictionary.pkl'), 'wb') as f:
        pickle.dump(final_dic, f)
    torch.save(model_sd.state_dict(), Path(saving_path, "model_sd"))


def get_mega_models(args, x_dim, device):
    def get_encoder(class_, **kwargs):
        return class_(
            num_dataset_features=x_dim,
            emb_dim=args["emb_dim"],
            num_gc_layers=args["num_gc_layers"],
            drop_ratio=args["drop_ratio"],
            pooling_type=args["pooling_type"],
            **kwargs,
        )

    model = MModel(get_encoder(_TUEncoderWrapper), args["emb_dim"]).to(device)
    model_sd = MModel_sd(get_encoder(_TUEncoderSdWrapper, device=device),
                         args["emb_dim"], device=device).to(device)
    LGA_learner = LGALearner(get_encoder(_TUEncoderWrapper),
                             mlp_edge_model_dim=args["mlp_edge_model_dim"]).to(device)
    return model, model_sd, LGA_learner


def load_mega_model_inplace(model, model_sd, LGA_learner, loading_path, device):
    loading_path_model = Path(loading_path, "MEGA_model")
    loading_path_model_sd = Path(loading_path, "model_sd")
    loading_path_LGA_learner = Path(loading_path, "LGALearner_model")

    model_statedict = torch.load(loading_path_model, map_location=device)
    model.load_state_dict(model_statedict)
    model.eval()
    model_sd_statedict = torch.load(loading_path_model_sd, map_location=device)
    model_sd.load_state_dict(model_sd_statedict)
    model_sd.eval()
    LGA_learner_statedict = torch.load(loading_path_LGA_learner, map_location=device)
    LGA_learner.load_state_dict(LGA_learner_statedict)
    LGA_learner.eval()


def _eval_one_batch(batch, model, LGA_learner):
    model.eval()
    LGA_learner.eval()

    x, _ = model(batch.batch, batch.x, batch.edge_index, edge_attr=batch.edge_attr, edge_weight=None)
    edge_logits = LGA_learner(batch.batch, batch.x, batch.edge_index, edge_attr=batch.edge_attr)

    batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()
    gt_edges_label = batch.edge_label.data.cpu().detach() if hasattr(batch, "edge_label") else None
    return x.detach(), batch.y, batch_aug_edge_weight.detach(), gt_edges_label

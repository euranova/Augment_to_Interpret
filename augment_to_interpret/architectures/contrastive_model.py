"""
Contain our model and some of its building bricks.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from ..basic_utils.helpers import get_laplacian_top_eigenvalues
from ..complex_utils import MLP, init_layer
from ..losses.contrastive_loss import simclr, barlow_twins, rationale_loss
from ..losses.contrastive_sampler import DualBranchContrast, InfoNCE, HardnessInfoNCE
from ..models import BatchNormSwitch

EPS = 1e-15


def _sampling(att_log_logit, training, temperature=1):
    if training:
        random_noise = torch.empty_like(att_log_logit).uniform_(
            1e-10, 1 - 1e-10)
        random_noise = torch.log(random_noise) - torch.log(1.0 -
                                                           random_noise)
        att_bern = ((att_log_logit + random_noise) / temperature).sigmoid()
    else:
        att_bern = (att_log_logit).sigmoid()
    return att_bern


def _get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
    r = init_r - current_epoch // decay_interval * decay_r
    if r < final_r:
        r = final_r
    return r


def _lift_node_att_to_edge_att(node_att, edge_index):
    src_lifted_att = node_att[edge_index[0]]
    dst_lifted_att = node_att[edge_index[1]]
    edge_att = src_lifted_att * dst_lifted_att
    return edge_att


class ModelNodeAttention(nn.Module):
    """Model level node feature attention.

    Args:
        nn (_type_): _description_
    """

    def __init__(self, nb_features):
        super(ModelNodeAttention, self).__init__()
        self.weight = nn.Parameter(torch.ones(nb_features))

    def forward(self, x):
        return x * F.gumbel_softmax(self.weight, hard=False, dim=0, tau=0.5)


class InstanceNodeAttention(nn.Module):

    def __init__(self, hidden_size, use_sigmoid=True):
        super().__init__()
        self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size], dropout=0.5)
        self.use_sigmoid = use_sigmoid

    def forward(self, x, batch=None, train=False):
        att_log_logits = self.feature_extractor(x, batch)
        att_log_logits = _sampling(att_log_logits, train)
        # if self.use_sigmoid:
        # att_log_logits = att_log_logits.sigmoid()
        new_x = x * att_log_logits
        return new_x, att_log_logits


class ContrastiveModel(nn.Module):
    def __init__(self,
                 clf,
                 extractor,
                 criterion,
                 optimizer,
                 optimizer_features,
                 task,
                 watchman=None,
                 loss_type='simclr_info_negative',
                 learn_edge_att=True,
                 use_watchman=False,
                 final_r=0.7,
                 decay_interval=10,
                 decay_r=0.1,
                 temperature_edge_sampling=1,
                 w_info_loss=1,
                 top_eigenvalues_nb=10,
                 watchman_lambda=.2,
                 features_extractor=None
                 ):
        super().__init__()

        if use_watchman:
            assert watchman is not None, (
                "If use_watchman is True, you must specify a watchman as parameter.")
        self.optimizer_features = optimizer_features
        self.features_extractor = features_extractor
        self.task = task
        self.extractor = extractor
        self.criterion = criterion
        self.watchman = watchman
        self.optimizer = optimizer
        self.device = next(self.parameters()).device
        self.w_info_loss = w_info_loss
        self.top_eigenvalues_nb = top_eigenvalues_nb
        self.watchman_lambda = watchman_lambda

        self.learn_edge_att = learn_edge_att
        self.use_watchman = use_watchman
        self.final_r = final_r
        self.decay_interval = decay_interval
        self.decay_r = decay_r
        self.temperature_edge_sampling = temperature_edge_sampling

        self.loss_method, self.norm_name = self.get_loss_norm(loss_type)
        self.negative_term = ('neg' in loss_type)
        self.double_augmentation = ('double_aug' in loss_type)
        if 'gsat' not in loss_type:
            print('Freezing classification head')
            for param in clf.fc_out.parameters():  # Shouldn't be useful, but let's be sure
                param.requires_grad = False
        else:
            print('Training classification head')
        self.clf = clf

        if self.loss_method in ['infonce', 'infonce_info']:
            self.dual_branch_contrast = DualBranchContrast(
                loss=InfoNCE(tau=0.2), mode='L2L',
                intraview_negs=True).to(self.device)

        if self.loss_method in ['hardinfonce', 'hardinfonce_info',
                                'hardinfonce_double_aug', 'hardinfonce_double_aug_info']:
            self.dual_branch_contrast = DualBranchContrast(
                loss=HardnessInfoNCE(tau=0.2), mode='L2L',
                intraview_negs=True).to(self.device)

    def get_loss_norm(self, loss_type):
        # the loss normalization is specified in the name after norm
        # e.g. 'ssc_info_norm_softmax'
        loss_tokens = loss_type.split('_norm_')
        loss_method = loss_tokens[0]
        norm_name = 'l2'
        if len(loss_tokens) == 2:
            norm_name = loss_tokens[1]
        return loss_method, norm_name

    def normalize(self, data, norm_name):
        if norm_name == 'l2':
            return F.normalize(data, dim=1)

        if norm_name == 'softmax':
            return F.softmax(data, dim=1)

        if norm_name == 'sigmoid':
            return F.sigmoid(data)

    def get_criterion_value(self, clf_logits, clf_labels, current_mask):
        if clf_logits is None:
            return 0
        if "graph" in self.task:
            ce_loss = self.criterion(clf_logits, clf_labels)
        else:

            # print("CS", current_mask.shape)
            # print("LO", clf_logits.shape)
            # print("LA", clf_labels.shape)

            ce_loss = self.criterion(
                clf_logits[current_mask],  # .view(-1, 1),
                clf_labels[current_mask]  # .view(-1, 1)
            )
        return ce_loss

    def one_sample_negative_loss(self, z_neg, norm_z):
        # add to self-self supervised contrastive loss the distance btw
        # a sample and its negative
        norm_z_neg = self.normalize(z_neg, self.norm_name)
        return torch.tensor([
            simclr(*(torch.stack([norm_z_i, norm_z_neg_i]) for _ in range(2)))
            for norm_z_i, norm_z_neg_i in zip(norm_z, norm_z_neg)
        ], device=norm_z.device).sum()

    def __loss__(self, att, clf_logits, clf_labels, epoch, z, z_aug, z_neg,
                 batch, watchman_prediction, graph_laplacian_top_eigenvalues, current_mask=None,
                 current_edge_mask=None):
        """ Compute the loss
        NOTE: The type of loss used and calculated depends on flags set in the self, it is not regulated by this function's arguments!

        :param att: tensor(Float); the attention values to be used for the info_loss
        :param clf_logits: tensor(Float); the predicted values to be used for the ce_loss
        :param clf_labels: tensor(Int | Float); the ground_truth labels to be used for the ce_loss and the supervised simclr losses (Float iff regression)
        :param epoch: int; epoch of the training (used for the info_loss if r is not fixed)
        :param z: tensor(Float); the sample embedding (or an augmentation of it, depending on self)
        :param z_aug: tensor(Float); the embedding of an (other) augmentation of the sample
        :param z_neg: tensor(Float); the embedding of the negative augmentation corresponding to z_aug
        :param batch: tensor(Int); unused
        :param watchman_prediction: tensor(Float); prediction for the watchman task
        :param graph_laplacian_top_eigenvalues: tensor(Float); label for the watchman task (rmse applied)
        :param current_mask: tensor(Int); used for node classification only: which nodes to use in the loss
        :param current_edge_mask: tensor(bool); which edges to take into account for the info_loss
        :return: tensor(Float), Dict(str, Float); (the final loss, a dictionary of losses for logging purposes)
        """
        assert current_edge_mask is not None
        implemented_losses = [
            "gsat", "simclr_double_aug", "simclr_double_aug_info",
            "simclr_double_aug_negative", "simclr_double_aug_info_negative"
        ]
        if self.loss_method not in implemented_losses:
            warnings.warn(f"The loss {self.loss_method} is not part of the paper.")

        if current_mask is None:
            if "node" in self.task:
                warnings.warn("You did not specify current_mask even though it is a node task. "
                              "This behaviour is not intended. Use at your own risks.")
            current_mask = torch.ones(z.shape[0]).bool()
        elif "graph" in self.task:
            warnings.warn("You specified current_mask even though it is a graph task. "
                          "This behaviour is not intended. Use at your own risks.")

        norm_z = self.normalize(z[current_mask.bool()], self.norm_name)
        norm_z_aug = self.normalize(z_aug[current_mask.bool()], self.norm_name)
        if z_neg is not None:
            z_neg = z_neg[current_mask.bool()]
        r = _get_r(self.decay_interval,
                   self.decay_r,
                   epoch,
                   final_r=self.final_r)
        info_loss = (att[current_edge_mask.bool()] * torch.log(att[current_edge_mask.bool()] / r + 1e-6) +
                     (1 - att[current_edge_mask.bool()]) * torch.log((1 - att[current_edge_mask.bool()]) /
                                                                     (1 - r + 1e-6) + 1e-6)).mean()
        loss_dict = {}

        ce_loss = self.get_criterion_value(clf_logits, clf_labels, current_mask.bool())
        if self.loss_method == 'gsat':
            loss = ce_loss + self.w_info_loss * info_loss
            loss_dict['ce_loss'] = ce_loss.item()
            loss_dict['info_loss'] = info_loss.item()
        elif self.loss_method == 'ssc_gsat':
            ssc_loss = simclr(norm_z, norm_z_aug, labels=clf_labels)
            loss_dict['ce_loss'] = ce_loss.item()
            loss_dict['info_loss'] = info_loss.item()
            loss_dict['ssc_loss'] = ssc_loss.item()
            loss = ce_loss + ssc_loss + self.w_info_loss * info_loss
        elif self.loss_method == 'ssc_info':
            ssc_loss = simclr(norm_z, norm_z_aug, labels=clf_labels)
            loss_dict['info_loss'] = info_loss.item()
            loss_dict['ssc_loss'] = ssc_loss.item()
            loss = ssc_loss + self.w_info_loss * info_loss
        elif self.loss_method == 'ssc_gsat_negative':
            ssc_loss = simclr(norm_z, norm_z_aug, labels=clf_labels)
            neg_loss = self.one_sample_negative_loss(z_neg, norm_z)  # try with norm_z_aug

            loss_dict['ce_loss'] = ce_loss.item()
            loss_dict['info_loss'] = info_loss.item()
            loss_dict['ssc_loss'] = ssc_loss.item()
            loss_dict['neg_loss'] = neg_loss.item()
            loss = ce_loss + ssc_loss + self.w_info_loss * info_loss + neg_loss
        elif self.loss_method in ['infonce', 'hardinfonce', 'hardinfonce_double_aug']:
            loss = self.dual_branch_contrast(z, z_aug)
            loss_dict['infonce_loss'] = loss.item()
        elif self.loss_method in ['infonce_info', 'hardinfonce_info', 'hardinfonce_double_aug_info']:
            c_loss = self.dual_branch_contrast(z, z_aug)
            loss_dict['info_loss'] = info_loss.item()
            loss_dict['infonce_info'] = c_loss.item()
            loss = c_loss + self.w_info_loss * info_loss
        elif self.loss_method in ['simclr', 'simclr_double_aug']:
            loss = simclr(norm_z, norm_z_aug)
            loss_dict['simclr_loss'] = loss.item()
        elif self.loss_method in ['simclr_simclrnegative', 'simclr_double_aug_simclrnegative']:
            norm_z_neg = self.normalize(z_neg, self.norm_name)
            simclr_loss = simclr(norm_z, norm_z_aug)
            simclrnegative_loss = simclr(norm_z_neg, norm_z_aug)
            loss = simclr_loss - simclrnegative_loss
            loss_dict['simclr_loss'] = simclr_loss.item()
            loss_dict['neg_loss'] = simclrnegative_loss.item()
        elif self.loss_method in ['simclr_double_aug_info_simclrnegative']:
            norm_z_neg = self.normalize(z_neg, self.norm_name)
            simclr_loss = simclr(norm_z, norm_z_aug)
            simclrnegative_loss = simclr(norm_z_neg, norm_z_aug)
            loss = simclr_loss - simclrnegative_loss + self.w_info_loss * info_loss
            loss_dict['simclr_loss'] = simclr_loss.item()
            loss_dict['neg_loss'] = simclrnegative_loss.item()
        elif self.loss_method in ['barlow', 'barlow_double_aug']:
            loss = barlow_twins(z, z_aug)
            loss_dict['barlow_loss'] = loss.item()
        elif self.loss_method in ['simclr_info', 'simclr_double_aug_info']:
            c_loss = simclr(norm_z, norm_z_aug)
            loss_dict['info_loss'] = info_loss.item()
            loss_dict['simclr_loss'] = c_loss.item()
            loss = c_loss + self.w_info_loss * info_loss
        elif self.loss_method in ['barlow_info', 'barlow_double_aug_info']:
            c_loss = barlow_twins(z, z_aug)
            loss_dict['info_loss'] = info_loss.item()
            loss_dict['barlow_loss'] = c_loss.item()
            loss = c_loss + self.w_info_loss * info_loss
        elif self.loss_method in ['simclr_info_negative', 'simclr_double_aug_info_negative']:
            c_loss = simclr(norm_z, norm_z_aug)
            neg_loss = self.one_sample_negative_loss(z_neg, norm_z)  # try with norm_z_aug

            loss_dict['info_loss'] = info_loss.item()
            loss_dict['simclr_loss'] = c_loss.item()
            loss_dict['neg_loss'] = neg_loss.item()
            loss = c_loss + self.w_info_loss * info_loss + neg_loss
        elif self.loss_method in ['simclr_negative', 'simclr_double_aug_negative']:
            c_loss = simclr(norm_z, norm_z_aug)
            neg_loss = self.one_sample_negative_loss(z_neg, norm_z)  # try with norm_z_aug
            loss_dict['simclr_loss'] = c_loss.item()
            loss_dict['neg_loss'] = neg_loss.item()
            loss = c_loss + neg_loss
        elif self.loss_method in ['rationale_negative']:
            norm_z_neg = self.normalize(z_neg, self.norm_name)
            #             rat_loss, sufficiency_loss, independence_loss = rationale_loss(z, z_aug, z_neg) # creates NANs
            rat_loss, sufficiency_loss, independence_loss = rationale_loss(norm_z, norm_z_aug, norm_z_neg)
            loss_dict['rat_loss'] = rat_loss.item()
            loss_dict['sufficiency_loss'] = sufficiency_loss.item()
            loss_dict['independence_loss'] = independence_loss.item()
            loss = rat_loss
        elif self.loss_method in ['simclr_rationale_negative']:
            norm_z_neg = self.normalize(z_neg, self.norm_name)
            simclr_loss = simclr(norm_z, norm_z_aug)
            #             rat_loss, sufficiency_loss, independence_loss = rationale_loss(z, z_aug, z_neg) # creates NANs
            rat_loss, sufficiency_loss, independence_loss = rationale_loss(norm_z, norm_z_aug, norm_z_neg)
            loss_dict['simclr_loss'] = simclr_loss.item()
            loss_dict['rat_loss'] = rat_loss.item()
            loss_dict['sufficiency_loss'] = sufficiency_loss.item()
            loss_dict['independence_loss'] = independence_loss.item()
            loss = simclr_loss + independence_loss
        elif self.loss_method in ['simclr_double_aug_info_simclrnegative']:
            norm_z_neg = self.normalize(z_neg, self.norm_name)
            simclr_loss = simclr(norm_z, norm_z_aug)
            simclrnegative_loss = simclr(norm_z_neg, norm_z_aug)
            loss = simclr_loss - simclrnegative_loss + self.w_info_loss * info_loss
            loss_dict['simclr_loss'] = simclr_loss.item()
            loss_dict['info_loss'] = info_loss.item()
            loss_dict['neg_loss'] = simclrnegative_loss.item()
        else:
            raise NotImplementedError(f"{self.loss_method} is not implemented.")

        # A posteriori : add watchman loss to any loss that was selected.

        if self.use_watchman:
            # Specific loss term for the watchman, trying to estimate the true eigenvalues
            # based on the embedding.
            # NOTE : the goal is to ensure the embeddings remain coherent, by ensuring that 
            # they can always be used to predict SOME information about the original graph

            # Euclidian distance between prediction and desired info (in this case, Laplacian top eigenvalues)
            watchman_distance = torch.cdist(
                watchman_prediction,
                graph_laplacian_top_eigenvalues
            )
            watchman_loss = torch.mean(watchman_distance)
            loss_dict['watchman_loss'] = watchman_loss.item()
            loss = loss + (self.watchman_lambda * watchman_loss)

        loss_dict['loss'] = loss.item()
        return loss, loss_dict

    def forward_pass(self, data, epoch, training, current_mask=None):
        assert getattr(data, "current_edge_mask", None) is not None
        with BatchNormSwitch.Switch(0):  # 0 for full graphs
            emb, emb_pool = self.clf.get_emb(data.x,
                                             data.edge_index,
                                             batch=data.batch,
                                             edge_attr=data.edge_attr,
                                             return_pool=True)
        att_log_logits = self.extractor(emb, data.edge_index, data.batch)
        att = _sampling(att_log_logits, training, temperature=self.temperature_edge_sampling)

        # TODO FOR ABLATION STUDY : REPLACE THE EDGE_ATT DIRECTLY BY GROUND TRUTH
        if self.learn_edge_att:
            edge_att = att  # same result as the (buggy) commented code below
            # if is_undirected(data.edge_index):
            #     nodesize = data.x.shape[0]
            #     edge_att = (att + transpose(
            #         data.edge_index, att, nodesize, nodesize,
            #         coalesced=False)[1]) / 2
            # else:
            #     edge_att = att
        else:
            edge_att = _lift_node_att_to_edge_att(att, data.edge_index)
        with BatchNormSwitch.Switch(1):  # 1 for positive augmentations of graphs
            clf_logits, emb_clf = self.clf(data.x,
                                           data.edge_index,
                                           data.batch,
                                           edge_attr=data.edge_attr,
                                           edge_atten=edge_att,
                                           return_emb=True)

        if self.double_augmentation:
            if self.features_extractor is None:
                att_log_logits2 = self.extractor(emb, data.edge_index, data.batch)
            else:
                with BatchNormSwitch.Switch(0):
                    emb_second, emb_pool_second = self.clf.get_emb(data.x_second,
                                                                   data.edge_index,
                                                                   batch=data.batch,
                                                                   edge_attr=data.edge_attr,
                                                                   return_pool=True)
                att_log_logits2 = self.extractor(emb_second, data.edge_index, data.batch)
            att2 = _sampling(att_log_logits2, training, temperature=self.temperature_edge_sampling)
            if self.learn_edge_att:
                edge_att2 = att2  # same result as the (buggy) commented code below
                # if is_undirected(data.edge_index):
                #     nodesize = data.x.shape[0]
                #     edge_att2 = (att2 + transpose(
                #         data.edge_index, att, nodesize, nodesize,
                #         coalesced=False)[1]) / 2
                # else:
                #     edge_att2 = att2
            else:
                edge_att2 = _lift_node_att_to_edge_att(att2, data.edge_index)
            with BatchNormSwitch.Switch(1):  # 1 for positive augmentations of graphs
                clf_logits2, emb_clf2 = self.clf(data.x,
                                                 data.edge_index,
                                                 data.batch,
                                                 edge_attr=data.edge_attr,
                                                 edge_atten=edge_att2,
                                                 return_emb=True)
            emb_pool = emb_clf2

        # compute the embedding of 1 - augmentation
        #         emb_clf_neg = None
        if self.negative_term:
            with BatchNormSwitch.Switch(2):  # 2 for negative augmentations of graphs
                clf_logits_neg, emb_clf_neg = self.clf(data.x,
                                                       data.edge_index,
                                                       data.batch,
                                                       edge_attr=data.edge_attr,
                                                       edge_atten=1 - edge_att,
                                                       return_emb=True)
        else:
            emb_clf_neg = None

        # Run the watchman and try to predict laplacian eigenvalues
        if self.use_watchman:
            watchman_prediction = self.watchman.forward(emb_pool)

            # Compute true eigenvalues for the Laplacian of each anchor graph and stack them
            # NOTE : data is a BATCH of graphs, not a single graph !
            graph_laplacian_top_eigenvalues = list()

            if isinstance(data, torch_geometric.data.batch.DataBatch):
                for i in range(len(data)):
                    try:
                        this_graph = data[i]
                    except Exception:
                        raise ValueError("""You tried to iterate over data, but it
                        failed. This error was likely raised because you attempted 
                        to use the (graph-level) watchman on a node classification task. """)

                    try:
                        this_graph_top_eiv = get_laplacian_top_eigenvalues(
                            this_graph, k=self.top_eigenvalues_nb
                        )
                    except ValueError:
                        if this_graph.num_nodes > 2:
                            raise
                        warnings.warn(
                            "A graph with two nodes only have been used. "
                            "Laplacian is not defined on those. "
                            "A default vector of 1s has been used instead."
                        )
                        this_graph_top_eiv = torch.ones(self.top_eigenvalues_nb, device=data.edge_index.device)

                    graph_laplacian_top_eigenvalues.append(this_graph_top_eiv)

            else:
                this_graph_top_eiv = get_laplacian_top_eigenvalues(
                    data, k=self.top_eigenvalues_nb
                )
                graph_laplacian_top_eigenvalues.append(this_graph_top_eiv)

            graph_laplacian_top_eigenvalues = torch.stack(
                graph_laplacian_top_eigenvalues
            )

            # print("---")
            # print("WP", watchman_prediction[0:3])
            # print("GL", graph_laplacian_top_eigenvalues[0:3])

        else:
            graph_laplacian_top_eigenvalues, watchman_prediction = None, None

        # print("-- CL",clf_logits.shape)
        # print("-- Y",data.y.shape)
        loss, loss_dict = self.__loss__(edge_att, clf_logits, data.y, epoch,
                                        emb_pool, emb_clf, emb_clf_neg,
                                        data.batch, current_mask=current_mask,
                                        watchman_prediction=watchman_prediction,
                                        graph_laplacian_top_eigenvalues=graph_laplacian_top_eigenvalues,
                                        current_edge_mask=data.current_edge_mask)
        #         print(f'emb {emb.shape}, clf_logits {clf_logits.shape}, edge_att {edge_att.shape}, emb_clf {emb_clf.shape}')

        # Compute sparsity
        loss_dict["sparsity"] = torch.mean(edge_att)

        return edge_att, loss, loss_dict, clf_logits, emb_clf


# --------------------------- SECONDARY MODULES ------------------------------ #

class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size, learn_edge_att):
        super().__init__()
        self.learn_edge_att = learn_edge_att

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=0.5)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=0.5)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col] if batch is not None else None)
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


class EmbeddingWatchmanMLP(nn.Module):
    """
    A simple MLP. It will be used to predict eigenvalues for the Laplacian of a graph based on its embedding.
    Its goal is to stabilize the training by forcing the embedding to remain meaningful enough to make this prediction.

    Arguments :
    - input_dim : should match the dimension of the graph embedding
    - hidden_size : size of each of the 3 hidden layers
    - output_size : should match the nb of eigenvalues to be predicted
    """

    def __init__(self, input_dim, hidden_size, output_size):
        super().__init__()
        self.layers = torch.nn.Sequential(
            init_layer(torch.nn.Linear(input_dim, hidden_size)),
            torch.nn.ReLU(),
            init_layer(torch.nn.Linear(hidden_size, hidden_size)),
            torch.nn.ReLU(),
            init_layer(torch.nn.Linear(hidden_size, hidden_size)),
            torch.nn.ReLU(),
            init_layer(torch.nn.Linear(hidden_size, output_size)),
            torch.nn.ReLU(),
            # NOTE : Use Relu for the output since we will predict eigenvalues, which on a graph Laplacian should always be positive.
        )

    def forward(self, x):
        return self.layers(x)

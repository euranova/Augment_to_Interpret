"""
Contain a method to train our model.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Batch

from ..basic_utils import process_data, visualize_results
from ..complex_utils import get_preds


class MyTrainer():

    def __init__(self, gsat, dataset_name, task="graph_classification", dataset_features_dim=-1, use_watchman=False):
        self.gsat = gsat
        self.dataset_name = dataset_name
        self.task = task
        self.dataset_features_dim = dataset_features_dim
        self.use_watchman = use_watchman

    def run_one_epoch_and_evaluate_downstream(self, data_loader, epoch, phase, dataset_name, seed, use_edge_attr,
                                              multi_label, writer, device, current_mask=None):
        result_dic = {}
        loader_len = len(data_loader)
        run_one_batch = self._train_one_batch if phase == 'train' else self._eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict = {}
        all_exp_labels, all_att, all_clf_labels, all_clf_logits, all_embs, all_features_att, all_features_exp_label = (
            [] for _ in range(7))

        for idx, data in enumerate(data_loader):
            data = data.to(device)
            att_features = None  # Initialization at None to match signatures
            if self.gsat.features_extractor is not None:
                data.x, att_features = self.gsat.features_extractor(data.x,
                                                                    data.batch,
                                                                    "test" not in phase)
                data.x_second, att_features_second = self.gsat.features_extractor(data.x,
                                                                                  data.batch,
                                                                                  "test" not in phase)
                all_features_att.append(att_features)
                all_features_exp_label.append(torch.cat((torch.ones(data.x.shape[0], self.dataset_features_dim),
                                                         torch.zeros(data.x.shape[0],
                                                                     data.x.shape[1] - self.dataset_features_dim)),
                                                        dim=1))
            if "node" in self.task:
                if phase == 'train':
                    current_mask = data.train_mask
                elif phase == 'test ':
                    current_mask = data.test_mask
                elif phase == "valid":
                    current_mask = data.val_mask
                else:
                    raise RuntimeError("There should not be any other phase value.")

                data = process_data(data, use_edge_attr, self.task, current_mask=current_mask)
                att, loss_dict, clf_logits, emb = run_one_batch(data, epoch, current_mask,
                                                                att_features=att_features)

            else:
                data = process_data(data, use_edge_attr, self.task)
                att, loss_dict, clf_logits, emb = run_one_batch(data, epoch,
                                                                att_features=att_features)

            if writer is not None:
                for k, v in loss_dict.items():
                    writer.add_scalar(f'Loss/{phase}_{k}', loss_dict[k], epoch + idx)

            if hasattr(data, "edge_label"):
                exp_labels = data.edge_label.data.cpu()
            else:
                # Write nonsense if edge_label is absent, but always the same value !
                exp_labels = 666 * torch.ones_like(data.edge_index[0])
            y = data.y.data.cpu() if "graph" in self.task else data.y.data.cpu()[current_mask]  # .view(-1, 1)

            logits = clf_logits if "graph" in self.task else clf_logits[current_mask]  # .view(-1, 1)
            desc, att_auroc_batch, clf_acc_batch, _, _ = self._log_epoch(epoch, phase, loss_dict, exp_labels, att, y,
                                                                         logits,
                                                                         dataset_name, seed, multi_label, batch=True)

            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v

            all_exp_labels.append(exp_labels)
            all_att.append(att)
            if "node" in self.task:
                all_embs.append(emb.data.cpu()[current_mask])
            else:
                all_embs.append(emb.data.cpu())
            all_clf_labels.append(y)
            all_clf_logits.append(logits)
            #         print(f'Iteration emb {emb.shape} clf_logits {clf_logits.shape}')
            #         if True: # remove

            if idx == loader_len - 1:
                all_exp_labels, all_att = torch.cat(all_exp_labels), torch.cat(all_att),
                all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)

                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                desc, att_auroc, clf_acc, clf_roc, avg_loss = self._log_epoch(epoch, phase, all_loss_dict,
                                                                              all_exp_labels, all_att, all_clf_labels,
                                                                              all_clf_logits,
                                                                              dataset_name, seed, multi_label,
                                                                              batch=False)
                if writer is not None:
                    writer.add_scalar('AttentionAcc/{}'.format(phase), att_auroc, epoch)
                    writer.add_scalar('ClassifAcc/{}'.format(phase), clf_acc, epoch)
                result_dic.update({"Interp_acc": att_auroc})
                result_dic.update({"Clf_acc": clf_acc})

            for k, v in all_loss_dict.items():
                all_loss_dict[k] = v / loader_len

                #
            # break # remove
            # pbar.set_description(desc)

        all_embs = torch.cat(all_embs, dim=0)

        if self.gsat.features_extractor is not None:
            all_features_att = torch.cat(all_features_att, dim=0)
            all_features_exp_label = torch.cat(all_features_exp_label, dim=0)
        else:
            all_features_att, all_features_exp_label = None, None

        if dataset_name == "ba_2motifs":
            batch = Batch.from_data_list(data_loader.dataset[:8])
            batch = process_data(batch, use_edge_attr, self.task)
            batch_att, *_ = self.gsat.forward_pass(batch, epoch, training=False)
            visualize_results(batch, batch_att, dataset_name, writer, tag=dataset_name, epoch=epoch)
        return result_dic, all_embs, all_clf_labels, all_features_att, all_features_exp_label

    @torch.no_grad()
    def _eval_one_batch(self, data, epoch, current_mask=None, att_features=None):
        self.gsat.extractor.eval()
        self.gsat.clf.eval()
        if self.use_watchman:
            self.gsat.watchman.eval()
        if "graph" in self.task:
            assert "node" not in self.task
            att, loss, loss_dict, clf_logits, emb = self.gsat.forward_pass(data, epoch, training=False)
        elif "node" in self.task:
            if current_mask is None:
                raise RuntimeError("Current mask cannot be None for node tasks.")
            att, loss, loss_dict, clf_logits, emb = self.gsat.forward_pass(
                data, epoch, training=False, current_mask=current_mask)
        else:
            raise ValueError(f"The task {self.task} should contain either 'node' or 'graph'.")

        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu(), emb.data.cpu()

    def _train_one_batch(self, data, epoch, current_mask=None, att_features=None):
        self.gsat.extractor.train()
        self.gsat.clf.train()
        if self.use_watchman:
            self.gsat.watchman.train()
        if "graph" in self.task:
            att, loss, loss_dict, clf_logits, emb = self.gsat.forward_pass(data, epoch, training=True)
        elif "node" in self.task and current_mask is not None:
            att, loss, loss_dict, clf_logits, emb = self.gsat.forward_pass(data, epoch, training=True,
                                                                           current_mask=current_mask)
        if att_features is not None:
            self.gsat.optimizer_features.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                r = 0.5
                info_loss = (att_features * torch.log(att_features / r + 1e-6) +
                             (1 - att_features) * torch.log((1 - att_features) /
                                                            (1 - r + 1e-6) + 1e-6)).mean()
                features_loss = loss + info_loss
                features_loss.backward(retain_graph=True)
                self.gsat.optimizer_features.step()

        self.gsat.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.gsat.optimizer.step()

        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu(), emb.data.cpu()

    def _log_epoch(self, epoch, phase, loss_dict, exp_labels, att, clf_labels, clf_logits, dataset_name, seed,
                   multi_label, batch):
        desc = f'[Seed {seed}, Epoch: {epoch}]: gsat_{phase}........., ' if batch else f'[Seed {seed}, Epoch: {epoch}]: gsat_{phase} finished, '
        for k, v in loss_dict.items():
            desc += f'{k}: {v:.3f}, '

        eval_desc, att_auroc, clf_acc, clf_roc = self._get_eval_score(exp_labels, att, clf_labels, clf_logits,
                                                                      dataset_name, multi_label, batch)
        desc += eval_desc
        return desc, att_auroc, clf_acc, clf_roc, loss_dict  # ['pred']

    def _get_eval_score(self, exp_labels, att, clf_labels, clf_logits, dataset_name, multi_label, batch):
        clf_preds = get_preds(clf_logits, multi_label)
        clf_acc = 0 if multi_label else (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]

        if batch:
            return f'clf_acc: {clf_acc:.3f}', None, None, None

        clf_roc = 0
        # Commented because it works only for the classification ogb datasets and not the regression ones
        # if 'ogb' in dataset_name:
        #     evaluator = Evaluator(name='-'.join(dataset_name.split('_')))
        #     clf_roc = evaluator.eval({'y_pred': clf_logits, 'y_true': clf_labels})['rocauc']
        att_auroc = roc_auc_score(exp_labels, att) if np.unique(exp_labels).shape[0] > 1 else 0

        desc = f'clf_acc: {clf_acc:.3f}, clf_roc: {clf_roc:.3f}, att_roc: {att_auroc:.3f}'

        return desc, att_auroc, clf_acc, clf_roc

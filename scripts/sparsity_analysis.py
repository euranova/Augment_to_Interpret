"""
Analyse the sparsity of the results
"""

import glob
import os
import warnings
from collections import OrderedDict
from itertools import chain
from pathlib import Path

import diptest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from augment_to_interpret.architectures.contrastive_model import ContrastiveModel
from augment_to_interpret.basic_utils import C, process_data, read_args, get_device
from augment_to_interpret.competitors.ad_gcl import (
    ADGCLClusteringEmbeddings, get_adgcl_models, load_adgcl_models_inplace)
from augment_to_interpret.competitors.mega import (
    MEGAClusteringEmbeddings, load_mega_model_inplace, get_mega_models)
from augment_to_interpret.complex_utils.model_utils import load_models
from augment_to_interpret.complex_utils.param_functions import (
    get_adgcl_args, get_mega_args, get_model_config)
from augment_to_interpret.datasets import get_task, get_data_loaders, load_node_dataset
from augment_to_interpret.models import batchnorm_switch

if __name__ == "__main__":
    MODEL_CRIT = "best_downstream_clf_model"

    splits = {'train': 0.8, 'valid': 0.1, 'test': 0.1}
    batch_size = 256

    device, C.CUDA_ID = get_device(C.CUDA_ID)


    # Load models
    def load_this(path_results, relative_path):
        if path_results.endswith("adgcl"):
            if not batchnorm_switch.use_old_version_of_the_code:
                raise RuntimeError("You cannot use adgcl with the corrected batchnorm.")
            this_dataset, this_seed = relative_path.split(os.path.sep)
            this_seed = int(this_seed[5:])  # e.g. seed_0
            this_loss = 'adgcl'
            args = get_adgcl_args()
        elif path_results.endswith("mega"):
            this_dataset, this_seed = relative_path.split(os.path.sep)
            this_seed = int(this_seed[5:])  # e.g. seed_0
            this_loss = 'mega'
            args = get_mega_args()
        else:
            argspath = Path(path_results, relative_path, "full_args.tsv")
            args = read_args(argspath)
            this_dataset, this_seed, this_loss = args["dataset"], args["seed"], args["loss"]

        # --------------------- Autodetect certain parameters -------------------- #
        task = get_task(this_dataset)
        if "graph" in task:
            model_name = "GIN"
        elif "node" in task:
            model_name = "GIN_node"
        else:
            raise NotImplementedError

        # Data loading
        if "graph" in task:
            loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(
                C.PATH_DATA,
                this_dataset,
                batch_size=batch_size,
                random_state=this_seed,
                splits=splits,
                shuffle_train=False,
            )
        elif "node" in task:
            dataloader, x_dim, edge_attr_dim, num_class, aux_info = load_node_dataset(
                this_dataset, batch_size)
            loaders = {
                "train": dataloader,
                "test": dataloader,
                "valid": dataloader
            }
        else:
            raise NotImplementedError

        if this_loss not in ['adgcl', 'mega']:
            model_config = get_model_config(
                model_name=model_name,
                dataset_name=this_dataset,
                hidden_size=args["hidden_size"],
                n_layers=args["n_layers"],
                dropout_ratio=args["dropout"],
                aux_info=aux_info
            )
        else:
            model_config = {"use_edge_attr": args["use_edge_attr"]}

        if this_loss == 'adgcl':
            model, view_learner = get_adgcl_models(args, x_dim, edge_attr_dim, device)
            load_adgcl_models_inplace(Path(path_results, relative_path, MODEL_CRIT),
                                      model, view_learner, device)
            gsat = ADGCLClusteringEmbeddings(model.encoder, view_learner)
        elif this_loss == "mega":
            model, _unused_model_sd, LGA_learner = get_mega_models(args, x_dim, device)
            load_mega_model_inplace(model, _unused_model_sd, LGA_learner,
                                    Path(path_results, relative_path, MODEL_CRIT), device)
            gsat = MEGAClusteringEmbeddings(model.encoder, LGA_learner)
        else:
            clf, extractor, watchman, optimizer, criterion, features_extractor, optimizer_features = load_models(
                x_dim, edge_attr_dim, num_class, aux_info, model_config, device, task,
                args["learning_rate"], args["learning_rate_watchman"], args["watchman_eigenvalues_nb"],
                args["hidden_dim_watchman"], args["use_watchman"], args["use_features_selector"])
            gsat = ContrastiveModel(clf,
                                    extractor,
                                    criterion,
                                    optimizer,
                                    optimizer_features=optimizer_features,
                                    watchman=watchman,
                                    task=task,
                                    loss_type=this_loss,
                                    learn_edge_att="node" in task,
                                    use_watchman=args["use_watchman"],
                                    final_r=args["r_info_loss"],
                                    temperature_edge_sampling=args["temperature_edge_sampling"],
                                    w_info_loss=args["loss_weight"],
                                    top_eigenvalues_nb=args["watchman_eigenvalues_nb"],
                                    watchman_lambda=args["watchman_lambda"],
                                    features_extractor=features_extractor)

            gsat.load_state_dict(
                torch.load(Path(path_results, relative_path, MODEL_CRIT, "gsat_model"), map_location=device)
            )
            gsat.eval()

        return gsat, loaders, model_config, task, this_dataset, this_loss, this_seed, args


    # Sparsity analysis functions

    def collect_embeddings(mymodel, myloaders, model_config, task):
        gt = []
        embs_list = []
        att_list = []
        att_auroc_list = []

        for data in myloaders["train"]:
            data = data.to(device)
            current_mask = None
            if "node" in task:
                current_mask = data.train_mask
            data = process_data(data, model_config["use_edge_attr"], task=task, current_mask=current_mask)

            att, _, _, _, emb = mymodel.forward_pass(
                data, 0, training=False, current_mask=current_mask)
            embs_list.append(emb)
            att_list.append(att)
            try:
                att_auroc = roc_auc_score(data.edge_label.reshape(-1).detach().cpu(), att.detach().cpu())
            except:
                # If anything goes wrong, ignore the AUC. It's due to node_classificaiton not having edge_label.
                # I'll deal with it at some point.
                att_auroc = -1
            att_auroc_list.append(att_auroc)

            gt.append(data.y)

        att_all = torch.cat(att_list).reshape(-1)
        embs_all = torch.cat(embs_list)
        gt_all = torch.cat(gt).reshape(-1)
        # att_auroc_all = torch.cat(att_auroc_list).reshape(-1)
        att_auroc_all = att_auroc_list

        return att_auroc_all, att_all, embs_all, gt_all


    def sparsity_analysis(att_all, embs_all, att_auroc_all, dataset_name, model_name, seed, path_results):
        result_dict = {}

        # Attentions
        att = att_all.detach().cpu().numpy()
        min_value = "{:.2f}".format(np.min(att))
        max_value = "{:.2f}".format(np.max(att))

        result_dict["att_amplitude"] = f"[{min_value} ; {max_value}]"
        result_dict["att_mean"] = np.mean(att)
        result_dict["att_std"] = np.std(att)

        dip, pval = diptest.diptest(att)
        result_dict["att_hartigan_multimodal_pval"] = pval

        for val in np.arange(0, 1.1, 0.1):
            result_dict["cdf_" + str(round(val, 2))] = np.where(att <= val, 1, 0).mean()

        plt.title(f" Dataset : {dataset_name} {model_name} : {np.mean(att_auroc_all)}  ")
        plt.hist(att, bins=40)
        outgraph_path = Path(path_results, "sparsity_plots")
        outgraph_path.mkdir(exist_ok=True, parents=True)
        plt.savefig(Path(outgraph_path, f"sparsity_{dataset_name}_{seed}_{model_name}.png"))
        plt.close()

        return result_dict


    # Analysis
    df = pd.DataFrame()
    for entry_point in ["main", "adgcl", "mega"]:
        path_results = str(Path(C.PATH_RESULTS, entry_point))
        save_old_bn = batchnorm_switch.use_old_version_of_the_code
        if entry_point == "adgcl":
            batchnorm_switch.use_old_version_of_the_code = True
        try:
            all_paths = glob.glob(path_results + '/**/done', recursive=True)
            if len(all_paths) == 0:
                warnings.warn(
                    f"The {path_results} directory does not contain any results denoted by a file "
                    f"called 'done'. Skipping that directory")
                continue
            for path in all_paths:
                relative_path = str(Path(*Path(path).parts[len(Path(path_results).parts):-1]))
                assert str(Path(path_results, relative_path, "done")) == path
                mymodel, myloaders, my_model_config, mytask, dataset, loss, seed, args = (
                    load_this(path_results, relative_path))

                att_auroc_all, att_all, embs_all, gt_all = collect_embeddings(
                    mymodel, myloaders, my_model_config, mytask)
                result_dict = sparsity_analysis(
                    att_all, embs_all, att_auroc_all, dataset, loss, seed, path_results)

                # Save result
                print(f">> Completed {dataset} {seed} {loss}.")
                fulld = args
                fulld["dataset"] = dataset
                fulld["seed"] = seed
                fulld["loss"] = loss

                fulld = OrderedDict(chain(fulld.items(), result_dict.items(), ))
                r = pd.Series(fulld)
                df = pd.concat([df, r.to_frame().T], ignore_index=True)
        finally:
            batchnorm_switch.use_old_version_of_the_code = save_old_bn
    df.to_csv(Path(C.PATH_RESULTS, "sparsity_final_results.tsv"), sep='\t')

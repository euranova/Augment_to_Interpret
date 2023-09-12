"""
Run the experiments for MEGA
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from augment_to_interpret.basic_utils import C, set_seed, get_device, read_final_res_dictionary
from augment_to_interpret.basic_utils.constants import mkdir_and_return
from augment_to_interpret.competitors.mega import (
    run_one_epoch, save_mega_models, MEGAClusteringEmbeddings, load_mega_model_inplace, get_mega_models)
from augment_to_interpret.complex_utils.param_functions import deduce_batch_size, get_mega_args
from augment_to_interpret.datasets import get_data_loaders, get_task
from augment_to_interpret.metrics import embedding_evaluation as EV_gsat
from augment_to_interpret.metrics.cluster_quality_metrics import analyse_embedding
from augment_to_interpret.metrics.metrics_utils import eucldist, correl
from augment_to_interpret.metrics.wasserstein_explanation_distance import wasserstein_analysis
from augment_to_interpret.models import batchnorm_switch

if __name__ == "__main__":
    # no correction of the batchnorm when running MEGA
    batchnorm_switch.use_old_version_of_the_code = True

    # ------------------------------ PARAMETERS ---------------------------------- #

    DATASETS = [
        'ba_2motifs',
        'mutag',
        'spmotif_0.5',
    ]

    SEEDS = np.arange(0, 3)

    max_batches = 6

    PATH_RESULTS = Path(C.PATH_RESULTS, "mega")

    num_tasks = 1
    splits = {'train': 0.8, 'valid': 0.1, 'test': 0.1}

    device, C.CUDA_ID = get_device(C.CUDA_ID)
    args = get_mega_args()


    def get_res_path(dataset_name, seed):
        return Path(PATH_RESULTS, dataset_name, f"seed_{seed}")


    for seed, dataset_name in itertools.product(SEEDS, DATASETS):
        set_seed(seed, device)
        task = get_task(dataset_name)
        print(f"seed: {seed}, dataset: {dataset_name}")
        path_results = mkdir_and_return(get_res_path(dataset_name, seed))
        path_best_interp_model = mkdir_and_return(path_results, "best_interp_model")
        path_best_clf_model = mkdir_and_return(path_results, "best_clf_model")
        path_best_downstream_clf_model = mkdir_and_return(path_results, "best_downstream_clf_model")
        path_best_silhouette_model = mkdir_and_return(path_results, "best_silhouette_model")
        path_final_model = mkdir_and_return(path_results, "final_model")

        writer = SummaryWriter(path_results)
        batch_size = deduce_batch_size(dataset_name=dataset_name)

        # Load datasets
        loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(
            C.PATH_DATA,
            dataset_name,
            batch_size=batch_size,
            random_state=seed,
            splits=splits,
        )

        # Embedding evaluator (downstream)
        get_base_model = lambda: LinearSVC(dual=False, fit_intercept=True) if "classification" in task else Ridge()
        evaluator_class = EV_gsat.GSATEvaluator() if "classification" in task else EV_gsat.RegressionEvaluator()
        evaluator = EV_gsat.EmbeddingEvaluation(
            base_model=get_base_model(),
            evaluator=evaluator_class,
            task=task,
            num_tasks=num_tasks,
            device=device,
            param_search=True,
        )

        model, model_sd, LGA_learner = get_mega_models(args, x_dim, device)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=args["model_lr"])
        LGA_optimizer = torch.optim.Adam(LGA_learner.parameters(), lr=args["LGA_lr"])

        ee = EV_gsat.EmbeddingEvaluation(
            get_base_model(), evaluator_class, task, num_tasks, device, param_search=True)

        best_interp_acc, best_clf_acc, best_downstream_clf_acc, best_silhouette = (
            -float('inf'), -float('inf'), -float('inf'), -float('inf'))

        for epoch in range(args["epochs"]):
            all_embs_train, all_clf_labels_train, edge_att_train, gt_edges_train = run_one_epoch(
                loaders['train'], model, model_sd, model_optimizer, LGA_learner, batch_size,
                args["LGA_lr"], args["reg_expect"], LGA_optimizer, device, train=True, epoch=epoch,
            )
            all_embs_val, all_clf_labels_val, edge_att_val, gt_edges_val = run_one_epoch(
                loaders['valid'], model, model_sd, model_optimizer, LGA_learner, batch_size,
                args["LGA_lr"], args["reg_expect"], LGA_optimizer, device, train=False, epoch=epoch,
            )
            all_embs_test, all_clf_labels_test, edge_att_test, gt_edges_test = run_one_epoch(
                loaders['test'], model, model_sd, model_optimizer, LGA_learner, batch_size,
                args["LGA_lr"], args["reg_expect"], LGA_optimizer, device, train=False, epoch=epoch,
            )

            model.eval()
            train_score, val_score, test_score = ee.get_embedding_evaluation_from_loaders(
                model.encoder, loaders['train'], loaders['valid'], loaders['test'])
            print("Performance: Train: {} Val: {} Test: {}".format(
                train_score, val_score, test_score))

            att_auroc_train = (roc_auc_score(gt_edges_train.cpu(), edge_att_train.cpu())
                               if np.unique(gt_edges_train).shape[0] > 1 else None)
            att_auroc_val = (roc_auc_score(gt_edges_val.cpu(), edge_att_val.cpu())
                             if np.unique(gt_edges_val).shape[0] > 1 else None)
            att_auroc_test = (roc_auc_score(gt_edges_test.cpu(), edge_att_test.cpu())
                              if np.unique(gt_edges_test).shape[0] > 1 else None)
            final_dic = {
                "train": {"Clf_acc": train_score, "Interp_acc": att_auroc_train},
                "valid": {"Clf_acc": val_score, "Interp_acc": att_auroc_val},
                "test": {"Clf_acc": test_score, "Interp_acc": att_auroc_test},
            }
            if "classification" in task:
                unsup_result_dic = evaluator.unsupervised_scores(
                    all_embs_train.cpu(),
                    all_clf_labels_train.cpu(),
                    all_embs_test.cpu(),
                    all_clf_labels_test.cpu(),
                    n_clusters=num_class)
                writer.add_scalar('Silhouette/train',
                                  unsup_result_dic['train_sil'], epoch)
                writer.add_scalar('ARI/train', unsup_result_dic['train_ari'],
                                  epoch)
                writer.add_scalar('Silhouette/test',
                                  unsup_result_dic['test_sil'], epoch)
                writer.add_scalar('ARI/test', unsup_result_dic['test_ari'],
                                  epoch)

                if unsup_result_dic['train_sil'] > best_silhouette:
                    best_silhouette = unsup_result_dic['train_sil']
                    save_mega_models(model, model_sd, LGA_learner, path_best_silhouette_model,
                                     final_dic, model_name="best silhouette model")

                final_dic = {**final_dic, **unsup_result_dic}
            writer.add_scalar('DownstreamClfAcc/train', train_score, epoch)
            final_dic.update({"DownstreamClf_acc_train": train_score})
            writer.add_scalar('DownstreamClfAcc/test', test_score, epoch)
            final_dic.update({"DownstreamClf_acc_test": test_score})
            writer.add_scalar('DownstreamClfAcc/valid', val_score, epoch)
            final_dic.update({"DownstreamClf_acc_valid": val_score})
            final_dic.update({"DownstreamClf_epoch": epoch})

            if att_auroc_val is not None and att_auroc_val > best_interp_acc:
                best_interp_acc = att_auroc_val
                save_mega_models(model, model_sd, LGA_learner, path_best_interp_model,
                                 final_dic, model_name="best interp model")
            if val_score > best_clf_acc:
                best_clf_acc = val_score
                save_mega_models(model, model_sd, LGA_learner, path_best_clf_model,
                                 final_dic, model_name="best clf model")
            downstream_score_valid = val_score
            if downstream_score_valid > best_downstream_clf_acc:  # To keep the same structure
                best_downstream_clf_acc = downstream_score_valid
                save_mega_models(model, model_sd, LGA_learner, path_best_downstream_clf_model,
                                 final_dic, model_name="best downstream model")
            if epoch + 1 == args["epochs"]:
                save_mega_models(model, model_sd, LGA_learner, path_final_model,
                                 final_dic, model_name="final model")
        Path(path_results, "done").touch()

    for analyse in ["final_model", "best_downstream_clf_model"]:
        loaders_dict = {seed: dict() for seed in SEEDS}
        models = {seed: dict() for seed in SEEDS}

        for dataset_name in DATASETS:
            for seed in SEEDS:
                set_seed(seed, device)
                path_results = get_res_path(dataset_name, seed)
                loading_path = Path(path_results, analyse)

                batch_size = deduce_batch_size(dataset_name=dataset_name)

                loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(
                    C.PATH_DATA,
                    dataset_name,
                    batch_size=batch_size,
                    random_state=seed,
                    splits=splits,
                    shuffle_train=False,
                )

                loaders_dict[seed][dataset_name] = loaders

                model, _unused_model_sd, LGA_learner = get_mega_models(args, x_dim, device)
                load_mega_model_inplace(model, _unused_model_sd, LGA_learner, loading_path, device)
                wrapper = MEGAClusteringEmbeddings(model.encoder, LGA_learner)

                models[seed][dataset_name] = wrapper
                print(f"Reloaded model for {dataset_name} {seed}")


        # Get embeddings, attentions and attention auc scores for all batch in loader

        def produce_for_this_dataset(this_dataset, this_seed):
            cluster_params = {
                'model': 'DBSCAN',
                'eps': 0.05,
                'min_samples': 25
            }

            gsat = models[this_seed][this_dataset]
            if args["use_edge_attr"]:
                raise NotImplementedError
            model_config = {
                "use_edge_attr": args["use_edge_attr"], "n_layers": args["num_gc_layers"]}

            path_results = get_res_path(this_dataset, this_seed)
            parent_path = Path(path_results, analyse)
            # Get embeddings and compute fidelity
            print("Analyse embeddings...")
            unsup_result, train_labels, embedding_dic = analyse_embedding(
                gsat,
                None,
                loaders_dict[this_seed][this_dataset]['train'],
                loaders_dict[this_seed][this_dataset]['test'],
                device,
                path=parent_path,
                cluster_params=cluster_params,
                task=get_task(this_dataset),
                return_clusters=True,
                model_config=model_config,
                max_batches=max_batches,
            )
            print("Done.")

            # Wasserstein distance between explanatory subgraphs
            # Are the subgraphs (important edges) for graphs of the same class/cluster closer ?
            was_result, gram_matrix_wasserstein_kernel, _ = wasserstein_analysis(
                gsat_model=gsat,
                metric='degree',  # Use the distribution of degrees in the subgraph
                data_loader=loaders_dict[this_seed][this_dataset]["train"],
                path_results=parent_path,
                use_optimal_threshold=False,  # Use fixed top N% edges
                model_config=model_config,
                device=device,
                task=task,
                y_label=train_labels,
                max_batches=max_batches,
            )

            # Euclidian distance between embeddings
            embs = embedding_dic["train_set"]["x"]
            # TODO : we use train, maybe make it a parameter to use test instead ? in all the code ?

            N = len(embs)
            embeddings_distance_matrix = np.zeros((N, N))
            embeddings_distance_matrix_shuffled = np.zeros((N, N))

            print("Computing a distance matrix...")
            for i in tqdm(range(N)):
                for j in range(i):
                    embeddings_distance_matrix[i, j] = eucldist(embs[i, :], embs[j, :])
                    embeddings_distance_matrix_shuffled[i, j] = eucldist(
                        embs[np.random.randint(N), :], embs[np.random.randint(N), :])
            # make it all symmetrical
            embeddings_distance_matrix = embeddings_distance_matrix + embeddings_distance_matrix.T
            embeddings_distance_matrix_shuffled = (
                embeddings_distance_matrix_shuffled + embeddings_distance_matrix_shuffled.T)

            # Results
            # NOTE : use embeddings euclidian distance first, those are the reference matrix
            corr_eucl_wass = correl(embeddings_distance_matrix, gram_matrix_wasserstein_kernel)

            corr_eucl_wass_shuffled = correl(
                embeddings_distance_matrix_shuffled, gram_matrix_wasserstein_kernel)

            da = {"eucl_wass_" + k: v for k, v in corr_eucl_wass.items()}
            db = {"eucl_wass_" + k + "_shuffled": v for k, v in corr_eucl_wass_shuffled.items()}
            corr_coefs = {**da, **db}

            flat_res = read_final_res_dictionary(parent_path)

            return {"dataset": this_dataset, "seed": this_seed, **unsup_result,
                    **flat_res, **was_result, **corr_coefs}


        all_results = {}
        for seed in SEEDS:
            for dataset_name in DATASETS:
                all_results[f"{dataset_name}_{seed}"] = produce_for_this_dataset(dataset_name, seed)

        final_result = pd.DataFrame.from_dict(all_results.values())
        final_result.to_csv(Path(PATH_RESULTS, f"posthoc_final_result_for_{analyse}.tsv"), sep='\t')

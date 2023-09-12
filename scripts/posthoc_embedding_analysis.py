"""
Analysis of the results from main.py, computation of unsupervised metrics.
"""

import argparse
import glob
import os
import time
from collections import OrderedDict
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

from augment_to_interpret.architectures.contrastive_model import ContrastiveModel
from augment_to_interpret.basic_utils import (
    C, read_args, set_seed, get_device, read_final_res_dictionary)
from augment_to_interpret.complex_utils.model_utils import load_models
from augment_to_interpret.complex_utils.param_functions import get_model_config
from augment_to_interpret.datasets import get_data_loaders, load_node_dataset, get_task
from augment_to_interpret.metrics.cluster_quality_metrics import analyse_embedding
from augment_to_interpret.metrics.embedding_evaluation import GSATEvaluator, EmbeddingEvaluation
from augment_to_interpret.metrics.metrics_utils import eucldist, correl
from augment_to_interpret.metrics.wasserstein_explanation_distance import wasserstein_analysis

if __name__ == "__main__":
    # Do we run the expensive analysis, that requires reloading the model ?
    parser = argparse.ArgumentParser(description='Launch Experiments')
    parser.add_argument('--run_expensive_analysis',
                        type=int,
                        help='If 1, reloads the model and runs clustering, perturbation and wasserstein analysis',
                        default=1)
    parser.add_argument('--max_batches',
                        type=int,
                        help='Maximum number of batches to analyse for some of the metrics',
                        default=6)
    args = parser.parse_args()
    run_expensive_analysis = bool(args.run_expensive_analysis)
    PATH_RESULTS = Path(C.PATH_RESULTS, "main")

    CLUSTER_PARAMS = {
        'model': 'DBSCAN',
        'eps': 0.05,
        'min_samples': 25
    }

    df = pd.DataFrame()

    # FETCH ALL RUNS
    all_paths = glob.glob(str(PATH_RESULTS) + '/**/done', recursive=True)

    print(all_paths)

    if len(all_paths) == 0:
        raise FileNotFoundError(
            f"The {PATH_RESULTS} directory does not contain any results, denoted by a file called 'done'. Check that your experiments were run properly, and that you have selected the correct result directory by setting the RESULT_DIR in the posthoc_embedding_analysis.py file.")

    # For each run ...
    current_posthoc_path_nb = 0
    for path in all_paths:
        print("------------------------------------------------------------")

        big_start = time.time()

        # Fetch full args for this run
        parent_path = str(Path(path).parent)
        argspath = Path(parent_path, "full_args.tsv")
        exp_args = read_args(argspath)
        task = get_task(exp_args["dataset"])
        exp_args["task"] = task
        print("ARGS:", exp_args)
        device, exp_args["cuda"] = get_device(exp_args["cuda"])

        set_seed(exp_args["seed"], device)

        # Use more nodes if the task is nodes
        max_batches = int(args.max_batches) * (20 if "node" in task else 1)

        if run_expensive_analysis:
            if "graph" in task:
                loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(
                    C.PATH_DATA, exp_args["dataset"],
                    batch_size=exp_args["batch_size"], random_state=exp_args["seed"],
                    splits={'train': 0.8, 'valid': 0.1, 'test': 0.1},
                    shuffle_train=False,
                    # For Wasserstein, we need the graphs to be in consistent order during the post hoc analysis only
                )
            else:
                dataloader, x_dim, edge_attr_dim, num_class, aux_info = load_node_dataset(
                    exp_args["dataset"], exp_args["batch_size"])
                # Pass the entire dataset at once
                loaders = {
                    "train": dataloader,
                    "valid": dataloader,
                    "test": dataloader
                }
            model_config = get_model_config(
                model_name=exp_args["model_name"],
                dataset_name=exp_args["dataset"],
                hidden_size=exp_args["hidden_size"],
                n_layers=exp_args["n_layers"],
                dropout_ratio=exp_args["dropout"],
                aux_info=aux_info,
            )
            model_config['deg'] = aux_info['deg']

        # For each of the three best saved models (one per criterion of interest)...
        for directory in [
            "best_interp_model", "best_clf_model", "best_downstream_clf_model"
        ]:

            print(directory)

            loading_path_root = str(Path(parent_path, directory))

            # Only needed if we run the expensive analysis
            # NOTE We run the expensive analysis ONLY on the best_downstream_clf_model
            if run_expensive_analysis and (directory == "best_downstream_clf_model"):
                # Reinstantiate and reload a model
                # NOTE : If you modify the ContrastiveModel as it is called in main.py, remember to modify this as well !
                clf, extractor, watchman, optimizer, criterion, features_extractor, optimizer_features = load_models(
                    x_dim=x_dim,
                    edge_attr_dim=edge_attr_dim,
                    num_class=num_class,
                    aux_info=aux_info,
                    model_config=model_config,
                    device=device,
                    task=exp_args["task"],
                    learning_rate=exp_args["learning_rate"],
                    learning_rate_watchman=exp_args["learning_rate_watchman"],
                    watchman_eigenvalues_nb=exp_args["watchman_eigenvalues_nb"],
                    hidden_dim_watchman=exp_args["hidden_dim_watchman"],
                    use_watchman=exp_args["use_watchman"],
                    use_features_selector=exp_args["use_features_selector"],
                )
                gsat = ContrastiveModel(
                    clf,
                    extractor,
                    criterion,
                    optimizer,
                    optimizer_features=optimizer_features,
                    watchman=watchman,
                    task=task,
                    loss_type=exp_args["loss"],
                    learn_edge_att="node" in task,
                    use_watchman=exp_args["use_watchman"],
                    final_r=exp_args["r_info_loss"],
                    temperature_edge_sampling=exp_args["temperature_edge_sampling"],
                    w_info_loss=exp_args["loss_weight"],
                    top_eigenvalues_nb=exp_args["watchman_eigenvalues_nb"],
                    watchman_lambda=exp_args["watchman_lambda"],
                    features_extractor=features_extractor
                )

                loading_path = Path(loading_path_root, "gsat_model")
                print(f"Loading contrastive model at : {loading_path}")

                if not os.path.isfile(loading_path):
                    raise FileNotFoundError

                gsat.load_state_dict(torch.load(loading_path, map_location=device))

                gsat.eval()

                gsat.clf.eval()
                gsat.extractor.eval()
                if exp_args["use_watchman"]:
                    gsat.watchman.eval()

                # Instantiate evaluators
                evaluator_class = GSATEvaluator()
                evaluator = EmbeddingEvaluation(
                    LinearSVC(dual=False, fit_intercept=True), evaluator_class,
                    task, 1, device, param_search=True)

                params_dict = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
                classifier = make_pipeline(StandardScaler(),
                                           GridSearchCV(evaluator.base_model, params_dict, cv=5,
                                                        scoring=evaluator.gscv_scoring_name, n_jobs=16, verbose=0)
                                           )

                # ------------------------ Run analysis ------------------------------ #
                # ---- Fetch and analyse embeddings
                print("Analyse embeddings...")
                unsup_result, train_labels, embedding_dic = analyse_embedding(
                    gsat=gsat,
                    classifier=None,
                    train_set=loaders['train'],
                    test_set=loaders['test'],
                    device=device,
                    path=parent_path,
                    model_config=model_config,
                    cluster_params=CLUSTER_PARAMS,
                    task=task,
                    return_clusters=True,
                    max_batches=max_batches,
                )
                print("Done.")

                # Wasserstein distance between explanatory subgraphs
                # Are the subgraphs (important edges) for graphs of the same class/cluster closer ?
                was_result, gram_matrix_wasserstein_kernel, final_indices = wasserstein_analysis(
                    gsat_model=gsat,
                    metric='degree',  # Use the distribution of degrees in the subgraph
                    data_loader=loaders["train"],
                    path_results=loading_path_root,
                    use_optimal_threshold=False,  # Use fixed top edges
                    model_config=model_config,
                    device=device,
                    task=task,
                    name="train_set",
                    y_label=train_labels,
                    max_batches=max_batches,
                )

                # Euclidian distance between embeddings
                embs = embedding_dic["train_set"]["x"]
                # NOTE TODO : we use train, maybe make it a parameter to use test instead ? in all the code ?

                # Did we subsample ? Usually the case for nodes
                # If yes, take it into account
                if final_indices is not None:
                    embs = embs[final_indices]

                N = len(embs)
                embeddings_distance_matrix = np.zeros((N, N))
                embeddings_distance_matrix_SHUFFLED = np.zeros((N, N))

                print("Computing a distance matrix...")
                for i in tqdm(range(N)):
                    for j in range(i):
                        embeddings_distance_matrix[i, j] = eucldist(embs[i, :], embs[j, :])
                        embeddings_distance_matrix_SHUFFLED[i, j] = eucldist(embs[np.random.randint(N), :],
                                                                             embs[np.random.randint(N), :])
                embeddings_distance_matrix = embeddings_distance_matrix + embeddings_distance_matrix.T  # make it symmetrical
                embeddings_distance_matrix_SHUFFLED = embeddings_distance_matrix_SHUFFLED + embeddings_distance_matrix_SHUFFLED.T  # make it symmetrical

                # Results
                # NOTE : use embeddings euclidian distance first, those are the reference matrix
                corr_eucl_wass = correl(
                    embeddings_distance_matrix,
                    gram_matrix_wasserstein_kernel
                )

                corr_eucl_wass_shuffled = correl(
                    embeddings_distance_matrix_SHUFFLED,
                    gram_matrix_wasserstein_kernel
                )

                da = {"eucl_wass_" + k: v for k, v in corr_eucl_wass.items()}
                db = {"eucl_wass_" + k + "_shuffled": v for k, v in corr_eucl_wass_shuffled.items()}
                corr_coefs = {**da, **db}


            else:
                unsup_result = {}
                was_result = {}
                corr_coefs = {}

            # Run this even if we don't run the expensive analysis
            # Read everything that was saved during the run when the model was made
            flat_res = read_final_res_dictionary(loading_path_root)

            # Save results in df
            fulld = OrderedDict(chain(
                exp_args.items(),
                unsup_result.items(),
                flat_res.items(),
                was_result.items(),
                {"clustering_method": str(CLUSTER_PARAMS)}.items(),
                corr_coefs.items()
            ))

            print("fulld", fulld)
            r = pd.Series(fulld)
            df = pd.concat([df, r.to_frame().T], ignore_index=True)

        big_end = time.time()
        current_posthoc_path_nb += 1
        print(
            f"-------> Completed posthoc analysis {current_posthoc_path_nb} out of {len(all_paths)} - took {big_end - big_start} seconds.")

    df.to_csv(Path(PATH_RESULTS, "posthoc_embedding_analysis_final.tsv"), sep='\t')

"""
Contain the wasserstein metric.
"""

import logging
import random
import warnings

import numpy as np
import scipy.stats as sps
import torch
import torch_geometric
from sklearn import metrics
from sklearn.metrics import silhouette_score
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

from .perturbation_metrics import mask_nan_edges_graph
from ..basic_utils import process_data
from ..basic_utils.helpers import get_laplacian_top_eigenvalues
from ..complex_utils.visualisation import print_visualisation


# Shorthands to compute degrees or eigenvalues

def compute_degree(graph):
    return torch_geometric.utils.degree(graph.edge_index[0], graph.num_nodes)


def wasserstein_silhouette_kernel(graph1, graph2, metric):
    if metric == "eigenvalues":
        metric_func = lambda graph: get_laplacian_top_eigenvalues(graph, 10)
    elif metric == "degree":
        metric_func = compute_degree
    else:
        raise ValueError(f"metric should be one of ['eigenvalues', 'degree'], not {metric}.")

    deg1 = metric_func(graph1).cpu().numpy()
    deg2 = metric_func(graph2).cpu().numpy()
    result = sps.wasserstein_distance(deg1, deg2)
    return result


def edge_att_only_in_k_hop_of(data, edge_att, idx_node, k_hop_distance=3):
    # Select relevant edge att
    _, _, _, new_edge_mask = k_hop_subgraph(idx_node, k_hop_distance, data.edge_index)
    new_edge_mask = new_edge_mask.unsqueeze(dim=1)
    # Set irrelevant to 0 (not to NaN, since we do not use the remove_top_features function in the wasserstein analysis)
    true_edge_att = torch.where(
        new_edge_mask == 0,
        torch.zeros_like(edge_att),
        edge_att
    )
    return true_edge_att


def get_subgraph(graph1,
                 optimal_threshold,
                 i,
                 path_results,
                 max_visualizations=8
                 ):
    """
    Return the subgraph containing only the edges marked as important by the model

    i is just an identifier.
    """
    new_edge_data = graph1.edge_index.float()  # shape = (2, nb_edges)

    # shorthands to make code more legible
    true_edge_labels = graph1.edge_label.detach().cpu().numpy()[:, 0]
    predicted_edge_labels = graph1.edge_label.detach().cpu().numpy()[:, 1]

    # If no optimal threshold is given, default to using as many as the ground truth
    if optimal_threshold is not None:
        criterion = (predicted_edge_labels < optimal_threshold)
        sel = criterion.nonzero()
    else:
        nb_edges = len(true_edge_labels)
        nb_ones = np.where(true_edge_labels == 1, 1, 0).sum()

        # Default to top 10% edges if no true_edge label is available
        if (nb_ones == 0) or (nb_ones == nb_edges): nb_ones = max(int(nb_edges / 10), 1)

        sel = np.argsort(predicted_edge_labels)[: nb_edges - nb_ones]
        criterion = np.zeros((nb_edges))
        criterion[sel] = 1

    new_edge_data[:, sel] = float('nan')
    new_graph = mask_nan_edges_graph(graph1, new_edge_data)

    # Print visualisations for the edges marked as important (up to a limit
    # to not slow down the analysis)
    if i < max_visualizations and path_results is not None:
        logging.debug("Printing visualisation...")
        # print truth
        print_visualisation(
            graph1,
            graph1.edge_label.detach().cpu().numpy()[:, 0],
            path_results,
            str(i) + '_truth'
        )

        # print estimated
        print_visualisation(
            graph1,
            1 - criterion.astype(int),
            path_results,
            str(i) + '_predicted'
        )

    return new_graph


def read_data_process(data_loader, gsat, path_results, model_config, task, name, device,
                      use_optimal_threshold=False,
                      use_top_percentile_threshold=90,
                      y_label=None,
                      max_batches=6,
                      ):
    assert max_batches >= 1, "max_batches should be 1 or higher."

    final_indices = None
    emb_clf_list = []

    subgraphs = []
    gt_list = []
    important_edges_proportion = []

    # Split by batch
    if y_label is not None:
        if "graph" in task:
            y_label = [
                torch.tensor(y_label[j:j + data_loader.batch_size])
                for j in range(0, len(y_label), data_loader.batch_size)
            ]
        # batch_size is meaningless in this case for the nodes, since the data loader will return the entire graph as a batch. So we don't split.
        elif "node" in task:
            y_label = [torch.tensor(y_label)]
        else:
            raise ValueError("There was neither 'graph' nor 'node' in task.")

    # For each batch...
    for idx_data, data in enumerate(data_loader, start=0):
        if idx_data >= max_batches:  # Cap the number of batches to save memory and time
            break
        data = data.to(device)
        if "graph" in task:
            current_mask = None
        elif "node" in task:
            if "train" in name and "test" in name:
                raise RuntimeError("There should be train OR test in name, not both.")
            elif "train" in name:
                current_mask = data.train_mask
            elif "test" in name:
                current_mask = data.test_mask
            else:
                raise RuntimeError("name should contain train or test.")
        else:
            raise RuntimeError(
                f"task should contain graph or node, and cannot therefore be {task}")

        data = process_data(
            data,
            model_config["use_edge_attr"],
            task=task,
            current_mask=current_mask,
        )

        edge_att, _, _, _, emb_clf = gsat.forward_pass(
            data,
            epoch=0,
            training=False,
            current_mask=current_mask,
        )

        emb_clf_list.append(emb_clf)

        # Ground truth labels for now,
        # to be replaced by clustering labels (in our analysis) or really anything
        # that was given as input
        if y_label is None:
            gt_list.append(data.y.cpu())
        else:
            gt_list.append(y_label[idx_data])

        # Add our computed edge attention to the edge labels
        if "graph" in task:
            data.edge_label = torch.stack(
                [
                    data.edge_label,
                    edge_att.reshape(-1)
                ]).T

            # Compute optimal threshold between true and estimated edge att
            # Use optimal or percentile
            true_edge_label = data.edge_label.detach().cpu().numpy()[:, 0]
            predicted_edge_label = data.edge_label.detach().cpu().numpy()[:, 1]

            fpr, tpr, thresholds = metrics.roc_curve(true_edge_label, predicted_edge_label)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold_calculated = thresholds[optimal_idx]

            # Create corresponding subgraphs
            print("Creating subgraphs...")
            for i, graph in enumerate(tqdm(data.to_data_list())):
                # Erase if -1
                if use_optimal_threshold:
                    optimal_threshold = optimal_threshold_calculated
                elif use_top_percentile_threshold == -1:
                    optimal_threshold = None
                else:
                    # Top edges graph-by-graph
                    use_top_percentile_threshold = np.clip(use_top_percentile_threshold, 1, 100)
                    pvs = np.sort(graph.edge_label.detach().cpu().numpy()[:, 1])
                    optimal_threshold = np.percentile(pvs, q=use_top_percentile_threshold)

                subgraphs.append(
                    get_subgraph(graph, optimal_threshold, i, path_results).cpu()
                )

            important_edges_proportion.append(
                (data.edge_label.detach().cpu().numpy()[:, 1] >= optimal_threshold_calculated).mean()
            )

        elif "node" in task:
            max_nb = max_batches

            # Draft only among real nodes, with truly important edges, if this info is available
            good_indices = [j for j in range(data.x.shape[0]) if current_mask[j]]

            if getattr(data, "edge_label_matrix", None) is not None:
                good_indices = [idx for idx in good_indices if torch.sum(data.edge_label_matrix[idx, :]) > 1]

            max_nb = min(max_nb, len(good_indices))
            final_indices = random.sample(good_indices, max_nb)

            original_data_edge_label = data.edge_label.clone()

            print("Iterating on nodes...")
            for node_idx in tqdm(final_indices):
                # First : stack the true edge label, then the predicted edge labels
                # The difference with graph is that we must eliminate everything outside
                # of the k-hop. This is the job of the edge_att_only_in_k_hop_of() function.
                truth_edge_label = edge_att_only_in_k_hop_of(
                    data=data, edge_att=original_data_edge_label.unsqueeze(dim=1), idx_node=node_idx
                )

                new_edge_label = edge_att_only_in_k_hop_of(
                    data=data, edge_att=edge_att, idx_node=node_idx
                )

                data.edge_label = torch.stack([
                    truth_edge_label.reshape(-1),
                    new_edge_label.reshape(-1)
                ]).T  # stack does not apply dtype promotion, whereas cat does, hence the use of stack

                # Compute optimal threshold between true and estimated edge att
                # Use optimal or percentile
                true_edge_label = data.edge_label.detach().cpu().numpy()[:, 0]
                predicted_edge_label = data.edge_label.detach().cpu().numpy()[:, 1]

                def has_info(x):
                    return np.min(x) != np.max(x)

                if has_info(true_edge_label) and has_info(predicted_edge_label):
                    fpr, tpr, thresholds = metrics.roc_curve(true_edge_label, predicted_edge_label)
                    optimal_idx = np.argmax(tpr - fpr)
                    optimal_threshold_calculated = thresholds[optimal_idx]
                else:
                    warnings.warn(
                        "No info in true_edge_label or predicted_edge_label. Defaulting calculated optimal threshold to 0.")
                    warnings.warn(
                        f"Information content : true_edge_label = {has_info(true_edge_label)} ; predicted_edge_label = {has_info(predicted_edge_label)}")
                    optimal_threshold_calculated = 0

                if use_optimal_threshold:
                    optimal_threshold = optimal_threshold_calculated
                elif use_top_percentile_threshold == -1:
                    optimal_threshold = None
                else:
                    # Top edges graph-by-graph
                    use_top_percentile_threshold = np.clip(use_top_percentile_threshold, 1, 100)
                    pvs = np.sort(data.edge_label.detach().cpu().numpy()[:, 1])
                    optimal_threshold = np.percentile(pvs, q=use_top_percentile_threshold)

                # Disabling visualisations for nodes, becuse the graphs tend to be larger
                # and they would be illegible (and take forever to compute).
                # TODO Re-enable for certain
                subgraphs.append(get_subgraph(
                        data, optimal_threshold, node_idx, path_results,
                        max_visualizations=0,
                ).cpu())

                important_edges_proportion.append(
                    (data.edge_label.detach().cpu().numpy()[:, 1] >= optimal_threshold_calculated).mean()
                )

        gt = torch.cat(gt_list).reshape(-1).cpu()

    return subgraphs, gt, np.mean(important_edges_proportion), final_indices


def precompute_gram_matrix(subgraphs, labels, metric):
    N = len(subgraphs)
    D = np.zeros((N, N))

    allsame = []
    alldiff = []

    print("Computing a Gram matrix...")

    for i in tqdm(range(N)):
        for j in range(i):
            myval = wasserstein_silhouette_kernel(subgraphs[i], subgraphs[j], metric)
            D[i, j] = myval

            if labels[i] == labels[j]:
                allsame.append(myval)
            else:
                alldiff.append(myval)
    D = D + D.T  # make it symmetrical

    print("Done.")

    return D, allsame, alldiff


# ---------------------------------------------------------------------------- #

def wasserstein_analysis(
    gsat_model,
    metric,
    data_loader,
    path_results,
    use_optimal_threshold,
    model_config,
    device,
    task="graph_classification",
    name="train_set",
    use_top_percentile_threshold=-1,
    y_label=None,
    max_batches=6,
):
    """
    Are the subgraphs (important edges) for graphs of the same class closer than those of opposite classes ?

    gsat_model
    metric : either 'degree' of "eigenvalues". We will compare the distribution of this value in the subgraph, using the wasserstein distance,
    use_optimal_threshold : if False, use top N% of edges instead of the best tpr-fpr compromise
    use_top_percentile_threshold : used only if use_optimal_threshold is False. It will default to using top values in percentage as given by this threshold. If instead this parameters equals -1, we will use as many edges as in the ground truth, or the top 10% edges if ground truth is unavailable.
    y_label : a list of training labels, in order. These are the clusters for which the Wasserstein score will be calculated
    """

    gsat_model.eval()

    subgraphs, labels, important_edges_proportion, final_indices = read_data_process(
        data_loader=data_loader,
        gsat=gsat_model,
        path_results=path_results,
        model_config=model_config,
        name=name,
        device=device,
        use_optimal_threshold=use_optimal_threshold,
        use_top_percentile_threshold=use_top_percentile_threshold,
        y_label=y_label,
        max_batches=max_batches,
        task=task,
    )

    # Did we subsample the labels ? For now this should only happen for node_classification
    if final_indices is not None:
        labels = labels[final_indices]

    gram_matrix_wasserstein_kernel, allsame, alldiff = precompute_gram_matrix(subgraphs, labels, metric)

    if len(np.unique(labels)) < 2:
        wass_aug_score = np.nan
    else:
        wass_aug_score = silhouette_score(gram_matrix_wasserstein_kernel, labels, metric='precomputed')

    fulld = {
        "wasserstein_kernel_metric": metric,
        "wass_aug_score": wass_aug_score,
        "important_edges_proportion": important_edges_proportion,
        "wass_allsame_mean": np.mean(allsame),
        "wass_allsame_var": np.var(allsame),
        "wass_alldiff_mean": np.mean(alldiff),
        "wass_alldif_var": np.var(alldiff),
    }

    # Return also the distance matrix for future processing
    return fulld, gram_matrix_wasserstein_kernel, final_indices

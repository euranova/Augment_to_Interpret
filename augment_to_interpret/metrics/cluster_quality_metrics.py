"""
Measure the quality of embedding clusters.
"""

import pickle
import time
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler

from .perturbation_metrics import compute_all_fidelities
from ..basic_utils import process_data

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Tensorflow not installed; ParametricUMAP will be unavailable",
        category=ImportWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="distutils Version classes are deprecated. Use packaging.version instead.",
        category=DeprecationWarning,
    )
    import umap
    import umap.plot

# Parameter
MAX_NB_SUBSAMPLING_FIDELITY_GRAPH = 20
MAX_NB_SUBSAMPLING_FIDELITY_NODES = 20


def dbscan_predict(model, X):
    """
    https://newbedev.com/scikit-learn-predicting-new-points-with-dbscan
    """
    nr_samples = X.shape[0]
    y_new = np.ones(shape=nr_samples, dtype=int) * -1
    for i in range(nr_samples):
        diff = model.components_ - X[i, :]  # NumPy broadcasting
        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance
        shortest_dist_idx = np.argmin(dist)
        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]
    return y_new


def cluster_embeddings(embeddings, cluster_params=None):
    """
    Clusters given embeddings with KMeans using the provided
    number of clusters
    """
    if cluster_params is None:
        cluster_params = {'model': 'KMeans', 'n_clusters': 2}
    if cluster_params['model'] == 'KMeans':
        kmeans = KMeans(n_clusters=cluster_params['n_clusters'], random_state=0).fit(embeddings)
        labels = kmeans.labels_
    elif cluster_params['model'] == 'DBSCAN':
        # perform dbscan on scaled embeddings to avoid differences in the range of values
        # taken by eps
        mapper = umap.UMAP(random_state=0).fit(embeddings)
        scaled_embeddings = MinMaxScaler().fit_transform(mapper.embedding_)
        clustering = DBSCAN(eps=cluster_params['eps'],
                            min_samples=cluster_params['min_samples']).fit(scaled_embeddings)
        labels = clustering.labels_
    else:
        raise NotImplementedError
    return labels


def cluster_embeddings_inductive(embeddings_train, embeddings_test,
                                 cluster_params=None):
    """
    Clusters given embeddings with KMeans using the provided
    number of clusters
    """
    if cluster_params is None:
        cluster_params = {'model': 'KMeans', 'n_clusters': 2}

    if cluster_params['model'] == 'KMeans':
        kmeans = KMeans(n_clusters=cluster_params['n_clusters'], random_state=0).fit(embeddings_train)
        labels = kmeans.labels_
        labels_test = kmeans.predict(embeddings_test)
    elif cluster_params['model'] == 'DBSCAN':
        # perform dbscan on scaled embeddings to avoid differences in the range of values
        # taken by eps
        mapper = umap.UMAP(random_state=0)
        mapper.fit(embeddings_train)
        scaled_embeddings_train = MinMaxScaler().fit_transform(mapper.transform(embeddings_train))
        # buggued implementation: scaled_embeddings_test = MinMaxScaler().fit_transform(mapper.transform(embeddings_test))
        clustering = DBSCAN(eps=cluster_params['eps'],
                            min_samples=cluster_params['min_samples']).fit(scaled_embeddings_train)
        labels = clustering.labels_
        labels_test = None  # dbscan_predict(clustering, scaled_embeddings_test)
    else:
        raise NotImplementedError
    return labels, labels_test


def get_class_medoid_index(embeddings, labels):
    """
    Computes the centroid of each class defined in the labels vector
    and returns it as a map associated with the original class name.
    """
    class_to_medoid = {}
    for cluster_id in np.unique(labels):
        cluster_idx = np.where(labels.reshape(-1) == cluster_id)[0]
        D = squareform(pdist(embeddings[cluster_idx]))
        medoid_id = cluster_idx[np.argmin(D.sum(axis=1))]
        class_to_medoid[cluster_id] = medoid_id
    return class_to_medoid


def _compute_entropy(sample_values):
    """ Estimate the entropy of a discrete random variable
    H(X) = - sum_x[ P(x) * log( P(x) ) ]

    Args:
        sample_values (:obj:`Array-like(Int)`): i.i.d. realisations of the random variable
    Returns:
        (:obj:`Float`): the estimated entropy (using ML estimation)
    """
    counts = np.array(list(Counter(np.array(sample_values)).values()))
    probs = counts / len(sample_values)
    return -np.sum(probs * np.log(probs) / np.log(2))


def _compute_mutual_information(first_sample_values, second_sample_values):
    """ Estimate the mutual information between two discrete random variables
    I(X, C) = H(X) + H(C) - H(X, C)
    Args:
        first_sample_values, second_sample_values (:obj:`Array-like(Int)`):
            i.i.d. realisations of the random variables
    Returns:
        (:obj:`Float`): the estimated mutual information (using ML estimation)
    """
    joint_entropy = _compute_entropy(
        np.array(first_sample_values) * (np.max(list(second_sample_values)) + 1)
        + np.array(second_sample_values)
    )
    return (
        _compute_entropy(first_sample_values)
        + _compute_entropy(second_sample_values)
        - joint_entropy
    )


def compute_normalized_mutual_information(pred_cluster_labels, ground_truth_downstream_labels):
    """ Compute a purity measure of the clusters, normalised to penalise having too many clusters
    NMI(W, C) = 2 * I(W, C) / (H(W) + H(C))
    with I the mutual information (under ML estimations) and H the entropy

    See https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    for more details

    Args:
        pred_cluster_labels (:obj:`Array-like(Int)`): the predicted clusters of (ungiven) samples
        ground_truth_downstream_labels (:obj:`Array-like(Int)`): the downstream class of the (ungiven) samples
    Returns:
        (:obj:`Float`): the normalized mutual information
    """
    return 2 * (
        _compute_mutual_information(pred_cluster_labels, ground_truth_downstream_labels)
        / (_compute_entropy(pred_cluster_labels) + _compute_entropy(ground_truth_downstream_labels))
    )


def compute_optimal_normalized_mutual_information(embeddings, ground_truth_downstream_labels):
    """ Find the optimal number of clusters to optimise the NMI

    Args:
        embeddings (:obj:`tensor(Float)`): the embeddings to cluster
        ground_truth_downstream_labels (:obj:`tensor(Int)`): the downstream class of the embeddings
    Returns:
        (:obj:`tensor(Int)`): the optimal normalized mutual information
    """
    best_score = 0
    embeddings = embeddings.detach().cpu().numpy()
    gt_entropy = _compute_entropy(ground_truth_downstream_labels)
    for n_clusters in range(2, len(ground_truth_downstream_labels)):
        cluster_params = {'model': 'KMeans', 'n_clusters': n_clusters}
        pred_cluster_labels = cluster_embeddings(embeddings, cluster_params=cluster_params)
        nmi = compute_normalized_mutual_information(pred_cluster_labels, ground_truth_downstream_labels)
        if nmi > best_score:
            best_score = nmi
        elif best_score > 2 * (gt_entropy / (gt_entropy + _compute_entropy(pred_cluster_labels))):
            return best_score  # very unlikely to find something better with more clusters


def compute_mean_cluster_std(pred_cluster_labels, ground_truth_downstream_value):
    """ Compute a purity measure of the clusters.
    This value is the mean of the (estimated) std of each cluster.

    Args:
        pred_cluster_labels (:obj:`Array-like(Int)`): the predicted clusters of (ungiven) samples
        ground_truth_downstream_value (:obj:`Array-like(Float)`): the downstream value of the (ungiven) samples
    Returns:
        (:obj:`Float`): the mean_cluster_std
    """
    pred_cluster_labels, ground_truth_downstream_value = (
        np.array(pred_cluster_labels), np.array(ground_truth_downstream_value))
    total_std = 0
    all_cluster_numbers = set(pred_cluster_labels)
    for cluster_number in all_cluster_numbers:
        total_std += np.std(ground_truth_downstream_value[pred_cluster_labels == cluster_number], ddof=1)
    return total_std / len(all_cluster_numbers)


def umap_embedding_plot(embeddings, labels, path, name):
    plt.figure()
    mapper = umap.UMAP(random_state=0).fit(embeddings)
    umap.plot.points(mapper, labels=labels, color_key_cmap='Paired')
    plt.title(name)
    plt.savefig(Path(path, name + "_umap.jpg"), format='jpg')
    scaled_embeddings = MinMaxScaler().fit_transform(mapper.embedding_)
    return scaled_embeddings


def analyse_embedding(gsat, classifier, train_set, test_set, device, path, model_config,
                      cluster_params=None, task="graph_classification",
                      return_clusters=False, max_batches=6):
    if cluster_params is None:
        cluster_params = {'model': 'KMeans', 'n_clusters': 2}
    gsat.eval()
    result_dic = {
        "train_set": {},
        "test_set": {},
        "train_and_test": {},
        "inductive": {}
    }

    embedding_dic = {
        "train_set": {},
        "test_set": {},
        "train_and_test": {},
        "inductive": {},
    }

    ## Obtain embeddings
    for name, data_set in zip(["train_set", "test_set"], [train_set, test_set]):
        edge_att_list, emb_clf_list = [], []
        gt_list = []

        print("Obtaining embeddings")
        current_batch = 0

        fidelity_list = []
        fidelity_opposite_list = []
        fidelity_gt_list = []
        fidelity_shuffled_list = []
        fidelity_scrambled_list = []

        for data in data_set:
            data = data.to(device)

            if current_batch < max_batches:  # Don't process more than max_batches batches, to save time and RAM.

                current_batch += 1

                current_mask = None
                if "node" in task:
                    if "train" in name and "test" in name:
                        raise RuntimeError("There should be train OR test in name, not both.")
                    elif "train" in name:
                        current_mask = data.train_mask
                    elif "test" in name:
                        current_mask = data.test_mask
                    else:
                        raise RuntimeError("name should contain train or test.")

                # NOTE : data here is an *entire batch of graphs*
                data = process_data(data, model_config["use_edge_attr"], task, current_mask)

                edge_att, _, _, _, emb_clf = gsat.forward_pass(
                    data, 0, training=False, current_mask=current_mask)

                edge_att_list.append(edge_att.detach().cpu().numpy())
                emb_clf_list.append(emb_clf)
                gt_list.append(data.y)

                ## Compute perturbation-based metrics

                # NOTES
                #   - gsat.forward_pass() is the FULL MODEL, gsat.clf() is the gnn encoder only !
                #   - edge_att is the "saliency map" so to speak

                s = time.time()

                # TODO : add a control metric of the variance/distribution/sparsity of
                # the edge attention values ?
                # histogram or variance of (edge_att.t().detach())

                if 'node' in task:
                    max_nb = MAX_NB_SUBSAMPLING_FIDELITY_NODES
                else:
                    max_nb = MAX_NB_SUBSAMPLING_FIDELITY_GRAPH

                # Compute perturbation-based metrics
                fidelity_results = compute_all_fidelities(
                    data_batch=data,
                    model=gsat,
                    graph_path=path,
                    max_nb=max_nb,
                    task=task,
                    name=name,
                    k_hop_distance=model_config["n_layers"],
                )

                # Register metrics, for each element
                fidelity_list.append(fidelity_results["fidelity"])
                fidelity_opposite_list.append(fidelity_results["fidelity_opposite"])
                fidelity_gt_list.append(fidelity_results["fidelity_gt"])
                fidelity_shuffled_list.append(fidelity_results["fidelity_shuffled"])
                fidelity_scrambled_list.append(fidelity_results["fidelity_scrambled"])

                e = time.time()
                print("Perturbation metrics computed in " + str(e - s) + " sec.")

        embedding_dic[name]["x"] = torch.cat(emb_clf_list).detach().cpu().numpy()
        embedding_dic[name]["gt"] = torch.cat(gt_list).reshape(-1).detach().cpu().numpy()
        embedding_dic[name]["edge_att"] = edge_att_list
        embedding_dic[name]["fidelity"] = np.array(fidelity_list)
        embedding_dic[name]["fidelity_opposite"] = np.array(fidelity_opposite_list)
        embedding_dic[name]["fidelity_gt"] = np.array(fidelity_gt_list)
        embedding_dic[name]["fidelity_shuffled"] = np.array(fidelity_shuffled_list)
        embedding_dic[name]["fidelity_scrambled"] = np.array(fidelity_scrambled_list)

    # Concatenate train and test elements of the dict
    for element in ["x", "gt", "fidelity", "fidelity_opposite", "fidelity_gt",
                    "fidelity_shuffled", "fidelity_scrambled"]:
        embedding_dic["train_and_test"][element] = np.concatenate(
            (embedding_dic["train_set"][element], embedding_dic["test_set"][element]))
    embedding_dic["train_and_test"]["edge_att"] = (embedding_dic["train_set"]["edge_att"]
                                                   + embedding_dic["test_set"]["edge_att"])

    ## Compute metrics based on saved embeddings
    # NOTE: x is data and gt is label

    print("Working on embeddings")

    for name in ["train_set", "test_set", "train_and_test"]:
        print("Dataset = " + name)

        x, gt = embedding_dic[name]["x"], embedding_dic[name]["gt"]
        if name == "test_set":
            umap_embedding_plot(x, gt, path=path, name=name)

        labels = cluster_embeddings(x, cluster_params=cluster_params)
        if name == "train_set" and return_clusters:
            train_labels = labels
        if len(np.unique(labels)) == 1:  # silhouette needs 2 classes at least
            score = 0
        else:
            score = silhouette_score(x, labels, metric='euclidean')

        s = time.time()

        if "classification" in task:
            ari = adjusted_rand_score(gt, labels)
            result_dic.update({name + "_ari": ari})

            print("Computing normalized mutual information...")
            snmi = time.time()
            result_dic[name + "_nmi"] = compute_normalized_mutual_information(labels, gt)
            enmi = time.time()
            print(f"Done in {str(enmi - snmi)} seconds.")
        else:
            result_dic[name + "_mean_cluster_std"] = compute_mean_cluster_std(labels, gt)

        e = time.time()
        print("Clustering metrics computed in " + str(e - s) + " sec.")

        result_dic.update({name + "_silhouette_score": score})

        # FIDELITY METRICS : average across dataset
        for fid_name in ["fidelity", "fidelity_opposite", "fidelity_gt", "fidelity_shuffled", "fidelity_scrambled"]:
            result_dic[f"{name}_{fid_name}"] = np.mean(embedding_dic[name][fid_name])

        sparsities = [np.mean(att) for att in embedding_dic[name]["edge_att"]]
        result_dic.update({name + "_sparsity": np.mean(sparsities)})

    name = "inductive"
    labels, _ = cluster_embeddings_inductive(embedding_dic["train_set"]["x"],
                                             embedding_dic["test_set"]["x"],
                                             cluster_params=cluster_params)
    x = embedding_dic["test_set"]["x"]
    all_labels = np.concatenate([labels])
    if len(np.unique(all_labels)) == 1:  # silhouette needs 2 classes at least
        score = 0
    else:
        score = silhouette_score(embedding_dic["train_set"]["x"], all_labels, metric='euclidean')
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     model_4clust = KMeans(n_clusters=n_clusters, random_state=0)
    #     sil_visualizer = SilhouetteVisualizer(model_4clust, colors='yellowbrick')
    #     sil_visualizer.fit(x)
    #     sil_visualizer.finalize()
    #     plt.title(f'SIL {score}')
    #     plt.savefig(path+name+"_SilhouetteVisualizer.jpg",
    #                 #,bbox_inches='tight_layout',
    #                 format='jpg')

    result_dic.update({name + "_silhouette_score_train": score})
    with open(Path(path, 'unsupervised_scores.pkl'), 'wb') as f:
        pickle.dump(result_dic, f)

    if return_clusters:
        return result_dic, train_labels, embedding_dic
    else:
        return result_dic

"""
Contain perturbation metrics such as fidelity.
"""

import random
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
from sklearn import metrics
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

from ..models import BatchNormSwitch

"""
TODO Currently each metric tests in steps of a few percent : number of steps is 
equal to resolution, the interval between 0 and 100 will be divided in as many
steps as the resolution
"""
RESOLUTION = 30


# ----------------------------- Mask graph edges ----------------------------- #

def remove_top_features(saliency: torch.Tensor,
                        x: torch.Tensor,
                        percentage=0.05,
                        perturbation_mode='mean'):
    """Remove top activated features

    Args:
        saliency (1-D torch.Tensor): saliency map
        x (1-D torch.Tensor): tensor from which to remove elements
        percentage (float, optional):
            percentage of elements to remove (rounded up). Defaults to 0.05.
        perturbation_mode: 'zero' or 'mean' or 'nan';
            If 'zero', perturbation will set to 0,
            if 'mean' set to mean of x,
            if 'nan' set to np.nan

    Returns:
        masked x (tensor): x with removed elements replaced by perturbation_mode
    """
    assert len(saliency.shape) == 1
    assert len(x.shape) == 1

    if perturbation_mode == 'zero':
        perturbed = 0
    elif perturbation_mode == 'mean':
        perturbed = torch.mean(x.float()).cpu()
    elif perturbation_mode == 'nan':
        perturbed = np.nan
    else:
        raise ValueError("perturbation_mode must be 'zero' or 'mean' or 'nan'")

    n_remove = ceil((~saliency.isnan()).sum() * percentage)
    shuffle_idx = torch.randperm(len(x))  # to take randomly in case of a tie
    ordered_idx = shuffle_idx[torch.argsort(-saliency[shuffle_idx], dim=0)]
    x_masked = x.detach().clone()
    x_masked[ordered_idx[:n_remove]] = perturbed
    return x_masked


def mask_nan_edges_graph(graph: torch_geometric.data.Data, new_edge_data: torch.Tensor):
    """
    Given a graph and new edge data, will create a new graph based on the original graph
    but with the new edge data instead.

    NOTE : NaNs in the new_edge_data will result in edges being removed !

    Arguments:
    - graph <torch_geometric.data.Data>: a torch Graph
    - new_edge_data:
        a new tensor of edge data, must have shape (2, nb_edges) with directional edges

    Returns:
    - a new graph with the new edge data and NaN edges removed
    """

    # Get indices of all nan 
    none_edge_id = torch.isnan(new_edge_data).any(dim=0)

    # Make new graph
    new_graph = Data(
        x=graph.x,
        edge_index=new_edge_data[:, ~none_edge_id].long(),  # Replace edges, cast to integer
        batch=graph.batch,
        y=graph.y,
        pos=graph.pos,
    )

    for attr in ["edge_attr", "edge_atten", "current_edge_mask", "edge_label"]:
        if getattr(graph, attr, None) is not None:
            setattr(new_graph, attr, getattr(graph, attr)[~none_edge_id])
    return new_graph


def remove_top_edges_graph(
    graph: torch_geometric.data.Data,
    saliency_edge: torch.Tensor,
    k=.05):
    """
    Given a graph and a saliency map on the edges, removes the top k% edges 
    and returns a new graph without them.

    Arguments:
    - graph: the graph
    - saliency_edge: 1-D torch.Tensor; an importance map on the edge
    - k: a float giving the proportion of edges to be removed (starting from highest importance)
    """

    new_edge_index = graph.edge_index.detach().clone().float()
    new_edge_index[0, :] = remove_top_features(
        saliency_edge, new_edge_index[0, :], k,
        perturbation_mode='nan')
    return mask_nan_edges_graph(graph, new_edge_index)


# --------------------------------- Metrics ---------------------------------- #

def euclidian_distance_row_wise(a: torch.Tensor, b: torch.Tensor):
    return (a - b).pow(2).sum(dim=-1).sqrt().mean()


def compute_embedding_fidelity(
    graph: torch_geometric.data.Data,
    saliency_edge: torch.Tensor,
    embedding_creation_model: torch.nn.Module,
    distance_function=euclidian_distance_row_wise):
    """
    Remove edges in order of importance, as given by known_edge_importance
    And evaluates the impact on the embedding (based on distance to the original embedding)

    Arguments:
    - graph : base graph
    - saliency_edge : 1-D torch.Tensor; a saliency map of edges
    - embedding_creation_model: model which when called as model(graph) returns an embedding.
        Embedding must have a batch dimension of length 1 as dimension 0 !
    - distance function : function to compare two embeddings. Default to euclidian distance.
    """

    scores_list = np.zeros(RESOLUTION + 1)

    embedding_original = embedding_creation_model(graph).cpu().detach()

    i = 0
    while i <= RESOLUTION:
        masked_graph = remove_top_edges_graph(
            graph,
            saliency_edge,
            k=(i / RESOLUTION),
        )

        # When we remove these edges, what is the distance between the embedding
        # for this graph without those edges, and the original embedding ?
        new_embedding = embedding_creation_model(
            masked_graph
        ).cpu().detach()

        embeddings_distance = distance_function(
            embedding_original.float(),
            new_embedding.float()
        )

        # Sum distance into a single value
        scores_list[i] = embeddings_distance  # .sum()

        i += 1

    # Normalize distances by the LAST distance observed (with maximal perturbation)
    scores_list = np.array(scores_list)
    scores_list = scores_list / scores_list[-1]

    fidelity = metrics.auc(
        np.arange(RESOLUTION + 1) / RESOLUTION,
        scores_list
    )

    return fidelity, scores_list


def compute_all_fidelities(data_batch,
                           model,
                           graph_path,
                           max_nb,
                           task,
                           name,
                           max_plot_nb=10,
                           k_hop_distance=3):
    results = {
        "fidelity": [],
        "fidelity_opposite": [],
        "fidelity_gt": [],
        "fidelity_shuffled": [],
        "fidelity_scrambled": []
    }
    scores = {
        "fidscores": [],
        "fidscores_opposite": [],
        "fidscores_shuffled": [],
        "fidscores_scrambled": []
    }

    # Classical version for graphs
    if "graph" in task:

        # Split the batch into its individual graphs
        batched_graphs = data_batch.to_data_list()

        # If requested, cap it at a certain number of graphs to save time
        batched_graphs = batched_graphs[:max_nb]

        print("Fidelity...")
        for i in tqdm(range(len(batched_graphs))):
            data = batched_graphs[i]

            # Fix edge mask not being kept, possible since we enumerate in predictable order
            row, col = data_batch.edge_index
            # Convert node-level batch indices to edge-level indices
            edge_batch = data_batch.batch[row]
            correct_id = (edge_batch == i).nonzero()
            correct_edge_mask = data_batch.current_edge_mask[correct_id]
            data.current_edge_mask = correct_edge_mask.reshape(data.edge_index.shape[1])

            edge_att, _, _, _, _ = model.forward_pass(
                data, epoch=0, training=False)

            assert (torch.min(edge_att) >= 0) and (torch.max(edge_att) <= 1), (
                "Attentions must be between 0 and 1")

            data.edge_atten = edge_att  # Adding edge attention as an arbitrary instance variable

            def emb_func(data):
                assert not model.clf.training
                with BatchNormSwitch.Switch(1):  # edge_atten is used, like a positive augmentation
                    return model.clf.get_emb(
                        data.x,
                        data.edge_index,
                        batch=data.batch,
                        edge_atten=data.edge_atten,  # transpose Needed for some reason ?
                        edge_attr=data.edge_attr
                    ).cpu().detach()  # shape (nb graphs, embedding_dim)

            estimated_edge_saliency = edge_att.squeeze(dim=1)
            true_edge_saliency = (
                data.edge_label.detach().clone().float() if hasattr(data, "edge_label")
                else torch.ones_like(estimated_edge_saliency)
            )

            # Other scenarios
            random_estimated_edge_saliency = torch.rand_like(estimated_edge_saliency)
            shuffled_estimated_edge_saliency = estimated_edge_saliency[
                torch.randperm(len(estimated_edge_saliency))]
            inverted_estimated_edge_saliency = 1 - estimated_edge_saliency

            # Random noise added to prevent threshold effects
            fid, fidscores = compute_embedding_fidelity(
                graph=data,
                embedding_creation_model=emb_func,
                saliency_edge=estimated_edge_saliency,
            )

            # Compare to other scenarios
            fid_opposite, fidscores_opposite = compute_embedding_fidelity(
                graph=data,
                embedding_creation_model=emb_func,
                saliency_edge=inverted_estimated_edge_saliency,
            )

            fidgt, fidscores_gt = compute_embedding_fidelity(
                graph=data,
                embedding_creation_model=emb_func,
                saliency_edge=true_edge_saliency,
            )

            fidscrambled, fidscores_scrambled = compute_embedding_fidelity(
                graph=data,
                embedding_creation_model=emb_func,
                saliency_edge=random_estimated_edge_saliency,
            )

            fidshuffled, fidscores_shuffled = compute_embedding_fidelity(
                graph=data,
                embedding_creation_model=emb_func,
                saliency_edge=shuffled_estimated_edge_saliency,
            )

            results["fidelity"].append(fid)
            results["fidelity_opposite"].append(fid_opposite)
            results["fidelity_gt"].append(fidgt)
            results["fidelity_shuffled"].append(fidshuffled)
            results["fidelity_scrambled"].append(fidscrambled)

            scores["fidscores"].append(fidscores)
            scores["fidscores_opposite"].append(fidscores_opposite)
            scores["fidscores_shuffled"].append(fidscores_shuffled)
            scores["fidscores_scrambled"].append(fidscores_scrambled)

            if i <= max_plot_nb:
                plt.figure()
                plt.title(str(i) + '-' + "Fidelity")
                plt.plot(fidscores)
                plt.plot(fidscores_opposite, color="red")
                plt.plot(fidscores_gt, color="green")
                plt.plot(fidscores_shuffled, color="purple")
                plt.plot(fidscores_scrambled, color="orange")

                outgraph_path = Path(graph_path, "fidelity")
                outgraph_path.mkdir(exist_ok=True, parents=True)
                plt.savefig(Path(outgraph_path, f"fidelity_{i}.png"))
                plt.close()

    elif "node" in task:
        # Do not iterate on data. It has only 1 graph, which contains all the nodes.
        # Instead, iterate on the nodes.
        data = data_batch

        if "train" in name and "test" in name:
            raise RuntimeError("There should be train OR test in name, not both.")
        elif "train" in name:
            current_mask = data.train_mask
        elif "test" in name:
            current_mask = data.test_mask
        else:
            raise RuntimeError("name should contain train or test.")

        att, _, _, _, _ = model.forward_pass(
            data, epoch=0, training=False, current_mask=current_mask
        )
        assert (torch.min(att) >= 0) and (torch.max(att) <= 1), "Attentions must be between 0 and 1"
        data.edge_atten = att  # Record edge atten !

        # For each node in data...

        # Draft only among real nodes, with truly important edges, if this info is available
        good_indices = [j for j in range(data.x.shape[0]) if current_mask[j]]
        if getattr(data, "edge_label_matrix", None) is not None:
            good_indices = [idx for idx in good_indices if torch.sum(data.edge_label_matrix[idx, :]) > 1]

        max_nb = min(max_nb, len(good_indices))
        final_indices = random.sample(good_indices, max_nb)

        for node_idx in tqdm(final_indices):
            def extract_node_embedding(graph):
                with BatchNormSwitch.Switch(1):  # edge_atten is used => looks like a positive augmentation
                    emb = model.clf.get_emb(
                        x=graph.x,
                        edge_index=graph.edge_index,
                        batch=None,
                        edge_attr=graph.edge_attr,
                        edge_atten=graph.edge_atten,
                        return_pool=False
                    )
                return emb[node_idx, :]

            # Select relevant edge att
            _, _, _, new_edge_mask = k_hop_subgraph(node_idx, k_hop_distance, data.edge_index)

            # Set irrelevant to NaN (so they will be filtered out by the remove_top_features function)
            nan = torch.tensor(float("nan"), dtype=torch.float).to(att.device)
            true_edge_att = torch.where(new_edge_mask == 0, nan, att.squeeze(dim=1))

            opposite_edge_att = 1 - true_edge_att

            random_att = torch.where(new_edge_mask == 0, nan, torch.rand_like(true_edge_att))

            shuffled_estimated_edge_saliency = true_edge_att.clone()
            shuffled_estimated_edge_saliency[new_edge_mask] = true_edge_att[new_edge_mask][
                torch.randperm(new_edge_mask.sum())]

            # use this new_edge_att in perturbations
            fid, fidscores = compute_embedding_fidelity(
                graph=data,
                embedding_creation_model=extract_node_embedding,
                saliency_edge=true_edge_att
            )

            fid_opposite, fidscores_opposite = compute_embedding_fidelity(
                graph=data,
                embedding_creation_model=extract_node_embedding,
                saliency_edge=opposite_edge_att
            )

            fid_scrambled, fidscores_scrambled = compute_embedding_fidelity(
                graph=data,
                embedding_creation_model=extract_node_embedding,
                saliency_edge=random_att
            )

            fid_shuffled, fidscores_shuffled = compute_embedding_fidelity(
                graph=data,
                embedding_creation_model=extract_node_embedding,
                saliency_edge=shuffled_estimated_edge_saliency
            )

            scores["fidscores"].append(fidscores)
            scores["fidscores_opposite"].append(fidscores_opposite)
            scores["fidscores_shuffled"].append(fidscores_shuffled)
            scores["fidscores_scrambled"].append(fidscores_scrambled)

            results["fidelity"].append(fid)
            results["fidelity_opposite"].append(fid_opposite)
            results["fidelity_gt"].append(-1)
            results["fidelity_shuffled"].append(fid_shuffled)
            results["fidelity_scrambled"].append(fid_scrambled)

            if node_idx <= max_plot_nb:
                plt.figure()
                plt.title(str(node_idx) + '-' + "Fidelity")
                plt.plot(fidscores)
                plt.plot(fidscores_opposite, color="red")
                plt.plot(fidscores_shuffled, color="purple")
                plt.plot(fidscores_scrambled, color="orange")

                outgraph_path = Path(graph_path, "fidelity")
                outgraph_path.mkdir(exist_ok=True, parents=True)
                plt.savefig(Path(outgraph_path, f"fidelity_{node_idx}.png"))
                plt.close()

    # Average over all graphs
    final = {
        "fidelity": np.mean(results["fidelity"]),
        "fidelity_opposite": np.mean(results["fidelity_opposite"]),
        "fidelity_gt": np.mean(results["fidelity_gt"]),
        "fidelity_shuffled": np.mean(results["fidelity_shuffled"]),
        "fidelity_scrambled": np.mean(results["fidelity_scrambled"])
    }

    # Final, merged plot
    merged_fidscores = np.mean(scores["fidscores"], axis=0)
    merged_fidscores_opposite = np.mean(scores["fidscores_opposite"], axis=0)
    merged_fidscores_shuffled = np.mean(scores["fidscores_shuffled"], axis=0)
    merged_fidscores_scrambled = np.mean(scores["fidscores_scrambled"], axis=0)

    plt.figure()
    plt.title(" Global Fidelity")
    plt.plot(merged_fidscores)
    plt.plot(merged_fidscores_opposite, color="red")
    plt.plot(merged_fidscores_shuffled, color="purple")
    plt.plot(merged_fidscores_scrambled, color="orange")

    plt.savefig(Path(graph_path, "fidelity_merged.png"))
    plt.close()

    return final

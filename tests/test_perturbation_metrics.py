""" Test all perturbation metrics """
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter

from augment_to_interpret.basic_utils import process_data
from augment_to_interpret.metrics.perturbation_metrics import (
    remove_top_features, remove_top_edges_graph, compute_embedding_fidelity, compute_all_fidelities)
from .utils import test_C


@pytest.fixture
def toy_graph():
    """ A toy graph for the tests:  0 <---> 1 <---> 2 """
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long, device=test_C.DEVICE)
    x = torch.tensor([[0], [1], [2]], dtype=torch.float, device=test_C.DEVICE)
    return Data(x=x, edge_index=edge_index)


@pytest.fixture
def toy_task_and_solver(toy_graph):
    toy_graph = toy_graph.detach().clone()
    toy_graph.edge_label = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=test_C.DEVICE)
    good_saliency = torch.tensor([1, 2, 3, 4], dtype=torch.float, device=test_C.DEVICE)
    bad_saliency = torch.tensor([4, 3, 2, 1], dtype=torch.float, device=test_C.DEVICE)

    def get_emb(x, edge_index, batch, edge_atten=None, edge_attr=None):
        if batch is None:
            batch = torch.zeros_like(x[:, 0], dtype=torch.long)
        s = scatter(
            x[edge_index[0]],
            batch[edge_index[0]],
            dim=0,
            dim_size=max(batch) + 1,
            reduce='sum',
        )
        return s

    def embedding_creation_model(g):  # Placeholder
        return get_emb(g.x, g.edge_index, g.batch)

    return {
        "graph": toy_graph,
        "good_saliency": good_saliency,
        "bad_saliency": bad_saliency,
        "get_emb": get_emb,
        "embedding_creation_model": embedding_creation_model,
    }


def test_remove_top_features():
    a = remove_top_features(torch.tensor([0.1, 5, 10, 2, 7]), x=torch.tensor([0.0, 10, 2, 3, 4]), percentage=0,
                            perturbation_mode="nan")
    assert np.array_equal(a, [0, 10, 2, 3, 4], equal_nan=True), a
    a = remove_top_features(torch.tensor([0.1, 5, 10, 2, 7]), x=torch.tensor([0.0, 10, 2, 3, 4]), percentage=0.2,
                            perturbation_mode="nan")
    assert np.array_equal(a, [0, 10, np.NaN, 3, 4], equal_nan=True), a
    a = remove_top_features(torch.tensor([0.1, 5, 10, 2, 7]), x=torch.tensor([0.0, 10, 2, 3, 4]), percentage=0.19,
                            perturbation_mode="nan")
    assert np.array_equal(a, [0, 10, np.NaN, 3, 4], equal_nan=True), a
    a = remove_top_features(torch.tensor([0.1, 5, 10, 2, 7]), x=torch.tensor([0.0, 10, 2, 3, 4]), percentage=0.21,
                            perturbation_mode="nan")
    assert np.array_equal(a, [0, 10, np.NaN, 3, np.NaN], equal_nan=True), a
    a = remove_top_features(torch.tensor([0.1, 5, 10, 2, 7]), x=torch.tensor([0.0, 10, 2, 3, 4]), percentage=1,
                            perturbation_mode="nan")
    assert np.array_equal(a, [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN], equal_nan=True), a

    count = 0
    for _ in range(1000):
        a = remove_top_features(torch.tensor([1, 1]), x=torch.tensor([1, 2]), percentage=0.5,
                                perturbation_mode="zero")
        count += a[0] == 0
    assert 400 < count < 600, "there should be some randomness in the choice of the removed one."

    a = remove_top_features(torch.tensor([0.1, np.NaN, 10, np.NaN, 7]), x=torch.tensor([0.0, 10, 2, 3, 4]),
                            percentage=0, perturbation_mode="nan")
    assert np.array_equal(a, [0, 10, 2, 3, 4], equal_nan=True), a
    a = remove_top_features(torch.tensor([0.1, np.NaN, 10, np.NaN, 7]), x=torch.tensor([0.0, 10, 2, 3, 4]),
                            percentage=0.33, perturbation_mode="nan")
    assert np.array_equal(a, [0, 10, np.NaN, 3, 4], equal_nan=True), a
    a = remove_top_features(torch.tensor([0.1, np.NaN, 10, np.NaN, 7]), x=torch.tensor([0.0, 10, 2, 3, 4]),
                            percentage=0.34, perturbation_mode="nan")
    assert np.array_equal(a, [0, 10, np.NaN, 3, np.NaN], equal_nan=True), a
    a = remove_top_features(torch.tensor([0.1, np.NaN, 10, np.NaN, 7]), x=torch.tensor([0.0, 10, 2, 3, 4]),
                            percentage=1, perturbation_mode="nan")
    assert np.array_equal(a, [np.NaN, 10, np.NaN, 3, np.NaN], equal_nan=True), a


def test_remove_top_edges_graph(toy_graph):
    saliency = torch.tensor([i for i in range(toy_graph.edge_index.shape[1])]).float().to(test_C.DEVICE)
    for keep_prop in [0, 0.5, 0.75, 1]:
        new_g = remove_top_edges_graph(toy_graph, saliency, 1 - keep_prop)
        assert new_g.edge_index.shape == (2, int(toy_graph.edge_index.shape[1] * keep_prop))


def test_embedding_fidelity(toy_task_and_solver):
    graph = toy_task_and_solver["graph"]
    good_saliency = toy_task_and_solver["good_saliency"]
    bad_saliency = toy_task_and_solver["bad_saliency"]
    embedding_creation_model = toy_task_and_solver["embedding_creation_model"]
    fidelity_g, sg = compute_embedding_fidelity(graph, good_saliency, embedding_creation_model)
    fidelity_b, sb = compute_embedding_fidelity(graph, bad_saliency, embedding_creation_model)

    assert fidelity_g > fidelity_b


def test_compute_all_fidelities_graphs(toy_task_and_solver):
    graph = toy_task_and_solver["graph"]
    get_emb = toy_task_and_solver["get_emb"]

    my_model = SimpleNamespace()
    my_model.forward_pass = lambda data, epoch, training, current_mask=None: (
        torch.rand_like(data.edge_index[0], dtype=torch.float).unsqueeze(1), None, None, None, None)
    my_model.clf = SimpleNamespace()
    my_model.clf.training = False
    my_model.clf.get_emb = get_emb
    dl = DataLoader([graph])
    for batch in dl:
        batch = process_data(batch, use_edge_attr=True, task="graph_classification")
        compute_all_fidelities(
            data_batch=batch,
            model=my_model,
            graph_path=test_C.PATH_TMP_TEST_FILES,
            max_nb=6,
            task="graph_classification",
            name="train_set",
            max_plot_nb=10,
            k_hop_distance=1,
        )


def test_compute_all_fidelities_nodes(toy_task_and_solver):
    graph = toy_task_and_solver["graph"]

    my_model = SimpleNamespace()
    my_model.forward_pass = lambda data, epoch, training, current_mask=None: (
        torch.rand_like(data.edge_index[0], dtype=torch.float).unsqueeze(1), None, None, None, None)
    my_model.clf = SimpleNamespace()
    my_model.clf.training = False

    def get_emb(x, edge_index, batch, edge_atten=None, edge_attr=None, return_pool=None):
        if batch is None:
            batch = torch.zeros_like(x[:, 0], dtype=torch.long)
        adj = to_dense_adj(edge_index, batch=batch).squeeze(dim=0)
        return torch.matmul(adj, x)

    my_model.clf.get_emb = get_emb
    dl = DataLoader([graph])
    for batch in dl:
        batch.train_mask = torch.ones(batch.x.shape[0], dtype=torch.long)
        batch = process_data(batch, use_edge_attr=True, task="node_classification", current_mask=batch.train_mask)
        compute_all_fidelities(
            data_batch=batch,
            model=my_model,
            graph_path=test_C.PATH_TMP_TEST_FILES,
            max_nb=6,
            task="node_classification",
            name="train_set",
            max_plot_nb=10,
            k_hop_distance=1,
        )

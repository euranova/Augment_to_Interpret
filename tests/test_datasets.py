""" Test the datasets for the pipeline """
import warnings
from dataclasses import dataclass, field
from typing import Tuple, List

import pytest
import torch

from augment_to_interpret.basic_utils import C
from augment_to_interpret.datasets import get_data_loaders, load_node_dataset, get_task
from .utils import test_C

SPLITS = {"train": 1 / 3, "valid": 1 / 3}
BATCH_SIZE = 256


def split_samples(total_n):
    n_train = int(total_n * SPLITS["train"])
    n_valid = int(total_n * SPLITS["valid"])
    return n_train, n_valid, total_n - n_train - n_valid


@dataclass
class DatasetInfo:
    task: str
    xdim: int
    edge_attr_dim: int
    num_classes: int
    set_lengths: Tuple[int]
    expected_fields: List[str] = field(default_factory=list)
    huge_resources: bool = False


GRAPH_DATASETS = {
    "ba_2motifs": DatasetInfo(
        task="graph_classification", xdim=10, edge_attr_dim=0, num_classes=2,
        set_lengths=split_samples(1000)),
    "mutag": DatasetInfo(
        task="graph_classification", xdim=14, edge_attr_dim=0, num_classes=2,
        set_lengths=split_samples(2951), huge_resources=True),
    "spmotif_0.5": DatasetInfo(
        task="graph_classification", xdim=4, edge_attr_dim=1, num_classes=3,
        set_lengths=(9000, 3000, 6000)),
    "mnist": DatasetInfo(
        task="graph_classification", xdim=5, edge_attr_dim=1, num_classes=3,
        set_lengths=(6000, 1500, 3147), huge_resources=True),
}

NODE_DATASETS = {
    "tree_cycle": DatasetInfo(
        task="node_classification", xdim=10, edge_attr_dim=0, num_classes=2,
        set_lengths=(696, 87, 88), expected_fields=["edge_label_matrix"]),
    "tree_grid": DatasetInfo(
        task="node_classification", xdim=10, edge_attr_dim=0, num_classes=2,
        set_lengths=(984, 123, 124), expected_fields=["edge_label_matrix"]),
    "cora": DatasetInfo(
        task="node_classification", xdim=1433, edge_attr_dim=0, num_classes=7,
        set_lengths=(140, 500, 1000)),
}

ALL_DATASETS = {**GRAPH_DATASETS, **NODE_DATASETS}


@pytest.mark.parametrize("dataset_name, dataset_info", list(GRAPH_DATASETS.items()))
def test_the_graph_dataset_matches_the_expected_information(dataset_name, dataset_info: DatasetInfo):
    assert get_task(dataset_name) == dataset_info.task
    if test_C.LOW_RESOURCES and dataset_info.huge_resources:
        warnings.warn(f"{dataset_name} uses lots of resources and LOW_RESOURCES in "
                      f"tests/utils/constants is set to True. Skipping the dataset.")
        return
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(
        data_dir=C.PATH_DATA,
        dataset_name=dataset_name,
        batch_size=BATCH_SIZE,
        splits=SPLITS,
        random_state=0,
        node_features=None,
        shuffle_train=True,
    )
    assert tuple(len(val.dataset) for val in loaders.values()) == dataset_info.set_lengths
    for loader in loaders:
        data = next(iter(loader))
        for expected_field in dataset_info.expected_fields:
            assert getattr(data, expected_field, None) is not None
        if getattr(data, "edge_label_matrix", None) is not None:
            assert torch.equal(data.edge_label_matrix, data.edge_label_matrix.T)
    assert x_dim == dataset_info.xdim
    assert edge_attr_dim == dataset_info.edge_attr_dim
    assert num_class == dataset_info.num_classes


@pytest.mark.parametrize("dataset_name, dataset_info", list(NODE_DATASETS.items()))
def test_the_node_dataset_matches_the_expected_information(dataset_name, dataset_info: DatasetInfo):
    assert get_task(dataset_name) == dataset_info.task
    if test_C.LOW_RESOURCES and dataset_info.huge_resources:
        warnings.warn(f"{dataset_name} uses lots of resources and LOW_RESOURCES in "
                      f"tests/utils/constants is set to True. Skipping the dataset.")
        return
    batch_size = 256
    dataloader, x_dim, edge_attr_dim, num_class, aux_info = load_node_dataset(
        dataset_name, batch_size, edge_attr_dim=0)
    assert len(dataloader) == 1
    data = next(iter(dataloader))

    for expected_field in dataset_info.expected_fields:
        assert getattr(data, expected_field, None) is not None
    if getattr(data, "edge_label_matrix", None) is not None:
        assert torch.equal(data.edge_label_matrix, data.edge_label_matrix.T)
    assert (data.train_mask.sum(), data.val_mask.sum(), data.test_mask.sum()) == dataset_info.set_lengths
    assert x_dim == dataset_info.xdim
    assert edge_attr_dim == dataset_info.edge_attr_dim
    assert num_class == dataset_info.num_classes

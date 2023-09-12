import torch
from torch.utils.tensorboard import SummaryWriter

from augment_to_interpret.basic_utils import visualize_results, C
from augment_to_interpret.datasets import get_data_loaders

from .utils import test_C

SPLITS = {"train": 1 / 3, "valid": 1 / 3}


def test_visualize_results():
    dataset_name = "ba_2motifs"
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(
        data_dir=C.PATH_DATA,
        dataset_name=dataset_name,
        batch_size=2,
        splits=SPLITS,
        random_state=0,
        node_features=None,
        shuffle_train=True,
    )
    batch = next(iter(loaders["train"]))
    batch_att = (batch.edge_label * (0.1 + torch.rand_like(batch.edge_label) * 0.02)
                 + (1 - batch.edge_label) * torch.rand_like(batch.edge_label) * 0.1)
    writer = SummaryWriter(test_C.PATH_TMP_TEST_FILES)
    visualize_results(batch, batch_att, dataset_name, writer, tag="example_non_bi", epoch=0)
    batch_att = 0.5 + batch.edge_label * 0.5
    visualize_results(batch, batch_att, dataset_name, writer, tag="example_bi", epoch=0)

"""
Contain functions to access datasets.
"""

import warnings
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from . import SynGraphDataset, Mutag, MutagNoise, SPMotif, MNIST75sp, graph_sst2, ZINC
from ..basic_utils import C, set_seed, get_device


def _get_indices(sample_labels, nb_elements, random_state=None, labels_to_keep=None):
    """ Get shuffled indices of samples with equal amount of samples in each selected class
    (and none in other classes)

    :param sample_labels: Tensor[Long]; labels of the samples
    :param nb_elements: Int; number of indices to keep
    :param random_state: Optional[Int]; random seed to use
    :param labels_to_keep: Optional[List[Any]]; the classes of interest. None to select from all classes.
    :return: List[int]; selected indices
    """

    sample_labels = sample_labels.detach().cpu().numpy()
    if labels_to_keep is None:
        labels_to_keep = list(set(sample_labels))

    rng = np.random.default_rng(random_state)
    one_more_labels = rng.choice(labels_to_keep, nb_elements % len(labels_to_keep),
                                 replace=False)  # to have the right number of samples in total
    indices = [
        idx
        for label in labels_to_keep
        for idx in rng.choice(
            np.nonzero(sample_labels == label)[0],
            nb_elements // len(labels_to_keep) + (label in one_more_labels),
            replace=False,
        )
    ]
    np.random.shuffle(indices)
    return indices


def get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, node_features=None,
                     shuffle_train=True):
    if node_features is None:
        node_features = {}
    set_seed(random_state, "cpu")
    dataset_name = dataset_name.lower()
    regression_datasets = ['ogbg_molesol', 'ogbg_mollipo', 'ogbg_molfreesolv', 'zinc']
    classification_datasets = [
        'ba_2motifs', 'mutag', "mutag_noise", 'graph-sst2', 'mnist',
        'spmotif_0.5', 'spmotif_0.7', 'spmotif_0.9',
        'ogbg_molhiv', 'ogbg_moltox21', 'ogbg_molbace',
        'ogbg_molbbbp', 'ogbg_molclintox', 'ogbg_molsider'
    ]
    assert dataset_name in regression_datasets + classification_datasets
    aux_info = {}

    if dataset_name == 'ba_2motifs':
        dataset = SynGraphDataset(data_dir, 'ba_2motifs')
        split_idx = _get_random_split_idx(dataset, splits)
        loaders, test_set = _get_loaders_and_test_set(
            batch_size, dataset=dataset, split_idx=split_idx, shuffle_train=shuffle_train)
        train_set = dataset[split_idx["train"]]

    elif dataset_name == 'mutag':
        dataset = Mutag(root=data_dir / 'mutag')
        split_idx = _get_random_split_idx(dataset, splits)
        loaders, test_set = _get_loaders_and_test_set(
            batch_size, dataset=dataset, split_idx=split_idx, shuffle_train=shuffle_train)
        train_set = dataset[split_idx['train']]

    elif dataset_name == 'mutag_noise':
        dataset = MutagNoise(root=data_dir / 'mutag', node_features=node_features)
        split_idx = _get_random_split_idx(dataset, splits)
        loaders, test_set = _get_loaders_and_test_set(
            batch_size, dataset=dataset, split_idx=split_idx, shuffle_train=shuffle_train)
        train_set = dataset[split_idx['train']]

    elif 'ogbg' in dataset_name:
        # Imported here as it may cause problems depending on the versions
        from ogb.graphproppred import PygGraphPropPredDataset

        dataset = PygGraphPropPredDataset(root=data_dir, name='-'.join(dataset_name.split('_')))
        split_idx = dataset.get_idx_split()
        loaders, test_set = _get_loaders_and_test_set(
            batch_size, dataset=dataset, split_idx=split_idx, shuffle_train=shuffle_train)
        train_set = dataset[split_idx['train']]

    elif dataset_name == 'graph-sst2':
        dataset = graph_sst2.get_dataset(dataset_dir=data_dir, dataset_name='Graph-SST2', task=None)
        dataloader, (train_set, valid_set, test_set) = graph_sst2.get_dataloader(
            dataset, batch_size=batch_size, degree_bias=True, seed=random_state)
        print('[INFO] Using default splits!')
        loaders = {
            'train': dataloader['train'], 'valid': dataloader['eval'], 'test': dataloader['test']}
        test_set = dataset  # used for visualization

    elif 'spmotif' in dataset_name:
        b = float(dataset_name.split('_')[-1])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Creating an ndarray from ragged nested sequences .* is deprecated.",
                category=np.VisibleDeprecationWarning
            )
            train_set = SPMotif(root=data_dir / dataset_name, b=b, mode='train')
            valid_set = SPMotif(root=data_dir / dataset_name, b=b, mode='val')
            test_set = SPMotif(root=data_dir / dataset_name, b=b, mode='test')
        loaders, test_set = _get_loaders_and_test_set(
            batch_size,
            dataset_splits={'train': train_set, 'valid': valid_set, 'test': test_set},
            shuffle_train=shuffle_train,
        )

    elif dataset_name == 'mnist':
        n_train_data, n_val_data = 6000, 1500
        labels_to_keep = [0, 1, 2]
        train_val = MNIST75sp(data_dir / 'mnist', mode='train')
        perm_idx = _get_indices(torch.cat([sample.y for sample in train_val]),
                                n_train_data + n_val_data,
                                labels_to_keep=labels_to_keep,
                                random_state=random_state)
        train_val = train_val[perm_idx]

        train_set, valid_set = train_val[:n_train_data], train_val[n_train_data:]
        assert len(valid_set) == n_val_data
        test_set = MNIST75sp(data_dir / 'mnist', mode='test')
        test_set = test_set[
            torch.isin(torch.cat([sample.y for sample in test_set]), torch.tensor(labels_to_keep))
        ]
        loaders, test_set = _get_loaders_and_test_set(
            batch_size, dataset_splits={'train': train_set, 'valid': valid_set, 'test': test_set},
            shuffle_train=shuffle_train)
        print('[INFO] Using default splits!')

    elif dataset_name == 'zinc':
        # Zinc dataset, inspired from AD-GCL:
        # https://github.com/susheels/adgcl/blob/2605ef8f980934c28d545f2556af5cc6ff48ed18/test_minmax_zinc.py#L16
        # Note subset is set to True (in AD-GCL too), to have 12.000 graphs in total
        # rather than ~250.000.
        train_set = ZINC(data_dir / "zinc", subset=True, split='train')
        val_set = ZINC(data_dir / "zinc", subset=True, split='val')
        test_set = ZINC(data_dir / "zinc", subset=True, split='test')
        for set_ in [train_set, val_set, test_set]:
            set_.data.edge_attr = set_.data.edge_attr.unsqueeze(dim=1)
        loaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train),
            'valid': DataLoader(val_set, batch_size=batch_size, shuffle=False),
            'test': DataLoader(test_set, batch_size=batch_size, shuffle=False),
        }
        aux_info = {
            **aux_info,
            "num_atom_type": train_set.num_atom_type,
            "num_bond_type": train_set.num_bond_type,
        }

    else:
        raise NotImplementedError

    x_dim = train_set[0].x.shape[1]
    edge_attr_dim = 0 if train_set[0].edge_attr is None else train_set[0].edge_attr.shape[1]
    if dataset_name in classification_datasets:
        num_class = len(set([sample.y.item() for sample in train_set]))
        multi_label = len(train_set.y.shape) == 2 and train_set.y.shape[-1] > 1
    else:
        num_class = None
        multi_label = False

    print('[INFO] Calculating degree...')
    # Compute in-degree histogram over training data.
    # deg = torch.zeros(10, dtype=torch.long)
    # for data in train_set:
    #     d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    #     deg += torch.bincount(d, minlength=deg.numel())
    batched_train_set = Batch.from_data_list(train_set)
    d = degree(
        batched_train_set.edge_index[1], num_nodes=batched_train_set.num_nodes, dtype=torch.long)
    deg = torch.bincount(d, minlength=10)

    aux_info = {**aux_info, **{'deg': deg, 'multi_label': multi_label}}
    return loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info


def load_node_dataset(dataset_name, batch_size, edge_attr_dim=0):
    assert dataset_name.lower() in ["cora", "tree_cycle", "tree_grid"]

    if "cora" in dataset_name.lower():
        dataset = Planetoid(root=Path(C.PATH_DATA, 'Cora'), name='Cora')
    else:
        def pre_transform(data):
            """ Symmetrize edge_label_matrix. """
            assert torch.tril(
                data.edge_label_matrix
            ).sum() == 0  # Assert it was upper triangular
            data.edge_label_matrix = (
                data.edge_label_matrix + data.edge_label_matrix.T)
            assert torch.equal(data.edge_label_matrix, data.edge_label_matrix.long().double())
            data.edge_label_matrix = data.edge_label_matrix.long()
            return data
        dataset = SynGraphDataset(root=C.PATH_DATA, name=dataset_name, pre_transform=pre_transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    x_dim = dataset[0].x.shape[1]

    is_2d = len(dataset[0].y.shape) > 1 and dataset[0].y.shape[-1] > 1

    if is_2d:
        if any(dataset[0].y.sum(axis=1) > 1):
            raise NotImplementedError(
                "You may have attempted to pass a multilabel dataset. Those are not handled yet.")
        else:
            raise NotImplementedError(
                "You may have attempted to pass one hot encoded data. Those are not handled yet.")

    num_class = len(torch.unique(dataset[0].y))

    aux_info = {
        'multi_label': False,
        'deg': None
    }

    print(dataloader, x_dim, edge_attr_dim, num_class, aux_info)

    return dataloader, x_dim, edge_attr_dim, num_class, aux_info


def _get_random_split_idx(dataset, splits):
    print('[INFO] Randomly split dataset!')
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    n_train, n_valid = int(splits['train'] * len(idx)), int(splits['valid'] * len(idx))
    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train + n_valid]
    test_idx = idx[n_train + n_valid:]
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def _get_loaders_and_test_set(batch_size, dataset=None, split_idx=None, dataset_splits=None,
                              shuffle_train=True):
    if split_idx is not None:
        assert dataset is not None
        train_loader = DataLoader(
            dataset[split_idx["train"]], batch_size=batch_size, shuffle=shuffle_train)
        valid_loader = DataLoader(
            dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            dataset[split_idx["test"]], batch_size=batch_size, shuffle=False)
        test_set = dataset.copy(split_idx["test"])  # For visualization
    else:
        assert dataset_splits is not None
        train_loader = DataLoader(
            dataset_splits['train'], batch_size=batch_size, shuffle=shuffle_train)
        valid_loader = DataLoader(
            dataset_splits['valid'], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            dataset_splits['test'], batch_size=batch_size, shuffle=False)
        test_set = dataset_splits['test']  # For visualization
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, test_set

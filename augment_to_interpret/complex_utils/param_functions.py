"""
Contain functions that compute parameters for all runs.
"""

from pathlib import Path

from ..datasets import get_task


def deduce_batch_size(wildcards=None, dataset_name=None):
    """ Deduce the batchsize given the dataset name. (provide only wildcards or only dataset_name)

    :param wildcards: Snakefile wildcard;
    :param dataset_name: str; name of the dataset
    :return: int; batch size
    """
    assert (wildcards is None) != (dataset_name is None), "provide only wildcards or only dataset_name"
    if dataset_name is None:
        dataset_name = wildcards.dataset
    return (
        128 if dataset_name.lower() == "zinc" else
        128 if dataset_name.lower() == "cora" else
        128 if "ogbg" in dataset_name.lower() else
        256
    )


def deduce_model(wildcards):
    task = get_task(wildcards.dataset)
    if task == "node_classification":
        return "GIN_node"
    if wildcards.dataset.lower() == "zinc":
        return "ZincEncoder"
    if "ogbg" in wildcards.dataset.lower():
        return "MoleculeEncoder"
    return "GIN"


def get_result_dir_format(root_dir):
    return Path(
        str(root_dir),
        "r_info_loss_{r_info_loss}",
        "temperature_edge_sampling_{temperature_edge_sampling}",
        "{dataset}",
        "{seed}",
        "{loss}",
        "loss_weight_{loss_weight}",
        "feature_select_{use_features_selector}",
        "watchman_{use_watchman}",
    )


def get_result_dir(root_dir, args):
    return Path(str(get_result_dir_format(root_dir)).format(
        r_info_loss=f"{args.r_info_loss:.2f}",
        temperature_edge_sampling=f"{args.temperature_edge_sampling:.2f}",
        dataset=f"{args.dataset}",
        seed=f"{args.seed:0}",
        loss=f"{args.loss}",
        loss_weight=f"{args.loss_weight:.2f}",
        use_features_selector=f"{args.use_features_selector:0}",
        use_watchman=f"{args.use_watchman:0}"
    ))


def get_adgcl_args():
    return {
        "model_lr": 0.001,
        "view_lr": 0.001,
        "num_gc_layers": 3,
        "pooling_type": 'standard',
        "emb_dim": 64,
        "mlp_edge_model_dim": 64,
        "drop_ratio": 0.3,
        "epochs": 150,
        "reg_lambda": 5.0,
        "use_edge_attr": True,
    }


def get_mega_args():
    return {
        "num_gc_layers": 3,
        "pooling_type": 'standard',
        "emb_dim": 64,
        "mlp_edge_model_dim": 64,
        "drop_ratio": 0.3,
        "use_edge_attr": False,
        "model_lr": 0.001,
        "LGA_lr": 0.0001,
        "epochs": 50,
        "reg_expect": 0.4,
    }


def get_model_config(model_name, dataset_name, hidden_size, n_layers, dropout_ratio, aux_info=None):
    if 'GIN' in model_name:
        return {
            'model_name': model_name,
            'hidden_size': hidden_size,
            'n_layers': n_layers,
            'dropout_p': dropout_ratio,
            'use_edge_attr': True
        }
    if model_name == 'MoleculeEncoder':
        return {
            'model_name': model_name,
            'hidden_size': hidden_size,
            'num_gc_layers': n_layers,
            'drop_ratio': dropout_ratio,
            'pooling_type': 'standard',
            'use_edge_attr': True
        }
    if model_name.lower() == 'zincencoder':
        assert dataset_name.lower() == 'zinc', "ZincEncoder can only be used with the Zinc dataset"
        return {
            'model_name': model_name,
            'hidden_size': hidden_size,
            'num_atom_type': aux_info['num_atom_type'],
            'num_bond_type': aux_info['num_bond_type'],
            'num_gc_layers': n_layers,
            'drop_ratio': dropout_ratio,
            'pooling_type': 'standard',
            'use_edge_attr': True,
        }
    raise NotImplementedError("The BatchNorm should be handled correctly before using this model.")

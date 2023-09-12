"""
Contain helpers to deal with models.
"""

import os
import pickle
from pathlib import Path

import torch

from . import get_model, Criterion
from ..architectures.contrastive_model import (
    ExtractorMLP, EmbeddingWatchmanMLP, InstanceNodeAttention)


def save_model(saving_path, gsat, final_dic, name_model):
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path)
    print(f"Saving {name_model}")
    torch.save(gsat.state_dict(), Path(saving_path, "gsat_model"))
    with open(Path(saving_path, 'final_res_dictionary.pkl'), 'wb') as f:
        pickle.dump(final_dic, f)


def load_models(x_dim, edge_attr_dim, num_class, aux_info, model_config, device, task,
                learning_rate, learning_rate_watchman, watchman_eigenvalues_nb, hidden_dim_watchman, use_watchman,
                use_features_selector):
    clf = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'],
                    model_config, device)
    extractor = ExtractorMLP(model_config['hidden_size'], learn_edge_att="node" in task).to(device)

    tensors_to_optim = []

    tensors_to_optim += [
        {'params': list(extractor.parameters())},
        {'params': list(clf.parameters())},
    ]

    optimizer_features = None,
    if use_features_selector:
        features_extractor = InstanceNodeAttention(
            x_dim, use_sigmoid=True).to(device)

        optimizer_features = torch.optim.Adam(
            list(features_extractor.parameters()),
            lr=learning_rate,
            weight_decay=3.0e-6)

    else:
        features_extractor = None

    if use_watchman:
        # Watchman trying to re-predict laplacian eigenvalues
        # based on the embedding, to help stabilize the training
        watchman = EmbeddingWatchmanMLP(
            input_dim=model_config["hidden_size"],  # Dimension of the graph embedding
            hidden_size=hidden_dim_watchman,
            output_size=watchman_eigenvalues_nb
        ).to(device)

        # Differentiated LRs for different parts of the model
        tensors_to_optim += [
            {'params': list(watchman.parameters()), 'lr': learning_rate_watchman},
        ]

    else:
        watchman = None

    # Collate everything into a single optimizer
    optimizer = torch.optim.Adam(tensors_to_optim,
                                 lr=learning_rate,
                                 weight_decay=3.0e-6)

    criterion = Criterion(num_class, aux_info['multi_label'])
    return clf, extractor, watchman, optimizer, criterion, features_extractor, optimizer_features

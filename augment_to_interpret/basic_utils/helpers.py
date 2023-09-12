"""
Contain helpers.
"""

import csv
import pickle
import random
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from scipy.sparse.linalg import eigs, eigsh, ArpackError
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, get_laplacian, to_scipy_sparse_matrix, subgraph


def process_data(data, use_edge_attr, task="graph_classification", current_mask=None):
    """ Add edge_label if necessary, current_edge_mask and current_edge_index to data inplace.
    Remove edge_attr if use_edge_attr is set to False.
    """
    if not use_edge_attr:
        data.edge_attr = None
    if "graph" in task:
        if data.get('edge_label', None) is None:
            data.edge_label = torch.zeros_like(data.edge_index[0])
        data.current_edge_mask = torch.ones_like(data.edge_index[0])
        data.current_edge_index = data.edge_index
    else:
        # edge_label has priority, but if absent try to rebuild it
        if data.get('edge_label', None) is None:
            if data.get('edge_label_matrix', None) is not None:
                data.edge_label = torch.tensor([
                    data['edge_label_matrix'][edge[0], edge[1]].item()
                    for edge in data['edge_index'].T
                ], device=data.edge_index.device)
            else:
                # If there was no edge label, default to full zeros
                data.edge_label = torch.zeros_like(data.edge_index[0])

            if current_mask is None:
                current_mask = torch.ones_like(data.edge_index[0])
            current_nodes = torch.where(current_mask)[0]
            frm_mask = (data.edge_index[0].unsqueeze(1) - current_nodes.unsqueeze(0) == 0).any(dim=1)
            to_mask = (data.edge_index[1].unsqueeze(1) - current_nodes.unsqueeze(0) == 0).any(dim=1)
            data.current_edge_mask = torch.logical_and(frm_mask, to_mask).long()
            data.current_edge_index = data.edge_index[:, torch.logical_and(frm_mask, to_mask)]
        else:
            data.current_edge_mask = torch.ones_like(data.edge_index[0])
            data.current_edge_index = data.edge_index
    return data


def set_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if 'cuda' in str(device):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True) # TODO : Make the code return an exception, meaning there is at least one non-deterministic operation remaining. Find it.


def visualize_a_graph(edge_index, edge_att, node_label, dataset_name, coor=None, norm=False, mol_type=None,
                      nodesize=300):
    """ Greatly inpired by GSAT code. """
    plt.clf()
    title = f"Scores between {edge_att.min():.3f} and {edge_att.max():.3f}"
    if norm:
        denom = edge_att.max() - edge_att.min()
        if denom == 0:
            denom = 1
        edge_att = (edge_att - edge_att.min()) / denom

    if mol_type is None or dataset_name == 'Graph-SST2':
        atom_colors = {0: '#E49D1C', 1: '#FF5357', 2: '#a1c569', 3: '#69c5ba'}
        node_colors = [None for _ in range(node_label.shape[0])]
        for y_idx in range(node_label.shape[0]):
            node_colors[y_idx] = atom_colors[node_label[y_idx].int().tolist()]
    else:
        node_color = ['#29A329', 'lime', '#F0EA00', 'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
        element_idxs = {k: Chem.PeriodicTable.GetAtomicNumber(Chem.GetPeriodicTable(), v) for k, v in mol_type.items()}
        node_colors = [node_color[(v - 1) % len(node_color)] for k, v in element_idxs.items()]

    data = Data(edge_index=edge_index, att=edge_att, y=node_label, num_nodes=node_label.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])

    # calculate Graph positions
    if coor is None:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = {idx: each.tolist() for idx, each in enumerate(coor)}

    ax = plt.gca()
    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->",
                lw=max(data['att'], 0) * 3,
                alpha=max(data['att'], 0),  # alpha control transparency
                color='black',  # color control color
                shrinkA=np.sqrt(nodesize) / 2.0 + 1,
                shrinkB=np.sqrt(nodesize) / 2.0 + 1,
                connectionstyle='arc3,rad=0.05',
            ))

    if mol_type is not None:
        nx.draw_networkx_labels(G, pos, mol_type, ax=ax)

    if dataset_name != 'Graph-SST2':
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=nodesize, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax,
                               connectionstyle='arc3,rad=0.4')

    fig = plt.gcf()
    fig.suptitle(title, size=25)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return fig, image


def visualize_results(batch, batch_att, dataset_name, writer, tag, epoch):
    """ Greatly inpired by GSAT code (though cleaned a bit). """
    imgs = []
    assert len(batch) == len(np.unique(batch.batch)) == max(batch.batch) + 1
    for sample_id in range(len(batch)):
        data = batch[sample_id]
        mol_type, coor = None, None
        if dataset_name == 'mutag':
            node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S',
                         8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
            mol_type = {k: node_dict[v.item()] for k, v in enumerate(data.node_type)}

        node_subset = batch.batch == sample_id
        _, edge_att = subgraph(node_subset, batch.edge_index, edge_attr=batch_att)

        node_label = (data.node_label.reshape(-1) if data.get('node_label', None) is not None
                      else torch.zeros(data.x.shape[0]))
        fig, img = visualize_a_graph(
            data.edge_index, edge_att, node_label, dataset_name,
            norm=True, mol_type=mol_type, coor=coor
        )
        imgs.append(img)
    imgs = np.stack(imgs)
    writer.add_images(tag, imgs, epoch, dataformats='NHWC')


def _string_to_bool(value):
    if value in ["0", "False", ""]:
        return False
    if value in ["1", "True"]:
        return True
    raise NotImplementedError


def read_args(argspath):
    with open(argspath, mode='r') as infile:
        reader = csv.reader(infile, delimiter='\t')
        args = {rows[0]: rows[1] for rows in reader}
    types = {
        "dropout": float,
        "learning_rate": float,
        "learning_rate_watchman": float,
        "watchman_lambda": float,
        "loss_weight": float,
        "r_info_loss": float,
        "temperature_edge_sampling": float,
        "seed": int,
        "hidden_size": int,
        "batch_size": int,
        "n_layers": int,
        "epochs": int,
        "cuda": int,
        "watchman_eigenvalues_nb": int,
        "hidden_dim_watchman": int,
        "use_watchman": _string_to_bool,
        "use_features_selector": _string_to_bool,
    }
    for key, type_ in types.items():
        try:
            args[key] = type_(args[key])
        except KeyError:
            warnings.warn(f"Skipping {key} as it was not found.")
    return args


def get_device(cuda_id):
    if cuda_id >= 0 and not torch.cuda.is_available():
        cuda_id = -1
        warnings.warn("CUDA was asked for but seems unavailable on your machine. Using CPU.")
    elif cuda_id < 0 and torch.cuda.is_available():
        warnings.warn("CUDA seems available, but you asked to run on the CPU.")
    return torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu'), cuda_id


def get_laplacian_top_eigenvalues(
    data: Data,
    k: int,
    normalization: Optional[str] = None,
    is_undirected: bool = False,
) -> torch.Tensor:
    r"""
    Computes the N highest eigenvalues of the graph Laplacian given by
    `torch_geometric.utils.get_laplacian`

    Source : torch_geometric.utils

    Args:
        data (Data): The graph to operate on

        k (int) : The number of eigenvalues to return. If higher than N-2 where N is the number of nodes, will be reduced to N-1

        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`None`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of the largest eigenvalue. (default: :obj:`False`)
    """
    assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

    # Cap nb. of eigenvalue to dimension of matrix - 2
    k_real = min(k, data.num_nodes - 2)
    if data.num_nodes < 3: raise ValueError("Cannot query eigenvalues of graph Laplacian with fewer than 3 nodes")

    # Collect edge weight (if absent, use None)
    edge_weight = data.edge_attr
    if edge_weight is not None and edge_weight.numel() != data.num_edges:
        edge_weight = None

    # Compute Laplacian and turn it to scipy sparse matrix
    edge_index, edge_weight = get_laplacian(data.edge_index, edge_weight,
                                            normalization,
                                            num_nodes=data.num_nodes)

    L = to_scipy_sparse_matrix(edge_index, edge_weight, data.num_nodes)

    # Compute Eigenvalues
    eig_fn = eigs
    if is_undirected and normalization != 'rw':
        eig_fn = eigsh

    n_fail = 0
    maxiter = 10 * L.shape[0]  # default value of the function
    try:
        while True:
            try:
                lambda_max = eig_fn(L,
                                    k=k_real,
                                    which='LM',  # Largest in magnitude
                                    return_eigenvectors=False,
                                    maxiter=maxiter)
                break
            except ArpackError:
                maxiter *= 2
                n_fail += 1
                warnings.warn(
                    f"Tryed {n_fail} times to obtain get_laplacian_top_eigenvalues, "
                    f"still an error. To stop trying, send a KeyboardInterrupt signal.")
    except KeyboardInterrupt:
        warnings.warn("get_laplacian_top_eigenvalues failed to converge, return a vector of 1")
        return torch.ones(k, device=edge_index.device)
    if n_fail > 0:
        warnings.warn(f"Succeeded at try {n_fail + 1}.")

    # Return float of real value only, and ensure the values are sorted in consistent order
    sorted_lambdamax_real = np.sort(
        lambda_max.real
    )
    lambda_max = torch.tensor(sorted_lambdamax_real, dtype=torch.float, device=edge_index.device)

    # Fill missing expected values
    result = torch.cat((
        torch.zeros(k - k_real, dtype=torch.float, device=edge_index.device),
        lambda_max
    ))
    return result.detach()


def read_final_res_dictionary(path_to_dir):
    file_path = Path(path_to_dir, "final_res_dictionary.pkl")
    with open(file_path, 'rb') as f:
        res = pickle.load(f)

    flat_res = {
        'exp': Path(path_to_dir).parts[-1],
        'path_root': str(path_to_dir)
    }
    for k1 in res.keys():
        if k1 in ['train', 'test', 'valid']:
            for k2 in res[k1].keys():
                flat_res[f'{k2}_{k1}'] = res[k1][k2]
        else:
            flat_res[f'{k1}'] = res[k1]
    return flat_res

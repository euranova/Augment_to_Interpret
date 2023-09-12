"""
Visualisation method for graph explanations.
"""

from pathlib import Path

import torch
from PIL import Image

from ..basic_utils import visualize_a_graph


def _visualise_edge_explanation(graph, explanation):
    edge_mask = explanation
    node_label = (
        graph.node_label.reshape(-1) if graph.get('node_label', None) is not None
        else torch.zeros(graph.x.shape[0])
    )
    fig, img = visualize_a_graph(
        graph.edge_index,
        edge_mask,
        node_label,
        "some_name",  # TODO Put real dataset name, but it seems to be only used to color the nodes
        norm=False,
        mol_type=None,
        coor=None
    )
    return img


def print_visualisation(graph, edge_attention_matrix, path_results, identifier):
    outdir = Path(path_results, "visualisation")
    outdir.mkdir(exist_ok=True, parents=True)
    imarray = _visualise_edge_explanation(graph, edge_attention_matrix)
    im = Image.fromarray(imarray)
    im.save(Path(outdir, f"{identifier}.jpeg"))

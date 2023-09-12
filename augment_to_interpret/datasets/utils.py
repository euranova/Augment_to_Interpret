"""
Functions and constants to deal with the datasets.
"""

NODE_CLASSIF_DATASETS = ["tree_grid", "tree_cycle", "cora"]
GRAPH_CLASSIF_DATASETS = ["ba_2motifs", "mutag", "spmotif_0.5", "mnist"]
DATASETS = NODE_CLASSIF_DATASETS + GRAPH_CLASSIF_DATASETS
assert len(set(DATASETS)) == len(NODE_CLASSIF_DATASETS) + len(GRAPH_CLASSIF_DATASETS), (
    "No dataset should be in both node and graph lists.")


def get_task(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name in NODE_CLASSIF_DATASETS:
        return "node_classification"
    if dataset_name in GRAPH_CLASSIF_DATASETS:
        return "graph_classification"
    raise NotImplementedError

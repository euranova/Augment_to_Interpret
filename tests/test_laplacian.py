""" Test the laplacian functions """
import torch
from torch_geometric.data import Data

from augment_to_interpret.basic_utils.helpers import get_laplacian_top_eigenvalues


def test_it_all():
    # Testing graph
    edge_index = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long).t()
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    graph = Data(x=x, edge_index=edge_index.contiguous())

    graph_laplacian_eigenvalues = get_laplacian_top_eigenvalues(graph, k=5)
    # NOTE : Intentionally querying more eigenvalues than possible to check if k is automatically reduced
    assert torch.allclose(graph_laplacian_eigenvalues, torch.Tensor([0, 0, 0, 0, 3.0]))

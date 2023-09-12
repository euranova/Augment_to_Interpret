""" Test all cluster quality metrics """
import numpy as np
import torch

from augment_to_interpret.metrics.cluster_quality_metrics import (
    _compute_entropy, _compute_mutual_information, compute_normalized_mutual_information,
    compute_optimal_normalized_mutual_information, compute_mean_cluster_std
)


def test_everything_at_once():
    assert abs(_compute_entropy(torch.tensor([1, 1, 2, 2, 5, 5, 0, 0])) - 2) < 1e-10
    assert abs(_compute_entropy([1, 0, 1, 0]) - 1) < 1e-10
    assert abs(_compute_entropy([1, 1, 1, 0]) - 0.8113) < 1e-3

    assert abs(_compute_mutual_information([1, 1, 2, 2], torch.tensor([5, 5, 3, 3])) - 1) < 1e-10
    assert abs(_compute_mutual_information([1, 1, 2, 2], [5, 3, 5, 3]) - 0) < 1e-10
    assert abs(_compute_mutual_information(torch.tensor([1, 1, 2, 2]), [5, 5, 2, 3]) - 1) < 1e-10

    assert abs(compute_normalized_mutual_information(
        [1, 1, 5, 3, 4, 5], torch.tensor([3, 3, 9, 0, 1, 9])
    ) - 1) < 1e-10
    assert abs(compute_normalized_mutual_information(np.random.randint(0, 10, (10000,)),
                                                     torch.randint(0, 10, (10000,)))) < 0.01
    assert compute_normalized_mutual_information([1, 1, 5, 3, 4, 5], [1, 2, 3, 4, 5, 3]) < 0.99

    gt_labels = torch.tensor([0, 0, 0, 1, 1, 0, 0, 0])
    embeddings = torch.tensor([[1], [2], [3], [10], [11], [20], [21], [22]])
    expected_clusters = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    assert (
        compute_optimal_normalized_mutual_information(embeddings, gt_labels)
        == compute_normalized_mutual_information(expected_clusters, gt_labels)
    )

    assert abs(compute_mean_cluster_std(
        torch.tensor([0, 1, 0, 0, 1, 1, 1]), [0.0, 2.0, 4.0, 8.0, 1.0, 6.0, 3.0]
    ) - (((32 / (3 - 1)) ** 0.5 + (14 / (4 - 1)) ** 0.5) / 2)) < 1e-7
    assert abs(compute_mean_cluster_std(
        [-3, 5, -3, -3, 5, 5, 5, 7, 7], torch.tensor([0.0, 2.0, 4.0, 8.0, 1.0, 6.0, 3.0, 9.0, 9.0])
    ) - (((32 / (3 - 1)) ** 0.5 + (14 / (4 - 1)) ** 0.5 + 0) / 3)) < 1e-7

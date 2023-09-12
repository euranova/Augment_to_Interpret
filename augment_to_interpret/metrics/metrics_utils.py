"""
Contain helpers for metric functions
"""

import numpy as np
import torch
from scipy.stats import pearsonr, ttest_ind


def collect_lower_triangle_without_diagonal(matrix):
    flat = []
    assert matrix.shape[0] == matrix.shape[1], "Matrix is not square"
    N = matrix.shape[0]
    for i in range(N):
        for j in range(i):
            flat.append(matrix[i, j])
    return flat


def eucldist(a, b):
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    return (a - b).pow(2).sum().sqrt()


def select_local_neighborhood(mat, k=.1, reverse=False):
    """
    Given a distance matrix (numpy format), returns a mask giving, for each row, the smallest values.
    
    Note that we assume the matrix is lower triangular (hence the diagnonal and the upper triangle will be ignored)
    
    Arguments:
        mat : the distance matrix
        k : the ratio to select, from 0 to 1 (for example, k=.1 means we take the smallest 10% of values each row)
        reverse : if true, returns the largest instead of the smallest
    """
    assert np.all(mat == mat.T)
    mat = mat.astype(float)
    mask = np.zeros(mat.shape)

    n = max(int(k * mat.shape[1]), 1)

    # Remove top triangle
    utind = np.diag_indices(mat.shape[0])
    parasite = np.inf if (not reverse) else -np.inf
    mat[utind] = parasite

    for i, row in enumerate(mat):

        if reverse:
            row_indices = np.argsort(row)[-n:]
        else:
            row_indices = np.argsort(row)[:n]
        mask[i, row_indices] = 1

    return mask.astype(bool)


def correl(first_distmat, second_distmat):
    """
    Takes as input two distance matrices (numpy format)
    Returns their local, global and distant correlations (see code for details)
    Locality is based on the closest elements in the FIRST distance matrix
    
    Note : distance matrices must be symmetrical.
    """
    assert np.all(first_distmat == first_distmat.T)
    assert np.all(second_distmat == second_distmat.T)

    firsttri = collect_lower_triangle_without_diagonal(first_distmat)
    secondtri = collect_lower_triangle_without_diagonal(second_distmat)
    global_correlation = pearsonr(firsttri, secondtri)[0]

    local_mask = select_local_neighborhood(first_distmat)
    distant_mask = select_local_neighborhood(first_distmat, reverse=True)

    local_correlation = pearsonr(first_distmat[local_mask], second_distmat[local_mask])[0]
    distant_correlation = pearsonr(first_distmat[distant_mask], second_distmat[distant_mask])[0]

    # In the second dist matrix, are the local values (local in the first distmatrix) significantly below the mean value ?
    local_second_signif_lower = ttest_ind(np.array(secondtri), second_distmat[local_mask].flatten())

    corr_results = {
        "global_correlation": global_correlation,
        "local_correlation": local_correlation,
        "distant_correlation": distant_correlation,
        "local_second_signif_lower": local_second_signif_lower,
        "mean_first_distance": np.mean(first_distmat),
        "mean_second_distance": np.mean(second_distmat),
        "mean_first_distance_local": np.mean(first_distmat[local_mask]),
        "mean_second_distance_local": np.mean(second_distmat[local_mask]),
    }

    return corr_results

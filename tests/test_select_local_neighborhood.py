"""
Test the select_local_neighborhood function
"""

import numpy as np

from augment_to_interpret.metrics.metrics_utils import select_local_neighborhood


def test_it_all():
    my_distance_matrix = np.array([
        [0, 7, 2, 4],
        [7, 0, 5, 3],
        [2, 5, 0, 6],
        [4, 3, 6, 0],
    ])
    expected_output = np.array([
        [False, False, True, True],
        [False, False, True, True],
        [True, True, False, False],
        [True, True, False, False],
    ])  # Not completely sure if we want to take the main diagonal into account: we select 0.5*3 samples or 0.5*4 samples?
    res = select_local_neighborhood(my_distance_matrix, k=0.5)
    assert np.all(res == expected_output)
    expected_output_reverse = np.array([
        [False, True, False, True],
        [True, False, True, False],
        [False, True, False, True],
        [True, False, True, False],
    ])
    res_reverse = select_local_neighborhood(my_distance_matrix, k=0.5, reverse=True)
    assert np.all(res_reverse == expected_output_reverse)

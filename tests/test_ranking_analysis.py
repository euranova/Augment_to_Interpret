""" Test the ranking analysis """
from pathlib import Path

import pandas as pd

from augment_to_interpret.complex_utils.ranking_analysis import ranking_regression_analysis
from .utils import test_C


def test_it_all():
    # Read tsv
    df = pd.read_csv(Path(test_C.PATH_TEST_FILES, "placeholder_node_features.tsv"), sep='\t', index_col=0)

    rd = ranking_regression_analysis(
        source_ranking=df["weights_ranking"].values,
        reference_ranking=df["gsat_ranking_weights"].values,
        background_ranking=df["entropy"].values
    )
    print(rd)

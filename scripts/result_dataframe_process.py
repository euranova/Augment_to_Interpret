"""
This script takes as input the posthoc_embedding_analysis_final.tsv and the training_curve_analysis_final.tsv
and produces nice tables.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from augment_to_interpret.basic_utils import C

if __name__ == "__main__":
    PATH_RESULTS = Path(C.PATH_RESULTS, "main")

    posthoc_df = pd.read_csv(Path(PATH_RESULTS, "posthoc_embedding_analysis_final.tsv"), sep='\t')
    training_curve_df = pd.read_csv(Path(PATH_RESULTS, "training_curve_analysis_final.tsv"), sep='\t')

    ## BEST MODEL
    # Statistics on the best saved models per criterion

    ids = [
        "dataset", "loss", "use_watchman", "exp", "loss_weight", "task"
    ]

    vars_of_interest = [
        "DownstreamClf_acc_train", "DownstreamClf_acc_test",
        "Interp_acc_train", "Interp_acc_test",
        "train_sil", "test_sil"
    ]
    if "wass_aug_score" in posthoc_df.columns: vars_of_interest.append("wass_aug_score")

    cols_of_interest = ids + vars_of_interest
    podf = posthoc_df[cols_of_interest + ["seed"]]

    grouped_podf = podf.groupby(ids)

    means = grouped_podf.mean().reset_index().round(6)
    stds = grouped_podf.std().reset_index().round(6)
    mins = grouped_podf.min().reset_index().round(6)
    maxs = grouped_podf.max().reset_index().round(6)

    excel_results = means[ids].copy()
    for col in vars_of_interest:
        excel_results.loc[:, col] = (means[col].astype(str) + ' ( +/-' +
                                     stds[col].astype(str) + ')').values
        excel_results.loc[:, col + "_interval"] = ('[' + mins[col].astype(str) + ' ; ' +
                                                   maxs[col].astype(str) + ']').values

    excel_results.to_csv(Path(PATH_RESULTS, "summary_results.tsv"), sep='\t')

    ## TRAINING CURVE
    # Statistics on the training process
    groupbyvars = ["use_watchman", "loss", "dataset", "loss_weight"]
    grouped_training = training_curve_df.groupby(groupbyvars)

    w_means = grouped_training.mean().round(6).reset_index()
    w_mins = grouped_training.min().round(6).reset_index()
    w_maxs = grouped_training.max().round(6).reset_index()

    vars = ["last_third_downstream_acc", "last_third_attention_acc"]
    for v in vars:
        w_means.loc[:, v] = (w_means[v + "_mean"].astype(str) + ' ( +/- '
                             + np.sqrt(w_means[v + "_variance"]).round(6).astype(str) + ')')
        w_means.loc[:, v + "_mean_interval"] = ('[' + w_mins[v + "_mean"].astype(str) + ' ; ' +
                                                w_maxs[v + "_mean"].astype(str) + ']').values

    res = w_means[groupbyvars + vars + [v + "_mean_interval" for v in vars]]
    res.to_csv(Path(PATH_RESULTS, "training_curve_grouped.tsv"), sep='\t')

"""
Post-hoc analysis of the events recorded by each Tensorboard summary.
"""

import csv
import glob
import pathlib
from collections import OrderedDict
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import element_blank, element_line, element_text
from plotnine import ggplot, aes, theme
from plotnine.geoms import geom_point
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm

from augment_to_interpret.basic_utils import C

if __name__ == "__main__":
    PATH_RESULTS = Path(C.PATH_RESULTS, "main")


    def fetch_values(event_acc, metric_name):
        # E. g. get wall clock, number of steps and value for a scalar named `metric_name`
        try:
            w_times, step_nums, vals = zip(*event_acc.Scalars(metric_name))
            return vals
        except Exception:
            # print("Could not find", metric_name, ", skipping")
            return [-1, -1, -1]


    df = pd.DataFrame()
    full_data_df = pd.DataFrame()

    full_data_chunks = []

    # For each run experiment...
    all_paths = glob.glob(str(PATH_RESULTS) + '/**/*.tfevents.*', recursive=True)

    if len(all_paths) == 0:
        raise FileNotFoundError(
            f"The {PATH_RESULTS} directory does not contain any results. Check that your experiments were run properly, and that you have selected the correct result directory by setting the RESULT_DIR in the training_curve_analysis.py file.")

    print("Reading Tensorboard logs...")
    for log_path in tqdm(all_paths):

        # print("Processing "+log_path)

        # Fetch the args dictionary which was saved by main.py, one directory up
        parentdir_path = pathlib.Path(log_path).parent.parent
        argspath = Path(parentdir_path, "full_args.tsv")
        with open(argspath, mode='r') as infile:
            reader = csv.reader(infile, delimiter='\t')
            args = {rows[0]: rows[1] for rows in reader}
        args = OrderedDict(args)

        # Read the tensorboard data
        event_acc = EventAccumulator(log_path)
        event_acc.Reload()

        # TODO : if there are any new metrics to be calculated, add them to the new_row dictionary.
        new_row = OrderedDict()
        # --------------------- Calculate values of interest --------------------- #

        # print(event_acc.Tags()['Scalars']) # List of available metrics

        downstream_accuracies_test = fetch_values(event_acc, 'DownstreamClfAcc/test')

        da1, da2, da3 = np.array_split(downstream_accuracies_test, 3)
        new_row["first_third_downstream_acc_mean"] = np.mean(da1)
        new_row["first_third_downstream_acc_variance"] = np.var(da1)
        new_row["middle_third_downstream_acc_mean"] = np.mean(da2)
        new_row["middle_third_downstream_acc_variance"] = np.var(da2)
        new_row["last_third_downstream_acc_mean"] = np.mean(da3)
        new_row["last_third_downstream_acc_variance"] = np.var(da3)

        attention_accuracies_valid = fetch_values(event_acc, 'AttentionAcc/valid')

        da1, da2, da3 = np.array_split(attention_accuracies_valid, 3)
        new_row["first_third_attention_acc_mean"] = np.mean(da1)
        new_row["first_third_attention_acc_variance"] = np.var(da1)
        new_row["middle_third_attention_acc_mean"] = np.mean(da2)
        new_row["middle_third_attention_acc_variance"] = np.var(da2)
        new_row["last_third_attention_acc_mean"] = np.mean(da3)
        new_row["last_third_attention_acc_variance"] = np.var(da3)

        watchman_losses_train = fetch_values(event_acc, 'Loss/train_watchman_loss')
        _, _, l3 = np.array_split(watchman_losses_train, 3)
        new_row["last_third_watchman_train_loss_mean"] = np.mean(l3)

        watchman_losses_test = fetch_values(event_acc, 'Loss/valid_watchman_loss')
        _, _, l3 = np.array_split(watchman_losses_test, 3)
        new_row["last_third_watchman_test_loss_mean"] = np.mean(l3)

        # -------------------------- Store results ------------------------------- #
        fulld = OrderedDict(chain(args.items(), new_row.items()))
        r = pd.Series(fulld)
        df = pd.concat([df, r.to_frame().T], ignore_index=True)

        # ---- Store full results (all epochs)
        for i, (x, y) in enumerate(zip(downstream_accuracies_test, attention_accuracies_valid)):
            new_row_epoch = {
                "epoch": i,
                "downstream_acc_test": x,
                "attention_acc_valid": y,
            }
            fulldf = OrderedDict(chain(args.items(), new_row_epoch.items()))
            full_data_chunks.append(pd.Series(fulldf).to_frame().T)

    full_data_df = pd.concat(full_data_chunks, ignore_index=True)

    ## Export results
    df.to_csv(Path(PATH_RESULTS, "training_curve_analysis_final.tsv"), sep='\t')
    full_data_df.to_csv(Path(PATH_RESULTS, "training_curve_full_raw_data.tsv"), sep='\t')


    ## Plot results

    def smart_plot(full_data_df, condition, value,
                   criterion="loss"):
        # Fix data type
        for k in ["epoch", "seed", value, "use_watchman", "loss_weight"]:
            full_data_df.loc[:, k] = pd.to_numeric(full_data_df[k])
        full_data_df.loc[:, "loss"] = full_data_df["loss"].astype("str")

        # Get only the rows that satisfy the condition in the dictionary
        subdf = full_data_df[
            np.logical_and.reduce(
                [full_data_df[k] == v for k, v in condition.items()]
            )
        ].copy()

        # Plot all seeds, colored by condition which is an argument
        for k in ["epoch", "seed", value, "use_watchman", "loss_weight"]:
            subdf.loc[:, k] = pd.to_numeric(subdf[k])
        subdf.loc[:, "loss"] = subdf["loss"].astype("str")

        subdf = subdf[["epoch", "loss", value, "seed", "use_watchman", "loss_weight"]].copy()

        # Unique ID for each loss (or other criterion) and seed combination
        subdf.loc[:, "id"] = subdf[criterion].astype(str) + '_' + subdf["seed"].astype(str)
        print(subdf["id"].unique())

        p = ggplot(
            subdf,
            aes(
                x='epoch',
                y=value,
                group="id",
                color=criterion  # Same criterion shares a color
            )
        )
        p += geom_point(size=.1, alpha=.2)
        # p += geom_smooth(method='loess', span=.1) # TODO : may cause segmentation fault ? Commented it to prevent it. Try re-enabling it later.

        # Visual theme
        p += theme(
            legend_title_align="center",
            legend_box_spacing=0.4,
            axis_line=element_line(size=1, colour="black"),
            panel_grid_major=element_line(colour="#d3d3d3"),
            panel_grid_minor=element_blank(),
            panel_border=element_blank(),
            panel_background=element_blank(),
            plot_title=element_text(size=15, family="Tahoma",
                                    face="bold"),
            text=element_text(size=11),
            axis_text_x=element_text(colour="black", size=10),
            axis_text_y=element_text(colour="black", size=10),
        )

        return p


    # For each dataset, then for each metric of interest, save the plot
    print("Plotting...")
    for dataset in full_data_df["dataset"].unique():
        for value in ["downstream_acc_test", "attention_acc_valid"]:

            # Output paths
            outdir = Path(PATH_RESULTS, "plots", dataset)
            outdir.mkdir(exist_ok=True, parents=True)

            # Compare losses, without watchman and loss weighting
            cond = {"dataset": dataset, "use_watchman": 0, "loss_weight": 1.0}
            p = smart_plot(full_data_df, cond, value, criterion="loss")
            p.save(Path(outdir, f"loss_impact_on_{value}.png"))

            # Assess *separately* impact of watchman and loss weighting
            # For each loss a separate plot
            all_losses = full_data_df["loss"].unique()

            for loss in all_losses:
                this_outdir = Path(outdir, loss)
                this_outdir.mkdir(exist_ok=True, parents=True)

                cond = {"dataset": dataset, "loss": loss, "loss_weight": 1.0}
                p = smart_plot(full_data_df, cond, value, criterion="use_watchman")
                p.save(Path(this_outdir, f"watchman_impact_on_{value}.png"))

                cond = {"dataset": dataset, "loss": loss, "use_watchman": 0}
                p = smart_plot(full_data_df, cond, value, criterion="loss_weight")
                p.save(Path(this_outdir, f"weight_impact_on_{value}.png"))

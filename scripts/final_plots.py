"""
Generate the plots present in the paper from the previously computed results.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_point, theme, element_blank, element_line, element_text, facet_wrap,
    xlab, ylab, labs, scale_color_manual, geom_tile, geom_text, geom_bar, coord_flip,
    scale_fill_manual, scale_y_discrete, scale_x_discrete, stat_summary, scale_fill_cmap, ggsave,
    geom_boxplot, scale_y_continuous,
)

from augment_to_interpret.basic_utils import C

pd.options.display.max_columns = 100
parser = argparse.ArgumentParser(description='Compute plots and tables.')
parser.add_argument('--baseline_type', type=str, help='Stop criterion to be used',
                    default="best_downstream_clf_model")
args = parser.parse_args()
baseline_type = args.baseline_type

# -------------------------------- Input files ------------------------------- #
PATH_MAIN_RESULTS = Path(C.PATH_RESULTS, "main")

posthoc_result = pd.read_csv(
    Path(PATH_MAIN_RESULTS, "posthoc_embedding_analysis_final.tsv"), sep='\t', index_col=0)
watchman_df = pd.read_csv(
    Path(PATH_MAIN_RESULTS, "training_curve_analysis_final.tsv"), sep='\t', index_col=0)
sparsity_result = pd.read_csv(
    Path(C.PATH_RESULTS, "sparsity_final_results.tsv"), sep='\t', index_col=0)
adgcl_posthoc_result = pd.read_csv(
    Path(C.PATH_RESULTS, "adgcl", f"posthoc_final_result_for_{baseline_type}.tsv"), sep='\t', index_col=0)
mega_posthoc_result = pd.read_csv(
    Path(C.PATH_RESULTS, "mega", f"posthoc_final_result_for_{baseline_type}.tsv"), sep='\t', index_col=0)

# Add the elements not given in the originals

adgcl_posthoc_result['loss'] = 'adgcl'
mega_posthoc_result['loss'] = 'mega'
adgcl_posthoc_result['use_watchman'] = False
mega_posthoc_result['use_watchman'] = False
adgcl_posthoc_result['task'] = mega_posthoc_result['task'] = 'graph_classification'
adgcl_posthoc_result['exp'] = mega_posthoc_result['exp'] = baseline_type

# Merge with existing results
merged_result = pd.concat([
    posthoc_result, adgcl_posthoc_result, mega_posthoc_result,
], axis=0, ignore_index=True)

N_SEEDS = merged_result["seed"].nunique()
# -------------------------------- Parameters -------------------------------- #
# Renaming and ordering
translation_dict = {
    'simclr_double_aug_info_negative': "\u2112",
    'simclr_double_aug': "\u2112 - Negative - Info",
    'gsat': "GSAT",
    'adgcl': "AD-GCL",
    'mega': "MEGA",
    'simclr_double_aug_info': "\u2112 - Negative",
    'simclr_double_aug_negative': "\u2112 - Info",
    'simclr_double_aug_info_simclrnegative': "DAISN",
    'simclr_double_aug_simclrnegative': "DASN"
}
ordered_losses = [
    'simclr_double_aug',
    'simclr_double_aug_info',
    'simclr_double_aug_negative',
    'simclr_double_aug_info_negative',
    'mega',
    'adgcl',
    'gsat',
]
facet_labels = {
    "ba_2motifs": "BA2Motifs",
    "mutag": "Mutag",
    "spmotif_0.5": "SPMotif 0.5",
    "tree_grid": "Tree Grid",
    "tree_cycle": "Tree Cycle",
    "cora": "Cora",
    "mnist": "MNIST",
    "graph_classification": "Graph classification",
    "node_classification": "Node classification"
}
translated_losses = [translation_dict[item] for item in ordered_losses]

repdi = {
    "use\_watchman": "Wm.",
    "False": "",
    "True": "\checkmark",
    "{train\_set\_fidelity}": "{Global Fid.}",
    "{train\_set\_fidelity\_scrambled}": "{Scrambled Fid.}",
    "{train\_set\_fidelity\_opposite}": "{Opposite Fid.}",
    "{train\_set\_sparsity}": "{Sparsity}",
    "DownstreamClf\_acc\_test": "Downstream AUC",
    "Interp\_acc\_test": "Intepretability AUC",
    "loss": "Loss",
    "eucl\_wass\_mean\_second\_distance\_local": "$W_1$ local",
    "eucl\_wass\_mean\_second\_distance": "$W_1$ global",

    'gsat': "GSAT",
    'adgcl': "AD-GCL",
    'mega': "MEGA",

    # Careful to apply longer first since they are applied in order
    'simclr\_double\_aug\_info\_negative': "$\mathcal{L}$",
    'simclr\_double\_aug\_info': "$\mathcal{L}$ - Negative",
    'simclr\_double\_aug\_negative': "$\mathcal{L}$ - Info",
    'simclr\_double\_aug': "$\mathcal{L}$ - Negative - Info",
    'simclr\_double\_aug\_info\_simclrnegative': "DAISN",
    'simclr\_double\_aug\_simclrnegative': "DASN",
    "ba\_2motifs": "BA2Motifs",
    "mutag": "Mutag",
    "spmotif\_0.5": "SPMotif 0.5",
    "tree\_grid": "Tree Grid",
    "tree\_cycle": "Tree Cycle",
    "cora": "Cora",
    "mnist": "MNIST",
    "graph\_classification": "Graph classification",
    "node\_classification": "Node classification",

    "0.": ".",
    "1.000": "1"
}

r_translate_dict = {
    'simclr_double_aug_info_negative': "\u2112",
    'gsat': "GSAT",
    "ba_2motifs": "BA2Motifs",
    "mutag": "Mutag",
}

################################### FIGURES ####################################

theme_common = theme(
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

# --------------------------------- Sparsity --------------------------------- #

sparids = ["dataset", "seed", "loss"]
cdfs = ["cdf_0." + str(i) for i in range(1, 10)] + ["cdf_1.0"]
sparsity_result = sparsity_result[
    (sparsity_result["r_info_loss"].isna() | (sparsity_result["r_info_loss"] == 0.7))
    & (sparsity_result["dataset"].isin(["ba_2motifs", "mutag", "spmotif_0.5"]))  # Only graph datasets
    & (sparsity_result["loss"].isin(["adgcl", "mega"]) | sparsity_result["use_watchman"])  # Only graph datasets
][sparids + cdfs].copy()

# Difference
for idx, cdf in enumerate(cdfs[1:], start=1):
    sparsity_result[f"temp_{cdf}"] = sparsity_result[cdf] - sparsity_result[cdfs[idx - 1]]
for cdf in cdfs[1:]:
    sparsity_result[cdf] = sparsity_result[f"temp_{cdf}"]
    del sparsity_result[f"temp_{cdf}"]

sparsity_result = sparsity_result.melt(id_vars=sparids)
grouped_sparsity = sparsity_result.groupby(["dataset", "loss", "variable"], dropna=False)
assert np.all(grouped_sparsity.count() == N_SEEDS), (
    "The number of entries for one of the sparsity results is not correct. "
    "This may be because different hyperparameters have been treated as the same experiment, "
    "or because some seeds have not run for all experiments.",
    grouped_sparsity.count()
)
sparsity_result = grouped_sparsity.mean().reset_index()

p = (
    ggplot(sparsity_result, aes('variable', 'loss', fill='value'))
    + geom_tile(aes(width=.95, height=.95))
    + geom_text(aes(label='np.around(value, 2)'), size=9)
)
p += facet_wrap("dataset", labeller=lambda x: facet_labels[x])
p += scale_x_discrete(limits=cdfs, labels=range(1, 11))
p += theme_common
p += theme(
    legend_position=None,
    figure_size=(14, 3)
)
p += xlab('Density decile') + ylab("Loss") + labs(fill="Density")

ticks_sparsity = [
    'gsat',
    'adgcl',
    'mega',
    'simclr_double_aug_info_negative',
    'simclr_double_aug_negative',
    'simclr_double_aug_info',
    'simclr_double_aug',
]
ticks_sparsity.reverse()
p += scale_y_discrete(
    limits=ticks_sparsity,
    labels=[translation_dict[item] for item in ticks_sparsity]
)
p += scale_fill_cmap(cmap_name='Wistia', guide=False)
ggsave(plot=p, filename=Path(C.PATH_FIGURES, "sparsities.png"), dpi=400)

# --------------------------------- Big results table --------------------------------- #

TASKS = ["graph_classification", "node_classification"]

new_index = [
    'gsat',
    'adgcl',
    'mega',
    'simclr_double_aug_info_negative',
    'simclr_double_aug_negative',
    'simclr_double_aug_info',
    'simclr_double_aug'
]

with open(Path(C.PATH_RESULTS, "final_table_latex.tex"), "w") as text_file:
    for task in TASKS:

        # filter
        final_result = merged_result[
            (merged_result["task"] == task)
            & (merged_result["exp"] == baseline_type)
            & (merged_result["dataset"] != "mnist")
            & (merged_result["loss"] != "simclr_double_aug_info_simclrnegative")
            & (merged_result["loss"] != "simclr_double_aug_simclrnegative")
            & (merged_result["r_info_loss"].isna() | (merged_result["r_info_loss"] == 0.7))
        ].copy()

        future_col = ["dataset"]
        future_row = ["use_watchman", "loss"]

        # Derived values
        final_result["fidelity_difference"] = (
            final_result["train_set_fidelity"]
            - final_result["train_set_fidelity_opposite"]
        )
        final_result["wasserstein_difference"] = (
            final_result["eucl_wass_mean_second_distance"]
            - final_result["eucl_wass_mean_second_distance_local"]
        )

        # Which tables do we want to make ? List them here
        VALUES_COL = [
            ["train_set_fidelity", "train_set_fidelity_opposite"],
            ["train_set_fidelity_scrambled", "train_set_sparsity"],
            ["DownstreamClf_acc_test", "Interp_acc_test"],
            ["eucl_wass_mean_second_distance", "eucl_wass_mean_second_distance_local"],
            ["fidelity_difference", "wasserstein_difference"],
        ]
        final_result = final_result[future_row + future_col + list(itertools.chain(*VALUES_COL))]
        grouped_final_result = final_result.groupby(future_row + future_col, dropna=False)
        assert np.all(grouped_final_result.count().isin([0, N_SEEDS])), (
            "The number of entries for one of the final results is not correct. "
            "This may be because different hyperparameters have been treated as the same experiment, "
            "or because some seeds have not run for all experiments.",
            grouped_final_result.count()
        )
        df_std = grouped_final_result.std().reset_index()
        df_mean = grouped_final_result.mean().reset_index()
        for values_col in VALUES_COL:
            df = df_mean[future_row + future_col].copy()
            for val_col in values_col:
                df[val_col] = (
                    df_mean[val_col]
                    .apply(lambda x: f"{x:.3f}")
                    .str.cat(df_std[val_col].apply(lambda x: f"{x:.3f}"), sep=' \\pm ')
                    .apply(lambda x: f"${x}$")
                )
            df = df.pivot(index=future_row, columns=future_col, values=values_col)

            # Reindex
            df = df.reindex(new_index, level=1)
            df = df.reset_index()

            # LaTeX export
            table_code_latex = df.to_latex(
                float_format="%.3f",
                escape=False,
                multirow=True,
                multicolumn_format="c",
                index=False  # Do not write the titles of the future_col and future_row
            ).replace("_", "\\_")

            # Cosmetic
            for key, value in repdi.items():
                table_code_latex = table_code_latex.replace(key, value)

            text_file.write(table_code_latex + "\n")

# --------------------------------- Watchman --------------------------------- #

# Ignore rows with negative values
watchman_df = watchman_df[
    (watchman_df["last_third_attention_acc_mean"] >= 0)
    & (watchman_df["r_info_loss"] == 0.7)
].copy()
wr = watchman_df.groupby(["dataset", "loss", "use_watchman"])
assert np.all(wr.count().isin([0, N_SEEDS])), (
    "The number of entries for one of the watchman results is not correct. "
    "This may be because different hyperparameters have been treated as the same experiment, "
    "or because some seeds have not run for all experiments.",
    wr.count()
)
wr = wr.mean().round(3)

# Replace variance with std
for k in ["first_third_downstream_acc_variance", "last_third_downstream_acc_variance",
          "first_third_attention_acc_variance", "last_third_attention_acc_variance"]:
    wr[k] = np.sqrt(wr[k])

keys = [
    'first_third_attention_acc',
    'last_third_attention_acc',
    'first_third_downstream_acc',
    'last_third_downstream_acc',
]

for val_col in keys:
    wr[val_col] = (
        wr[val_col + "_mean"]
        .apply(lambda x: f"{x:.3f}")
        .str.cat(wr[val_col + "_variance"].apply(lambda x: f"{x:.3f}"), sep=' \\pm ')
        .apply(lambda x: f"${x}$")
    )
wr = wr[keys].reset_index()

wr.to_csv(
    Path(C.PATH_RESULTS, "watchman.tsv"),
    sep='\t'
)

# --------------------------------- Fidelity --------------------------------- #

# filter
df = merged_result[
    (merged_result["exp"] == baseline_type)
    & (merged_result["loss"] != "adgcl")
    & (merged_result["loss"] != "mega")
    & (merged_result["dataset"] != "mnist")
].copy()  # Include all results from "main" for the correct stopping criterion

# Melt
df_melted = (
    df[["train_set_fidelity", "train_set_fidelity_opposite", "train_set_sparsity", "dataset", "task"]]
    .melt(id_vars=["task", "dataset", "train_set_sparsity"])
)

p = ggplot(
    df_melted,
    aes(
        y='train_set_sparsity',
        x='value',
        color="variable"
    )
)
p += geom_point(size=1, alpha=1, na_rm=False)
p += facet_wrap('~ task + dataset',
                labeller=lambda x: facet_labels[x],
                nrow=2)

p += theme_common
p += theme(
    figure_size=(12, 7)
)
p += xlab('Fidelity') + ylab("Train set sparsity index") + labs(color="Fidelity")
p += scale_color_manual(
    values=['dodgerblue', 'darkred'],
    labels=['Real', 'Opposite']
)
ggsave(plot=p, filename=Path(C.PATH_FIGURES, "fidelity_sparsity.png"), dpi=400)

# --------------------------------- AUCs --------------------------------- #

# filter again from scratch so adgcl and mega are present
df = merged_result[
    (merged_result["exp"] == baseline_type)
    & (merged_result["dataset"] != "mnist")
    & (merged_result["r_info_loss"].isna() | (merged_result["r_info_loss"] == 0.7))
    & (
        merged_result["loss"].isin(["adgcl", "mega"])
        | ((merged_result["task"] == "node_classification") & ~merged_result["use_watchman"])
        | ((merged_result["task"] != "node_classification") & merged_result["use_watchman"])
    )
][["DownstreamClf_acc_test", "Interp_acc_test", "loss", "dataset", "task"]].copy()

df_melted = df.melt(id_vars=["task", "dataset", "loss"])

assert np.all(df_melted.groupby(["dataset", "loss", "variable"]).count().isin([0, N_SEEDS])), (
    "The number of entries for one of the watchman results is not correct. "
    "This may be because different hyperparameters have been treated as the same experiment, "
    "or because some seeds have not run for all experiments.",
    df_melted.groupby(["dataset", "loss", "variable"]).count()
)

p = ggplot(
    df_melted,
    aes(
        x='loss',
        y='value',
        fill="variable",
    )
)
p += geom_bar(position="dodge", stat="summary", fun_y=np.mean)
p += stat_summary(fun_data='mean_sdl', fun_args={'mult': 1}, geom='errorbar', position="dodge")
p += facet_wrap('~ task + dataset',
                labeller=lambda x: facet_labels[x],
                drop=True,
                nrow=2)
p += scale_x_discrete(
    limits=ordered_losses,
    labels=translated_losses
)
p += theme_common
p += theme(
    axis_text_x=element_text(colour="black", size=10, rotation=90),
    figure_size=(10, 7)
)
p += coord_flip()
p += xlab('Loss') + ylab('AUC') + labs(fill="AUC")
p += scale_fill_manual(
    values=['orange', 'teal'],
    labels=['Downstream', 'Interpretability']
)
ggsave(plot=p, filename=Path(C.PATH_FIGURES, "aucs.png"), dpi=400)

# --------------------------------- Explore R --------------------------------- #

explore_r_figsize = (3, 8)

y_cols = ["train_set_sparsity", "Interp_acc_test", "train_set_fidelity_opposite",
          "DownstreamClf_acc_test"]
r_res = merged_result[
    (merged_result["exp"] == baseline_type)
    & (merged_result["loss"].isin(["simclr_double_aug_info_negative", "gsat"]))
    & (merged_result["dataset"].isin(["ba_2motifs", "mutag"]))  # Only graph datasets
    & (merged_result["use_watchman"])  # Only graph datasets
][["r_info_loss", "loss", "dataset"] + y_cols].copy()
r_res["R_cat"] = r_res["r_info_loss"].astype("category")
assert np.all(r_res.groupby(["R_cat", "loss", "dataset"]).count().isin([0, N_SEEDS])), (
    "The number of entries for one of the watchman results is not correct. "
    "This may be because different hyperparameters have been treated as the same experiment, "
    "or because some seeds have not run for all experiments.",
    r_res.groupby(["R_cat", "loss", "dataset"]).count()
)

## Figures

p = ggplot(
    r_res,
    aes(
        x='R_cat',
        y='train_set_sparsity',
    )
)
p += geom_boxplot(width=.8, fill="dodgerblue")
p += facet_wrap("~ loss + dataset",
                labeller=lambda x: r_translate_dict[x],
                ncol=1)
p += theme_common
p += theme(
    axis_text_x=element_text(colour="black", size=10, rotation=90),
    figure_size=explore_r_figsize
)
p += coord_flip()
p += xlab('R sparsity parameter') + ylab('Train set sparsity index')
p += scale_y_continuous(labels=lambda x: ["{:.2f}".format(i) for i in x])
ggsave(plot=p, filename=Path(C.PATH_FIGURES, "explore_r.png"), dpi=400)

p = ggplot(
    r_res,
    aes(
        x='R_cat',
        y='Interp_acc_test',
    )
)
p += geom_boxplot(width=.8, fill="teal")
p += facet_wrap("~ loss + dataset",
                labeller=lambda x: r_translate_dict[x],
                ncol=1)
p += theme_common
p += theme(
    axis_text_x=element_text(colour="black", size=10, rotation=90),
    figure_size=explore_r_figsize
)
p += coord_flip()
p += xlab('') + ylab('Interpretability AUC')
p += scale_y_continuous(labels=lambda x: ["{:.2f}".format(i) for i in x])
ggsave(plot=p, filename=Path(C.PATH_FIGURES, "explore_r_interp.png"), dpi=400)

p = ggplot(
    r_res,
    aes(
        x='R_cat',
        y='train_set_fidelity_opposite',
    )
)
p += geom_boxplot(width=.8, fill="red")
p += facet_wrap("~ loss + dataset",
                labeller=lambda x: r_translate_dict[x],
                ncol=1)
p += theme_common
p += theme(
    axis_text_x=element_text(colour="black", size=10, rotation=90),
    figure_size=explore_r_figsize
)
p += coord_flip()
p += xlab('') + ylab('Opposite fidelity (lower is better)')
p += scale_y_continuous(labels=lambda x: ["{:.2f}".format(i) for i in x])
ggsave(plot=p, filename=Path(C.PATH_FIGURES, "explore_r_fid.png"), dpi=400)

p = ggplot(
    r_res,
    aes(
        x='R_cat',
        y='DownstreamClf_acc_test'
    )
)
p += geom_boxplot(width=.8, fill="orange")
p += facet_wrap("~ loss + dataset",
                labeller=lambda x: r_translate_dict[x],
                ncol=1)
p += theme_common
p += theme(
    axis_text_x=element_text(colour="black", size=10, rotation=90),
    figure_size=explore_r_figsize
)
p += coord_flip()
p += xlab('') + ylab('Downstream AUC')
p += scale_y_continuous(labels=lambda x: ["{:.2f}".format(i) for i in x])
ggsave(plot=p, filename=Path(C.PATH_FIGURES, "explore_r_downstream.png"), dpi=400)

Path(C.PATH_RESULTS, "figures_done").touch()

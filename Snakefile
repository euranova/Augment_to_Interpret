"""
HOW TO RUN

# Preparation 
1) Prepare an appropriate configfile in the snakeconfig directory, so you can pass it in the command line.
2) Ensure snakemake is installed (use the newest Docker image)

# Running
Here is a list of bash commands, with their signification:
```
# Run it, specifying that 2 golden_ticket at once are available, so 2 jobs will run in parallel. Failed jobs will be restarted up to 3 times.
# You must also specify a config file, otherwise it will default to the snakeconfig/default.yml
snakemake --resources golden_ticket=2 --cores all --restart-times 3 --configfile snakeconfig/myconfig.yml

# Other commands
snakemake -n -p                              # Dry run : nothing will be done, but preview the commands that will be run.                                         
snakemake --dag | dot -Tpdf > dag.pdf        # Produce workflow graph
snakemake -c "sbatch ..."                    # Run on SLURM sbatch, replace '...' with the sbatch parameters
```
I have also added the important commands to the Makefile.

NOTE : 
    - By default, Snakemake will try to run a file called "Snakefile" present at the point of execution. If you change the name, you will need to tell it to Snakemake.
    - Snakemake is Python ! You can write custom Python code in this file, for example to create custom lists of final files, and it will still work !
    - Recall that the snakefile is run where you call it, so ensure all paths are relative to the execution point.
"""
import itertools
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch

from augment_to_interpret.basic_utils import C
from augment_to_interpret.complex_utils.param_functions import (
    deduce_batch_size, deduce_model, get_result_dir, get_result_dir_format)


# ---------------------------- Deduce parameters ----------------------------- #
configfile: Path(C.PATH_SNAKE_CONFIG, "default.yml")
PATH_MAIN_RESULTS = Path(C.PATH_RESULTS, "main")
BASELINE_TYPE = "best_downstream_clf_model"  # For the baselines, which model to use? Best downstream or final ?

if config["run_params"]["cuda"] >= 0 and not torch.cuda.is_available():
    config["run_params"]["cuda"] = -1
    warnings.warn("CUDA was asked for but seems unavailable on your machine. Using CPU.")
elif config["run_params"]["cuda"] < 0 and torch.cuda.is_available():
    warnings.warn("CUDA seems available, but you asked to run on the CPU.")


# Final output
def deduce_final_output():
    final = []

    if config["run_posthoc_analysis"]:
        final = [Path(PATH_MAIN_RESULTS, "summary_results.tsv"), Path(PATH_MAIN_RESULTS, "training_curve_grouped.tsv")]
    else:
        final.append(Path(PATH_MAIN_RESULTS, "final"))
        if config["run_baselines"]: 
            raise ValueError("You asked for run_baselines but not for run_posthoc_analysis. However, running baselines always includes a posthoc analysis.")

    if config["run_baselines"]:
        # If we run the baselines, also run the figures.
        # The figures always necessitate the baselines, so the simplest way
        # to ensure there is no error is to call the figure generation only
        # when we are also calling the baselines.
        final.append(Path(C.PATH_RESULTS, "figures_done"))
        final.append(Path(C.PATH_RESULTS, "baselines_done"))

    return final

rule all:
    input: deduce_final_output()


rule produce_figures:
    input:
        Path(PATH_MAIN_RESULTS, "posthoc_embedding_analysis_final.tsv"),
        Path(PATH_MAIN_RESULTS, "training_curve_analysis_final.tsv"),
        Path(C.PATH_RESULTS, "sparsity_final_results.tsv"),
        Path(C.PATH_RESULTS, "baselines_done")
    output: Path(C.PATH_RESULTS, "figures_done")
    resources: golden_ticket = 1
    log: Path(C.PATH_RESULTS, "produce_figures.log")
    shell:
        "\n".join([
            f"python {Path(C.PATH_SCRIPTS, 'final_plots.py')} \\",
            f"    --baseline_type {BASELINE_TYPE} \\",
            f"    &> {{log}}",
        ])


# ---------------------------- Post-hoc analysis ----------------------------- #
# Run analysis on the saved results of all runs.

rule analysis_summary:
    # Re-arrange the results computed below (fast)
    input: Path(PATH_MAIN_RESULTS, "training_curve_analysis_final.tsv"), Path(PATH_MAIN_RESULTS, "posthoc_embedding_analysis_final.tsv")
    output: Path(PATH_MAIN_RESULTS, "summary_results.tsv"), Path(PATH_MAIN_RESULTS, "training_curve_grouped.tsv")
    resources: golden_ticket = 1
    log: Path(PATH_MAIN_RESULTS, "result_dataframe_process.log")
    shell: f"python {Path(C.PATH_SCRIPTS, 'result_dataframe_process.py')} &> {{log}}"

rule analysis_training_curve: 
    # Pick interesting metrics in the TensorBoard logs (fast)
    input: Path(PATH_MAIN_RESULTS, "final")
    output: Path(PATH_MAIN_RESULTS, "training_curve_analysis_final.tsv")
    resources: golden_ticket = 1
    log: Path(PATH_MAIN_RESULTS, "training_curve_analysis.log")
    shell: f"python {Path(C.PATH_SCRIPTS, 'training_curve_analysis.py')} &> {{log}}"

rule analysis_posthoc_embedding: 
    # Run posthoc embedding analysis, including clustering and perturbation metrics (slow)
    input: Path(PATH_MAIN_RESULTS, "final")
    output: Path(PATH_MAIN_RESULTS, "posthoc_embedding_analysis_final.tsv")
    resources: golden_ticket = 1
    log: Path(PATH_MAIN_RESULTS, "posthoc_embedding_analysis.log")
    shell: f"python {Path(C.PATH_SCRIPTS, 'posthoc_embedding_analysis.py')} &> {{log}}"

rule sparsity_analysis:
    # Sparsity analysis on the edge attention (moderately slow)
    # NOTE Since this rule necessitates the baselines, it should never be called
    # unless config["run_baselines"] is True (the get_final_output() function 
    # takes care of that).
    input:
        Path(PATH_MAIN_RESULTS, "final"),
        Path(C.PATH_RESULTS, "baselines_done")
    output:  Path(C.PATH_RESULTS, "sparsity_final_results.tsv")
    resources: golden_ticket = 1
    log: Path(C.PATH_RESULTS, "sparsity_analysis.log")
    shell: f"python {Path(C.PATH_SCRIPTS, 'sparsity_analysis.py')} &> {{log}}"

# ---------------------------- Run experiments ------------------------------- #

def get_namespaces(**kwargs):
    keys, values = zip(*kwargs.items())
    for current_values in itertools.product(*values):
        yield SimpleNamespace(**{key: val for key, val in zip(keys, current_values)})

all_inputs = [
    Path(get_result_dir(PATH_MAIN_RESULTS, args), "done")
    for args in get_namespaces(
        dataset=config["grid_search_params"]["datasets"],
        seed=config["grid_search_params"]["seeds"],
        loss=config["grid_search_params"]["losses"],
        use_watchman=config["grid_search_params"]["watchman_status"],
        use_features_selector=config["grid_search_params"]["feature_selector_status"],
        loss_weight=config["grid_search_params"]["loss_weights"],
        r_info_loss=config["grid_search_params"]["r_info_loss"],
        temperature_edge_sampling=config["grid_search_params"]["temperature_edge_sampling"],
    )
]

rule concat_runs:
    """
    Master rule which asks for the results for all possible combinations of parameters.
    """
    input: all_inputs
    output: Path(PATH_MAIN_RESULTS, "final")
    shell: "touch {output}"  # Signal we are done
    
rule run:
    output:
        Path(get_result_dir_format(PATH_MAIN_RESULTS), "done")
    params:
        epochs = config["run_params"]["epochs"],
        cuda = config["run_params"]["cuda"],
        batch_size = deduce_batch_size,
        model = deduce_model,
    resources:
        golden_ticket = 1  # Each run instance of this rule requires one ticket
    log: Path(get_result_dir_format(PATH_MAIN_RESULTS), "main.log")
    
    shell:
        "\n".join([
            f"python {Path(C.PATH_SCRIPTS, 'main.py')} \\",
            f"    --dataset {{wildcards.dataset}} \\",
            f"    --loss {{wildcards.loss}} \\",
            f"    --epochs {{params.epochs}} \\",
            f"    --seed {{wildcards.seed}} \\",
            f"    --cuda {{params.cuda}} \\",
            f"    --batch_size {{params.batch_size}} \\",
            f"    --model_name {{params.model}} \\",
            f"    --use_watchman {{wildcards.use_watchman}} \\",
            f"    --watchman_lambda 1.0 \\",
            f"    --loss_weight {{wildcards.loss_weight}} \\",
            f"    --use_features_selector {{wildcards.use_features_selector}} \\",
            f"    --r_info_loss {{wildcards.r_info_loss}} \\",
            f"    --temperature_edge_sampling {{wildcards.temperature_edge_sampling}} \\",
            f"    &> {{log}}",
            f"touch {{output}}",  # Signal we are done
        ])


# ---------------------------- Run baselines --------------------------------- #

rule run_baselines:
    """
    This will run the ADGCL and MEGA baseline experiments, as well as their post-hoc analysis.
    """
    input:
        adgcl_posthoc_full = Path(C.PATH_RESULTS, "adgcl", f"posthoc_final_result_for_{BASELINE_TYPE}.tsv"),
        mega_posthoc_full = Path(C.PATH_RESULTS, "mega", f"posthoc_final_result_for_{BASELINE_TYPE}.tsv")
    output: Path(C.PATH_RESULTS, "baselines_done")
    shell: """
        touch {output}
    """

rule run_individual_baseline:
    output:
        Path(C.PATH_RESULTS, "{method_baseline}", f"posthoc_final_result_for_{BASELINE_TYPE}.tsv")
    resources:
        golden_ticket = 1  # Each run instance of this rule requires one ticket
    log: Path(C.PATH_RESULTS, "{method_baseline}", "{method_baseline}.log")
    shell: f"python {Path(C.PATH_SCRIPTS, 'baselines', '{wildcards.method_baseline}.py')} &> {{log}}"
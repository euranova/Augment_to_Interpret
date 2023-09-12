# Scripts

The entrypoints to the augment_to_interpret library are the following:
- `main.py` trains and saves a single model using our losses or the GSAT loss (see arguments).
- `posthoc_embedding_analysis.py` runs the post-hoc analysis for models trained using `main.py`.
- `baselines/adgcl.py` (resp. `baselines/mega.py`) trains and saves all models using AD-GCL 
(resp. MEGA) method (see constants in the file), and run the post-hoc analysis for these models.
- `generate_superpixels` generates the MNIST dataset.
- `result_dataframe_process.py`, `sparsity_analysis.py` and `training_curve_analysis.py` 
run additional analyses of all the results.
- `final_plots` generates figures of the paper from all the previous results.

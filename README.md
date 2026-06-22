# SCREAM_Autotuning

Code accompanying the paper **"Machine Learning for Optimized Tuning of the Simple Cloud-Resolving Earth Atmosphere Model (SCREAM)"** (Paige et al., *JAMES*).

This repository contains the full workflow for building Gaussian-process (and comparison) surrogate models of SCREAM output as a function of tunable parameters, using those surrogates to optimize parameters against observations, and evaluating the resulting simulations.

## Repository structure

```
01_data_preprocessing/   Build the zonal/regional/global (ZRG) training dataset from raw SCREAM output
02_surrogate_modeling/   Train and cross-validate surrogate models (GP, CNN, RF, spline) and compare kernels
03_optimization/         Optimize SCREAM parameters against observations using the trained GP surrogate
03_optimization/cost_functions/   Alternative cost-function weightings (variable and zonal/regional/global)
04_simulation_evaluation/   Compare optimized vs. default SCREAM simulations against observations
environment.yml          Conda environment used for all stages
```

### 01_data_preprocessing
- `Preprocessing.ipynb` — loads raw SCREAM simulation output and observations, computes zonal/regional/global (ZRG) statistics, and writes the pickled tables used as surrogate-model training data in the rest of the pipeline.

### 02_surrogate_modeling
- `Final_Surrogate_Kfolds_masked.ipynb` — interactive preprocessing + K-fold surrogate evaluation.
- `Final_surrogate_Kfolds_masked_10seeds_rawr2.py` — trains GP, CNN, and RF surrogates with 5-fold CV repeated over 10 random seeds; reports mean ± std out-of-sample R².
- `GP_Kfold_subsetting_10seed_rawr2.py` — out-of-sample R² as a function of training-set size, for choosing how many simulations are needed to train the surrogate.
- `GP_gridcell_kfolds_masked_10seeds_rawr2.py` — same evaluation at native grid-cell resolution rather than ZRG-aggregated.
- `GP_kernel_comparison_Kfolds_masked_10seeds_rawr2.py` — compares GP kernels (RBF, Matern 1/2, 3/2, 5/2).
- `spline_knot_degree_search_10seed_rawr2.py` — hyperparameter search over spline knot count/degree as an alternative surrogate.
- `Final_Surrogate_Optimizing_Visualizing.ipynb` — earlier version of the optimization/visualization notebook in `03_optimization/`, kept for reference.

### 03_optimization
- `Final_Surrogate_Optimizing_Visualizing.ipynb` — trains the GP surrogate on the full dataset and uses it to optimize SCREAM parameters against observations; produces barcode/comparison plots across cost functions.
- `run_GPsurrogate_fromsave_final_efficient.py` / `run_GPsurrogate_fromsave_efficient.sh` — script + launcher for running the GP-surrogate optimization from saved training data on HPC.
- `cost_functions/` — variable- and region-weighting definitions used by the optimizer (`standard`, `precip_cost_fun`, `precip4_cost_fun`, `tropics_weighted_cost_fun`, `no_region_cost_fun`).

### 04_simulation_evaluation
- `40daysim_visualization.ipynb` — time-series and monthly bias analysis comparing an optimized simulation to the default-parameter control.
- `Compare_visualize_new_simulations.ipynb` — visualizes newly run default vs. optimal simulations.
- `Validation_of_other_variables.ipynb` — validation of additional output variables not covered by the cost function.

## Installation

```bash
conda env create -f environment.yml
conda activate autotune_env
```

The environment includes [ESEm](https://github.com/duncanwp/ESEm) (Gaussian-process surrogate modeling), GPflow, TensorFlow, scikit-learn, and the standard climate-data stack (xarray, iris, cartopy, netCDF4).

## Workflow

1. **Preprocess** raw SCREAM ensemble output and observations into ZRG training tables (`01_data_preprocessing/`).
2. **Train and validate** surrogate models via K-fold cross-validation to select model type, kernel, and training-set size (`02_surrogate_modeling/`).
3. **Optimize** SCREAM tunable parameters against observations using the chosen surrogate and cost function (`03_optimization/`).
4. **Evaluate** the optimized parameter set by running SCREAM and comparing against the default configuration and observations (`04_simulation_evaluation/`).

## Data availability

The SCREAM simulation output, observational datasets, and intermediate pickled training tables used in this study are archived separately on Zenodo (DOI to be added upon publication). See [DATA.md](DATA.md) for what is included and how it maps to each stage of this repository.

## Citation

If you use this code, please cite the associated paper (see [CITATION.cff](CITATION.cff)):

> Paige, J. N., Caldwell, P., Beydoun, H., Hannah, W., Rebassoo, F., Mahfouz, N., Bertagna, L., Bogenschutz, P., Bradley, A., Collins, G., Donahue, A., Guba, O., Hillman, B., Keen, N., Lee, J., Lin, W., Ma, H.-Y., Salinger, A., Shand, L., Terai, C., & Wagman, B. Machine Learning for Optimized Tuning of the Simple Cloud-Resolving Earth Atmosphere Model (SCREAM). *Journal of Advances in Modeling Earth Systems (JAMES)*, in review.

This repository itself is archived on Zenodo via GitHub's release integration — see the badge/DOI once a release is tagged.

## License

Code is released under the MIT License — see [LICENSE](LICENSE).

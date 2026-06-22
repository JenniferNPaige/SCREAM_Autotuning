# SCREAM_Autotuning

Code accompanying the paper **"Machine Learning for Optimized Tuning of the Simple Cloud-Resolving Earth Atmosphere Model (SCREAM)"** (Paige, Caldwell, Beydoun, Hannah, Rebassoo, Mahfouz, Bertagna, Bogenschutz, Bradley, Collins, Donahue, Guba, Hillman, Keen, Lee, Lin, Ma, Salinger, Shand, Terai, & Wagman; submitted to *Journal of Advances in Modeling Earth Systems*, JAMES).

## Summary

Global storm-resolving models (GSRMs) like SCREAM are too computationally expensive to tune by hand or with long simulations. This work builds an automated tuning ("autotuning") framework for SCREAM:

1. A **153-member perturbed parameter ensemble (PPE)** of 2-day SCREAM simulations (DYAMOND1 summer and DYAMOND2 winter cases) is run, varying 16 SHOC/P3 parameters via Latin hypercube sampling, and output is averaged to zonal/regional/global (ZRG) values.
2. Several **machine-learning surrogate models** (Gaussian process, CNN, random forest, multiple linear regression, spline regression) are trained on the PPE to emulate SCREAM's response (PCP, OSR, OLR, TLWP) to parameter changes; the GP surrogate is most skillful under this data-limited regime.
3. The GP surrogate is used to **optimize** SCREAM's parameters by minimizing a weighted cost function (bias relative to observations, aggregated across variables/regions/seasons) via SciPy basinhopping.
4. The optimized parameters are **evaluated** in real SCREAM simulations (2-day and a 35-day DYAMOND2 run) against the default tuning and observations, including variables not used in tuning.

Key result: the optimized tuning reduces 2-day bias by >20% in tuned variables and improves several untuned variables (notably the mid-level dry bias), but some improvements degrade in the 35-day evaluation — short-simulation autotuning is a useful step but not sufficient by itself for final GSRM tuning.

## Repository structure

```
01_data_preprocessing/   Build the zonal/regional/global (ZRG) training dataset from raw SCREAM output (Sec. 3)
02_surrogate_modeling/   Train and cross-validate surrogate models (GP, CNN, RF, MLR, spline) (Sec. 4, App. A)
03_optimization/         Optimize SCREAM parameters against observations using the trained GP surrogate (Sec. 5)
03_optimization/cost_functions/   Alternative cost-function weightings (variable and zonal/regional/global)
04_simulation_evaluation/   Compare optimized vs. default SCREAM simulations against observations (Sec. 6, App. B)
environment.yml          Conda environment used for all stages
```

### 01_data_preprocessing
- `Preprocessing.ipynb` — loads raw SCREAM PPE output (153 members × DY1/DY2 2-day simulations) and observations (IMERG, CERES-SYN, MAC), masks to observational coverage, computes area-weighted zonal/regional/global (ZRG) statistics (Table 3 regions), and writes the pickled ZRG tables used as surrogate-model training data throughout the rest of the pipeline. Produces the correlation heatmaps (Fig. 3) and PPE spinup/spread plots (Fig. 2).

### 02_surrogate_modeling
Implements the surrogate comparison in Sec. 4 and the appendix sensitivity studies in App. A.
- `Final_Surrogate_Kfolds_masked.ipynb` — interactive preprocessing + K-fold surrogate evaluation.
- `Final_surrogate_Kfolds_masked_10seeds_rawr2.py` — trains GP, CNN, RF, MLR, and spline surrogates with 5-fold CV repeated over 10 random seeds; reports mean ± std out-of-sample R² and RMSE (Figs. 4–5).
- `GP_Kfold_subsetting_10seed_rawr2.py` — out-of-sample R² as a function of training-set size (Fig. 6), supporting the choice of PPE size (skill plateaus above ~100 samples).
- `GP_gridcell_kfolds_masked_10seeds_rawr2.py` — ZRG-trained vs. native-grid-cell-trained GP comparison (Table A1–A2); ZRG training is selected as marginally more skillful.
- `GP_kernel_comparison_Kfolds_masked_10seeds_rawr2.py` — compares GP kernels: RBF, Matérn 1/2, 3/2, 5/2, rational quadratic, linear, and combinations, against the ESEm default RBF+linear+polynomial kernel (Table A3).
- `spline_knot_degree_search_10seed_rawr2.py` — hyperparameter search over spline knot count/degree (Table A4); 3 knots, degree 2 is selected.
- `Final_Surrogate_Optimizing_Visualizing.ipynb` — earlier version of the optimization/visualization notebook in `03_optimization/`, kept for reference.

### 03_optimization
Implements Sec. 5 (cost function definition and optimization).
- `Final_Surrogate_Optimizing_Visualizing.ipynb` — trains the final ZRG-trained GP surrogate on the full 153-member PPE and optimizes SCREAM's 16 tunable parameters by minimizing cost via SciPy `basinhopping` (100 random starts, 5 basin-hopping iterations each); produces the optimization barcode plot (Fig. 7) and cost-function comparison (Fig. 8).
- `run_GPsurrogate_fromsave_final_efficient.py` / `run_GPsurrogate_fromsave_efficient.sh` — script + HPC launcher for running the same GP-surrogate optimization from saved training data.
- `cost_functions/` — variable- and region-weighting definitions used by the optimizer, corresponding to the four cost functions compared in Fig. 8:
  - `standard_cost_fun.py` — equal weight across variables and across zonal/regional/global averages.
  - `precip_cost_fun.py` / `precip4_cost_fun.py` — upweighted precipitation contribution.
  - `no_region_cost_fun.py` — regional contribution to cost removed.
  - `tropics_weighted_cost_fun.py` — tropical areas upweighted.

### 04_simulation_evaluation
Implements Sec. 6 (evaluation of optimized SCREAM simulations) and App. B.
- `40daysim_visualization.ipynb` — time series (Fig. 17) and monthly bias analysis (Fig. 18, App. B4) comparing the 35-day optimized DYAMOND2 simulation to the default-parameter control, used to assess whether 2-day tuning improvements hold over longer integrations (Sec. 6.1).
- `Compare_visualize_new_simulations.ipynb` — geographic-difference and zonal/regional/global bias plots (Figs. 9–14, App. B1) comparing newly run default vs. optimal 2-day SCREAM simulations against observations for the cost-function-trained variables (PCP, TLWP, OSR, OLR).
- `Validation_of_other_variables.ipynb` — validation of variables not used in the cost function (WVP, IWP, T200/T850, U200/U850 vs. ERA5; Figs. 15–16, 19, App. B2–B3), including the day-2-vs.-monthly correlation analysis used to assess simulation-length sufficiency.

## Installation

```bash
conda env create -f environment.yml
conda activate autotune_env
```

The environment includes [ESEm](https://github.com/duncanwp/ESEm) (Gaussian-process surrogate modeling; Watson-Parris et al., 2021), GPflow, TensorFlow, scikit-learn, and the standard climate-data stack (xarray, iris, cartopy, netCDF4).

## Workflow

1. **Preprocess** raw SCREAM PPE output and observations into ZRG training tables (`01_data_preprocessing/`).
2. **Train and validate** surrogate models via K-fold cross-validation to select model type, kernel, and training-set size (`02_surrogate_modeling/`).
3. **Optimize** SCREAM's 16 tunable parameters against observations using the chosen GP surrogate and cost function (`03_optimization/`).
4. **Evaluate** the optimized parameter set by running SCREAM (2-day and 35-day) and comparing against the default configuration and observations, including untuned variables (`04_simulation_evaluation/`).

## Data availability

Per the paper's Open Research statement, the code and data for this study are intended to be available from this repository. The repository currently contains code only; see [DATA.md](DATA.md) for the manifest of SCREAM PPE output, observational datasets (IMERG, CERES-SYN, MAC, ERA5), and intermediate pickled ZRG training tables that still need to be staged here or in a companion Zenodo deposit (DOI to be added upon publication).

## Citation

If you use this code, please cite the associated paper (see [CITATION.cff](CITATION.cff)):

> Paige, J., Caldwell, P., Beydoun, H., Hannah, W., Rebassoo, F., Mahfouz, N., Bertagna, L., Bogenschutz, P., Bradley, A., Collins, G., Donahue, A., Guba, O., Hillman, B., Keen, N., Lee, J., Lin, W., Ma, H.-Y., Salinger, A., Shand, L., Terai, C., & Wagman, B. Machine Learning for Optimized Tuning of the Simple Cloud-Resolving Earth Atmosphere Model (SCREAM). *Journal of Advances in Modeling Earth Systems (JAMES)*, in review.

This repository itself is archived on Zenodo via GitHub's release integration — see the badge/DOI once a release is tagged.

## License

Code is released under the MIT License — see [LICENSE](LICENSE).

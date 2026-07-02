# SCREAM_Autotuning

Code accompanying **"Machine Learning for Optimized Tuning of the Simple Cloud-Resolving Earth Atmosphere Model (SCREAM)"** (Paige et al., *Journal of Advances in Modeling Earth Systems*, in review).

## Summary

Global storm-resolving models (GSRMs) like SCREAM are too computationally expensive to tune by hand or with long simulations. This repository implements an automated tuning framework for SCREAM:

1. A **153-member perturbed parameter ensemble (PPE)** of 2-day SCREAM simulations (DYAMOND1 summer and DYAMOND2 winter cases) is run, varying 16 SHOC/P3 parameters via Latin hypercube sampling, and output is averaged to zonal/regional/global (ZRG) values.
2. Several **machine-learning surrogate models** (Gaussian process, CNN, random forest, multiple linear regression, spline regression) are trained on the PPE to emulate SCREAM's response (PCP, OSR, OLR, TLWP) to parameter changes; the GP surrogate is most skillful under this data-limited regime.
3. The GP surrogate is used to **optimize** SCREAM's parameters by minimizing a weighted cost function (bias relative to observations, aggregated across variables/regions/seasons) via SciPy basinhopping.
4. The optimized parameters are **evaluated** in real SCREAM simulations (2-day and a 35-day DYAMOND2 run) against the default tuning and observations, including variables not used in tuning.

Key result: the optimized tuning reduces 2-day bias by >20% in tuned variables and improves several untuned variables (notably the mid-level dry bias), but some improvements degrade in the 35-day evaluation — short-simulation autotuning is a useful step but not sufficient by itself for final GSRM tuning.

## Repository structure

```
01_data_preprocessing/      Build ZRG training tables from raw SCREAM output (Sec. 3)
02_surrogate_modeling/      Train and cross-validate surrogate models (Sec. 4, App. A)
03_optimization/            Optimize parameters using the GP surrogate (Sec. 5)
03_optimization/cost_functions/   Cost-function weighting definitions
04_simulation_evaluation/   Evaluate optimized vs. default simulations (Sec. 6, App. B)
paths.py                    Central data-path configuration (see DATA.md)
environment.yml             Conda environment
```

### 01_data_preprocessing
- `Preprocessing.ipynb` — loads raw PPE output (153 members × DY1/DY2) and observations (IMERG, CERES-SYN, MAC), masks to observational coverage, computes area-weighted ZRG statistics, and writes the pickled training tables used throughout the pipeline. Produces the PPE spread plots (Fig. 2) and correlation heatmaps (Fig. 3).

### 02_surrogate_modeling
- `Final_Surrogate_Kfolds_masked.ipynb` — interactive surrogate evaluation notebook.
- `Final_surrogate_Kfolds_masked_10seeds_rawr2.py` — GP, CNN, RF, MLR, and spline surrogates with 5-fold CV × 10 seeds; reports mean ± std R² and RMSE (Figs. 4–5).
- `GP_Kfold_subsetting_10seed_rawr2.py` — R² vs. training-set size (Fig. 6).
- `GP_gridcell_kfolds_masked_10seeds_rawr2.py` — ZRG vs. native-grid GP comparison (Tables A1–A2).
- `GP_kernel_comparison_Kfolds_masked_10seeds_rawr2.py` — GP kernel comparison (Table A3).
- `spline_knot_degree_search_10seed_rawr2.py` — spline hyperparameter search (Table A4).

### 03_optimization
- `Final_Surrogate_Optimizing_Visualizing.ipynb` — trains the final GP surrogate on the full PPE and optimizes via `basinhopping` (100 starts × 5 iterations); produces the optimization barcode plot (Fig. 7) and cost-function comparison (Fig. 8).
- `run_GPsurrogate_fromsave_final_efficient.py` / `run_GPsurrogate_fromsave_efficient.sh` — HPC script and launcher for running the optimization from saved training data.
- `cost_functions/` — four cost-function weightings compared in Fig. 8: `standard_cost_fun.py`, `precip_cost_fun.py`, `no_region_cost_fun.py`, `tropics_weighted_cost_fun.py`.

### 04_simulation_evaluation
- `Compare_visualize_new_simulations.ipynb` — geographic and ZRG bias plots (Figs. 9–14, App. B1) for the tuned variables (PCP, TLWP, OSR, OLR).
- `Validation_of_other_variables.ipynb` — validation of untuned variables (WVP, IWP, T200/T850, U200/U850 vs. ERA5; Figs. 15–16, 19, App. B2–B3) and day-2 vs. monthly correlation analysis.
- `40daysim_visualization.ipynb` — time series (Fig. 17) and monthly bias analysis (Fig. 18, App. B4) for the 35-day DYAMOND2 simulation.

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

## Data

Data is split across a Zenodo deposit (derived artifacts, DOI forthcoming) and a NERSC HPSS archive (raw simulation output — public portal: <https://portal.nersc.gov/archive/home/j/jpaige3/www/SCREAM-autotuning/>). See [DATA.md](DATA.md) for the full manifest and path configuration instructions.

## Citation

> Paige, J., Caldwell, P., Beydoun, H., Hannah, W., Rebassoo, F., Mahfouz, N., Keen, N., Elsaesser, G. S., Bertagna, L., Bogenschutz, P., Collins, G., Donahue, A., Golaz, J.-C., Guba, O., Hillman, B., Lee, J., Lin, W., Ma, H.-Y., Salinger, A., Shand, L., Terai, C., & Wagman, B. Machine Learning for Optimized Tuning of the Simple Cloud-Resolving Earth Atmosphere Model (SCREAM). *Journal of Advances in Modeling Earth Systems (JAMES)*, in review.

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## License

MIT — see [LICENSE](LICENSE).

## Related work

[rebassoo/autotune](https://github.com/rebassoo/autotune) — a modified, actively maintained version of this workflow.

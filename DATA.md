# Data manifest (for Zenodo deposit)

The code in this repository expects data that currently lives on NERSC/LLNL HPC storage (paths referenced in the notebooks point at `/global/cfs/cdirs/...`). None of that data is included in this GitHub repository. This file lists what needs to be pulled down and archived as a companion Zenodo data deposit, and which pipeline stage consumes it, so the Zenodo record and this code repository can be cross-linked in the paper.

Once the deposit exists, add its DOI to the **Data availability** section of `README.md` and to the paper's data availability statement.

## 1. Raw SCREAM ensemble output
- **Used by:** `01_data_preprocessing/Preprocessing.ipynb`
- **What:** Per-ensemble-member SCREAM output (daily-averaged fields) for the tunable-parameter ensemble used to train the surrogate, plus the 40-day default/optimized comparison runs used in `04_simulation_evaluation/`.
- **Format:** NetCDF (`.nc`), `ne30pg2` grid.
- **Action needed:** Identify the minimal set of variables/timesteps actually needed (PCP, TLWP, OSR, OLR, plus anything used in `Validation_of_other_variables.ipynb`) to keep the deposit a reasonable size — full raw output is likely too large for Zenodo (50 GB community limit).

## 2. Observational datasets
- **Used by:** `01_data_preprocessing/Preprocessing.ipynb`, `03_optimization/`, `04_simulation_evaluation/`
- **What:** Observational reference products referenced in the notebooks, e.g. `IMERG.precip_total_surf_mass_flux.daily_AVERAGE.ne30pg2.20160807_mahf708.nc` (IMERG precipitation) and any equivalents for TLWP/OSR/OLR.
- **Action needed:** Confirm redistribution license for each observational product (e.g. IMERG terms) before including in a public Zenodo deposit — some products may need to be linked to their original archive instead of re-hosted.

## 3. Preprocessed ZRG training tables
- **Used by:** `02_surrogate_modeling/`, `03_optimization/`
- **What:** The pickled zonal/regional/global (ZRG) summary tables produced by `01_data_preprocessing/Preprocessing.ipynb` — this is the actual surrogate-model training data, and the most important thing to archive since it's what makes the surrogate results reproducible without re-running the full SCREAM ensemble.
- **Format:** `.pkl` (pandas/xarray objects).
- **Action needed:** Locate the saved pickle files on HPC (referenced as "pre-saved ZRG data" in `02_surrogate_modeling/*.py` headers) and stage them for upload.

## 4. Optimization results
- **Used by:** `03_optimization/`, `04_simulation_evaluation/`
- **What:** Optimizer output (parameter sets per cost function) used to generate the barcode plots and to configure the optimized SCREAM runs evaluated in `04_simulation_evaluation/`.
- **Format:** CSV/pickle.

## Suggested Zenodo deposit structure

```
data/
  raw_ensemble/          # subset of SCREAM ensemble output (or omit, link to source archive)
  observations/          # obs products with redistribution rights confirmed
  zrg_training_tables/   # pickled ZRG tables — core reproducibility artifact
  optimization_results/  # per-cost-function optimized parameter sets
README.md                # describes contents, units, grid, provenance, and which repo/script consumes each folder
```

## Status
Data has not yet been pulled down from HPC into this local workspace. This file is a planning manifest only — no files have been transferred or uploaded yet.

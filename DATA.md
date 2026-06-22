# Data manifest (for Zenodo deposit)

The code in this repository expects data that currently lives on NERSC/LLNL/OLCF HPC storage (paths referenced in the notebooks point at `/global/cfs/cdirs/...`). None of that data is included in this GitHub repository yet. The paper's Open Research statement points readers to this GitHub repo for "all data and code used to produce these results," so closing this gap (here or in a linked Zenodo deposit) is required before submission/publication.

This file lists what needs to be pulled down and archived, and which pipeline stage and paper section/figure consumes it. Once a deposit exists, add its DOI to the **Data availability** section of `README.md` and to the paper's data availability statement.

## 1. Perturbed parameter ensemble (PPE) SCREAM output
- **Used by:** `01_data_preprocessing/Preprocessing.ipynb`
- **Paper:** Sec. 3 (PPE design), Figs. 2–3
- **What:** Daily-averaged SCREAM output (native ~5 km, regridded to ~100 km) for the 153-member PPE: 130 Latin-hypercube-sampled members (of an original 300) plus 23 additional exploratory members, each run for 2 days following a 5-day nudged spinup, for both DYAMOND1 (Aug 1, 2016) and DYAMOND2 (Jan 20, 2020) cases. Variables needed: PCP, OSR, OLR, TLWP (the four cost-function variables), plus the 16 tunable parameter values per member (Table 1: thl2tune, qw2tune, length_fac, c_diag_3rd_mom, Ckh, Ckm, lambda_low, lambda_high, spa_to_nc, eci, eri, k_acc, dep_nuc_exponent, max_total_ni, ice_sed_knob, D_breakup_cutoff).
- **Format:** NetCDF (`.nc`), `ne30pg2` grid.
- **Size note:** ~500,000 Frontier node hours went into producing this ensemble — the raw 5 km output is almost certainly too large for Zenodo (50 GB community limit); only the regridded/ZRG-reduced fields need to be archived (see #3 below), with raw output linked to its HPC archive location instead if retained at all.

## 2. Observational datasets (tuning targets and validation)
- **Used by:** `01_data_preprocessing/Preprocessing.ipynb`, `03_optimization/`, `04_simulation_evaluation/Validation_of_other_variables.ipynb`
- **Paper:** Table 2, Sec. 3.2
- **What:**
  - IMERG — precipitation (PCP) target, e.g. `IMERG.precip_total_surf_mass_flux.daily_AVERAGE.ne30pg2.20160807_mahf708.nc`
  - CERES-SYN — outgoing shortwave (OSR) and longwave (OLR) radiation targets
  - MAC — total liquid water path (TLWP) target
  - ERA5 — reference for untuned variables (WVP, IWP, T200, T850, U200, U850) in `04_simulation_evaluation/Validation_of_other_variables.ipynb`
- **Action needed:** Confirm redistribution rights for each product before re-hosting on Zenodo — some (e.g. IMERG, ERA5) may need to be linked to their original archive (NASA/Copernicus) rather than re-distributed.

## 3. Preprocessed ZRG training tables
- **Used by:** `02_surrogate_modeling/`, `03_optimization/`
- **Paper:** Sec. 3.4, 4.2
- **What:** The pickled zonal/regional/global (ZRG; Table 3 regions — globe, poles, extratropical/tropical land and ocean, ascending/descending tropical ocean, 10° latitude zones) summary tables produced by `01_data_preprocessing/Preprocessing.ipynb`. This is the actual surrogate-model training data (153 members × DY1/DY2 × PCP/OSR/OLR/TLWP × ZRG areas) and the single most important artifact to archive, since it makes the surrogate comparison and optimization results (Secs. 4–5) reproducible without re-running the SCREAM ensemble.
- **Format:** `.pkl` (pandas/xarray objects), referenced as "pre-saved ZRG data" in the `02_surrogate_modeling/*.py` script headers.

## 4. Optimization results
- **Used by:** `03_optimization/`, `04_simulation_evaluation/`
- **Paper:** Sec. 5, Figs. 7–8
- **What:** The 100 basinhopping optimization runs (per cost function: standard, precip-upweighted ×2, no-region, tropics-upweighted) with resulting parameter sets and costs, used to produce the barcode plot (Fig. 7) and select the final optimal parameter set used in evaluation.
- **Format:** CSV/pickle.

## 5. Evaluation simulations (optimized vs. default)
- **Used by:** `04_simulation_evaluation/`
- **Paper:** Sec. 6, Figs. 9–19, App. B
- **What:**
  - 2-day default- and optimal-parameter SCREAM simulations for both DYAMOND1 and DYAMOND2, including the trained variables (PCP, OSR, OLR, TLWP) and untuned variables (WVP, IWP, T200, T850, U200, U850).
  - A 35-day DYAMOND2 evaluation run (default and optimal parameters, same 5-day spinup procedure) used to test whether 2-day-tuning improvements persist over longer integrations (Sec. 6.1, Figs. 17–18).
- **Format:** NetCDF (`.nc`).

## Suggested Zenodo deposit structure

```
data/
  ppe_raw/                 # subset of PPE SCREAM output, or omit and link to HPC/source archive
  observations/             # IMERG/CERES-SYN/MAC/ERA5, or links if redistribution isn't permitted
  zrg_training_tables/      # pickled ZRG tables — core reproducibility artifact (Secs. 4-5)
  optimization_results/     # per-cost-function basinhopping results (Sec. 5)
  evaluation_simulations/   # 2-day and 35-day default/optimal SCREAM runs (Sec. 6)
README.md                   # describes contents, units, grid, provenance, and which repo/script consumes each folder
```

## Status
Data has not yet been pulled down from HPC into this local workspace. This file is a planning manifest only — no files have been transferred or uploaded yet.

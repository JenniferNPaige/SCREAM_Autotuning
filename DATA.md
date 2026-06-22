# Data manifest (for Zenodo deposit)

The code in this repository expects data that currently lives on NERSC/LLNL/OLCF HPC storage (paths referenced via `paths.py` point at `/global/cfs/cdirs/...` and `/pscratch/...` on the source systems). None of that data is included in this GitHub repository. The paper's Open Research statement points readers to this GitHub repo for "all data and code used to produce these results," so closing this gap — via a companion Zenodo data deposit — is required before submission/publication.

Sizing rule of thumb: the 153-member PPE was built from raw SCREAM output at native ~5 km resolution across two DYAMOND cases — far too large for Zenodo's 50 GB community limit. The things that are actually small are the *reduced* artifacts: ZRG-aggregated training tables, optimization result CSVs, cross-validation summary JSON/pickles, and single time-averaged 2D fields (rather than full multi-day native-grid output). Below, each item is marked **Include** or **Exclude/link only** on that basis.

## Include in the Zenodo deposit

| # | Item | Paper ref. | Format | Why it's small enough |
|---|------|-----------|--------|------------------------|
| 1 | ZRG training tables (`obs.pkl`, `GP_ZRG_masked_proj.pkl`, `ppe_param.pkl`) | Sec. 3.4, 4.2 | `.pkl` | Zonal/regional/global-aggregated tables for 153 members × 4 variables × ~22 areas — a few MB at most. This is *the* core reproducibility artifact: it lets Secs. 4–5 (surrogate comparison, optimization) be rerun without ever touching raw SCREAM output. |
| 2 | Cross-validation outputs (`CV_summary_*.json`, per-fold `GP_ZRG_r2output_*.json`, `Fold_*_ZRG_masked_data_*.pkl`) | Sec. 4, App. A, Figs. 4–6, Tables A1–A4 | `.json` / `.pkl` | Summary statistics (R², RMSE per seed/fold/kernel/knot-degree), not raw data. |
| 3 | Optimization results (`optimization_results/<cost_fun>/results*.csv`) | Sec. 5, Figs. 7–8 | `.csv` | 100 basinhopping runs × 16 parameters × 5 cost functions — small text tables. |
| 4 | Default- and optimal-parameter **2nd-day-average** fields (one time-averaged snapshot each, DY1 and DY2, default and optimal) | Sec. 6, Figs. 9–16 | `.nc` (single time slice, regridded ~100 km) | This is the specific slice you flagged as worth keeping — a single averaged day's worth of the ~10 evaluation variables (PCP, OSR, OLR, TLWP, WVP, IWP, T200/T850, U200/U850) at ne30pg2, not the full native-grid multi-day archive. |
| 5 | 35-day DYAMOND2 evaluation run, **monthly-mean** fields only (default and optimal) | Sec. 6.1, Figs. 17–18 | `.nc` (monthly mean, regridded) | Same logic — the monthly mean used for Figs. 17–18 is one averaged field per variable, not the 35 daily files. |
| 6 | Regridded observational fields actually used as tuning targets/validation (IMERG PCP, CERES-SYN OSR/OLR, MAC TLWP, ERA5 WVP/IWP/T200/T850/U200/U850 — all already regridded to ne30pg2 in this workflow) | Table 2, Sec. 3.2 | `.nc` | Small, already-regridded single-field files. **Action needed:** confirm redistribution rights for IMERG/ERA5/CERES/MAC before re-hosting; if not permitted, include a script/instructions to regenerate them from the original archives (NASA GES DISC, Copernicus CDS, CERES, MAC) instead of the files themselves. |
| 7 | Tunable-parameter table per ensemble member (`ppe_params.json`, Table 1 values for all 153 members) | Table 1, Sec. 3.1 | `.json` | Tiny — 153 rows × 16 parameters. |

## Exclude / link only (too large for Zenodo)

| Item | Why excluded | What to do instead |
|------|--------------|---------------------|
| Raw 153-member PPE SCREAM output (native ~5 km, both DYAMOND cases, all output variables) | ~500,000 Frontier node-hours of simulation; native-grid multi-variable output for 153 members is on the order of many 10s–100s of GB | Not needed for reproducibility (item 1 above already captures everything downstream code uses). Link to the HPC archive location/DOE data portal if a permanent copy exists; otherwise note it's available on request. |
| Full daily (not 2nd-day-only) native-grid output for the 2-day default/optimal evaluation runs | Multi-day, multi-variable, native-grid — large | Only the 2nd-day average (item 4) is actually used in the paper's figures. |
| Full 35 daily files of the 35-day DYAMOND2 evaluation run | 35× the size of the monthly mean for no analysis benefit beyond what Figs. 17–18 (time series) need | Item 5's monthly mean covers Figs. 17–18 bias plots; if the time-series figure (Fig. 17) needs the daily values, archive just the global-mean daily time series (a tiny CSV/pickle), not the full daily 3D fields. |

## Suggested Zenodo deposit structure

```
data/
  ppe_raw/
    ppe_params.json              # item 7
  observations/
    dy1/, dy2/, era5/             # item 6
    regions.nc
  zrg_training_tables/            # item 1
    obs.pkl
    GP_ZRG_masked_proj.pkl
    ppe_param.pkl
  cv_results/                     # item 2
  optimization_results/           # item 3
    default_cost_fun/
    precip_cost_fun/
    precip4_cost_fun/
    no_region_cost_fun/
    tropics_weighted_cost_fun/
  evaluation_simulations/
    dy1/default/, dy1/optimal/    # item 4 (2nd-day average only)
    dy2/default/, dy2/optimal/    # item 4 (2nd-day average only)
    dy2_35day/default/, dy2_35day/optimal/   # item 5 (monthly mean only)
README.md                         # contents, units, grid, provenance per folder
```

This mirrors the `SCREAM_AUTOTUNE_DATA` directory layout expected by `paths.py` — unpacking the Zenodo deposit into `data/` at the repo root (or pointing `SCREAM_AUTOTUNE_DATA` at it) should make every script/notebook runnable without further path changes.

## Status
Data has not yet been pulled down from HPC into this local workspace. This file is a planning manifest only — no files have been transferred or uploaded yet.

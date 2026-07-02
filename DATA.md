# Data

Data for this study is split across two sources.

## Zenodo deposit

Small derived artifacts needed to reproduce the surrogate-model training,
optimization, and evaluation analyses (Secs. 4–6). DOI to be added upon
publication.

| Path | Contents | Paper ref. |
|------|----------|-----------|
| `ppe_raw/ppe_params.json` | Tunable-parameter values for all 153 PPE members | Table 1 |
| `zrg_training_tables/` | Area-weighted ZRG training tables (`obs.pkl`, `GP_ZRG_masked_proj.pkl`) | Sec. 3.4 |
| `cv_results/` | K-fold CV outputs (R², RMSE per surrogate/kernel/knot-degree) | Sec. 4, App. A |
| `optimization_results/` | Basinhopping results per cost function (`.csv`) | Sec. 5 |
| `evaluation_simulations/` | 35-day simulation pickles (time series, monthly means, obs) | Sec. 6.1 |
| `observations/masks/` | Observational-coverage masks | Sec. 3.2 |
| `observations/regions.nc` | Zonal/regional/global area definitions | Table 3 |
| `observations/dy1/`, `dy2/`, `era5/` | Regridded observational fields (redistribution rights vary by source) | Table 2 |

Full contents and usage instructions: [data_Zenodo/DATA.md](data_Zenodo/DATA.md).

## NERSC HPSS archive

Raw and large simulation output stored on NERSC long-term tape archive.

Public portal: <https://portal.nersc.gov/archive/home/j/jpaige3/www/SCREAM-autotuning/>

| Path | Contents | Paper ref. |
|------|----------|-----------|
| `dy1/m0000–m0152/` | 153-member PPE, DYAMOND1 (Aug 2016) | Sec. 3 |
| `dy2/m0000–m0152/` | 153-member PPE, DYAMOND2 (Jan 2020) | Sec. 3 |
| `dy1/optimal/`, `dy2/optimal/` | 2-day optimal-parameter evaluation runs | Sec. 6 |
| `default_40days/` | 35-day default-parameter DYAMOND2 run | Sec. 6.1 |
| `opt_mar_26_40days/` | 35-day optimal-parameter DYAMOND2 run | Sec. 6.1 |

The default 2-day evaluation run is PPE member `m0000`. See [data_HPSS/readme](data_HPSS/readme) for the system path and archive layout.

## Configuring paths

All scripts and notebooks resolve data paths through `paths.py`. Set these
environment variables to point at your local copies:

```bash
export SCREAM_AUTOTUNE_ZENODO=/path/to/unpacked/zenodo/deposit
export SCREAM_AUTOTUNE_HPSS=/path/to/hpss/mount   # on NERSC systems
```

If unset, the code defaults to `data_Zenodo/` and `data_HPSS/` relative to the
repository root.

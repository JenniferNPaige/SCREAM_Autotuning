"""
Central data-path configuration for the SCREAM_Autotuning pipeline.

Data is split across two sources:

1. **Zenodo deposit** (data_Zenodo/ in this repo, excluded from git):
   Small derived artifacts — ZRG training tables, CV outputs, optimization CSVs,
   40-day simulation pickles, observational masks/regions, and ppe_params.json.
   Set SCREAM_AUTOTUNE_ZENODO to point at the unpacked Zenodo deposit; defaults
   to data_Zenodo/ next to this file.

       export SCREAM_AUTOTUNE_ZENODO=/path/to/unpacked/zenodo/deposit

2. **NERSC HPSS archive** (data_HPSS/ in this repo contains only a readme):
   Raw PPE output, 2-day default/optimal evaluation runs, and 35-day simulations
   stored on NERSC tape archive.
   Public portal: https://portal.nersc.gov/archive/home/j/jpaige3/www/SCREAM-autotuning/
   See data_HPSS/readme for the system path and archive layout.
   Set SCREAM_AUTOTUNE_HPSS to the local mount path; defaults to data_HPSS/
   (which on your local machine contains only the readme).

       export SCREAM_AUTOTUNE_HPSS=/path/to/hpss/mount   # e.g. on NERSC

Directory layout under SCREAM_AUTOTUNE_ZENODO:

    ppe_raw/ppe_params.json          tunable-parameter values per ensemble member
    observations/dy1/, dy2/          IMERG/CERES-SYN/MAC observational files
    observations/era5/               ERA5 regridded fields (WVP, T, U, omega)
    observations/regions.nc          zonal/regional/global area definitions
    observations/masks/              observational-coverage masks
    zrg_training_tables/             pickled ZRG tables (obs + GP projections)
    cv_results/                      K-fold cross-validation outputs
    optimization_results/            per-cost-function basinhopping CSVs
    evaluation_simulations/          40-day simulation pickles

Directory layout under SCREAM_AUTOTUNE_HPSS:

    dy1/                  PPE DY1 members (dy1/m0000 … dy1/m0152) + dy1/optimal/
    dy2/                  PPE DY2 members (dy2/m0000 … dy2/m0152) + dy2/optimal/
    default_40days/       35-day default-parameter DYAMOND2 simulation
    opt_mar_26_40days/    35-day optimal-parameter DYAMOND2 simulation
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

ZENODO_ROOT = Path(os.environ.get("SCREAM_AUTOTUNE_ZENODO", REPO_ROOT / "data_Zenodo"))
HPSS_ROOT   = Path(os.environ.get("SCREAM_AUTOTUNE_HPSS",   REPO_ROOT / "data_HPSS"))
OUTPUT_ROOT = Path(os.environ.get("SCREAM_AUTOTUNE_OUTPUT",  REPO_ROOT / "outputs"))

# --- Zenodo: PPE parameter table (Sec. 3.1, Table 1) -----------------------
PPE_PARAMS_JSON = ZENODO_ROOT / "ppe_raw" / "ppe_params.json"

# --- Zenodo: Observations and region definitions (Table 2–3) ---------------
OBS_DIR      = ZENODO_ROOT / "observations"
DY1_OBS_DIR  = OBS_DIR / "dy1"
DY2_OBS_DIR  = OBS_DIR / "dy2"
ERA5_DIR     = OBS_DIR / "era5"      # ERA5 regridded fields (WVP, T, U, omega)
REGIONS_FILE = OBS_DIR / "regions.nc"
MASK_DIR     = OBS_DIR / "masks"

# --- Zenodo: Preprocessed ZRG training tables (Sec. 3.4, 4.2) -------------
ZRG_DIR        = ZENODO_ROOT / "zrg_training_tables"
OBS_PICKLE     = ZRG_DIR / "obs.pkl"
GP_PROJ_PICKLE = ZRG_DIR / "GP_ZRG_masked_proj.pkl"

# --- Zenodo: Cross-validation outputs (Sec. 4, App. A) --------------------
CV_RESULTS_DIR = ZENODO_ROOT / "cv_results"

# --- Zenodo: Optimization (Sec. 5) -----------------------------------------
COST_FUNCTIONS_DIR = REPO_ROOT / "03_optimization" / "cost_functions"
OPT_RESULTS_DIR    = ZENODO_ROOT / "optimization_results"

# --- Zenodo: 35-day simulation pickles (Sec. 6.1, App. B4) ----------------
EVAL_SIM_DIR              = ZENODO_ROOT / "evaluation_simulations"
EVAL_40DAY_TS_PICKLE      = EVAL_SIM_DIR / "40daysim_timeseries.pkl"
EVAL_40DAY_MONTHLY_PICKLE = EVAL_SIM_DIR / "40daysim_monthlymean.pkl"
EVAL_40DAY_OBS_PICKLE     = EVAL_SIM_DIR / "40daysim_obs.pkl"

# --- HPSS: PPE raw simulation output (Sec. 3) ------------------------------
# See data_HPSS/readme for archive layout and public portal link.
DY1_DIR = HPSS_ROOT / "dy1"    # 153 PPE members: dy1/m0000 … dy1/m0152
DY2_DIR = HPSS_ROOT / "dy2"    # 153 PPE members: dy2/m0000 … dy2/m0152

# Control (member 0) used as baseline and for grid/region metadata
CONTROL_FILE = (
    DY2_DIR / "m0000"
    / "output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc"
)

# --- HPSS: 2-day default/optimal evaluation runs (Sec. 6) -----------------
DY1_OPTIMAL_RUN_DIR = DY1_DIR / "optimal"
DY2_OPTIMAL_RUN_DIR = DY2_DIR / "optimal"

# --- HPSS: 35-day DYAMOND2 evaluation simulations (Sec. 6.1) --------------
EVAL_40DAY_CTL_DIR = HPSS_ROOT / "default_40days"
EVAL_40DAY_OPT_DIR = HPSS_ROOT / "opt_mar_26_40days"

# --- Figure/plot output (not part of either data deposit) ------------------
FIGURES_DIR = OUTPUT_ROOT / "figures"


def ensure_output_dirs():
    """Create output directories this pipeline writes into, if missing."""
    for d in (CV_RESULTS_DIR, OPT_RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

"""
Central data-path configuration for the SCREAM_Autotuning pipeline.

All scripts/notebooks import paths from here instead of hardcoding
HPC-specific absolute paths (e.g. /global/cfs/cdirs/...). Set the
SCREAM_AUTOTUNE_DATA environment variable to point at wherever the
companion Zenodo data deposit has been unpacked; it defaults to a
`data/` directory next to this file, matching the layout described
in DATA.md.

    export SCREAM_AUTOTUNE_DATA=/path/to/unpacked/zenodo/data

Directory layout expected under SCREAM_AUTOTUNE_DATA (see DATA.md):

    ppe_raw/dy1/                     per-member DYAMOND1 (summer) PPE output
    ppe_raw/dy2/                     per-member DYAMOND2 (winter) PPE output
    ppe_raw/ppe_params.json          tunable-parameter values per ensemble member
    observations/dy1/, dy2/          IMERG/CERES-SYN/MAC observational files
    observations/era5/               ERA5 reference fields for untuned variables
    observations/regions.nc          zonal/regional/global area definitions (Table 3)
    masks/                           observational-coverage masks
    zrg_training_tables/             pickled ZRG tables (obs + GP projections; Sec. 3.4, 4.2)
    cv_results/                      K-fold cross-validation outputs (Sec. 4, App. A)
    optimization_results/            per-cost-function basinhopping results (Sec. 5)
    evaluation_simulations/          2-day and 35-day default/optimal SCREAM runs (Sec. 6)

Outputs (figures, etc.) are written under SCREAM_AUTOTUNE_OUTPUT,
defaulting to an `outputs/` directory next to this file.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_ROOT = Path(os.environ.get("SCREAM_AUTOTUNE_DATA", REPO_ROOT / "data"))
OUTPUT_ROOT = Path(os.environ.get("SCREAM_AUTOTUNE_OUTPUT", REPO_ROOT / "outputs"))

# --- PPE raw simulation output (Sec. 3) -------------------------------------
PPE_RAW_DIR = DATA_ROOT / "ppe_raw"
DY1_DIR = PPE_RAW_DIR / "dy1"
DY2_DIR = PPE_RAW_DIR / "dy2"
PPE_PARAMS_JSON = PPE_RAW_DIR / "ppe_params.json"

# --- Observations (Table 2) and region definitions (Table 3) ---------------
OBS_DIR = DATA_ROOT / "observations"
DY1_OBS_DIR = OBS_DIR / "dy1"
DY2_OBS_DIR = OBS_DIR / "dy2"
ERA5_DIR = OBS_DIR / "era5"
REGIONS_FILE = OBS_DIR / "regions.nc"
MASK_DIR = DATA_ROOT / "masks"

# --- Default-parameter control simulation (used as a baseline in several
#     notebooks for regridding/region info) ---------------------------------
CONTROL_FILE = (
    DY2_DIR
    / "m0000"
    / "output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc"
)

# --- Preprocessed ZRG training tables (Sec. 3.4, 4.2) -----------------------
ZRG_DIR = DATA_ROOT / "zrg_training_tables"
OBS_PICKLE = ZRG_DIR / "obs.pkl"
GP_PROJ_PICKLE = ZRG_DIR / "GP_ZRG_masked_proj.pkl"

# --- Cross-validation outputs (Sec. 4, App. A) ------------------------------
CV_RESULTS_DIR = DATA_ROOT / "cv_results"

# --- Optimization (Sec. 5) ---------------------------------------------------
COST_FUNCTIONS_DIR = REPO_ROOT / "03_optimization" / "cost_functions"
OPT_RESULTS_DIR = DATA_ROOT / "optimization_results"

# --- Evaluation simulations (Sec. 6, App. B) --------------------------------
EVAL_SIM_DIR = DATA_ROOT / "evaluation_simulations"

# 2nd-day-average default/optimal runs (item 4 in DATA.md), per DYAMOND period
DY1_DEFAULT_RUN_DIR = EVAL_SIM_DIR / "dy1" / "default"
DY1_OPTIMAL_RUN_DIR = EVAL_SIM_DIR / "dy1" / "optimal"
DY2_DEFAULT_RUN_DIR = EVAL_SIM_DIR / "dy2" / "default"
DY2_OPTIMAL_RUN_DIR = EVAL_SIM_DIR / "dy2" / "optimal"

# --- Figure/plot output (not part of the Zenodo data deposit) --------------
FIGURES_DIR = OUTPUT_ROOT / "figures"


def ensure_output_dirs():
    """Create the output directories this pipeline writes into, if missing."""
    for d in (CV_RESULTS_DIR, OPT_RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

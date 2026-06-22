"""
GP-only KFolds surrogate — grid cell level data, ZRG-level R².

Training:
  GP is trained on normalized per-grid-cell values (masked to obs coverage),
  using the same preprocessing steps as Final_Surrogate_Preprocessing.ipynb.

Evaluation:
  After prediction, grid cells are inverse-transformed to physical space, then
  aggregated to zonal/regional/global (ZRG) averages — identical to the
  original script. R² and RMSE are computed on those ZRG averages so results
  are directly comparable to Final_surrogate_Kfolds_masked_10seeds.py.
  The same 6 polar zone columns are dropped (DY1/DY2 × {-85.0, -75.0, 85.0}).

Structure:
  10-seed, 5-fold cross-validation — identical to the original script.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import json
import numpy as np
import pandas as pd
import xarray as xr

from esem import gp_model

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn import preprocessing

from datetime import datetime


# ── User configuration ────────────────────────────────────────────────────────

N_SEEDS   = 10
BASE_SEED = 0
folds     = 5

# Zones dropped in the original script (NaN-heavy polar bands)
DROPPED_ZONES = ['DY1_85.0', 'DY1_-85.0', 'DY1_-75.0',
                 'DY2_85.0', 'DY2_-85.0', 'DY2_-75.0']


# ── JSON encoder ──────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# ── Averaging helpers (from Final_Surrogate_Preprocessing-2.ipynb) ─────────────

def zonal_means_native(data, area, lat, lon):
    data        = np.asarray(data).squeeze()
    masked_area = np.where(np.isnan(data), np.nan, area).squeeze()
    lat         = np.asarray(lat).squeeze()
    lat_bands   = np.linspace(-90, 90, 19)  # 10-degree bands
    zonal_means = {}
    for i in range(len(lat_bands) - 1):
        mask_zone  = (lat >= lat_bands[i]) & (lat < lat_bands[i + 1])
        data_zone  = np.where(mask_zone, data, np.nan)
        area_zone  = np.where(mask_zone, masked_area, np.nan)
        zone_center = lat_bands[i] + (lat_bands[i + 1] - lat_bands[i]) / 2
        if np.all(np.isnan(data_zone)) or np.nansum(area_zone) == 0:
            zonal_means[zone_center] = np.nan
        else:
            zonal_means[zone_center] = np.nansum(data_zone * area_zone) / np.nansum(area_zone)
    return zonal_means


def regional_means_native(data, area, region_data):
    data        = np.asarray(data).squeeze()
    masked_area = np.where(np.isnan(data), np.nan, area).squeeze()
    regions_list = ['poles', 'extratropical_land', 'extratropical_ocean',
                    'tropical_land', 'ascending_tropical_ocean', 'descending_tropical_ocean']
    region_means = {}
    for reg_name in regions_list:
        mask_reg = np.asarray(region_data[reg_name]).squeeze()
        data_reg = np.where(mask_reg > 0, data, np.nan)
        area_reg = np.where(mask_reg > 0, masked_area, np.nan)
        if np.all(np.isnan(data_reg)) or np.nansum(area_reg) == 0:
            region_means[reg_name] = np.nan
        else:
            region_means[reg_name] = np.nansum(data_reg * area_reg) / np.nansum(area_reg)
    return region_means


def global_means_native(data, area):
    data        = np.asarray(data).squeeze()
    masked_area = np.where(np.isnan(data), np.nan, area).squeeze()
    return np.nansum(data * masked_area) / np.nansum(masked_area)


def gridcells_to_zrg_df(phys_array, valid_mask, n_ncol, run_labels,
                         area, lat, lon, regions_file, day_prefix):
    """
    Convert (n_runs, n_valid_cells) physical predictions to a ZRG DataFrame.

    phys_array  : (n_runs, n_valid_cells) ndarray in physical space
    valid_mask  : bool array shape (n_ncol,) — which ncol indices are valid
    n_ncol      : total number of grid columns (e.g. 21600 for ne30pg2)
    run_labels  : list of run names (row index)
    day_prefix  : 'DY1' or 'DY2'
    """
    zonal_rows, regional_rows, global_vals = [], [], []
    full = np.full(n_ncol, np.nan)
    for row in phys_array:
        full[:] = np.nan
        full[valid_mask] = row
        zonal_rows.append(zonal_means_native(full, area, lat, lon))
        regional_rows.append(regional_means_native(full, area, regions_file))
        global_vals.append(global_means_native(full, area))

    z_df  = pd.DataFrame(zonal_rows,    index=run_labels)
    r_df  = pd.DataFrame(regional_rows, index=run_labels)
    df    = pd.concat([z_df, r_df], axis=1)
    df['global'] = global_vals
    df.columns   = [f'{day_prefix}_{c}' for c in df.columns]
    return df


def build_zrg_df(dy1_phys, dy2_phys, dy1_valid, dy2_valid, n_ncol,
                 run_labels, area, lat, lon, regions_file):
    """
    Combine DY1 and DY2 ZRG DataFrames and drop the polar zone columns.
    dy1_phys / dy2_phys : (n_runs, n_valid_cells) ndarrays
    """
    dy1_zrg = gridcells_to_zrg_df(dy1_phys, dy1_valid, n_ncol, run_labels,
                                   area, lat, lon, regions_file, 'DY1')
    dy2_zrg = gridcells_to_zrg_df(dy2_phys, dy2_valid, n_ncol, run_labels,
                                   area, lat, lon, regions_file, 'DY2')
    combined = pd.concat([dy1_zrg, dy2_zrg], axis=1)
    cols_to_drop = [c for c in DROPPED_ZONES if c in combined.columns]
    return combined.drop(columns=cols_to_drop)


def compute_metrics(y_true_norm, y_pred_norm, y_true_phys, y_pred_phys, label):
    r2_norm     = r2_score(y_true_norm, y_pred_norm, multioutput='uniform_average')
    r2_norm_raw = r2_score(y_true_norm, y_pred_norm, multioutput='raw_values')
    rmse_norm   = root_mean_squared_error(y_true_norm, y_pred_norm)
    r2_phys     = r2_score(y_true_phys, y_pred_phys, multioutput='uniform_average')
    r2_phys_raw = r2_score(y_true_phys, y_pred_phys, multioutput='raw_values')
    rmse_phys   = root_mean_squared_error(y_true_phys, y_pred_phys)
    print(f"  {label}: R²(norm)={r2_norm:.4f}  RMSE(norm)={rmse_norm:.4f} | "
          f"R²(phys)={r2_phys:.4f}  RMSE(phys)={rmse_phys:.4f}")
    return r2_norm, rmse_norm, r2_phys, rmse_phys, r2_norm_raw, r2_phys_raw


# ── Paths ─────────────────────────────────────────────────────────────────────

params_json  = '/global/cfs/cdirs/e3sm/jpaige3/ESEm/SCREAM.2024-autocal-00.ne1024pg2-params.json'

DY1_path     = '/global/cfs/cdirs/e3sm/jpaige3/dy1ne1024'
DY2_path     = '/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/SCREAM.2024-autocal-00.ne1024pg2/'

DY1_obs_dir  = '/global/cfs/cdirs/e3smdata/simulations/ecp-autotune/obs/'
DY2_obs_dir  = '/global/cfs/projectdirs/e3smdata/simulations/SCREAM.2024-autocal-00.ne1024pg2/obs/'

control_file = ('/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/'
                'SCREAM.2024-autocal-00.ne1024pg2/m0000/SCREAM.2024-autocal-00.ne1024pg2/run/'
                'output.scream.AutoCal.daily_avg_ne30pg2.AVERAGE.nhours_x24.2020-01-26-00000.nc')

regions_path = '/global/cfs/projectdirs/e3smdata/simulations/ecp-autotune/regions.nc'

save_dir     = '/global/cfs/cdirs/e3sm/jpaige3/ESEm/CV_Saved_Model_Data_masked/10seeds_GP_gridcell_raw'
os.makedirs(os.path.join(save_dir, 'GP'), exist_ok=True)


# ── Load parameters ───────────────────────────────────────────────────────────

ppe_params_all = pd.read_json(params_json)


# ── Collect valid run folders and file paths ──────────────────────────────────

def _ice_sed_ok(folder, ppe_params_all):
    try:
        return float(ppe_params_all['p3_ice_sed_knob'][folder]) >= 1.0
    except KeyError:
        return False


def collect_dy2(path, ppe_params_all):
    folders = []
    for m in range(0, 301):
        folder = 'm{:04d}'.format(m)
        if os.path.exists(os.path.join(path, folder)):
            folders.append(folder)
    for f in os.listdir(path):
        if f.startswith('opt'):
            folders.append(f)
    for bad in ['m0230', 'optmar20day5', 'm0024', 'm0025', 'm0061',
                'optmar22hd', 'optmar15', 'optmar20day2']:
        if bad in folders:
            folders.remove(bad)
    folders = sorted(folders)
    nc_file = ('output.scream.AutoCal.daily_avg_ne30pg2.'
               'AVERAGE.nhours_x24.2020-01-26-00000.nc')
    file_list = [
        os.path.join(path, f, 'SCREAM.2024-autocal-00.ne1024pg2', 'run', nc_file)
        for f in folders
    ]
    mask = np.array([_ice_sed_ok(f, ppe_params_all) for f in folders])
    return np.array(file_list)[mask], np.array(folders)[mask]


def collect_dy1(path, ppe_params_all):
    folders = []
    for m in range(0, 301):
        folder = 'm{:04d}'.format(m)
        if os.path.exists(os.path.join(path, folder)):
            folders.append(folder)
    for f in os.listdir(path):
        if f.startswith('opt'):
            folders.append(f)
    bad = [
        'm0024', 'm0025', 'm0061', 'optmar22hd',
        'm0262', 'm0263', 'm0264', 'm0266', 'm0267', 'm0270', 'm0272',
        'm0274', 'm0275', 'm0279', 'm0289', 'm0290', 'm0292', 'm0293',
        'm0294', 'm0295', 'm0296', 'm0299', 'm0300',
        'optmar15seed0', 'optmar27a', 'optmar20dayAll', 'optmar20day2-fail',
        'optmar15b', 'optmar20day2-ltend', 'm0230', 'optmar20day5',
    ]
    for b in bad:
        if b in folders:
            folders.remove(b)
    folders = sorted(folders)
    nc_file = ('output.scream.AutoCal.daily_avg_ne30pg2.'
               'AVERAGE.nhours_x24.2016-08-07-00000.nc')
    file_list = [
        os.path.join(path, f, 'SCREAM.2024-autocal-00.ne1024pg2', 'run', nc_file)
        for f in folders
    ]
    mask = np.array([_ice_sed_ok(f, ppe_params_all) for f in folders])
    return np.array(file_list)[mask], np.array(folders)[mask]


DY1_files, DY1_names = collect_dy1(DY1_path, ppe_params_all)
DY2_files, DY2_names = collect_dy2(DY2_path, ppe_params_all)

sim_names = sorted(set(DY1_names) & set(DY2_names))
print(f"Runs in both DY1 and DY2: {len(sim_names)}")

DY1_files = np.array([DY1_files[list(DY1_names).index(s)] for s in sim_names])
DY2_files = np.array([DY2_files[list(DY2_names).index(s)] for s in sim_names])

ppe_params = ppe_params_all.loc[sim_names]


# ── Load xarray datasets ──────────────────────────────────────────────────────

print("Loading DY1 xarray dataset...")
DY1_ds = xr.open_mfdataset(DY1_files, concat_dim='run_label', combine='nested')
DY1_ds = DY1_ds.assign_coords(run_label=('run_label', sim_names)).squeeze('time')

print("Loading DY2 xarray dataset...")
DY2_ds = xr.open_mfdataset(DY2_files, concat_dim='run_label', combine='nested')
DY2_ds = DY2_ds.assign_coords(run_label=('run_label', sim_names)).squeeze('time')


# ── Filter variables and compute TotalLiqWaterPath ────────────────────────────

to_keep = ['precip_total_surf_mass_flux', 'LiqWaterPath', 'RainWaterPath',
           'SW_flux_up_at_model_top', 'LW_flux_up_at_model_top']
DY1_ds = DY1_ds[to_keep]
DY2_ds = DY2_ds[to_keep]
DY1_ds['TotalLiqWaterPath'] = DY1_ds['LiqWaterPath'] + DY1_ds['RainWaterPath']
DY2_ds['TotalLiqWaterPath'] = DY2_ds['LiqWaterPath'] + DY2_ds['RainWaterPath']


# ── Load observations ─────────────────────────────────────────────────────────

DY1_PCP_obs  = xr.open_dataset(
    DY1_obs_dir + 'IMERG.precip_total_surf_mass_flux.daily_AVERAGE.ne30pg2.20160807_mahf708.nc'
)['precip_total_surf_mass_flux'].squeeze('time')
DY1_TLWP_obs = (xr.open_dataset(
    DY1_obs_dir + 'mac.clwp-tlwp-wvp.20160807.ne30pg2.nc'
)['tlwp'] * 1e-3).squeeze('time')
DY1_OSR_obs  = xr.open_dataset(
    DY1_obs_dir + 'CERES.SW_flux_up_at_model_top.daily_AVERAGE.ne30pg2.20160807_mahf708.nc'
)['SW_flux_up_at_model_top'].squeeze('time')
DY1_OLR_obs  = xr.open_dataset(
    DY1_obs_dir + 'CERES.LW_flux_up_at_model_top.daily_AVERAGE.ne30pg2.20160807_mahf708.nc'
)['LW_flux_up_at_model_top'].squeeze('time')

DY2_PCP_obs  = xr.open_dataset(
    DY2_obs_dir + 'IMERG.precip_total_surf_mass_flux.AVERAGE.ne30pg2.20200126.nc'
)['precip_total_surf_mass_flux'].squeeze('time')
DY2_TLWP_obs = (xr.open_dataset(
    DY2_obs_dir + 'mac.clwp-tlwp-wvp.20200126.ne30pg2.nc'
)['tlwp'] * 1e-3).squeeze('time')
DY2_OSR_obs  = xr.open_dataset(
    DY2_obs_dir + 'CERES.SW_flux_up_at_model_top.AVERAGE.ne30pg2.20200126.nc'
)['SW_flux_up_at_model_top'].squeeze('time')
DY2_OLR_obs  = xr.open_dataset(
    DY2_obs_dir + 'CERES.LW_flux_up_at_model_top.AVERAGE.ne30pg2.20200126.nc'
)['LW_flux_up_at_model_top'].squeeze('time')


# ── Valid (non-NaN) ncol indices per variable and day ─────────────────────────

DY1_PCP_valid  = ~np.isnan(DY1_PCP_obs.values)
DY1_TLWP_valid = ~np.isnan(DY1_TLWP_obs.values)
DY1_OSR_valid  = ~np.isnan(DY1_OSR_obs.values)
DY1_OLR_valid  = ~np.isnan(DY1_OLR_obs.values)

DY2_PCP_valid  = ~np.isnan(DY2_PCP_obs.values)
DY2_TLWP_valid = ~np.isnan(DY2_TLWP_obs.values)
DY2_OSR_valid  = ~np.isnan(DY2_OSR_obs.values)
DY2_OLR_valid  = ~np.isnan(DY2_OLR_obs.values)

n_ncol = len(DY1_PCP_obs.values)   # total grid columns (e.g. 21600 for ne30pg2)

print(f"Total ncol: {n_ncol}")
print(f"Valid grid cells — PCP:  DY1={DY1_PCP_valid.sum()},  DY2={DY2_PCP_valid.sum()}")
print(f"                  TLWP: DY1={DY1_TLWP_valid.sum()}, DY2={DY2_TLWP_valid.sum()}")
print(f"                  OSR:  DY1={DY1_OSR_valid.sum()},  DY2={DY2_OSR_valid.sum()}")
print(f"                  OLR:  DY1={DY1_OLR_valid.sum()},  DY2={DY2_OLR_valid.sum()}")


# ── Grid metadata for ZRG averaging ──────────────────────────────────────────

print("Loading grid metadata (area, lat, lon, regions)...")
control_ds   = xr.open_dataset(control_file)
area         = control_ds['area'].values.squeeze()
lat          = control_ds['lat'].values.squeeze()
lon          = control_ds['lon'].values.squeeze()
regions_file = xr.open_dataset(regions_path)


# ── Build grid-cell DataFrames (runs × ncol) ──────────────────────────────────
# Bulk-loads full (n_runs, ncol) arrays, then slices to valid cells.
# These are the GP's Y training space.

def build_gridcell_df(dy1_ds, dy2_ds, varname, dy1_valid, dy2_valid, sim_names):
    dy1_idx = np.where(dy1_valid)[0]
    dy2_idx = np.where(dy2_valid)[0]
    cols    = [f'DY1_{i}' for i in dy1_idx] + [f'DY2_{i}' for i in dy2_idx]
    dy1_arr = dy1_ds[varname].values   # (n_runs, ncol)
    dy2_arr = dy2_ds[varname].values
    data    = np.concatenate([dy1_arr[:, dy1_valid], dy2_arr[:, dy2_valid]], axis=1)
    return pd.DataFrame(data, index=sim_names, columns=cols)


print("Building grid-cell DataFrames (triggers xarray I/O)...")
PCP_gc_ppedataset  = build_gridcell_df(
    DY1_ds, DY2_ds, 'precip_total_surf_mass_flux',
    DY1_PCP_valid, DY2_PCP_valid, sim_names)

TLWP_gc_ppedataset = build_gridcell_df(
    DY1_ds, DY2_ds, 'TotalLiqWaterPath',
    DY1_TLWP_valid, DY2_TLWP_valid, sim_names)

OSR_gc_ppedataset  = build_gridcell_df(
    DY1_ds, DY2_ds, 'SW_flux_up_at_model_top',
    DY1_OSR_valid, DY2_OSR_valid, sim_names)

OLR_gc_ppedataset  = build_gridcell_df(
    DY1_ds, DY2_ds, 'LW_flux_up_at_model_top',
    DY1_OLR_valid, DY2_OLR_valid, sim_names)

print(f"Shapes — PCP:{PCP_gc_ppedataset.shape}  TLWP:{TLWP_gc_ppedataset.shape}  "
      f"OSR:{OSR_gc_ppedataset.shape}  OLR:{OLR_gc_ppedataset.shape}")

assert list(ppe_params.index) == list(PCP_gc_ppedataset.index) == sim_names


# ── Pre-compute ZRG truth for all runs (done once, reused across folds) ────────
# These are the reference DataFrames used for R² in physical space.
# Number of DY1 valid cells per variable (needed to split grid-cell arrays):
n_DY1_PCP  = DY1_PCP_valid.sum()
n_DY1_TLWP = DY1_TLWP_valid.sum()
n_DY1_OSR  = DY1_OSR_valid.sum()
n_DY1_OLR  = DY1_OLR_valid.sum()

print("Pre-computing ZRG truth DataFrames for all runs...")
PCP_zrg_truth  = build_zrg_df(
    PCP_gc_ppedataset.values[:, :n_DY1_PCP],
    PCP_gc_ppedataset.values[:, n_DY1_PCP:],
    DY1_PCP_valid, DY2_PCP_valid, n_ncol,
    sim_names, area, lat, lon, regions_file)

TLWP_zrg_truth = build_zrg_df(
    TLWP_gc_ppedataset.values[:, :n_DY1_TLWP],
    TLWP_gc_ppedataset.values[:, n_DY1_TLWP:],
    DY1_TLWP_valid, DY2_TLWP_valid, n_ncol,
    sim_names, area, lat, lon, regions_file)

OSR_zrg_truth  = build_zrg_df(
    OSR_gc_ppedataset.values[:, :n_DY1_OSR],
    OSR_gc_ppedataset.values[:, n_DY1_OSR:],
    DY1_OSR_valid, DY2_OSR_valid, n_ncol,
    sim_names, area, lat, lon, regions_file)

OLR_zrg_truth  = build_zrg_df(
    OLR_gc_ppedataset.values[:, :n_DY1_OLR],
    OLR_gc_ppedataset.values[:, n_DY1_OLR:],
    DY1_OLR_valid, DY2_OLR_valid, n_ncol,
    sim_names, area, lat, lon, regions_file)

print(f"ZRG truth shapes — PCP:{PCP_zrg_truth.shape}  TLWP:{TLWP_zrg_truth.shape}  "
      f"OSR:{OSR_zrg_truth.shape}  OLR:{OLR_zrg_truth.shape}")


# ── Parameter bounds ──────────────────────────────────────────────────────────

dict_range_pars = {
    'length_fac':                  [0.1, 10],
    'p3_spa_to_nc':                [0.1, 10],
    'p3_k_accretion':              [0.01, 100],
    'p3_ice_sed_knob':             [1, 2],
    'thl2tune':                    [0.1, 10],
    'qw2tune':                     [0.1, 10],
    'c_diag_3rd_mom':              [0.1, 10],
    'Ckh':                         [0.1, 1],
    'Ckm':                         [0.1, 1],
    'lambda_low':                  [0.0001, 0.1],
    'lambda_high':                 [0.0001, 0.1],
    'p3_eci':                      [0.1, 1],
    'p3_eri':                      [0.1, 1],
    'p3_dep_nucleation_exponent':  [0.2, 0.304],
    'p3_d_breakup_cutoff':         [0, 500e-6],
    'max_total_ni':                [5e5, 1e7],
}

param_bounds = np.array([dict_range_pars[param] for param in ppe_params.columns])


# ── K-fold cross-validation across multiple seeds ─────────────────────────────

all_results = {'GP': []}

for seed_offset in range(N_SEEDS):
    seed = BASE_SEED + seed_offset
    kf   = KFold(n_splits=folds, shuffle=True, random_state=seed)

    print(f"\n{'#'*60}")
    print(f"Seed {seed_offset + 1}/{N_SEEDS}  (random_state={seed})")
    print('#'*60)

    for k, (train_index, test_index) in enumerate(kf.split(ppe_params)):
        print(f"\n{'='*60}")
        print(f"Seed {seed_offset + 1}/{N_SEEDS}  Fold {k + 1}/{folds}")
        print('='*60)

        X_train = ppe_params.iloc[train_index]
        X_test  = ppe_params.iloc[test_index]
        train_run_labels = X_train.index.to_list()
        test_run_labels  = X_test.index.to_list()
        print("Test runs:", test_run_labels)

        # ── Grid-cell train/test splits (GP training space) ───────────────────
        PCP_gc_train  = PCP_gc_ppedataset.loc[train_run_labels]
        TLWP_gc_train = TLWP_gc_ppedataset.loc[train_run_labels]
        OSR_gc_train  = OSR_gc_ppedataset.loc[train_run_labels]
        OLR_gc_train  = OLR_gc_ppedataset.loc[train_run_labels]

        PCP_gc_test  = PCP_gc_ppedataset.loc[test_run_labels]
        TLWP_gc_test = TLWP_gc_ppedataset.loc[test_run_labels]
        OSR_gc_test  = OSR_gc_ppedataset.loc[test_run_labels]
        OLR_gc_test  = OLR_gc_ppedataset.loc[test_run_labels]

        for df in [PCP_gc_train, TLWP_gc_train, OSR_gc_train, OLR_gc_train,
                   PCP_gc_test,  TLWP_gc_test,  OSR_gc_test,  OLR_gc_test]:
            df.columns = df.columns.astype(str)

        # ── ZRG truth splits (evaluation space) ───────────────────────────────
        PCP_zrg_train  = PCP_zrg_truth.loc[train_run_labels]
        TLWP_zrg_train = TLWP_zrg_truth.loc[train_run_labels]
        OSR_zrg_train  = OSR_zrg_truth.loc[train_run_labels]
        OLR_zrg_train  = OLR_zrg_truth.loc[train_run_labels]

        PCP_zrg_test  = PCP_zrg_truth.loc[test_run_labels]
        TLWP_zrg_test = TLWP_zrg_truth.loc[test_run_labels]
        OSR_zrg_test  = OSR_zrg_truth.loc[test_run_labels]
        OLR_zrg_test  = OLR_zrg_truth.loc[test_run_labels]

        print(f"Y_train grid-cell shape (PCP): {PCP_gc_train.shape}")
        print(f"Y_train ZRG shape      (PCP): {PCP_zrg_train.shape}")

        # ── Normalisation (on grid-cell data — same StandardScaler as original)
        X_pipe_sk_minmax = preprocessing.MinMaxScaler()
        X_pipe_sk_minmax.fit(param_bounds.T)
        X_train_norm = X_pipe_sk_minmax.transform(X_train)
        X_test_norm  = X_pipe_sk_minmax.transform(X_test)

        gc_scalers   = {}   # fitted on grid-cell training data
        gc_norm_data = {}
        for name, gc_train_df, gc_test_df in [
            ('PCP',  PCP_gc_train,  PCP_gc_test),
            ('TLWP', TLWP_gc_train, TLWP_gc_test),
            ('OSR',  OSR_gc_train,  OSR_gc_test),
            ('OLR',  OLR_gc_train,  OLR_gc_test),
        ]:
            scaler = preprocessing.StandardScaler()
            scaler.fit(gc_train_df)
            gc_scalers[name] = scaler
            gc_norm_data[f'{name}_train_norm'] = scaler.transform(gc_train_df)
            gc_norm_data[f'{name}_test_norm']  = scaler.transform(gc_test_df)

        print(f"X_train_norm: {X_train_norm.shape}")

        # ── One GP per variable (each has a different number of valid grid cells)
        # R²_norm : normalized grid-cell predictions vs normalized truth
        # R²_phys : ZRG averages of physical predictions vs ZRG truth

        var_names     = ['PCP', 'TLWP', 'OSR', 'OLR']
        n_dy1_per_var = [n_DY1_PCP, n_DY1_TLWP, n_DY1_OSR, n_DY1_OLR]
        dy1_valids    = [DY1_PCP_valid, DY1_TLWP_valid, DY1_OSR_valid, DY1_OLR_valid]
        dy2_valids    = [DY2_PCP_valid, DY2_TLWP_valid, DY2_OSR_valid, DY2_OLR_valid]
        var_zrg_trains = [PCP_zrg_train, TLWP_zrg_train, OSR_zrg_train, OLR_zrg_train]
        var_zrg_tests  = [PCP_zrg_test,  TLWP_zrg_test,  OSR_zrg_test,  OLR_zrg_test]

        gp_results = {'model': 'GP', 'seed': seed, 'fold': k}

        for i, varname in enumerate(var_names):
            y_tr_norm = gc_norm_data[f'{varname}_train_norm']   # (n_train, n_cells)
            y_te_norm = gc_norm_data[f'{varname}_test_norm']    # (n_test,  n_cells)

            # esem gp_model expects (n_samples, n_spatial, n_vars)
            print(f"\nTraining GP [{varname}]  Y shape: {y_tr_norm.shape}")
            gp_var = gp_model(X_train_norm, y_tr_norm[:, :, np.newaxis])
            gp_var.train()

            m_tr, _ = gp_var.predict(X_train_norm)   # (n_train, n_cells, 1)
            m_te, _ = gp_var.predict(X_test_norm)

            pred_tr_norm = m_tr[:, :, 0]   # (n_train, n_cells)
            pred_te_norm = m_te[:, :, 0]

            n_dy1 = n_dy1_per_var[i]
            pred_tr_phys = gc_scalers[varname].inverse_transform(pred_tr_norm)
            pred_te_phys = gc_scalers[varname].inverse_transform(pred_te_norm)

            zrg_tr = build_zrg_df(
                pred_tr_phys[:, :n_dy1], pred_tr_phys[:, n_dy1:],
                dy1_valids[i], dy2_valids[i], n_ncol,
                train_run_labels, area, lat, lon, regions_file)
            zrg_te = build_zrg_df(
                pred_te_phys[:, :n_dy1], pred_te_phys[:, n_dy1:],
                dy1_valids[i], dy2_valids[i], n_ncol,
                test_run_labels, area, lat, lon, regions_file)

            print(f"\n  [GP/{varname}] In-sample  (norm=grid-cell | phys=ZRG):")
            r2_n, rmse_n, r2_p, rmse_p, r2_n_raw, r2_p_raw = compute_metrics(
                y_tr_norm, pred_tr_norm, var_zrg_trains[i], zrg_tr, varname + "(train)")
            gp_results[f'{varname}_train_r2_norm']      = r2_n
            gp_results[f'{varname}_train_rmse_norm']    = rmse_n
            gp_results[f'{varname}_train_r2_phys']      = r2_p
            gp_results[f'{varname}_train_rmse_phys']    = rmse_p
            gp_results[f'{varname}_train_r2_norm_raw']  = r2_n_raw
            gp_results[f'{varname}_train_r2_phys_raw']  = r2_p_raw

            print(f"\n  [GP/{varname}] Out-of-sample (norm=grid-cell | phys=ZRG):")
            r2_n, rmse_n, r2_p, rmse_p, r2_n_raw, r2_p_raw = compute_metrics(
                y_te_norm, pred_te_norm, var_zrg_tests[i], zrg_te, varname + "(test)")
            gp_results[f'{varname}_test_r2_norm']      = r2_n
            gp_results[f'{varname}_test_rmse_norm']    = rmse_n
            gp_results[f'{varname}_test_r2_phys']      = r2_p
            gp_results[f'{varname}_test_rmse_phys']    = rmse_p
            gp_results[f'{varname}_test_r2_norm_raw']  = r2_n_raw
            gp_results[f'{varname}_test_r2_phys_raw']  = r2_p_raw

        all_results['GP'].append(gp_results)

        timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        gp_r2_file = os.path.join(
            save_dir, 'GP',
            f"GP_gridcell_r2output_uniform_avg_seed={seed}_k={k}_{timestamp}.json"
        )
        with open(gp_r2_file, 'w') as f:
            json.dump({'timestamp': timestamp, 'model': 'Gaussian Process',
                       'seed': seed, 'fold': k, 'metrics': gp_results},
                      f, indent=2, cls=NumpyEncoder)


# ── Summary across all seeds ──────────────────────────────────────────────────

print("\n" + "="*60)
print(f"Cross-validation summary  ({N_SEEDS} seeds × {folds} folds)  — GP grid-cell")
print(f"Values: mean ± std across per-seed means")
print("="*60)

fold_list   = all_results['GP']
metric_keys = [mk for mk in fold_list[0] if mk not in ('model', 'seed', 'fold')]
all_seeds   = sorted(set(r['seed'] for r in fold_list))

seed_means = {mk: [] for mk in metric_keys}
for s in all_seeds:
    seed_folds = [r for r in fold_list if r['seed'] == s]
    for mk in metric_keys:
        vals = [r[mk] for r in seed_folds if mk in r]
        if vals:
            seed_means[mk].append(np.mean(vals))

print("\nGP:")
for mk in metric_keys:
    means = seed_means[mk]
    if means:
        print(f"  {mk}: mean={np.mean(means):.4f}  std={np.std(means, ddof=1):.4f}")

timestamp    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
summary_file = os.path.join(save_dir, f"CV_summary_gridcell_{timestamp}.json")
with open(summary_file, 'w') as f:
    json.dump(all_results, f, indent=2, cls=NumpyEncoder)
print("\nSaved CV summary to", summary_file)

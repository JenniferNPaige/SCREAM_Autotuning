"""
GP-only KFolds cross-validation — saves per-component (raw) R² values.

Runs N_SEEDS independent KFold shuffles of the full dataset.  For each
(seed, fold) the GP is trained and the out-of-sample R² is saved as a
full array of per-spatial-column values (multioutput='raw_values'), in
addition to the variance-weighted scalar for comparison.

Output per (seed, fold):
  - JSON with scalar R² and RMSE (physical + normalized)
  - npz with raw per-column R² arrays: shape (n_spatial_cols,) per variable

Summary JSON at the end: mean ± std of the scalar metrics across seeds.
"""

# ── User configuration ────────────────────────────────────────────────────────

N_SEEDS   = 10
BASE_SEED = 0
folds     = 5

obs_filename     = '/global/cfs/cdirs/e3sm/jpaige3/ESEm/GP_Saved_Model_Data/obs_2026-03-23_12-24-55.pkl'
GP_proj_filename = '/global/cfs/cdirs/e3sm/jpaige3/ESEm/GP_Saved_Model_Data/GP_ZRG_masked_proj_2026-03-23_12-24-55.pkl'
save_dir         = "/global/cfs/cdirs/e3sm/jpaige3/ESEm/CV_Saved_Model_Data_masked/GP_raw_r2"

# ── Imports ───────────────────────────────────────────────────────────────────

import os
import json
import pickle
import numpy as np
import pandas as pd

from esem import gp_model
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn import preprocessing
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Load data ─────────────────────────────────────────────────────────────────

with open(obs_filename, 'rb') as f:
    loaded_obs = pickle.load(f)

with open(GP_proj_filename, 'rb') as f:
    loaded = pickle.load(f)

ppe_params          = loaded['X_train']
PCP_zrg_ppedataset  = loaded['PCP_train']
TLWP_zrg_ppedataset = loaded['TLWP_train']
OSR_zrg_ppedataset  = loaded['OSR_train']
OLR_zrg_ppedataset  = loaded['OLR_train']

sim_names = list(ppe_params.index)
print(f"Loaded {len(sim_names)} runs")

VAR_NAMES = ['PCP', 'TLWP', 'OSR', 'OLR']
var_datasets = {
    'PCP':  PCP_zrg_ppedataset,
    'TLWP': TLWP_zrg_ppedataset,
    'OSR':  OSR_zrg_ppedataset,
    'OLR':  OLR_zrg_ppedataset,
}

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
X_pipe_sk_minmax = preprocessing.MinMaxScaler()
X_pipe_sk_minmax.fit(param_bounds.T)

os.makedirs(save_dir, exist_ok=True)

# ── Cross-validation loop ─────────────────────────────────────────────────────

# scalar results for the summary
all_scalar_results = []

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
        train_labels = X_train.index.to_list()
        test_labels  = X_test.index.to_list()

        X_train_norm = X_pipe_sk_minmax.transform(X_train)
        X_test_norm  = X_pipe_sk_minmax.transform(X_test)

        # Normalise Y per variable with StandardScaler fit on train split
        scalers         = {}
        train_norm_list = []
        for var in VAR_NAMES:
            y_train = var_datasets[var].loc[train_labels].copy()
            y_train.columns = y_train.columns.astype(str)
            scaler = preprocessing.StandardScaler().fit(y_train)
            scalers[var] = scaler
            train_norm_list.append(scaler.transform(y_train))

        Y_train_norm = np.transpose(np.stack(train_norm_list, axis=0), (1, 2, 0))

        # ── Train GP ──────────────────────────────────────────────────────────
        print(f"  Training GP on {len(train_labels)} runs...")
        model_gp = gp_model(X_train_norm, Y_train_norm)
        model_gp.train()

        m_test, _ = model_gp.predict(X_test_norm)

        # ── Compute metrics ───────────────────────────────────────────────────
        scalar_record  = {'seed': seed, 'fold': k}
        raw_r2_arrays  = {}   # var -> np.ndarray shape (n_cols,)
        raw_rmse_arrays = {}  # var -> np.ndarray shape (n_cols,)

        for i, var in enumerate(VAR_NAMES):
            y_test = var_datasets[var].loc[test_labels].copy()
            y_test.columns = y_test.columns.astype(str)
            y_test_norm = scalers[var].transform(y_test)
            pred_norm   = m_test[:, :, i]

            # Scalar metrics (normalized space)
            r2_vw_norm   = r2_score(y_test_norm, pred_norm, multioutput='variance_weighted')
            rmse_norm     = root_mean_squared_error(y_test_norm, pred_norm)

            # Physical space
            pred_phys  = scalers[var].inverse_transform(pred_norm)
            r2_vw_phys = r2_score(y_test.values, pred_phys, multioutput='variance_weighted')
            rmse_phys  = root_mean_squared_error(y_test.values, pred_phys)

            # Raw per-column R² and RMSE (physical space)
            r2_raw   = r2_score(y_test.values, pred_phys, multioutput='raw_values')
            rmse_raw = np.sqrt(np.mean((y_test.values - pred_phys) ** 2, axis=0))
            raw_r2_arrays[var]   = r2_raw
            raw_rmse_arrays[var] = rmse_raw

            scalar_record[f'{var}_r2_vw_norm']  = float(r2_vw_norm)
            scalar_record[f'{var}_rmse_norm']    = float(rmse_norm)
            scalar_record[f'{var}_r2_vw_phys']  = float(r2_vw_phys)
            scalar_record[f'{var}_rmse_phys']   = float(rmse_phys)

            print(f"    {var}: R²(vw,phys)={r2_vw_phys:.4f}  RMSE(phys)={rmse_phys:.4f}  "
                  f"raw R²/RMSE shape={r2_raw.shape}")

        all_scalar_results.append(scalar_record)

        # ── Save per-fold outputs ─────────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        stem = f"GP_seed={seed}_fold={k}_{timestamp}"

        # scalar JSON
        json_path = os.path.join(save_dir, f"{stem}_scalar.json")
        with open(json_path, 'w') as f:
            json.dump(scalar_record, f, indent=2, cls=NumpyEncoder)

        # raw per-column R² and RMSE as npz — arrays keyed as e.g. 'PCP_r2', 'PCP_rmse'
        npz_path = os.path.join(save_dir, f"{stem}_raw.npz")
        np.savez(npz_path,
                 **{f'{var}_r2':   raw_r2_arrays[var]   for var in VAR_NAMES},
                 **{f'{var}_rmse': raw_rmse_arrays[var]  for var in VAR_NAMES})
        print(f"  Saved scalar : {json_path}")
        print(f"  Saved raw    : {npz_path}")

# ── Summary across seeds ──────────────────────────────────────────────────────
# Average folds within each seed → per-seed mean; then mean ± std across seeds.

df = pd.DataFrame(all_scalar_results)
scalar_cols = [c for c in df.columns if c not in ('seed', 'fold')]

seed_means = df.groupby('seed')[scalar_cols].mean()
summary_mean = seed_means.mean()
summary_std  = seed_means.std(ddof=1)

print(f"\n{'='*60}")
print(f"Summary: mean ± std across {N_SEEDS} seeds (each seed = mean of {folds} folds)")
print('='*60)
for col in scalar_cols:
    print(f"  {col}: {summary_mean[col]:+.4f} ± {summary_std[col]:.4f}")

# Save full per-fold scalar CSV
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
csv_path = os.path.join(save_dir, f"GP_raw_r2_scalar_summary_{timestamp}.csv")
df.to_csv(csv_path, index=False)
print(f"\nPer-fold scalar results saved to {csv_path}")

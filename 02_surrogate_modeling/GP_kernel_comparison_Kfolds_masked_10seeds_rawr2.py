"""
GP kernel comparison — KFolds cross-validation workflow.
Loads pre-saved ZRG data from pickle files.
Trains a GP surrogate with each of several kernel choices using K-fold CV.

Runs N_SEEDS independent KFold shuffles.  For each seed the 5-fold CV produces
a per-seed mean metric.  The final summary reports mean ± std across those
per-seed means for every kernel, making it easy to compare kernel performance.

Kernels evaluated
-----------------
  RBF
  Matern12      — Matern ν=1/2
  Matern32      — Matern ν=3/2
  Matern52      — Matern ν=5/2
  RatQuad       — Rational Quadratic
  Linear
  RBF+Linear   
  Matern52+Lin  
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import json
import pickle
import pandas as pd
import numpy as np
import gpflow

from esem import gp_model

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn import preprocessing

from datetime import datetime


# ── User configuration ────────────────────────────────────────────────────────

N_SEEDS   = 10   # number of independent KFold shuffles
BASE_SEED = 0    # seeds will be BASE_SEED … BASE_SEED+N_SEEDS-1
folds     = 5


# ── Kernel definitions ────────────────────────────────────────────────────────
# Each entry maps a short name to a callable that returns a fresh kernel
# instance (a new object is needed for every model fit).

def make_kernels():
    return {
        'RBF':          lambda: gpflow.kernels.SquaredExponential(),
        'Matern12':     lambda: gpflow.kernels.Matern12(),
        'Matern32':     lambda: gpflow.kernels.Matern32(),
        'Matern52':     lambda: gpflow.kernels.Matern52(),
        'RatQuad':      lambda: gpflow.kernels.RationalQuadratic(),
        'Linear':       lambda: gpflow.kernels.Linear(),
        'RBF+Linear':   lambda: gpflow.kernels.SquaredExponential() + gpflow.kernels.Linear(),
        'Matern52+Lin': lambda: gpflow.kernels.Matern52() + gpflow.kernels.Linear(),
    }


# ── JSON encoder that handles numpy scalar types ──────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Helpers ───────────────────────────────────────────────────────────────────

def variance_inverse_transform(scaler, v_norm):
    """
    Correctly invert a StandardScaler on variance values.
    inverse_transform(v) gives v*std + mean, which is wrong for variances.
    The correct inverse is: v_physical = v_normalized * std**2
    """
    return v_norm * (scaler.scale_ ** 2)


def compute_metrics(y_true_norm, y_pred_norm, y_true_phys, y_pred_phys, label):
    """Compute and print R² and RMSE in both normalized and physical space."""
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

obs_filename     = '/global/cfs/cdirs/e3sm/jpaige3/ESEm/GP_Saved_Model_Data/obs_2026-03-23_12-24-55.pkl'
GP_proj_filename = '/global/cfs/cdirs/e3sm/jpaige3/ESEm/GP_Saved_Model_Data/GP_ZRG_masked_proj_2026-03-23_12-24-55.pkl'
save_dir         = "/global/cfs/cdirs/e3sm/jpaige3/ESEm/CV_Saved_Model_Data_masked/10seed_GP_kernelselect_raw"

# ── Load pre-saved observations ───────────────────────────────────────────────

with open(obs_filename, 'rb') as f:
    loaded_obs = pickle.load(f)

zrg_obs      = loaded_obs['zrg_obs']
PCP_zrg_obs  = loaded_obs['PCP_zrg_obs']
TLWP_zrg_obs = loaded_obs['TLWP_zrg_obs']
OSR_zrg_obs  = loaded_obs['OSR_zrg_obs']
OLR_zrg_obs  = loaded_obs['OLR_zrg_obs']
n_cols_per_df = zrg_obs.shape[1] // 4

# ── Load pre-saved GP projections (contains training data) ────────────────────

with open(GP_proj_filename, 'rb') as f:
    loaded = pickle.load(f)

ppe_params          = loaded['X_train']
PCP_zrg_ppedataset  = loaded['PCP_train']
TLWP_zrg_ppedataset = loaded['TLWP_train']
OSR_zrg_ppedataset  = loaded['OSR_train']
OLR_zrg_ppedataset  = loaded['OLR_train']
zrg_ppedataset = pd.concat(
    [PCP_zrg_ppedataset, TLWP_zrg_ppedataset, OSR_zrg_ppedataset, OLR_zrg_ppedataset], axis=1
)

sim_names = list(ppe_params.index)
assert list(zrg_ppedataset.index) == sim_names
print(f"Loaded {len(sim_names)} runs, zrg_ppedataset shape: {zrg_ppedataset.shape}")

# ── Parameter ranges and bounds ───────────────────────────────────────────────

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

# ── K-fold cross-validation across multiple seeds and kernels ─────────────────
# all_results[kernel_name] is a flat list of per-fold dicts, each with a 'seed' key.

KERNELS = make_kernels()
all_results = {k: [] for k in KERNELS}

for kernel_name, kernel_factory in KERNELS.items():
    print(f"\n{'*'*60}")
    print(f"KERNEL: {kernel_name}")
    print('*'*60)

    for seed_offset in range(N_SEEDS):
        seed = BASE_SEED + seed_offset
        kf   = KFold(n_splits=folds, shuffle=True, random_state=seed)

        print(f"\n{'#'*60}")
        print(f"  Seed {seed_offset + 1}/{N_SEEDS}  (random_state={seed})")
        print('#'*60)

        for k, (train_index, test_index) in enumerate(kf.split(ppe_params)):
            print(f"\n{'='*60}")
            print(f"  Kernel={kernel_name}  Seed {seed_offset + 1}/{N_SEEDS}  Fold {k + 1}/{folds}")
            print('='*60)

            X_train = ppe_params.iloc[train_index]
            X_test  = ppe_params.iloc[test_index]
            train_run_labels = X_train.index.to_list()
            test_run_labels  = X_test.index.to_list()
            print("Test runs:", test_run_labels)

            # Per-variable train/test splits
            PCP_train  = PCP_zrg_ppedataset.loc[train_run_labels]
            TLWP_train = TLWP_zrg_ppedataset.loc[train_run_labels]
            OSR_train  = OSR_zrg_ppedataset.loc[train_run_labels]
            OLR_train  = OLR_zrg_ppedataset.loc[train_run_labels]

            PCP_test  = PCP_zrg_ppedataset.loc[test_run_labels]
            TLWP_test = TLWP_zrg_ppedataset.loc[test_run_labels]
            OSR_test  = OSR_zrg_ppedataset.loc[test_run_labels]
            OLR_test  = OLR_zrg_ppedataset.loc[test_run_labels]

            # Ensure column names are strings (required by some scalers/models)
            for df in [PCP_train, TLWP_train, OSR_train, OLR_train,
                       PCP_test,  TLWP_test,  OSR_test,  OLR_test]:
                df.columns = df.columns.astype(str)

            # ── Normalisation ─────────────────────────────────────────────────

            X_pipe_sk_minmax = preprocessing.MinMaxScaler()
            X_pipe_sk_minmax.fit(param_bounds.T)   # fit on [lower, upper] rows
            X_train_norm = X_pipe_sk_minmax.transform(X_train)
            X_test_norm  = X_pipe_sk_minmax.transform(X_test)

            scalers   = {}
            norm_data = {}
            for name, train_df, test_df in [
                ('PCP',  PCP_train,  PCP_test),
                ('TLWP', TLWP_train, TLWP_test),
                ('OSR',  OSR_train,  OSR_test),
                ('OLR',  OLR_train,  OLR_test),
            ]:
                scaler = preprocessing.StandardScaler()
                scaler.fit(train_df)
                scalers[name] = scaler
                norm_data[f'{name}_train_norm'] = scaler.transform(train_df)
                norm_data[f'{name}_test_norm']  = scaler.transform(test_df)

            Y_train_norm = np.transpose(
                np.stack([norm_data['PCP_train_norm'], norm_data['TLWP_train_norm'],
                          norm_data['OSR_train_norm'], norm_data['OLR_train_norm']], axis=0), (1, 2, 0)
            )
            Y_test_norm = np.transpose(
                np.stack([norm_data['PCP_test_norm'], norm_data['TLWP_test_norm'],
                          norm_data['OSR_test_norm'], norm_data['OLR_test_norm']], axis=0), (1, 2, 0)
            )
            print("X_train_norm:", X_train_norm.shape, "Y_train_norm:", Y_train_norm.shape)

            # ── Helper: evaluate GP on train and test sets ────────────────────

            def evaluate_gp(kernel_label, m_train, m_test):
                results = {'kernel': kernel_label, 'seed': seed, 'fold': k}
                var_names      = ['PCP', 'TLWP', 'OSR', 'OLR']
                var_trains     = [PCP_train,  TLWP_train,  OSR_train,  OLR_train]
                var_tests      = [PCP_test,   TLWP_test,   OSR_test,   OLR_test]
                var_train_norms = [norm_data[f'{n}_train_norm'] for n in var_names]
                var_test_norms  = [norm_data[f'{n}_test_norm']  for n in var_names]

                print(f"\n  [GP-{kernel_label}] In-sample metrics (normalized | physical):")
                for i, vname in enumerate(var_names):
                    scaler = scalers[vname]
                    pred_train_norm = pd.DataFrame(m_train[:, :, i], index=X_train.index)
                    pred_train_phys = pd.DataFrame(scaler.inverse_transform(pred_train_norm))
                    r2_n, rmse_n, r2_p, rmse_p, r2_n_raw, r2_p_raw = compute_metrics(
                        var_train_norms[i], pred_train_norm,
                        var_trains[i], pred_train_phys, vname + "(train)"
                    )
                    results[f'{vname}_train_r2_norm']      = r2_n
                    results[f'{vname}_train_rmse_norm']    = rmse_n
                    results[f'{vname}_train_r2_phys']      = r2_p
                    results[f'{vname}_train_rmse_phys']    = rmse_p
                    results[f'{vname}_train_r2_norm_raw']  = r2_n_raw
                    results[f'{vname}_train_r2_phys_raw']  = r2_p_raw

                print(f"\n  [GP-{kernel_label}] Out-of-sample metrics (normalized | physical):")
                for i, vname in enumerate(var_names):
                    scaler = scalers[vname]
                    pred_test_norm = pd.DataFrame(m_test[:, :, i], index=X_test.index)
                    pred_test_phys = pd.DataFrame(scaler.inverse_transform(pred_test_norm))
                    r2_n, rmse_n, r2_p, rmse_p, r2_n_raw, r2_p_raw = compute_metrics(
                        var_test_norms[i], pred_test_norm,
                        var_tests[i], pred_test_phys, vname + "(test)"
                    )
                    results[f'{vname}_test_r2_norm']      = r2_n
                    results[f'{vname}_test_rmse_norm']    = rmse_n
                    results[f'{vname}_test_r2_phys']      = r2_p
                    results[f'{vname}_test_rmse_phys']    = rmse_p
                    results[f'{vname}_test_r2_norm_raw']  = r2_n_raw
                    results[f'{vname}_test_r2_phys_raw']  = r2_p_raw

                return results

            # ── Train GP with selected kernel ─────────────────────────────────

            print(f"\nTraining GP [{kernel_name}]...")
            kernel_instance = kernel_factory()
            model_gp = gp_model(X_train_norm, Y_train_norm, kernel=kernel_instance)
            model_gp.train()

            m_gp_train, _v_gp_train = model_gp.predict(X_train_norm)
            m_gp_test,  _v_gp_test  = model_gp.predict(X_test_norm)

            gp_results = evaluate_gp(kernel_name, m_gp_train, m_gp_test)
            all_results[kernel_name].append(gp_results)

            # ── Save per-fold results ─────────────────────────────────────────

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_dir   = os.path.join(save_dir, kernel_name)
            os.makedirs(out_dir, exist_ok=True)
            result_file = os.path.join(
                out_dir,
                f"GP_{kernel_name}_ZRG_r2output_uniform_avg_seed={seed}_k={k}_{timestamp}.json"
            )
            with open(result_file, 'w') as f:
                json.dump(
                    {'timestamp': timestamp, 'model': 'Gaussian Process',
                     'kernel': kernel_name, 'seed': seed, 'fold': k,
                     'metrics': gp_results},
                    f, indent=2, cls=NumpyEncoder
                )
            print("Saved results to", result_file)


# ── Summary across all seeds and kernels ──────────────────────────────────────
# For each kernel: average folds within each seed → per-seed mean.
# Then report mean ± std across the N_SEEDS per-seed means.

print("\n" + "="*60)
print(f"GP Kernel comparison summary  ({N_SEEDS} seeds × {folds} folds)")
print(f"Values: mean ± std across per-seed means")
print("="*60)

for kernel_name, fold_list in all_results.items():
    print(f"\nKernel: {kernel_name}")
    if not fold_list:
        continue
    metric_keys = [mk for mk in fold_list[0] if mk not in ('kernel', 'seed', 'fold')]
    all_seeds   = sorted(set(r['seed'] for r in fold_list))

    # Compute per-seed mean for each metric
    seed_means = {mk: [] for mk in metric_keys}
    for s in all_seeds:
        seed_folds = [r for r in fold_list if r['seed'] == s]
        for mk in metric_keys:
            vals = [r[mk] for r in seed_folds if mk in r]
            if vals:
                seed_means[mk].append(np.mean(vals))

    for mk in metric_keys:
        means = seed_means[mk]
        if means:
            print(f"  {mk}: mean={np.mean(means):.4f}  std={np.std(means, ddof=1):.4f}")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
summary_file = os.path.join(save_dir, f"GP_kernel_comparison_summary_{timestamp}.json")
os.makedirs(save_dir, exist_ok=True)
with open(summary_file, 'w') as f:
    json.dump(all_results, f, indent=2, cls=NumpyEncoder)
print("\nSaved kernel comparison summary to", summary_file)

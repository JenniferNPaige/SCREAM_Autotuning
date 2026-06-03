"""
KFolds surrogate training workflow — loads pre-saved ZRG data from pickle files.
Trains GP, CNN, and RF surrogate models with K-fold cross-validation.

Runs N_SEEDS independent KFold shuffles.  For each seed the 5-fold CV produces
a per-seed mean metric.  The final summary reports mean ± std across those
per-seed means, giving a robust estimate of model performance that is not
sensitive to a particular fold assignment.
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import json
import pickle
import pandas as pd
import numpy as np

from esem import gp_model, cnn_model, rf_model

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline

from datetime import datetime


# ── User configuration ────────────────────────────────────────────────────────

N_SEEDS  = 10   # number of independent KFold shuffles
BASE_SEED = 0   # seeds will be BASE_SEED … BASE_SEED+N_SEEDS-1
folds    = 5


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
    r2_norm = r2_score(y_true_norm, y_pred_norm, multioutput='variance_weighted')
    rmse_norm = root_mean_squared_error(y_true_norm, y_pred_norm)
    r2_phys = r2_score(y_true_phys, y_pred_phys, multioutput='variance_weighted')
    rmse_phys = root_mean_squared_error(y_true_phys, y_pred_phys)
    print(f"  {label}: R²(norm)={r2_norm:.4f}  RMSE(norm)={rmse_norm:.4f} | "
          f"R²(phys)={r2_phys:.4f}  RMSE(phys)={rmse_phys:.4f}")
    return r2_norm, rmse_norm, r2_phys, rmse_phys


# ── Paths ─────────────────────────────────────────────────────────────────────

obs_filename = '/global/cfs/cdirs/e3sm/jpaige3/ESEm/GP_Saved_Model_Data/obs_2026-03-23_12-24-55.pkl'
GP_proj_filename = '/global/cfs/cdirs/e3sm/jpaige3/ESEm/GP_Saved_Model_Data/GP_ZRG_masked_proj_2026-03-23_12-24-55.pkl'
save_dir = "/global/cfs/cdirs/e3sm/jpaige3/ESEm/CV_Saved_Model_Data_masked"

# ── Load pre-saved observations ───────────────────────────────────────────────

with open(obs_filename, 'rb') as f:
    loaded_obs = pickle.load(f)

zrg_obs = loaded_obs['zrg_obs']
PCP_zrg_obs = loaded_obs['PCP_zrg_obs']
TLWP_zrg_obs = loaded_obs['TLWP_zrg_obs']
OSR_zrg_obs = loaded_obs['OSR_zrg_obs']
OLR_zrg_obs = loaded_obs['OLR_zrg_obs']
n_cols_per_df = zrg_obs.shape[1] // 4

# ── Load pre-saved GP projections (contains training data) ────────────────────

with open(GP_proj_filename, 'rb') as f:
    loaded = pickle.load(f)

ppe_params = loaded['X_train']
PCP_zrg_ppedataset = loaded['PCP_train']
TLWP_zrg_ppedataset = loaded['TLWP_train']
OSR_zrg_ppedataset = loaded['OSR_train']
OLR_zrg_ppedataset = loaded['OLR_train']
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

# ── K-fold cross-validation across multiple seeds ────────────────────────────

# all_results[model] is a flat list of per-fold dicts, each with a 'seed' key.
all_results = {'GP': [], 'CNN': [], 'RF': [], 'MLR': [], 'Spline': []}

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
        X_test = ppe_params.iloc[test_index]
        train_run_labels = X_train.index.to_list()
        test_run_labels = X_test.index.to_list()
        print("Test runs:", test_run_labels)

        # Per-variable train/test splits
        PCP_train = PCP_zrg_ppedataset.loc[train_run_labels]
        TLWP_train = TLWP_zrg_ppedataset.loc[train_run_labels]
        OSR_train = OSR_zrg_ppedataset.loc[train_run_labels]
        OLR_train = OLR_zrg_ppedataset.loc[train_run_labels]

        PCP_test = PCP_zrg_ppedataset.loc[test_run_labels]
        TLWP_test = TLWP_zrg_ppedataset.loc[test_run_labels]
        OSR_test = OSR_zrg_ppedataset.loc[test_run_labels]
        OLR_test = OLR_zrg_ppedataset.loc[test_run_labels]

        # Ensure column names are strings (required by some scalers/models)
        for df in [PCP_train, TLWP_train, OSR_train, OLR_train,
                   PCP_test, TLWP_test, OSR_test, OLR_test]:
            df.columns = df.columns.astype(str)

        # Stack for RF (no preprocessing)
        Y_train_ZRG = np.transpose(
            np.stack([PCP_train, TLWP_train, OSR_train, OLR_train], axis=0), (1, 2, 0)
        )
        Y_test_ZRG = np.transpose(
            np.stack([PCP_test, TLWP_test, OSR_test, OLR_test], axis=0), (1, 2, 0)
        )
        print("Y_train_ZRG shape:", Y_train_ZRG.shape)
        print("Y_test_ZRG shape:", Y_test_ZRG.shape)

        # ── Normalisation ─────────────────────────────────────────────────────

        X_pipe_sk_minmax = preprocessing.MinMaxScaler()
        X_pipe_sk_minmax.fit(param_bounds.T)  # fit on [lower, upper] rows
        X_train_norm = X_pipe_sk_minmax.transform(X_train)
        X_test_norm = X_pipe_sk_minmax.transform(X_test)

        scalers = {}
        norm_data = {}
        for name, train_df, test_df in [
            ('PCP', PCP_train, PCP_test),
            ('TLWP', TLWP_train, TLWP_test),
            ('OSR', OSR_train, OSR_test),
            ('OLR', OLR_train, OLR_test),
        ]:
            scaler = preprocessing.StandardScaler()
            scaler.fit(train_df)
            scalers[name] = scaler
            norm_data[f'{name}_train_norm'] = scaler.transform(train_df)
            norm_data[f'{name}_test_norm'] = scaler.transform(test_df)

        Y_train_norm = np.transpose(
            np.stack([norm_data['PCP_train_norm'], norm_data['TLWP_train_norm'],
                      norm_data['OSR_train_norm'], norm_data['OLR_train_norm']], axis=0), (1, 2, 0)
        )
        Y_test_norm = np.transpose(
            np.stack([norm_data['PCP_test_norm'], norm_data['TLWP_test_norm'],
                      norm_data['OSR_test_norm'], norm_data['OLR_test_norm']], axis=0), (1, 2, 0)
        )
        print("X_train_norm:", X_train_norm.shape, "Y_train_norm:", Y_train_norm.shape)

        # ── Save fold preprocessing data ──────────────────────────────────────

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fold_filename = os.path.join(
            save_dir, f"Fold_{k}_seed_{seed}_ZRG_masked_data_{timestamp}.pkl"
        )
        with open(fold_filename, 'wb') as f:
            pickle.dump({
                'seed': seed,
                'X_pipeline': X_pipe_sk_minmax,
                'Y_pipeline_PCP': scalers['PCP'],
                'Y_pipeline_TLWP': scalers['TLWP'],
                'Y_pipeline_OSR': scalers['OSR'],
                'Y_pipeline_OLR': scalers['OLR'],
                'X_train_index': train_run_labels,
                'X_test_index': test_run_labels,
                'X_train': X_train, 'X_test': X_test,
                'Y_train': Y_train_ZRG, 'Y_test': Y_test_ZRG,
                'PCP_train': PCP_train, 'TLWP_train': TLWP_train,
                'OSR_train': OSR_train, 'OLR_train': OLR_train,
                'PCP_test': PCP_test, 'TLWP_test': TLWP_test,
                'OSR_test': OSR_test, 'OLR_test': OLR_test,
                'X_train_norm': X_train_norm, 'X_test_norm': X_test_norm,
                'Y_train_norm': Y_train_norm, 'Y_test_norm': Y_test_norm,
                **{f'{n}_train_norm': norm_data[f'{n}_train_norm'] for n in ['PCP', 'TLWP', 'OSR', 'OLR']},
                **{f'{n}_test_norm': norm_data[f'{n}_test_norm'] for n in ['PCP', 'TLWP', 'OSR', 'OLR']},
            }, f)
        print("Saved fold data to", fold_filename)

        # ── Helper: evaluate one model on train and test sets ─────────────────

        def evaluate_model(model_name, m_train, _v_train, m_test, _v_test):
            results = {'model': model_name, 'seed': seed, 'fold': k}
            var_names = ['PCP', 'TLWP', 'OSR', 'OLR']
            var_trains = [PCP_train, TLWP_train, OSR_train, OLR_train]
            var_tests = [PCP_test, TLWP_test, OSR_test, OLR_test]
            var_train_norms = [norm_data[f'{n}_train_norm'] for n in var_names]
            var_test_norms = [norm_data[f'{n}_test_norm'] for n in var_names]

            print(f"\n  [{model_name}] In-sample metrics (normalized | physical):")
            for i, name in enumerate(var_names):
                scaler = scalers[name]
                pred_train_norm = pd.DataFrame(m_train[:, :, i], index=X_train.index)
                pred_train_phys = pd.DataFrame(scaler.inverse_transform(pred_train_norm))
                r2_n, rmse_n, r2_p, rmse_p = compute_metrics(
                    var_train_norms[i], pred_train_norm,
                    var_trains[i], pred_train_phys, name + "(train)"
                )
                results[f'{name}_train_r2_norm'] = r2_n
                results[f'{name}_train_rmse_norm'] = rmse_n
                results[f'{name}_train_r2_phys'] = r2_p
                results[f'{name}_train_rmse_phys'] = rmse_p

            print(f"\n  [{model_name}] Out-of-sample metrics (normalized | physical):")
            for i, name in enumerate(var_names):
                scaler = scalers[name]
                pred_test_norm = pd.DataFrame(m_test[:, :, i], index=X_test.index)
                pred_test_phys = pd.DataFrame(scaler.inverse_transform(pred_test_norm))
                r2_n, rmse_n, r2_p, rmse_p = compute_metrics(
                    var_test_norms[i], pred_test_norm,
                    var_tests[i], pred_test_phys, name + "(test)"
                )
                results[f'{name}_test_r2_norm'] = r2_n
                results[f'{name}_test_rmse_norm'] = rmse_n
                results[f'{name}_test_r2_phys'] = r2_p
                results[f'{name}_test_rmse_phys'] = rmse_p

            return results

        # ── Gaussian Process ──────────────────────────────────────────────────

        print("\nTraining GP...")
        model_gp = gp_model(X_train_norm, Y_train_norm)
        model_gp.train()

        m_gp_train, v_gp_train = model_gp.predict(X_train_norm)
        m_gp_test, v_gp_test = model_gp.predict(X_test_norm)

        gp_results = evaluate_model('GP', m_gp_train, v_gp_train, m_gp_test, v_gp_test)
        all_results['GP'].append(gp_results)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        gp_r2_file = os.path.join(
            save_dir, 'GP',
            f"GP_ZRG_r2output_varweight_seed={seed}_k={k}_{timestamp}.json"
        )
        os.makedirs(os.path.dirname(gp_r2_file), exist_ok=True)
        with open(gp_r2_file, 'w') as f:
            json.dump({'timestamp': timestamp, 'model': 'Gaussian Process',
                       'seed': seed, 'fold': k, 'metrics': gp_results},
                      f, indent=2, cls=NumpyEncoder)

        # ── CNN ───────────────────────────────────────────────────────────────

        print("\nTraining CNN...")
        model_cnn = cnn_model(X_train_norm, Y_train_norm)
        model_cnn.train()

        m_cnn_train, v_cnn_train = model_cnn.predict(X_train_norm)
        m_cnn_test, v_cnn_test = model_cnn.predict(X_test_norm)

        cnn_results = evaluate_model('CNN', m_cnn_train, v_cnn_train, m_cnn_test, v_cnn_test)
        all_results['CNN'].append(cnn_results)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cnn_r2_file = os.path.join(
            save_dir, 'CNN',
            f"CNN_ZRG_r2output_varweight_seed={seed}_k={k}_{timestamp}.json"
        )
        os.makedirs(os.path.dirname(cnn_r2_file), exist_ok=True)
        with open(cnn_r2_file, 'w') as f:
            json.dump({'timestamp': timestamp, 'model': 'CNN',
                       'seed': seed, 'fold': k, 'metrics': cnn_results},
                      f, indent=2, cls=NumpyEncoder)

        # ── Random Forest ─────────────────────────────────────────────────────

        print("\nTraining RF...")
        # RF uses raw (un-normalised) X and the stacked Y_ZRG array
        model_rf = rf_model(X_train.to_numpy(), Y_train_ZRG)
        model_rf.train()

        m_rf_train, v_rf_train = model_rf.predict(X_train.to_numpy())
        m_rf_test, v_rf_test = model_rf.predict(X_test.to_numpy())

        # RF operates in physical space; compute metrics directly
        rf_results = {'model': 'RF', 'seed': seed, 'fold': k}
        var_names = ['PCP', 'TLWP', 'OSR', 'OLR']
        var_trains = [PCP_train, TLWP_train, OSR_train, OLR_train]
        var_tests = [PCP_test, TLWP_test, OSR_test, OLR_test]

        print("\n  [RF] In-sample metrics (physical):")
        for i, name in enumerate(var_names):
            pred_train = pd.DataFrame(m_rf_train[:, :, i], index=X_train.index)
            r2_p = r2_score(var_trains[i], pred_train, multioutput='variance_weighted')
            rmse_p = root_mean_squared_error(var_trains[i], pred_train)
            print(f"  {name}(train): R²(phys)={r2_p:.4f}  RMSE(phys)={rmse_p:.4f}")
            rf_results[f'{name}_train_r2_phys'] = r2_p
            rf_results[f'{name}_train_rmse_phys'] = rmse_p

        print("\n  [RF] Out-of-sample metrics (physical):")
        for i, name in enumerate(var_names):
            pred_test = pd.DataFrame(m_rf_test[:, :, i], index=X_test.index)
            r2_p = r2_score(var_tests[i], pred_test, multioutput='variance_weighted')
            rmse_p = root_mean_squared_error(var_tests[i], pred_test)
            print(f"  {name}(test): R²(phys)={r2_p:.4f}  RMSE(phys)={rmse_p:.4f}")
            rf_results[f'{name}_test_r2_phys'] = r2_p
            rf_results[f'{name}_test_rmse_phys'] = rmse_p

        all_results['RF'].append(rf_results)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rf_r2_file = os.path.join(
            save_dir, 'RF',
            f"RF_ZRG_r2output_varweight_seed={seed}_k={k}_{timestamp}.json"
        )
        os.makedirs(os.path.dirname(rf_r2_file), exist_ok=True)
        with open(rf_r2_file, 'w') as f:
            json.dump({'timestamp': timestamp, 'model': 'Random Forest',
                       'seed': seed, 'fold': k, 'metrics': rf_results},
                      f, indent=2, cls=NumpyEncoder)

        # ── Helper: fit a sklearn multi-output model and return esem-style arrays
        # Returns m_train, m_test with shape (n_samples, n_spatial_cols, n_vars)
        # matching the shape expected by evaluate_model. Variance is None (not used).

        def run_sklearn_model(sk_model_factory):
            """
            Trains one sklearn multi-output regressor per variable on normalised data.
            Returns (m_train, m_test) stacked into (n_samples, n_cols, n_vars).
            """
            preds_train, preds_test = [], []
            for var in ['PCP', 'TLWP', 'OSR', 'OLR']:
                y_tr = norm_data[f'{var}_train_norm']
                model_sk = sk_model_factory()
                model_sk.fit(X_train_norm, y_tr)
                preds_train.append(model_sk.predict(X_train_norm))  # (n_train, n_cols)
                preds_test.append(model_sk.predict(X_test_norm))    # (n_test,  n_cols)
            # stack to (n_vars, n_samples, n_cols) then transpose to (n_samples, n_cols, n_vars)
            m_train = np.transpose(np.stack(preds_train, axis=0), (1, 2, 0))
            m_test  = np.transpose(np.stack(preds_test,  axis=0), (1, 2, 0))
            return m_train, m_test

        # ── Multiple Linear Regression ────────────────────────────────────────

        print("\nTraining MLR...")
        m_mlr_train, m_mlr_test = run_sklearn_model(LinearRegression)

        mlr_results = evaluate_model('MLR', m_mlr_train, None, m_mlr_test, None)
        all_results['MLR'].append(mlr_results)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        mlr_r2_file = os.path.join(
            save_dir, 'MLR',
            f"MLR_ZRG_r2output_varweight_seed={seed}_k={k}_{timestamp}.json"
        )
        os.makedirs(os.path.dirname(mlr_r2_file), exist_ok=True)
        with open(mlr_r2_file, 'w') as f:
            json.dump({'timestamp': timestamp, 'model': 'MLR',
                       'seed': seed, 'fold': k, 'metrics': mlr_results},
                      f, indent=2, cls=NumpyEncoder)

        # ── Spline Regression ─────────────────────────────────────────────────

        print("\nTraining Spline...")
        # n_knots=5, degree=3 (cubic) — adjust as needed
        m_spl_train, m_spl_test = run_sklearn_model(
            lambda: make_pipeline(SplineTransformer(n_knots=3, degree=2), LinearRegression())
        )

        spl_results = evaluate_model('Spline', m_spl_train, None, m_spl_test, None)
        all_results['Spline'].append(spl_results)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        spl_r2_file = os.path.join(
            save_dir, 'Spline',
            f"Spline_ZRG_r2output_varweight_seed={seed}_k={k}_{timestamp}.json"
        )
        os.makedirs(os.path.dirname(spl_r2_file), exist_ok=True)
        with open(spl_r2_file, 'w') as f:
            json.dump({'timestamp': timestamp, 'model': 'Spline',
                       'seed': seed, 'fold': k, 'metrics': spl_results},
                      f, indent=2, cls=NumpyEncoder)

# ── Summary across all seeds ──────────────────────────────────────────────────
# For each model: average folds within each seed → per-seed mean.
# Then report mean ± std across the N_SEEDS per-seed means.

print("\n" + "="*60)
print(f"Cross-validation summary  ({N_SEEDS} seeds × {folds} folds)")
print(f"Values: mean ± std across per-seed means")
print("="*60)

for model_name, fold_list in all_results.items():
    print(f"\n{model_name}:")
    if not fold_list:
        continue
    metric_keys = [mk for mk in fold_list[0] if mk not in ('model', 'seed', 'fold')]
    all_seeds = sorted(set(r['seed'] for r in fold_list))

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
summary_file = os.path.join(save_dir, f"CV_summary_{timestamp}.json")
with open(summary_file, 'w') as f:
    json.dump(all_results, f, indent=2, cls=NumpyEncoder)
print("\nSaved CV summary to", summary_file)

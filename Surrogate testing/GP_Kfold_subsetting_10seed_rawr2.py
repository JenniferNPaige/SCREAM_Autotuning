"""
GP surrogate: out-of-sample R² vs. training subset size.

For each subset size in SUBSET_SIZES, N_SEEDS independent random subsets are
drawn from the full dataset.  Each subset is evaluated via 5-fold CV.  The
per-seed result is the mean R² across the 5 folds.  The final reported values
are the mean and standard deviation of those per-seed results, giving a robust
picture of how much performance varies with subset composition.
"""

# ── User configuration ────────────────────────────────────────────────────────

SUBSET_SIZES = [50, 75, 100, 125, 153]   # number of runs to sample per trial
N_SEEDS      = 10                          # independent subset draws per size
BASE_SEED    = 0                           # seeds will be BASE_SEED … BASE_SEED+N_SEEDS-1

obs_filename    = '/global/cfs/cdirs/e3sm/jpaige3/ESEm/GP_Saved_Model_Data/obs_2026-03-23_12-24-55.pkl'
GP_proj_filename = '/global/cfs/cdirs/e3sm/jpaige3/ESEm/GP_Saved_Model_Data/GP_ZRG_masked_proj_2026-03-23_12-24-55.pkl'
save_dir        = "/global/cfs/cdirs/e3sm/jpaige3/ESEm/CV_Saved_Model_Data_masked/10seed_GP_Subsetting_raw"

# ── Imports ───────────────────────────────────────────────────────────────────

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from esem import gp_model
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn import preprocessing
from datetime import datetime

# ── Load data ─────────────────────────────────────────────────────────────────

with open(GP_proj_filename, 'rb') as f:
    loaded = pickle.load(f)

ppe_params         = loaded['X_train']
PCP_zrg_ppedataset  = loaded['PCP_train']
TLWP_zrg_ppedataset = loaded['TLWP_train']
OSR_zrg_ppedataset  = loaded['OSR_train']
OLR_zrg_ppedataset  = loaded['OLR_train']

sim_names = list(ppe_params.index)
n_total   = len(sim_names)
print(f"Full dataset: {n_total} runs")

# ── Parameter bounds (MinMaxScaler fitted on full bounds, not data) ───────────

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

# ── Validate requested subset sizes ──────────────────────────────────────────

for s in SUBSET_SIZES:
    if s > n_total:
        raise ValueError(
            f"Subset size {s} exceeds total number of runs ({n_total})."
        )
    if s < 5:
        raise ValueError(
            f"Subset size {s} is too small for 5-fold cross-validation (need ≥ 5)."
        )

# ── Cross-validation loop ─────────────────────────────────────────────────────

VAR_NAMES = ['PCP', 'TLWP', 'OSR', 'OLR']
var_datasets = {
    'PCP':  PCP_zrg_ppedataset,
    'TLWP': TLWP_zrg_ppedataset,
    'OSR':  OSR_zrg_ppedataset,
    'OLR':  OLR_zrg_ppedataset,
}

# seed_means[size][var] = list of N_SEEDS per-seed mean R² values
# (each per-seed mean is itself the mean of 5 CV folds)
seed_means         = {size: {v: [] for v in VAR_NAMES} for size in SUBSET_SIZES}
seed_means_raw     = {size: {v: [] for v in VAR_NAMES} for size in SUBSET_SIZES}
seed_means_phys    = {size: {v: [] for v in VAR_NAMES} for size in SUBSET_SIZES}
seed_means_raw_phys = {size: {v: [] for v in VAR_NAMES} for size in SUBSET_SIZES}

# full per-fold records for the saved CSV
records = []

for size in SUBSET_SIZES:
    print(f"\n{'='*60}")
    print(f"Subset size: {size}/{n_total}  ({N_SEEDS} seeds × 5 folds)")
    print('='*60)

    for seed_offset in range(N_SEEDS):
        seed = BASE_SEED + seed_offset
        rng  = np.random.default_rng(seed)

        # Random subset of run indices (without replacement)
        subset_idx    = np.sort(rng.choice(n_total, size=size, replace=False))
        subset_params = ppe_params.iloc[subset_idx]
        subset_labels = subset_params.index.to_list()

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)

        fold_r2          = {var: [] for var in VAR_NAMES}
        fold_r2_raw      = {var: [] for var in VAR_NAMES}
        fold_r2_phys     = {var: [] for var in VAR_NAMES}
        fold_r2_raw_phys = {var: [] for var in VAR_NAMES}

        for fold, (train_rel, test_rel) in enumerate(kf.split(subset_params)):
            train_labels = [subset_labels[i] for i in train_rel]
            test_labels  = [subset_labels[i] for i in test_rel]

            X_train = subset_params.iloc[train_rel]
            X_test  = subset_params.iloc[test_rel]

            X_train_norm = X_pipe_sk_minmax.transform(X_train)
            X_test_norm  = X_pipe_sk_minmax.transform(X_test)

            # Normalise each variable's Y with a StandardScaler fit on train split
            train_norm_list = []
            scalers = {}
            for var in VAR_NAMES:
                ds = var_datasets[var]
                y_train = ds.loc[train_labels].copy()
                y_train.columns = y_train.columns.astype(str)
                scaler = preprocessing.StandardScaler().fit(y_train)
                scalers[var] = scaler
                train_norm_list.append(scaler.transform(y_train))

            Y_train_norm = np.transpose(np.stack(train_norm_list, axis=0), (1, 2, 0))

            # Train GP
            print(f"  Seed {seed_offset+1}/{N_SEEDS}  Fold {fold+1}/5 — "
                  f"training GP on {len(train_labels)} runs...")
            model = gp_model(X_train_norm, Y_train_norm)
            model.train()

            # Predict on test set
            m_test, _ = model.predict(X_test_norm)

            # Compute out-of-sample R² per variable in normalized and physical space
            for i, var in enumerate(VAR_NAMES):
                ds = var_datasets[var]
                y_test = ds.loc[test_labels].copy()
                y_test.columns = y_test.columns.astype(str)
                y_test_norm  = scalers[var].transform(y_test)
                pred_norm    = m_test[:, :, i]
                pred_phys    = scalers[var].inverse_transform(pred_norm)

                r2          = r2_score(y_test_norm, pred_norm, multioutput='uniform_average')
                r2_raw      = r2_score(y_test_norm, pred_norm, multioutput='raw_values')
                r2_phys     = r2_score(y_test,      pred_phys, multioutput='uniform_average')
                r2_phys_raw = r2_score(y_test,      pred_phys, multioutput='raw_values')

                fold_r2[var].append(r2)
                fold_r2_raw[var].append(r2_raw)
                fold_r2_phys[var].append(r2_phys)
                fold_r2_raw_phys[var].append(r2_phys_raw)

                records.append({
                    'subset_size': size,
                    'seed':        seed,
                    'variable':    var,
                    'fold':        fold,
                    'r2':          r2,
                    'r2_phys':     r2_phys,
                })
                print(f"    {var}: R²(norm)={r2:.4f}  R²(phys)={r2_phys:.4f}")

        # Record the mean-across-folds for this seed
        for var in VAR_NAMES:
            seed_means[size][var].append(np.mean(fold_r2[var]))
            seed_means_raw[size][var].append(np.mean(fold_r2_raw[var], axis=0))
            seed_means_phys[size][var].append(np.mean(fold_r2_phys[var]))
            seed_means_raw_phys[size][var].append(np.mean(fold_r2_raw_phys[var], axis=0))

# ── Aggregate results ─────────────────────────────────────────────────────────

mean_r2      = {var: [] for var in VAR_NAMES}
std_r2       = {var: [] for var in VAR_NAMES}
mean_r2_phys = {var: [] for var in VAR_NAMES}
std_r2_phys  = {var: [] for var in VAR_NAMES}

for size in SUBSET_SIZES:
    for var in VAR_NAMES:
        vals      = seed_means[size][var]
        vals_phys = seed_means_phys[size][var]
        mean_r2[var].append(np.mean(vals))
        std_r2[var].append(np.std(vals, ddof=1))
        mean_r2_phys[var].append(np.mean(vals_phys))
        std_r2_phys[var].append(np.std(vals_phys, ddof=1))

# ── Save results ─────────────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(save_dir, exist_ok=True)

# Full per-fold R² values
df_results = pd.DataFrame(records)
csv_path = os.path.join(save_dir, f"GP_subset_scaling_r2_{timestamp}.csv")
df_results.to_csv(csv_path, index=False)
print(f"Per-fold results saved to {csv_path}")

# Aggregated summary (mean ± std across seeds) as JSON
summary = []
for j, size in enumerate(SUBSET_SIZES):
    for var in VAR_NAMES:
        summary.append({
            'subset_size':          size,
            'variable':             var,
            'n_seeds':              N_SEEDS,
            'mean_r2':              mean_r2[var][j],
            'std_r2':               std_r2[var][j],
            'seed_means':           seed_means[size][var],
            'seed_means_raw':       [arr.tolist() for arr in seed_means_raw[size][var]],
            'mean_r2_phys':         mean_r2_phys[var][j],
            'std_r2_phys':          std_r2_phys[var][j],
            'seed_means_phys':      seed_means_phys[size][var],
            'seed_means_raw_phys':  [arr.tolist() for arr in seed_means_raw_phys[size][var]],
        })

json_path = os.path.join(save_dir, f"GP_subset_scaling_r2_{timestamp}.json")
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Aggregated summary saved to {json_path}")

# ── Print summary ─────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Summary: mean ± std R² across {N_SEEDS} seeds (each seed = mean of 5 CV folds)")
print('='*60)
header = f"{'Size':>6}" + "".join(f"  {v:>18}" for v in VAR_NAMES)
print(header)
for j, size in enumerate(SUBSET_SIZES):
    row = f"{size:>6}"
    for var in VAR_NAMES:
        row += f"  {mean_r2[var][j]:+.4f} ± {std_r2[var][j]:.4f}"
    print(row)

# ── Plot ──────────────────────────────────────────────────────────────────────

colors = {'PCP': '#1f77b4', 'TLWP': '#ff7f0e', 'OSR': '#2ca02c', 'OLR': '#d62728'}

fig, ax = plt.subplots(figsize=(8, 6))

for var in VAR_NAMES:
    mu  = np.array(mean_r2[var])
    sd  = np.array(std_r2[var])
    ax.errorbar(SUBSET_SIZES, mu, yerr=sd,
                marker='o', color=colors[var], linewidth=2,
                capsize=4, capthick=1.5, label=var)

ax.set_xlabel('Training size')
ax.set_ylabel('R²')
ax.set_xticks(SUBSET_SIZES)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title(f'GP surrogate: out-of-sample R² vs. training subset size\n'
             f'(mean ± 1 std across {N_SEEDS} seeds, each seed = mean of 5 CV folds)')

plt.tight_layout()

plot_path = os.path.join(save_dir, f"GP_subset_scaling_r2_{timestamp}.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {plot_path}")
plt.show()

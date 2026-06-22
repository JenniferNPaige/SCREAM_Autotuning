"""
Spline regression hyperparameter search via K-fold cross-validation.
Evaluates several (n_knots, degree) combinations and plots out-of-sample R².
"""

# ── User configuration ────────────────────────────────────────────────────────

# (n_knots, degree) pairs to evaluate — degree must be < n_knots
KNOT_DEGREE_COMBOS = [
    (2, 1),
    (3, 2),
    (4, 2),
    (4, 3),
    (5, 3),
    (5, 4),
    (6, 3),
    (8, 3),
    (8, 4),
]

N_SEEDS    = 10
BASE_SEED  = 0
N_FOLDS    = 5

# ── Imports ───────────────────────────────────────────────────────────────────

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import paths

# ── Paths (see paths.py / DATA.md for the expected data layout) ──────────────

GP_proj_filename = str(paths.GP_PROJ_PICKLE)
save_dir         = str(paths.CV_RESULTS_DIR)
os.makedirs(save_dir, exist_ok=True)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn import preprocessing
from datetime import datetime

# ── Validate combos ───────────────────────────────────────────────────────────

for n_knots, degree in KNOT_DEGREE_COMBOS:
    if degree >= n_knots:
        raise ValueError(
            f"Invalid combo (n_knots={n_knots}, degree={degree}): "
            f"degree must be less than n_knots for SplineTransformer."
        )

# ── Load data ─────────────────────────────────────────────────────────────────

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
X_scaler = preprocessing.MinMaxScaler()
X_scaler.fit(param_bounds.T)

# ── K-fold cross-validation loop ──────────────────────────────────────────────

# results[(n_knots, degree)][var] = list of per-fold out-of-sample R²
results     = {combo: {v: [] for v in VAR_NAMES} for combo in KNOT_DEGREE_COMBOS}
results_raw = {combo: {v: [] for v in VAR_NAMES} for combo in KNOT_DEGREE_COMBOS}

for combo in KNOT_DEGREE_COMBOS:
    n_knots, degree = combo
    print(f"\n{'='*60}")
    print(f"n_knots={n_knots}, degree={degree}")
    print('='*60)

    for seed_offset in range(N_SEEDS):
        seed = BASE_SEED + seed_offset
        kf   = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        print(f"\n  Seed {seed_offset + 1}/{N_SEEDS}  (random_state={seed})")

        for fold, (train_idx, test_idx) in enumerate(kf.split(ppe_params)):
            X_train = ppe_params.iloc[train_idx]
            X_test  = ppe_params.iloc[test_idx]
            train_labels = X_train.index.to_list()
            test_labels  = X_test.index.to_list()

            X_train_norm = X_scaler.transform(X_train)
            X_test_norm  = X_scaler.transform(X_test)

            print(f"    Fold {fold + 1}/{N_FOLDS}")
            for var in VAR_NAMES:
                ds = var_datasets[var]

                y_train = ds.loc[train_labels].copy()
                y_test  = ds.loc[test_labels].copy()
                y_train.columns = y_train.columns.astype(str)
                y_test.columns  = y_test.columns.astype(str)

                y_scaler = preprocessing.StandardScaler().fit(y_train)
                y_train_norm = y_scaler.transform(y_train)
                y_test_norm  = y_scaler.transform(y_test)

                model = make_pipeline(
                    SplineTransformer(n_knots=n_knots, degree=degree),
                    LinearRegression()
                )
                model.fit(X_train_norm, y_train_norm)
                pred_norm = model.predict(X_test_norm)

                r2     = r2_score(y_test_norm, pred_norm, multioutput='uniform_average')
                r2_raw = r2_score(y_test_norm, pred_norm, multioutput='raw_values')
                results[combo][var].append(r2)
                results_raw[combo][var].append(r2_raw)
                print(f"      {var}: R²={r2:.4f}")

# ── Aggregate ─────────────────────────────────────────────────────────────────
# Per-seed means → mean ± std across seeds (consistent with other scripts).
# min/max are across all individual folds (used for plot error bars).

mean_r2 = {combo: {} for combo in KNOT_DEGREE_COMBOS}
std_r2  = {combo: {} for combo in KNOT_DEGREE_COMBOS}
min_r2  = {combo: {} for combo in KNOT_DEGREE_COMBOS}
max_r2  = {combo: {} for combo in KNOT_DEGREE_COMBOS}

for combo in KNOT_DEGREE_COMBOS:
    for var in VAR_NAMES:
        vals = results[combo][var]   # N_SEEDS * N_FOLDS values
        seed_means = [np.mean(vals[s * N_FOLDS:(s + 1) * N_FOLDS]) for s in range(N_SEEDS)]
        mean_r2[combo][var] = np.mean(seed_means)
        std_r2[combo][var]  = np.std(seed_means, ddof=1)
        min_r2[combo][var]  = np.min(vals)
        max_r2[combo][var]  = np.max(vals)

# ── Print summary table ───────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"Summary: mean ± std out-of-sample R²  ({N_SEEDS} seeds × {N_FOLDS} folds)")
print('='*70)
header = f"{'(knots,deg)':>12}" + "".join(f"  {v:>26}" for v in VAR_NAMES)
print(header)
for combo in KNOT_DEGREE_COMBOS:
    row = f"{str(combo):>12}"
    for var in VAR_NAMES:
        mu = mean_r2[combo][var]
        sd = std_r2[combo][var]
        row += f"  {mu:+.4f} ± {sd:.4f}          "
    print(row)

# ── Save results to JSON ──────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output = []
for combo in KNOT_DEGREE_COMBOS:
    n_knots, degree = combo
    entry = {'n_knots': n_knots, 'degree': degree}
    for var in VAR_NAMES:
        entry[f'{var}_mean_r2']     = mean_r2[combo][var]
        entry[f'{var}_std_r2']      = std_r2[combo][var]
        entry[f'{var}_min_r2']      = min_r2[combo][var]
        entry[f'{var}_max_r2']      = max_r2[combo][var]
        entry[f'{var}_fold_r2']     = results[combo][var]
        entry[f'{var}_fold_r2_raw'] = [arr.tolist() for arr in results_raw[combo][var]]
    output.append(entry)

os.makedirs(save_dir, exist_ok=True)
json_path = os.path.join(save_dir, f"Spline_knot_degree_search_{timestamp}.json")
with open(json_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {json_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────

combo_labels = [f"k={k}\nd={d}" for k, d in KNOT_DEGREE_COMBOS]
x = np.arange(len(KNOT_DEGREE_COMBOS))

colors = {'PCP': '#1f77b4', 'TLWP': '#ff7f0e', 'OSR': '#2ca02c', 'OLR': '#d62728'}
offsets = {'PCP': -0.3, 'TLWP': -0.1, 'OSR': 0.1, 'OLR': 0.3}
width = 0.18

fig, ax = plt.subplots(figsize=(max(10, len(KNOT_DEGREE_COMBOS) * 1.2), 6))

for var in VAR_NAMES:
    mu  = np.array([mean_r2[c][var] for c in KNOT_DEGREE_COMBOS])
    lo  = mu - np.array([min_r2[c][var] for c in KNOT_DEGREE_COMBOS])
    hi  = np.array([max_r2[c][var] for c in KNOT_DEGREE_COMBOS]) - mu
    xpos = x + offsets[var]
    ax.bar(xpos, mu, width=width, color=colors[var], alpha=0.8, label=var)
    ax.errorbar(xpos, mu, yerr=[lo, hi], fmt='none',
                color='black', capsize=3, capthick=1, linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(combo_labels, fontsize=9)
ax.set_ylabel('Mean out-of-sample R² (normalized space)')
ax.set_xlabel('Spline configuration (knots, degree)')
ax.set_title(f'Spline regression: out-of-sample R² across (n_knots, degree) combinations\n'
             f'({N_SEEDS} seeds × {N_FOLDS}-fold CV, error bars show fold range)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')

plt.tight_layout()
plot_path = os.path.join(save_dir, f"Spline_knot_degree_search_{timestamp}.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to {plot_path}")
plt.show()

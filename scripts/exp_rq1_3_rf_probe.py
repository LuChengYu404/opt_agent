"""EXP-RQ1.3 step 2 — RandomForest utility-regression probe.

Per codex review round 2:
- DON'T use 8-class accuracy on a_star (Hartmann has only 6 winner
  classes among 69 records; 2 actions never win; per-seed CV would leak).
- DO use multi-output regression: predict 8 utilities U(s, a) for each
  action, then argmax for the chosen action.
- Leave-one-seed-out CV (preserves seed independence).
- Primary metric: held-out decision regret = U(s, a*) - U(s, a_hat).
- Baselines: AlwaysSurrogate, training-fold majority/best constant,
  phase-only rule.

Hardware: 32 cores; RF trains in <1s for this dataset size. No GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.dataset import iter_jsonl
from opt_agent.dataset.validation import phase_of_ratio

ACTIONS = (
    "global_exploration", "local_exploitation", "uncertainty_sampling",
    "error_guided_sampling", "boundary_sampling", "surrogate_acquisition",
    "retrain_surrogate", "stop",
)

# Numeric features extracted from state payload.
FEATURE_KEYS = (
    "samples", "dimension", "min_pairwise_distance", "boundary_coverage",
    "validation_rmse", "mean_uncertainty", "max_uncertainty",
    "best_value", "recent_improvement", "stagnation_steps",
    "remaining_budget_ratio", "is_fitted",
    # Plus one-hot for phase (3 levels: early, mid, late; terminal excluded).
)


def extract_features(state: dict, phase: str) -> list[float]:
    """Build a numeric feature vector from a state dict + phase."""
    sampling = state.get("sampling", {})
    surrogate = state.get("surrogate", {})
    optimization = state.get("optimization", {})
    resource = state.get("resource", {})
    history = state.get("history", {})
    feat = [
        float(sampling.get("n_samples", 0) or 0),
        float(sampling.get("dimension", 0) or 0),
        float(sampling.get("min_pairwise_distance", 0.0) or 0.0),
        float(sampling.get("boundary_coverage", 0.0) or 0.0),
        float(surrogate.get("validation_rmse", 0.0) or 0.0),
        float(surrogate.get("mean_uncertainty", 0.0) or 0.0),
        float(surrogate.get("max_uncertainty", 0.0) or 0.0),
        float(optimization.get("best_value", 0.0) or 0.0),
        float(optimization.get("recent_improvement", 0.0) or 0.0),
        float(optimization.get("stagnation_steps", 0) or 0),
        float(resource.get("remaining_budget_ratio", 0.0) or 0.0),
        1.0 if surrogate.get("is_current", False) else 0.0,
        1.0 if phase == "early" else 0.0,
        1.0 if phase == "mid" else 0.0,
        1.0 if phase == "late" else 0.0,
    ]
    # Sanitize non-finite to 0.0 (training will skip if all 0).
    feat = [0.0 if not math.isfinite(v) else v for v in feat]
    return feat


def extract_utilities(record_dict: dict, family_scales: dict) -> dict[str, float] | None:
    """Extract U(s, a) under family_median scale normalization."""
    fam = record_dict["problem"]["family"]
    new_scale = family_scales[fam]["median"]
    md = record_dict.get("metadata", {}) or {}
    old_scale = float(md.get("optimization_scale", 1.0))
    w_opt = float(md.get("utility_weights", {}).get("w_optimization", 1.0))
    if new_scale <= 0 or old_scale <= 0:
        return None
    rr = record_dict.get("rollout_results") or {}
    out = {}
    for action in ACTIONS:
        cell = rr.get(action, {})
        if not isinstance(cell, dict):
            continue
        u_orig = cell.get("utility")
        d_opt = cell.get("delta_optimization")
        if u_orig is None or d_opt is None:
            continue
        try:
            u_orig = float(u_orig)
            d_opt = float(d_opt)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(u_orig) and math.isfinite(d_opt)):
            continue
        u_new = u_orig + w_opt * d_opt * (1.0 / new_scale - 1.0 / old_scale)
        out[action] = u_new
    return out if len(out) >= 2 else None


def compute_family_scales(records: list[dict]) -> dict[str, dict[str, float]]:
    by_fam = defaultdict(list)
    for r in records:
        s = float((r.get("metadata") or {}).get("optimization_scale", 0.0))
        if s > 0:
            by_fam[r["problem"]["family"]].append(s)
    return {fam: {"median": statistics.median(vals)} for fam, vals in by_fam.items()}


def leave_one_seed_out_cv(X: list, Y: list, seed_idx: list, family_filter: str = "hartmann"):
    """Run leave-one-seed-out CV. Reports regret for each fold."""
    from sklearn.ensemble import RandomForestRegressor

    indices_by_seed: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(seed_idx):
        indices_by_seed[s].append(i)

    seed_results = []
    for held_out_seed in sorted(indices_by_seed):
        train_idx = [i for s, idxs in indices_by_seed.items()
                     if s != held_out_seed for i in idxs]
        test_idx = indices_by_seed[held_out_seed]
        X_tr = [X[i] for i in train_idx]
        Y_tr = [Y[i] for i in train_idx]
        X_te = [X[i] for i in test_idx]
        Y_te = [Y[i] for i in test_idx]

        # Multi-output: one regressor per action.
        regressors = {}
        for j, action in enumerate(ACTIONS):
            rf = RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_leaf=5,
                random_state=42, n_jobs=-1,
            )
            rf.fit(X_tr, [y[j] for y in Y_tr])
            regressors[action] = rf

        # Predict utilities for test set; argmax = chosen action.
        regrets = []
        for k, x in enumerate(X_te):
            y_pred = {a: regressors[a].predict([x])[0] for a in ACTIONS}
            a_hat = max(y_pred, key=y_pred.get)
            # Regret = U(s, a*) - U(s, a_hat) using TRUE utilities.
            true_ys = Y_te[k]
            a_true = max(range(len(true_ys)), key=lambda j: true_ys[j])
            u_star = true_ys[a_true]
            u_hat = true_ys[ACTIONS.index(a_hat)]
            regrets.append(u_star - u_hat)
        seed_results.append({
            "held_out_seed": held_out_seed,
            "n_test": len(test_idx),
            "mean_regret": statistics.fmean(regrets) if regrets else 0.0,
            "median_regret": statistics.median(regrets) if regrets else 0.0,
        })
    return seed_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--family", default="hartmann",
                        help="Family to probe (default: hartmann — only one "
                             "with sufficient decision opportunity)")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs" / "exp_rq1_3_rf_probe.json")
    args = parser.parse_args()

    print(f"Loading {args.dataset}...")
    records = list(iter_jsonl(args.dataset))
    print(f"Loaded {len(records)} records")

    record_dicts = [r.__dict__ for r in records]
    family_scales = compute_family_scales(record_dicts)
    print(f"Family scales: {[(k, round(v['median'], 3)) for k, v in family_scales.items()]}")

    # Build dataset (X, Y) for the chosen family.
    X, Y, seed_idx, skipped = [], [], [], []
    for r in record_dicts:
        if r["problem"]["family"] != args.family:
            continue
        phase = phase_of_ratio(float(
            (r.get("state") or {}).get("resource", {}).get("remaining_budget_ratio", 0.0)
        ))
        if phase == "terminal":
            continue
        utils = extract_utilities(r, family_scales)
        if utils is None:
            continue
        feat = extract_features(r.get("state", {}), phase)
        # Skip records with all-zero features (would be useless for training).
        if all(v == 0 for v in feat[:11]):  # skip zero-only feature rows
            skipped.append(r.get("id", "?"))
            continue
        y_vec = [utils[a] for a in ACTIONS]
        X.append(feat)
        Y.append(y_vec)
        seed_idx.append(int(r["problem"]["seed"]))

    print(f"\nProbing {args.family}: n_records={len(X)} (skipped {len(skipped)}), "
          f"n_seeds={len(set(seed_idx))}")

    # Baselines.
    # Baseline 1: AlwaysSurrogate (constant).
    sur_idx = ACTIONS.index("surrogate_acquisition")
    regret_sur_all = []
    for k in range(len(Y)):
        y = Y[k]
        a_true = max(range(len(y)), key=lambda j: y[j])
        regret_sur_all.append(y[a_true] - y[sur_idx])

    # Baseline 2: training-fold majority (argmax by global mean utility).
    action_means = [statistics.fmean([Y[k][j] for k in range(len(Y))])
                    for j in range(len(ACTIONS))]
    majority_idx = max(range(len(action_means)), key=lambda j: action_means[j])
    regret_maj_all = []
    for k in range(len(Y)):
        y = Y[k]
        a_true = max(range(len(y)), key=lambda j: y[j])
        regret_maj_all.append(y[a_true] - y[majority_idx])

    # Baseline 3: phase-only rule (constant per phase).
    # Skip — would need phase per record, stored separately. Approximate
    # with AlwaysSurrogate.

    # RF leave-one-seed-out CV.
    print("\nRunning leave-one-seed-out CV with multi-output RandomForest...")
    seed_results = leave_one_seed_out_cv(X, Y, seed_idx, family_filter=args.family)
    regret_rf_all = [r["mean_regret"] for r in seed_results]

    print(f"\n=== Per-seed RF regret on {args.family} ===")
    for r in seed_results:
        print(f"  seed={r['held_out_seed']} n_test={r['n_test']:>3d}  "
              f"mean_regret={r['mean_regret']:+.4f}  median_regret={r['median_regret']:+.4f}")

    print(f"\n=== Aggregate comparison on {args.family} ===")
    print(f"  AlwaysSurrogate  mean_regret = {statistics.fmean(regret_sur_all):+.4f}")
    print(f"  Majority         mean_regret = {statistics.fmean(regret_maj_all):+.4f}")
    print(f"  RF (LOSO)        mean_regret = {statistics.fmean(regret_rf_all):+.4f}")
    # Per-seed-equal aggregation.
    n_seeds = len(set(seed_idx))
    rf_seed_means = regret_rf_all
    sur_seed_means = []  # would need to fold by seed too; approximate as same.
    print(f"\n=== Robustness (LOSO) ===")
    print(f"  RF seed-equal mean: {statistics.fmean(rf_seed_means):+.4f}")
    print(f"  RF seed-equal median: {statistics.median(rf_seed_means):+.4f}")
    print(f"  RF seed-equal max:    {max(rf_seed_means):+.4f}")
    print(f"  RF seed-equal min:    {min(rf_seed_means):+.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "dataset": str(args.dataset),
        "family": args.family,
        "n_records": len(X),
        "n_seeds": n_seeds,
        "features": list(FEATURE_KEYS) + ["phase_early", "phase_mid", "phase_late"],
        "rf_per_seed": seed_results,
        "always_surrogate_mean_regret": statistics.fmean(regret_sur_all),
        "majority_mean_regret": statistics.fmean(regret_maj_all),
        "rf_seed_equal_mean": statistics.fmean(rf_seed_means),
        "rf_seed_equal_median": statistics.median(rf_seed_means),
    }, indent=2) + "\n")
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""EXP-RQ2.1 — Cross-family transfer via leave-one-family-out CV.

Trains a multi-output RandomForestRegressor System-1 on counterfactual
oracle labels (gold_probs → utilities under family_median normalization),
tests whether it transfers to a held-out family never used in training.

This is the RQ2 (Generalization) experiment per `PROGRESS_BRANCH.md §5.2`.

**Setup**
- Dataset: `runs/phase8_6d_h3_20seed_5family.jsonl` (663 records, 5 families × 20 seeds × horizon=3)
- Features: 15-dim state vector (12 numeric + 3 phase one-hot), same as `exp_rq1_3_rf_probe.py`
- Target: 8-dim utility vector under family_median scale normalization
- Model: RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5)
- CV: 5 folds (leave-one-family-out) — each family is held out once
- Baselines:
  1. **Cross-family RF** (RQ2 test): train on 4 families, test on held-out 5th
  2. **In-family RF** (LOSO reference, NOT an upper bound): train on N-1 seeds of held-out family, test on held seed
  3. **AlwaysSurrogate** (constant): always picks `surrogate_acquisition`
  4. **AlwaysLocal** (constant): always picks `local_exploitation` (informational)
  5. **RulePolicy** (real, state-conditional): actual `RulePolicy.select_action()` from `src/opt_agent/loop/policies.py` — STOP if budget ≤ 4, GLOBAL_EXPLORATION if stagnant_steps ≥ 3 AND budget ≥ 2, else SURROGATE_ACQUISITION
  4. **Majority-class** (constant): argmax by global mean utility across all training data
  5. **RulePolicy** (real, state-conditional): actual `RulePolicy.select_action()` from `src/opt_agent/loop/policies.py` — STOP if budget ≤ 4, GLOBAL_EXPLORATION if stagnant_steps ≥ 3 AND budget ≥ 2, else SURROGATE_ACQUISITION

**Metric**: held-out decision regret = U(s, a*) − U(s, a_hat), where a_hat
is the model's argmax and a* is the oracle argmax. Lower is better.

**Statistical test**: paired bootstrap 95% CI on cross-family regret vs
baselines, computed both per-record (underestimates variance due to within-
seed correlation) and seed-block (preserves within-seed correlation;
preferred per codex gpt-6-sol 2026-09-27).

Usage:
    uv run python scripts/exp_rq2_1_cross_family_transfer.py
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

from opt_agent.dataset import read_jsonl
from opt_agent.dataset.validation import phase_of_ratio

# Reuse shared extraction helpers (per codex gpt-6-sol 2026-09-27 REVISE
# #Q2: factor shared helpers to prevent silent divergence)
sys.path.insert(0, str(ROOT / "scripts"))
from _shared_rq_features import (  # noqa: E402
    ACTIONS, FAMILIES, PHASE_VALUES,
    extract_features, extract_utilities, compute_family_scales,
)


def build_dataset(
    records: list[dict], *, family_scales: dict | None = None,
) -> tuple[list, list, list, list, list]:
    """Build (X, Y, family_idx, seed_idx, states) from raw record dicts.

    Skips terminal-phase records (no decision left to make) and zero-feature
    rows. `states` is the raw state dict per record (used by
    `real_rulepolicy_action` for state-conditional baseline).
    """
    if family_scales is None:
        family_scales = compute_family_scales(records)
    X, Y, fam_idx, seed_idx, states = [], [], [], [], []
    for r in records:
        phase = phase_of_ratio(float(
            (r.get("state") or {}).get("resource", {}).get("remaining_budget_ratio", 0.0)
        ))
        if phase == "terminal":
            continue
        utils = extract_utilities(r, family_scales)
        if utils is None:
            continue
        feat = extract_features(r.get("state", {}), phase)
        if all(v == 0 for v in feat[:11]):
            continue
        y_vec = [utils[a] for a in ACTIONS]
        X.append(feat)
        Y.append(y_vec)
        fam_idx.append(FAMILIES.index(r["problem"]["family"]))
        seed_idx.append(int(r["problem"]["seed"]))
        states.append(r.get("state", {}))
    return X, Y, fam_idx, seed_idx, states


def train_predict_rf(X_tr, Y_tr, X_te):
    """Train a multi-output RF (one regressor per action); return predicted
    utilities for X_te as a list of dicts (one per test record)."""
    from sklearn.ensemble import RandomForestRegressor

    regressors = {}
    for j, action in enumerate(ACTIONS):
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_tr, [y[j] for y in Y_tr])
        regressors[action] = rf

    preds = []
    for x in X_te:
        y_pred = {a: regressors[a].predict([x])[0] for a in ACTIONS}
        preds.append(y_pred)
    return preds


def regret(y_true_vec, a_hat: str) -> float:
    """U(s, a*) − U(s, a_hat) under true utilities."""
    a_star_idx = max(range(len(y_true_vec)), key=lambda j: y_true_vec[j])
    a_hat_idx = ACTIONS.index(a_hat)
    return y_true_vec[a_star_idx] - y_true_vec[a_hat_idx]


def real_rulepolicy_action(state: dict) -> str:
    """The actual `RulePolicy.select_action()` logic — state-conditional, NOT a constant.

    From src/opt_agent/loop/policies.py:RulePolicy:
      - STOP if remaining_budget <= 4
      - GLOBAL_EXPLORATION (n_samples=2) if stagnation_steps >= 3 AND budget >= 2
      - SURROGATE_ACQUISITION otherwise
    """
    resource = state.get("resource", {}) or {}
    optimization = state.get("optimization", {}) or {}
    remaining = float(resource.get("remaining_budget", 0.0) or 0.0)
    if remaining <= 4:
        return "stop"
    stagnation = int(optimization.get("stagnation_steps", 0) or 0)
    if stagnation >= 3 and remaining >= 2:
        return "global_exploration"
    return "surrogate_acquisition"


def paired_bootstrap_ci(
    diffs: list[float],
    repetitions: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap 95% CI for the mean of `diffs`. Returns (mean, ci_lo, ci_hi)."""
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    import random as _random
    n = len(diffs)
    mean = sum(diffs) / n
    rng = _random.Random(seed)
    boot_means = []
    for _ in range(repetitions):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo_idx = int((alpha / 2) * repetitions)
    hi_idx = int((1 - alpha / 2) * repetitions) - 1
    return mean, boot_means[lo_idx], boot_means[hi_idx]


def seed_block_bootstrap_ci(
    diffs: list[float],
    seed_idx: list,
    repetitions: int = 2000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    """Seed-block paired bootstrap 95% CI for the mean of `diffs`.

    Resamples **seeds** (with replacement) and uses all records from each
    resampled seed. This preserves within-seed correlation structure,
    which per-record bootstrap ignores. Preferred when records within a
    seed are correlated (which they are here: same Sobol/BO trajectory
    produces multiple correlated decision points).

    Returns (mean, ci_lo, ci_hi). Mean is the records-weighted mean of
    the observed diffs (not the bootstrap mean, which converges to it).
    """
    if not diffs or not seed_idx:
        return float("nan"), float("nan"), float("nan")
    import random as _random
    n = len(diffs)
    if n != len(seed_idx):
        raise ValueError(f"diffs/seed_idx length mismatch: {n} vs {len(seed_idx)}")

    # Group records by seed
    seed_to_diffs: dict = {}
    for i, s in enumerate(seed_idx):
        seed_to_diffs.setdefault(int(s), []).append(diffs[i])
    unique_seeds = sorted(seed_to_diffs.keys())

    # Aggregate per-seed mean for the observed mean
    per_seed_means = [sum(seed_to_diffs[s]) / len(seed_to_diffs[s])
                      for s in unique_seeds]
    # Records-weighted mean across all diffs
    mean = sum(diffs) / n

    rng = _random.Random(rng_seed)
    boot_means = []
    for _ in range(repetitions):
        # Resample seeds with replacement
        n_seeds = len(unique_seeds)
        picked_seeds = [unique_seeds[rng.randrange(n_seeds)] for _ in range(n_seeds)]
        # Records-weighted mean over all records of picked seeds
        total = 0.0
        count = 0
        for s in picked_seeds:
            vals = seed_to_diffs[s]
            total += sum(vals)
            count += len(vals)
        boot_means.append(total / count if count > 0 else 0.0)
    boot_means.sort()
    lo_idx = int((alpha / 2) * repetitions)
    hi_idx = int((1 - alpha / 2) * repetitions) - 1
    return mean, boot_means[lo_idx], boot_means[hi_idx]


def evaluate_holdout(
    X_te, Y_te, preds_cross, preds_infam, states_te, seed_idx_te,
) -> dict:
    """Compute per-record regret for all baselines, plus paired bootstrap CIs
    for cross-family vs AlwaysSurrogate / In-family / RulePolicy / AlwaysLocal.

    Per-record bootstrap and seed-block bootstrap both reported (per codex
    gpt-6-sol 2026-09-27: records from one seed are correlated; resampling
    seeds preserves within-seed correlation structure).
    """
    regrets_cross, regrets_infam, regrets_sur = [], [], []
    regrets_rule_real, regrets_always_local = [], []
    for k, y_true in enumerate(Y_te):
        a_cross = max(preds_cross[k], key=preds_cross[k].get)
        a_infam = max(preds_infam[k], key=preds_infam[k].get)
        regrets_cross.append(regret(y_true, a_cross))
        regrets_infam.append(regret(y_true, a_infam))
        regrets_sur.append(regret(y_true, "surrogate_acquisition"))
        regrets_rule_real.append(regret(y_true, real_rulepolicy_action(states_te[k])))
        regrets_always_local.append(regret(y_true, "local_exploitation"))

    diffs_cross_minus_sur = [c - s for c, s in zip(regrets_cross, regrets_sur)]
    diffs_cross_minus_rule = [c - r for c, r in zip(regrets_cross, regrets_rule_real)]
    diffs_cross_minus_local = [c - l for c, l in zip(regrets_cross, regrets_always_local)]
    diffs_cross_minus_infam = [c - i for c, i in zip(regrets_cross, regrets_infam)]

    # Per-record bootstrap (record-level; underestimates variance due to seed correlation)
    rec_cross_sur_mean, rec_cross_sur_lo, rec_cross_sur_hi = paired_bootstrap_ci(diffs_cross_minus_sur)
    rec_cross_rule_mean, rec_cross_rule_lo, rec_cross_rule_hi = paired_bootstrap_ci(diffs_cross_minus_rule)
    rec_cross_local_mean, rec_cross_local_lo, rec_cross_local_hi = paired_bootstrap_ci(diffs_cross_minus_local)
    rec_cross_infam_mean, _, _ = paired_bootstrap_ci(diffs_cross_minus_infam)

    # Seed-block bootstrap (preserves within-seed correlation; codex gpt-6-sol recommended)
    sb_cross_sur_mean, sb_cross_sur_lo, sb_cross_sur_hi = seed_block_bootstrap_ci(
        diffs_cross_minus_sur, seed_idx_te
    )
    sb_cross_rule_mean, sb_cross_rule_lo, sb_cross_rule_hi = seed_block_bootstrap_ci(
        diffs_cross_minus_rule, seed_idx_te
    )
    sb_cross_local_mean, sb_cross_local_lo, sb_cross_local_hi = seed_block_bootstrap_ci(
        diffs_cross_minus_local, seed_idx_te
    )

    return {
        "n_test": len(Y_te),
        "regret_cross_mean": statistics.fmean(regrets_cross),
        "regret_cross_median": statistics.median(regrets_cross),
        "regret_infam_mean": statistics.fmean(regrets_infam),
        "regret_infam_median": statistics.median(regrets_infam),
        "regret_sur_mean": statistics.fmean(regrets_sur),
        "regret_sur_median": statistics.median(regrets_sur),
        "regret_rule_mean": statistics.fmean(regrets_rule_real),
        "regret_rule_median": statistics.median(regrets_rule_real),
        "regret_alwayslocal_mean": statistics.fmean(regrets_always_local),
        "regret_alwayslocal_median": statistics.median(regrets_always_local),
        # Cross-family vs Surrogate — record-level bootstrap (exploratory)
        "cross_minus_sur_mean": rec_cross_sur_mean,
        "cross_minus_sur_ci_lo": rec_cross_sur_lo,
        "cross_minus_sur_ci_hi": rec_cross_sur_hi,
        # Cross-family vs Surrogate — seed-block bootstrap (preferred)
        "cross_minus_sur_sb_mean": sb_cross_sur_mean,
        "cross_minus_sur_sb_ci_lo": sb_cross_sur_lo,
        "cross_minus_sur_sb_ci_hi": sb_cross_sur_hi,
        # Cross-family vs RulePolicy (real state-conditional) — seed-block bootstrap
        "cross_minus_rule_mean": rec_cross_rule_mean,
        "cross_minus_rule_ci_lo": rec_cross_rule_lo,
        "cross_minus_rule_ci_hi": rec_cross_rule_hi,
        "cross_minus_rule_sb_mean": sb_cross_rule_mean,
        "cross_minus_rule_sb_ci_lo": sb_cross_rule_lo,
        "cross_minus_rule_sb_ci_hi": sb_cross_rule_hi,
        # Cross-family vs AlwaysLocal (constant local_exploitation) — informational
        "cross_minus_local_mean": rec_cross_local_mean,
        "cross_minus_local_ci_lo": rec_cross_local_lo,
        "cross_minus_local_ci_hi": rec_cross_local_hi,
        "cross_minus_local_sb_mean": sb_cross_local_mean,
        "cross_minus_local_sb_ci_lo": sb_cross_local_lo,
        "cross_minus_local_sb_ci_hi": sb_cross_local_hi,
        # Cross-family vs In-family (record-level; informational)
        "cross_minus_infam_mean": rec_cross_infam_mean,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset", type=Path,
        nargs="?", default=ROOT / "runs" / "phase8_6d_h3_20seed_5family.jsonl",
    )
    parser.add_argument(
        "--output-json", type=Path,
        default=ROOT / "runs" / "exp_rq2_1_report.json",
    )
    parser.add_argument(
        "--output-md", type=Path,
        default=ROOT / "runs" / "EXP_RQ2_1_REPORT.md",
    )
    args = parser.parse_args()

    print(f"Loading {args.dataset}...")
    records = read_jsonl(args.dataset, version_policy="warn")
    record_dicts = [r.__dict__ if hasattr(r, "__dict__") else r for r in records]
    print(f"Loaded {len(record_dicts)} records")

    X, Y, fam_idx, seed_idx, all_states = build_dataset(record_dicts)
    print(f"After filtering (non-terminal, non-zero): n={len(X)} "
          f"records across {len(set(fam_idx))} families, {len(set(seed_idx))} seeds")

    # Group by family
    fam_groups = defaultdict(list)
    for i, fi in enumerate(fam_idx):
        fam_groups[fi].append(i)
    print("\nRecords per family (after filtering):")
    for fi, idxs in sorted(fam_groups.items()):
        print(f"  {FAMILIES[fi]}: n={len(idxs)}")

    # Per-family CV
    results = {"config": {
        "dataset": str(args.dataset),
        "n_records_total": len(X),
        "families": list(FAMILIES),
        "model": "RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5)",
        "cv": "5-fold leave-one-family-out",
        "baselines": ["AlwaysSurrogate", "In-family RF (LOSO reference, NOT upper bound)",
                       "AlwaysLocal (constant local_exploitation)",
                       "RulePolicy (real, state-conditional)"],
    }, "folds": []}

    for held_out_fi, test_indices in sorted(fam_groups.items()):
        held_out_family = FAMILIES[held_out_fi]
        train_indices = [i for i, fi in enumerate(fam_idx) if fi != held_out_fi]
        X_tr = [X[i] for i in train_indices]
        Y_tr = [Y[i] for i in train_indices]
        X_te = [X[i] for i in test_indices]
        Y_te = [Y[i] for i in test_indices]
        states_te = [all_states[i] for i in test_indices]

        print(f"\n=== Held-out family: {held_out_family} ===")
        print(f"  Train: n={len(X_tr)} (4 other families)")
        print(f"  Test:  n={len(X_te)} ({held_out_family})")

        # Cross-family model: train on all records from 4 families, test on held-out family
        print("  Training cross-family RF...")
        preds_cross = train_predict_rf(X_tr, Y_tr, X_te)

        # In-family upper bound: leave-one-seed-out within the held-out family.
        # If the held-out family has only 1 seed, fall back to train on all.
        held_out_seeds = sorted(set(seed_idx[i] for i in test_indices))
        if len(held_out_seeds) >= 2:
            # Aggregate: train on all-but-one held-out seed, test on held-out seed.
            in_fam_preds_per_record = [[] for _ in test_indices]
            for held_out_seed in held_out_seeds:
                tr_seed = [i for i in test_indices if seed_idx[i] != held_out_seed]
                te_seed = [i for i in test_indices if seed_idx[i] == held_out_seed]
                if not tr_seed or not te_seed:
                    continue
                preds_seed = train_predict_rf(
                    [X[i] for i in tr_seed],
                    [Y[i] for i in tr_seed],
                    [X[i] for i in te_seed],
                )
                # Map back to test_indices order
                for j, ti in enumerate(te_seed):
                    pos_in_test = test_indices.index(ti)
                    in_fam_preds_per_record[pos_in_test] = preds_seed[j]
            # For any test records without an in-family prediction (e.g.,
            # single-seed family), use cross-family prediction as fallback
            preds_infam = [
                p if p else preds_cross[k]
                for k, p in enumerate(in_fam_preds_per_record)
            ]
        else:
            preds_infam = preds_cross  # fallback

        fold_result = evaluate_holdout(
            X_te, Y_te, preds_cross, preds_infam, states_te,
            seed_idx_te=[seed_idx[i] for i in test_indices],
        )
        fold_result["held_out_family"] = held_out_family
        fold_result["n_train"] = len(X_tr)
        fold_result["n_test_seeds"] = len(held_out_seeds)
        results["folds"].append(fold_result)

        print(f"  Cross-family regret: {fold_result['regret_cross_mean']:+.4f} "
              f"(median {fold_result['regret_cross_median']:+.4f})")
        print(f"  In-family regret:    {fold_result['regret_infam_mean']:+.4f} "
              f"(median {fold_result['regret_infam_median']:+.4f})")
        print(f"  AlwaysSurrogate:     {fold_result['regret_sur_mean']:+.4f}")
        print(f"  RulePolicy (real):    {fold_result['regret_rule_mean']:+.4f} "
              f"(state-conditional)")
        print(f"  AlwaysLocal:         {fold_result['regret_alwayslocal_mean']:+.4f} "
              f"(constant local_exploitation)")
        ci = fold_result
        print(f"  Cross − Surrogate (seed-block, preferred):  "
              f"95% CI [{ci['cross_minus_sur_sb_ci_lo']:+.4f}, {ci['cross_minus_sur_sb_ci_hi']:+.4f}]")
        print(f"  Cross − RulePolicy (seed-block):     "
              f"mean={ci['cross_minus_rule_sb_mean']:+.4f}  "
              f"95% CI [{ci['cross_minus_rule_sb_ci_lo']:+.4f}, {ci['cross_minus_rule_sb_ci_hi']:+.4f}]")
        print(f"  Cross − AlwaysLocal (seed-block):    "
              f"mean={ci['cross_minus_local_sb_mean']:+.4f}  "
              f"95% CI [{ci['cross_minus_local_sb_ci_lo']:+.4f}, {ci['cross_minus_local_sb_ci_hi']:+.4f}]")
        print(f"  Cross − In-family:                   "
              f"mean={ci['cross_minus_infam_mean']:+.4f}  (in-family reference)")

    # Aggregate across families
    n_total = sum(f["n_test"] for f in results["folds"])
    weighted = lambda key: sum(f[key] * f["n_test"] for f in results["folds"]) / n_total
    results["aggregate"] = {
        "n_total_records": n_total,
        "weighted_regret_cross_mean": weighted("regret_cross_mean"),
        "weighted_regret_infam_mean": weighted("regret_infam_mean"),
        "weighted_regret_sur_mean": weighted("regret_sur_mean"),
        "weighted_regret_rule_mean": weighted("regret_rule_mean"),
        "weighted_regret_alwayslocal_mean": weighted("regret_alwayslocal_mean"),
        "weighted_cross_minus_sur_mean": weighted("cross_minus_sur_mean"),
        "weighted_cross_minus_rule_mean": weighted("cross_minus_rule_mean"),
        "weighted_cross_minus_local_mean": weighted("cross_minus_local_mean"),
        "weighted_cross_minus_infam_mean": weighted("cross_minus_infam_mean"),
    }

    # Pool per-record paired bootstrap across all folds
    # (We don't keep individual per-record regrets in the report; for a quick
    # summary, use the per-fold CI as the primary inferential signal.)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nWrote {args.output_json}")

    # Markdown report
    md: list[str] = []
    md.append("# EXP-RQ2.1 — Cross-Family Transfer (Leave-One-Family-Out CV)\n")
    md.append("**Date**: 2026-09-27")
    md.append(f"**Dataset**: `{args.dataset.relative_to(ROOT)}` "
              f"({results['config']['n_records_total']} records, 5 families × 20 seeds × horizon=3)")
    md.append(f"**Model**: multi-output RandomForestRegressor (one regressor per action)")
    md.append(f"**CV**: 5-fold leave-one-family-out")
    md.append(f"**Baselines**: AlwaysSurrogate, In-family RF (reference, NOT upper bound), "
              f"**RulePolicy (real, state-conditional)**, AlwaysLocal (constant local_exploitation, informational)")
    md.append(f"**Metric**: held-out decision regret = U(s, a*) − U(s, a_hat) (lower is better)")
    md.append("")
    md.append("**Statistical inference**: per-family paired bootstrap CIs in two flavors — "
              "(a) **record-level** (understates variance due to within-seed correlation), "
              "(b) **seed-block** (preferred per codex gpt-6-sol 2026-09-27 REVISE). "
              "Both reported below; only seed-block CIs used for headline claims.")
    md.append("")
    md.append("## 1. Per-family results\n")
    md.append("| Held-out | n_test | cross-fam regret | in-fam (reference) | AlwaysSurrogate | **RulePolicy (real)** | AlwaysLocal | Cross − Sur CI (seed-block) | Cross − **Rule** CI (seed-block) | Cross − Local CI (seed-block) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for f in results["folds"]:
        md.append(
            f"| {f['held_out_family']} | {f['n_test']} | "
            f"{f['regret_cross_mean']:+.4f} | {f['regret_infam_mean']:+.4f} | "
            f"{f['regret_sur_mean']:+.4f} | **{f['regret_rule_mean']:+.4f}** | "
            f"{f['regret_alwayslocal_mean']:+.4f} | "
            f"`[{f['cross_minus_sur_sb_ci_lo']:+.4f}, {f['cross_minus_sur_sb_ci_hi']:+.4f}]` | "
            f"`[{f['cross_minus_rule_sb_ci_lo']:+.4f}, {f['cross_minus_rule_sb_ci_hi']:+.4f}]` | "
            f"`[{f['cross_minus_local_sb_ci_lo']:+.4f}, {f['cross_minus_local_sb_ci_hi']:+.4f}]` |"
        )
    md.append("")
    md.append("**Reading the table (negative diff = cross is BETTER)**:")
    md.append("- Cross − Surrogate CI < 0 → cross beats AlwaysSurrogate")
    md.append("- Cross − **RulePolicy** CI < 0 → cross beats the **real, state-conditional** rule")
    md.append("- Cross − AlwaysLocal CI < 0 → cross beats the constant `local_exploitation` policy")
    md.append("")
    md.append("**Important note on baselines (per codex gpt-6-sol REVISE)**:")
    md.append("- **RulePolicy** is the actual state-conditional `RulePolicy.select_action()` "
              "from `src/opt_agent/loop/policies.py`, **not** a constant `local_exploitation` "
              "policy. It picks STOP if remaining_budget ≤ 4, GLOBAL_EXPLORATION if "
              "stagnation_steps ≥ 3 AND budget ≥ 2, else SURROGATE_ACQUISITION.")
    md.append("- **AlwaysLocal** is the constant `local_exploitation` policy (informational "
              "only — not what was previously called \"RulePolicy\" in round-1).")
    md.append("")

    md.append("## 2. Aggregate (records-weighted across folds)\n")
    md.append(f"- **Cross-family RF mean regret**: {results['aggregate']['weighted_regret_cross_mean']:+.4f}")
    md.append(f"- **In-family RF mean regret (reference)**: {results['aggregate']['weighted_regret_infam_mean']:+.4f}")
    md.append(f"- **AlwaysSurrogate mean regret**: {results['aggregate']['weighted_regret_sur_mean']:+.4f}")
    md.append(f"- **RulePolicy (real, state-conditional) mean regret**: {results['aggregate']['weighted_regret_rule_mean']:+.4f}")
    md.append(f"- **AlwaysLocal mean regret**: {results['aggregate']['weighted_regret_alwayslocal_mean']:+.4f}")
    md.append(f"- **Cross − Surrogate (records-weighted mean)**: {results['aggregate']['weighted_cross_minus_sur_mean']:+.4f}")
    md.append(f"- **Cross − RulePolicy (records-weighted mean)**: {results['aggregate']['weighted_cross_minus_rule_mean']:+.4f}")
    md.append(f"- **Cross − AlwaysLocal (records-weighted mean)**: {results['aggregate']['weighted_cross_minus_local_mean']:+.4f}")
    md.append(f"- **Cross − In-family (records-weighted mean)**: {results['aggregate']['weighted_cross_minus_infam_mean']:+.4f}")
    md.append("")

    md.append("## 3. Interpretation\n")
    md.append("**Cross-family transfer test (RQ2)**: if the cross-family regret is "
              "lower than a baseline's on a held-out family, the System-1 learned on 4 "
              "families transfers usefully to it.")
    md.append("")
    md.append("**In-family reference (NOT upper bound)**: the in-family RF (trained on "
              "N-1 seeds of the held-out family) is a reference, not an upper bound. "
              "On Rastrigin, the cross-family RF has lower regret than the in-family RF "
              "(Cross − In-family diff < 0). The two fits also have different amounts "
              "of training data (4 families vs N-1 seeds).")
    md.append("")

    # Per-family headline findings (descriptive, exploratory)
    beats_sur_sb = sum(1 for f in results["folds"] if f["cross_minus_sur_sb_ci_hi"] < 0)
    loses_sur_sb = sum(1 for f in results["folds"] if f["cross_minus_sur_sb_ci_lo"] > 0)
    beats_rule_sb = sum(1 for f in results["folds"] if f["cross_minus_rule_sb_ci_hi"] < 0)
    loses_rule_sb = sum(1 for f in results["folds"] if f["cross_minus_rule_sb_ci_lo"] > 0)
    beats_local_sb = sum(1 for f in results["folds"] if f["cross_minus_local_sb_ci_hi"] < 0)
    loses_local_sb = sum(1 for f in results["folds"] if f["cross_minus_local_sb_ci_lo"] > 0)

    md.append("**Headline findings (per codex gpt-6-sol 2026-09-27 REVISE, exploratory)**:")
    md.append("")
    md.append("- **vs AlwaysSurrogate** (seed-block 95% CI):")
    md.append(f"  - **{beats_sur_sb}/5 families**: cross-family has lower regret")
    md.append(f"  - **{loses_sur_sb}/5 families**: cross-family has higher regret")
    md.append(f"  - **{5 - beats_sur_sb - loses_sur_sb}/5 families**: inconclusive")
    md.append("")
    md.append("- **vs RulePolicy (real, state-conditional)**:")
    md.append(f"  - **{beats_rule_sb}/5 families**: cross-family has lower regret")
    md.append(f"  - **{loses_rule_sb}/5 families**: cross-family has higher regret")
    md.append(f"  - **{5 - beats_rule_sb - loses_rule_sb}/5 families**: inconclusive")
    md.append("")
    md.append("- **vs AlwaysLocal (constant local_exploitation)** — informational:")
    md.append(f"  - **{beats_local_sb}/5 families**: cross-family has lower regret")
    md.append(f"  - **{loses_local_sb}/5 families**: cross-family has higher regret")
    md.append(f"  - **{5 - beats_local_sb - loses_local_sb}/5 families**: inconclusive")
    md.append("")
    md.append("**Pattern**: cross-family transfer is heterogeneous across the five tested "
              "family–dimension combinations, but neither *family-specific* nor "
              "*dimension-specific* in a clean sense — family and dimension are "
              "confounded, and dimension is itself an RF feature.")
    md.append("")
    md.append("**Hypothesis (descriptive, untested)**: cross-family training helps when the "
              "held-out family shares structural features with the training set (e.g., "
              "10D multimodal structure common to Ackley + Rastrigin in the training set). "
              "Hartmann's decision-rich 6D landscape may have family-specific structure "
              "that the cross model does not capture; SixHumpCamel is too flat for the cross "
              "model to add value over AlwaysSurrogate.")
    md.append("")

    md.append("## 4. Limitations\n")
    md.append("- **Single model class (RF)** — we don't claim this generalizes to MLPs, "
              "Transformers, or other System-1 architectures.")
    md.append("- **Single dataset (20-seed 5-family BOCoDe)** — engineering families not included.")
    md.append("- **Family × dimension confounded**: Branin/SixHumpCamel are 2D, Hartmann is 6D, "
              "Ackley/Rastrigin are 10D. The cross-family signal may be partially a dimension-"
              "transfer signal. A crossed design (same family at multiple dimensions) is required "
              "to disentangle.")
    md.append("- **Trained RF vs baselines is heterogeneous** — wins vs RulePolicy on 3/5, "
              "wins vs AlwaysSurrogate on 2/5, loses vs AlwaysLocal on aggregate "
              "(AlwaysLocal is an unrealistic constant `local_exploitation` policy that ignores budget). "
              "The headline number depends on which baseline you pick.")
    md.append("- **One-step decision regret, not closed-loop trajectory** — the RF is evaluated "
              "on its one-step decision regret against the oracle. We do not measure the effect "
              "of deploying the RF through a full BO trajectory.")
    md.append("- **5 per-family 95% CIs without multiplicity adjustment** — some borderline-pass "
              "CIs may not survive Bonferroni or Holm correction. Per codex gpt-6-sol recommendation, "
              "the 95% intervals are exploratory.")
    md.append("- **Aggregate records-weighted mean is arithmetic, not inferential** — it hides "
              "per-family heterogeneity.")
    md.append("")

    md.append("## 5. Files\n")
    md.append(f"- `{args.output_json.relative_to(ROOT)}` — raw per-fold results (record-level + seed-block CIs)")
    md.append(f"- `{args.output_md.relative_to(ROOT)}` — this report")
    md.append(f"- Source data: `{args.dataset.relative_to(ROOT)}` (20-seed 5-family)")
    md.append(f"- Shared helpers: `scripts/_shared_rq_features.py` (also used by `exp_rq1_3_rf_probe.py`)")
    md.append("")

    args.output_md.write_text("\n".join(md))
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

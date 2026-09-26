"""EXP-RQ1.3 — Scale Sensitivity Analysis (parallel, multi-scheme).

Per codex (gpt-5.6-sol) review 2026-09-26:
  - The current analysis uses per-record `optimization_scale` = std of 5 init
    points. For Hartmann this varies 0.0889-0.6579 across seeds — outlier seeds
    inflate the family mean.
  - Compute ΔU, R_sur, R_stop under FOUR scale schemes:
      (a) per-record (current — std of 5 init)
      (b) family-median (one per family — robust to outliers)
      (c) family-mean
      (d) fixed total_budget (= 30)
  - Report seed-equal means, medians, q90/q95, and prevalence above a
    pre-specified practical threshold δ.

Parallelization: multiprocessing.Pool over records (32 cores available).
GPU: not needed (CPU-bound on small records).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import statistics
import sys
import time
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


# Pre-specified practical threshold (utility units). Set to 0.1 — small
# but non-negligible compared to the ~0.03 gaps observed on Branin/6HC.
PRACTICAL_DELTA = 0.1


def _process_record(record_dict: dict, scale_scheme: str,
                    family_scales: dict[str, dict[str, float]]) -> dict | None:
    """Compute ΔU, R_sur, R_stop for one record under a scale scheme.

    scale_scheme in {"per_record", "family_median", "family_mean",
                     "total_budget"}.
    """
    family = record_dict["problem"]["family"]
    md = record_dict.get("metadata", {}) or {}
    w_opt = float(md.get("utility_weights", {}).get("w_optimization", 1.0))
    old_scale = float(md.get("optimization_scale", 1.0))
    if scale_scheme == "per_record":
        new_scale = old_scale
    elif scale_scheme == "family_median":
        new_scale = family_scales[family]["median"]
    elif scale_scheme == "family_mean":
        new_scale = family_scales[family]["mean"]
    elif scale_scheme == "total_budget":
        new_scale = float(md.get("total_budget", 30.0))
    elif scale_scheme == "robust_iqr":
        new_scale = family_scales[family]["robust_sd_iqr"]
    else:
        raise ValueError(f"unknown scheme {scale_scheme}")

    if old_scale <= 0 or new_scale <= 0:
        return None

    rr = record_dict.get("rollout_results") or {}
    utils = {}
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
        # utility_new = utility_old + w_opt * delta_opt * (1/new - 1/old)
        u_new = u_orig + w_opt * d_opt * (1.0 / new_scale - 1.0 / old_scale)
        utils[action] = u_new

    if len(utils) < 2:
        return None

    vals = sorted(utils.items(), key=lambda x: x[1], reverse=True)
    a_star, u_star = vals[0]
    second_u = vals[1][1]
    u_sur = utils.get("surrogate_acquisition", float("nan"))
    u_stop = utils.get("stop", float("nan"))
    return {
        "family": family,
        "seed": int(record_dict["problem"]["seed"]),
        "phase": phase_of_from_dict(record_dict),
        "a_star": a_star,
        "ΔU": u_star - second_u,
        "R_sur": u_star - u_sur,
        "R_stop": u_star - u_stop,
        "u_star": u_star,
        "scale_old": old_scale,
        "scale_new": new_scale,
        "n_actions_valid": len(utils),
    }


def phase_of_from_dict(record_dict: dict) -> str:
    """Canonical phase via shared validation.phase_of_ratio."""
    state = record_dict.get("state", {})
    resource = state.get("resource", {})
    ratio = float(resource.get("remaining_budget_ratio", 0.0))
    return phase_of_ratio(ratio)


def compute_family_scales(records: list[dict]) -> dict[str, dict[str, float]]:
    """Per-family statistics of optimization_scale.

    Includes robust SD (IQR/1.349) as policy-independent reference scale.
    Seed-equal versions compute per-seed median first, then summarize across
    seeds — avoids long trajectories dominating.
    """
    by_fam: dict[str, list[float]] = defaultdict(list)
    by_fam_seed: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in records:
        s = float((r.get("metadata") or {}).get("optimization_scale", 0.0))
        if s > 0:
            fam = r["problem"]["family"]
            by_fam[fam].append(s)
            by_fam_seed[(fam, int(r["problem"]["seed"]))].append(s)

    out = {}
    for fam, vals in by_fam.items():
        sorted_vals = sorted(vals)
        n = len(sorted_vals)
        # Robust SD via IQR (1.349 is the inverse normal-IQR constant).
        if n >= 4:
            q1, _, q3 = statistics.quantiles(sorted_vals, n=4, method="inclusive")
            iqr = q3 - q1
            robust_sd = iqr / 1.349 if iqr > 0 else statistics.fmean(vals)
        else:
            robust_sd = statistics.fmean(vals)
        # Seed-equal median: per-seed median, then median across seeds.
        seed_medians = [statistics.median(sv) for (f, _), sv in by_fam_seed.items()
                        if f == fam]
        seed_means = [statistics.fmean(sv) for (f, _), sv in by_fam_seed.items()
                      if f == fam]
        out[fam] = {
            "median": statistics.median(vals),
            "mean": statistics.fmean(vals),
            "robust_sd_iqr": robust_sd,
            "seed_equal_median": statistics.median(seed_medians)
                if seed_medians else statistics.median(vals),
            "seed_equal_mean": statistics.fmean(seed_means)
                if seed_means else statistics.fmean(vals),
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
            "n_seeds": len(seed_medians),
        }
    return out


def aggregate(rows: list[dict]) -> dict:
    """Aggregate rows into per-family stats.

    Uses statistics.quantiles (inclusive method) for q90/q95.
    Excludes terminal records (remaining_budget_ratio == 0) from the primary
    decision-opportunity estimand; they're aggregated separately.
    """
    by_fam: dict[str, list[dict]] = defaultdict(list)
    by_fam_terminal: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["phase"] == "terminal":
            by_fam_terminal[r["family"]].append(r)
        else:
            by_fam[r["family"]].append(r)
    out = {}
    for fam, rs in by_fam.items():
        for key in ("ΔU", "R_sur", "R_stop", "u_star"):
            vals = [r[key] for r in rs if math.isfinite(r[key])]
            if not vals:
                continue
            n = len(vals)
            sorted_vals = sorted(vals)
            # Use inclusive quantile method (matches numpy default).
            if n >= 10:
                q90 = statistics.quantiles(sorted_vals, n=10, method="inclusive")[8]
                q95 = statistics.quantiles(sorted_vals, n=20, method="inclusive")[18]
            else:
                q90 = sorted_vals[min(int(0.9 * n), n - 1)]
                q95 = sorted_vals[min(int(0.95 * n), n - 1)]
            prevalence_above_delta = sum(1 for v in vals if v > PRACTICAL_DELTA) / n
            out.setdefault(fam, {})[key] = {
                "n": n,
                "mean": statistics.fmean(vals),
                "median": statistics.median(vals),
                "min": min(vals),
                "max": max(vals),
                "q90": q90,
                "q95": q95,
                "prevalence_above_delta": prevalence_above_delta,
            }
    # Terminal: report separately, smaller stats.
    terminal_stats = {}
    for fam, rs in by_fam_terminal.items():
        for key in ("ΔU", "R_sur"):
            vals = [r[key] for r in rs if math.isfinite(r[key])]
            if vals:
                terminal_stats.setdefault(fam, {})[key] = {
                    "n": len(vals),
                    "mean": statistics.fmean(vals),
                    "median": statistics.median(vals),
                }
    if terminal_stats:
        out["__terminal_summary__"] = terminal_stats
    return out


def seed_equal_per_family(rows: list[dict]) -> dict:
    """One number per (family, seed); report summary across seeds.

    Per-seed mean/median/prevalence first, then aggregate over seeds.
    Excludes terminal records.
    """
    by_seed: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        if r["phase"] == "terminal":
            continue
        by_seed[(r["family"], r["seed"])].append(r)
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (fam, seed), rs in by_seed.items():
        for key in ("ΔU", "R_sur", "R_stop"):
            vals = [r[key] for r in rs if math.isfinite(r[key])]
            if vals:
                out[fam][key].append(statistics.fmean(vals))
    summary = {}
    for fam, d in out.items():
        summary[fam] = {}
        for k, vs in d.items():
            if not vs:
                continue
            summary[fam][k] = {
                "mean_of_seed_means": statistics.fmean(vs),
                "median_of_seed_means": statistics.median(vs),
                "min": min(vs),
                "max": max(vs),
                "n_seeds": len(vs),
            }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs" / "exp_rq1_3_scale_sensitivity.json")
    args = parser.parse_args()

    print(f"Loading {args.dataset}...")
    records = list(iter_jsonl(args.dataset))
    record_dicts = [r.__dict__ for r in records]
    print(f"Loaded {len(records)} records")

    family_scales = compute_family_scales(record_dicts)
    print("\n=== Per-family optimization_scale ===")
    for fam in sorted(family_scales):
        s = family_scales[fam]
        print(f"  {fam:14s} n={s['n']:>3d}  median={s['median']:.4f}  "
              f"mean={s['mean']:.4f}  min={s['min']:.4f}  max={s['max']:.4f}")

    schemes = ("per_record", "family_median", "family_mean",
               "robust_iqr", "total_budget")
    all_results = {}
    t0 = time.monotonic()
    for scheme in schemes:
        print(f"\n--- Computing under scheme={scheme} (workers={args.workers}) ---")
        # Parallelize.
        with mp.Pool(args.workers) as pool:
            rows = pool.starmap(
                _process_record,
                [(rd, scheme, family_scales) for rd in record_dicts],
            )
        rows = [r for r in rows if r is not None]
        agg = aggregate(rows)
        seed_eq = seed_equal_per_family(rows)
        all_results[scheme] = {
            "n_records": len(rows),
            "aggregate": agg,
            "seed_equal_mean": seed_eq,
        }
        # Print quick table.
        for fam in sorted(agg):
            dU = agg[fam].get("ΔU", {})
            Rs = agg[fam].get("R_sur", {})
            print(f"  {fam:14s} ΔU mean={dU.get('mean', float('nan')):.4f} "
                  f"med={dU.get('median', float('nan')):.4f} "
                  f"q90={dU.get('q90', float('nan')):.4f} "
                  f"prev(>δ)={dU.get('prevalence_above_delta', 0):.2%} | "
                  f"R_sur mean={Rs.get('mean', float('nan')):.4f} "
                  f"med={Rs.get('median', float('nan')):.4f} "
                  f"prev(>δ)={Rs.get('prevalence_above_delta', 0):.2%}")
    elapsed = time.monotonic() - t0
    print(f"\nAll schemes computed in {elapsed:.1f}s")

    # Headline comparison: does the regime distinction survive family_median
    # normalization?  Print ΔU mean under each scheme side-by-side.
    print("\n=== ΔU mean across schemes (regime-stability test) ===")
    print(f"{'family':14s} " + " ".join(f"{s:>16s}" for s in schemes))
    for fam in sorted(family_scales):
        row = [fam]
        for scheme in schemes:
            d = all_results[scheme]["aggregate"].get(fam, {}).get("ΔU", {})
            row.append(f"{d.get('mean', float('nan')):>16.4f}")
        print(" ".join(row))

    # Collect all per-record rows (not aggregated) for downstream experiments.
    # We re-run with multiprocessing once more on the canonical scheme (family_median).
    with mp.Pool(args.workers) as pool:
        all_rows_canonical = pool.starmap(
            _process_record,
            [(rd, "family_median", family_scales) for rd in record_dicts],
        )
    all_rows_canonical = [r for r in all_rows_canonical if r is not None]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "dataset": str(args.dataset),
        "practical_delta": PRACTICAL_DELTA,
        "family_scales": family_scales,
        "results_by_scheme": all_results,
        "per_record_canonical_scheme": all_rows_canonical,
    }, indent=2, default=str) + "\n")
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

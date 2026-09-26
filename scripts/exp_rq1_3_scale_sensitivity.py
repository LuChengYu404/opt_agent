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

from opt_agent.dataset import iter_jsonl, phase_of

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
    """Reproduce validation.phase_of without instantiating DecisionRecord."""
    state = record_dict.get("state", {})
    resource = state.get("resource", {})
    ratio = float(resource.get("remaining_budget_ratio", 0.0))
    # Same boundaries as validation.phase_of_ratio (Phase 8.6B).
    if ratio > 0.7:
        return "early"
    if ratio > 0.3:
        return "mid"
    if ratio > 0.0:
        return "late"
    return "terminal"


def compute_family_scales(records: list[dict]) -> dict[str, dict[str, float]]:
    """Per-family median / mean of optimization_scale."""
    by_fam: dict[str, list[float]] = defaultdict(list)
    for r in records:
        s = float((r.get("metadata") or {}).get("optimization_scale", 0.0))
        if s > 0:
            by_fam[r["problem"]["family"]].append(s)
    out = {}
    for fam, vals in by_fam.items():
        out[fam] = {
            "median": statistics.median(vals),
            "mean": statistics.fmean(vals),
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }
    return out


def aggregate(rows: list[dict]) -> dict:
    """Aggregate rows into per-family stats."""
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    out = {}
    for fam, rs in by_fam.items():
        for key in ("ΔU", "R_sur", "R_stop", "u_star"):
            vals = [r[key] for r in rs if math.isfinite(r[key])]
            if not vals:
                continue
            q90 = (sorted(vals)[int(0.9 * len(vals))] if len(vals) >= 2
                   else vals[0])
            q95 = (sorted(vals)[int(0.95 * len(vals))] if len(vals) >= 2
                   else vals[0])
            prevalence_above_delta = sum(1 for v in vals if v > PRACTICAL_DELTA) / len(vals)
            out.setdefault(fam, {})[key] = {
                "n": len(vals),
                "mean": statistics.fmean(vals),
                "median": statistics.median(vals),
                "min": min(vals),
                "max": max(vals),
                "q90": q90,
                "q95": q95,
                "prevalence_above_delta": prevalence_above_delta,
            }
    return out


def seed_equal_per_family(rows: list[dict]) -> dict:
    """One number per (family, seed); report mean across seeds.

    This avoids long-trajectory records dominating.
    """
    by_seed: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_seed[(r["family"], r["seed"])].append(r)
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (fam, seed), rs in by_seed.items():
        for key in ("ΔU", "R_sur", "R_stop"):
            vals = [r[key] for r in rs if math.isfinite(r[key])]
            if vals:
                out[fam][key].append(statistics.fmean(vals))
    return {fam: {k: statistics.fmean(vs) if vs else float("nan")
                   for k, vs in d.items()}
            for fam, d in out.items()}


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

    schemes = ("per_record", "family_median", "family_mean", "total_budget")
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "dataset": str(args.dataset),
        "practical_delta": PRACTICAL_DELTA,
        "family_scales": family_scales,
        "results_by_scheme": all_results,
    }, indent=2, default=str) + "\n")
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Numerical verification of the THEORY.md propositions on real data.

P1 (affine invariance): for every rollout branch in the canonical 20-seed
dataset, recompute `compute_utility` with an affinely transformed
optimization delta and matching transformed scale, and assert the utility
is unchanged — verifying the proposition through the actual code path.

P2 (adaptive-value bound): recompute `D = E[max_a U] - max_a E[U]` per
family from the same dataset under family-median normalization, verify
D >= 0, and report D with the best constant action per family.

Example:
  uv run python scripts/verify_theory_propositions.py
"""
from __future__ import annotations

import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _shared_rq_features import (  # noqa: E402
    ACTIONS, FAMILIES, compute_family_scales, extract_utilities,
)
from opt_agent.dataset import iter_jsonl  # noqa: E402
from opt_agent.rollout.utility import UtilityWeights, compute_utility  # noqa: E402

DATASET = ROOT / "runs/phase8_6d_h3_20seed_5family.jsonl"


def verify_p1(records: list[dict], rng: random.Random) -> dict:
    checked, max_rel_err = 0, 0.0
    weights = UtilityWeights()
    for record in records:
        md = record.get("metadata") or {}
        scale = float(md.get("optimization_scale", 1.0))
        if not (math.isfinite(scale) and scale > 0):
            continue
        for cell in (record.get("rollout_results") or {}).values():
            if not isinstance(cell, dict):
                continue
            delta = cell.get("delta_optimization")
            if delta is None or not math.isfinite(float(delta)):
                continue
            alpha = rng.uniform(0.05, 20.0)
            beta = rng.uniform(-50.0, 50.0)
            base = compute_utility(
                delta_optimization=float(delta),
                n_new_samples=int(cell.get("n_new_samples", 1)),
                n_model_fits=int(cell.get("n_model_fits", 0)),
                weights=weights, optimization_scale=scale,
                evaluation_scale=float(md.get("evaluation_scale", 30.0)),
                fit_scale=float(md.get("fit_scale", 30.0)),
                is_stop=cell.get("action_id") == "stop",
                remaining_budget_ratio=float(
                    (record.get("state") or {}).get("resource", {})
                    .get("remaining_budget_ratio", 0.0)),
            )
            # affine transform of the objective axis: delta and scale both
            # scale by alpha (beta cancels in the difference by construction)
            transformed = compute_utility(
                delta_optimization=alpha * float(delta),
                n_new_samples=int(cell.get("n_new_samples", 1)),
                n_model_fits=int(cell.get("n_model_fits", 0)),
                weights=weights, optimization_scale=alpha * scale,
                evaluation_scale=float(md.get("evaluation_scale", 30.0)),
                fit_scale=float(md.get("fit_scale", 30.0)),
                is_stop=cell.get("action_id") == "stop",
                remaining_budget_ratio=float(
                    (record.get("state") or {}).get("resource", {})
                    .get("remaining_budget_ratio", 0.0)),
            )
            if not (math.isfinite(base) and math.isfinite(transformed)):
                continue
            rel = (abs(transformed - base)
                   / max(1e-12, abs(base), abs(alpha * float(delta)) * 1e-9))
            max_rel_err = max(max_rel_err, rel)
            checked += 1
    return {"n_branches_checked": checked, "max_relative_error": max_rel_err}


def verify_p2(records: list[dict]) -> dict:
    scales = compute_family_scales(records)
    per_family: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record in records:
        utilities = extract_utilities(record, scales)
        if utilities:
            per_family[record["problem"]["family"]].append(utilities)
    out = {}
    for family in FAMILIES:
        rows = per_family.get(family) or []
        if not rows:
            continue
        oracle_mean = statistics.fmean(max(u[a] for a in u) for u in rows)
        mean_by_action = {
            action: statistics.fmean(u[action] for u in rows if action in u)
            for action in ACTIONS
        }
        best_constant = max(mean_by_action, key=mean_by_action.get)
        gap = oracle_mean - mean_by_action[best_constant]
        out[family] = {
            "n_records": len(rows),
            "adaptive_value_D": gap,
            "nonnegative": gap >= -1e-12,
            "oracle_mean_utility": oracle_mean,
            "best_constant_action": best_constant,
            "best_constant_mean_utility": mean_by_action[best_constant],
        }
    return out


def main() -> None:
    records = [r.__dict__ for r in iter_jsonl(DATASET)]
    rng = random.Random(20260927)
    p1 = verify_p1(records, rng)
    p2 = verify_p2(records)
    report = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "n_records": len(records),
        "p1_affine_invariance": p1,
        "p2_adaptive_value": p2,
    }
    print(f"dataset: {report['dataset']} ({len(records)} records)")
    print(f"P1 affine invariance: {p1['n_branches_checked']} branches "
          f"re-checked through compute_utility, "
          f"max relative error {p1['max_relative_error']:.2e}")
    print("P2 adaptive-value per family (family-median normalization):")
    for family, row in p2.items():
        print(f"  {family:14s} D={row['adaptive_value_D']:+.4f} "
              f"(n={row['n_records']}) best constant action = "
              f"{row['best_constant_action']} "
              f"mean {row['best_constant_mean_utility']:+.4f} vs oracle "
              f"{row['oracle_mean_utility']:+.4f} nonneg={row['nonnegative']}")
    output = ROOT / "runs/verify_theory_propositions.json"
    output.write_text(
        __import__("json").dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output}")
    if p1["max_relative_error"] > 1e-9:
        raise SystemExit("P1 FAILED: utility changed under affine transform")
    if not all(row["nonnegative"] for row in p2.values()):
        raise SystemExit("P2 FAILED: negative adaptive-value (Jensen violation)")


if __name__ == "__main__":
    main()

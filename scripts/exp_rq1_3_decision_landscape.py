"""EXP-RQ1.3 — Decision Landscape Analysis.

Per the reframing proposed 2026-09-26: before arguing that a learned policy
is valuable, we need to establish whether the benchmark contains
"decision opportunity" — i.e., whether the utility difference between
the best and second-best action is real and state-dependent.

For each (problem, phase, seed) record in the 5-family dataset, compute:

  - U(s, a): utility of action a in state s (already in rollout_results)
  - ΔU(s): max_a U(s, a) - second_max_a U(s, a)
  - R_sur(s): U(s, a*) - U(s, surrogate_acquisition)
  - R_stop(s): U(s, a*) - U(s, stop)
  - H(U)(s): Shannon entropy of the softmax-normalized utility vector

Hypothesis (to test, NOT to assume):
  - ΔU is small on Branin/Hartmann/SixHumpCamel/Ackley — utility landscape is
    flat → no decision opportunity → no policy can add much value.
  - ΔU is large on Rastrigin — utility landscape is peaked → decision
    opportunity exists → explains why LLM beats AlwaysSurrogate there.

Output: per-family distribution of these metrics, plus a CSV for the
optional RandomForest probe in the next experiment.
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

from opt_agent.dataset import iter_jsonl, phase_of

ACTIONS = (
    "global_exploration", "local_exploitation", "uncertainty_sampling",
    "error_guided_sampling", "boundary_sampling", "surrogate_acquisition",
    "retrain_surrogate", "stop",
)


def utility_vector(record: dict) -> dict[str, float]:
    """Extract U(s, a) for all 8 actions from a DecisionRecord."""
    rr = record.get("rollout_results") or {}
    out = {}
    for action in ACTIONS:
        cell = rr.get(action, {})
        u = cell.get("utility") if isinstance(cell, dict) else None
        if u is None or not math.isfinite(float(u)):
            u = float("nan")
        out[action] = float(u)
    return out


def metrics_for_state(u: dict[str, float]) -> dict:
    """Compute ΔU, R_sur, R_stop, H(U) from a utility vector."""
    vals = [(a, v) for a, v in u.items() if math.isfinite(v)]
    if not vals:
        return {"ΔU": float("nan"), "R_sur": float("nan"),
                "R_stop": float("nan"), "H_U": float("nan"),
                "a_star": None, "u_star": float("nan")}
    vals.sort(key=lambda x: x[1], reverse=True)
    a_star, u_star = vals[0]
    second_u = vals[1][1] if len(vals) > 1 else u_star
    u_sur = u.get("surrogate_acquisition", float("nan"))
    u_stop = u.get("stop", float("nan"))
    # Entropy over softmax-normalized utilities.
    if u_star == vals[-1][1]:  # all equal
        H = math.log(len(vals))
    else:
        shifted = [v - u_star for _, v in vals]
        exps = [math.exp(s) for s in shifted]
        Z = sum(exps)
        ps = [e / Z for e in exps]
        H = -sum(p * math.log(p) for p in ps if p > 0)
    return {
        "ΔU": u_star - second_u,
        "R_sur": u_star - u_sur,
        "R_stop": u_star - u_stop,
        "H_U": H,
        "a_star": a_star,
        "u_star": u_star,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs" / "exp_rq1_3_landscape.json")
    args = parser.parse_args()

    records = list(iter_jsonl(args.dataset))
    print(f"Loaded {len(records)} records from {args.dataset}")

    # Aggregate per (family, phase).
    by_fam_phase: dict[tuple[str, str], list[dict]] = defaultdict(list)
    all_rows: list[dict] = []
    for r in records:
        u = utility_vector(r.__dict__)
        m = metrics_for_state(u)
        family = r.problem["family"]
        # Use shared phase_of from validation.py (budget-ratio based).
        phase = phase_of(r)
        row = {"family": family, "phase": phase,
               "record_id": r.id, "seed": int(r.problem["seed"])}
        row.update(m)
        all_rows.append(row)
        by_fam_phase[(family, phase)].append(row)

    summary = {}
    for (fam, phase), rows in sorted(by_fam_phase.items()):
        def stats(key):
            vals = [r[key] for r in rows if math.isfinite(r[key])]
            if not vals:
                return None
            return {
                "n": len(vals),
                "mean": statistics.fmean(vals),
                "median": statistics.median(vals),
                "min": min(vals),
                "max": max(vals),
            }
        summary[f"{fam}/{phase}"] = {
            "n_records": len(rows),
            "ΔU": stats("ΔU"),
            "R_sur": stats("R_sur"),
            "R_stop": stats("R_stop"),
            "H_U": stats("H_U"),
        }

    # Print per-family aggregates (collapsed across phases).
    print()
    print("=== Per-family aggregates (across phases) ===")
    print(f"{'family':14s} {'n':>4s} "
          f"{'ΔU mean':>10s} {'ΔU med':>10s} "
          f"{'R_sur mean':>11s} {'R_stop mean':>12s} "
          f"{'H_U mean':>10s}")
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_fam[r["family"]].append(r)
    for fam in sorted(by_fam):
        rows = by_fam[fam]
        n = len(rows)
        dUs = [r["ΔU"] for r in rows if math.isfinite(r["ΔU"])]
        Rs = [r["R_sur"] for r in rows if math.isfinite(r["R_sur"])]
        Rst = [r["R_stop"] for r in rows if math.isfinite(r["R_stop"])]
        Hs = [r["H_U"] for r in rows if math.isfinite(r["H_U"])]
        print(f"{fam:14s} {n:>4d} "
              f"{statistics.fmean(dUs):>10.4f} {statistics.median(dUs):>10.4f} "
              f"{statistics.fmean(Rs):>11.4f} {statistics.fmean(Rst):>12.4f} "
              f"{statistics.fmean(Hs):>10.4f}")

    print()
    print("=== Per (family, phase) ===")
    for key in sorted(summary):
        s = summary[key]
        dU = s["ΔU"] or {}
        Rs = s["R_sur"] or {}
        Rst = s["R_stop"] or {}
        Hu = s["H_U"] or {}
        print(f"  {key:30s} n={s['n_records']:>3d} "
              f"ΔU_mean={dU.get('mean', float('nan')):.4f} "
              f"R_sur_mean={Rs.get('mean', float('nan')):.4f} "
              f"R_stop_mean={Rst.get('mean', float('nan')):.4f} "
              f"H_U_mean={Hu.get('mean', float('nan')):.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "dataset": str(args.dataset),
        "n_records": len(records),
        "summary": summary,
        "per_record": all_rows[:200],  # sample; full too big for json
    }, indent=2, default=str) + "\n")
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

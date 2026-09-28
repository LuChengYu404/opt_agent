"""EXP: trajectory-granularity learnability probe (2026-09-28).

Branch ``exp/lenz-state-jev-presentation`` — the survey's option-2 question:
does the FULL trial log (D_t, reconstructable from record metadata) contain
state→action signal that the frozen 15-scalar summary misses?

Protocol (per family, leave-one-seed-out, mirroring exp_rq1_3_rf_probe):
multi-output RandomForestRegressor over action utilities; three feature
sets compared on the SAME folds:
  - frozen15 : the frozen 15-dim state features (_shared_rq_features);
  - traj_ext  : frozen15 + trajectory features from D_t (per-dim unit-cube
                coverage stats, best-x location, pairwise-distance
                quantiles, top-gap / improvement-curve stats);
  - traj_only : the trajectory features alone.

Metrics: top-1 agreement with the oracle argmax, and one-step decision
regret U(a*) − U(â) over raw gold utilities. d-dependent features are fine
because models are per-family (dimension is constant within a family).

Usage:
    .venv/bin/python scripts/exp_lenz_gate2_probe.py [--dataset ...]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _shared_rq_features import ACTIONS, extract_features  # noqa: E402
from opt_agent.dataset import iter_jsonl, phase_of_ratio  # noqa: E402

FAMILIES = ("ackley", "branin", "hartmann", "rastrigin", "sixhumpcamel")


def _trajectory_features(meta: dict) -> list[float]:
    """d-dependent trajectory features from the replay metadata (D_t)."""
    X = meta["backend_X"]
    Y = [row[0] for row in meta["backend_objectives"]]
    bounds = meta["backend_problem_bounds"]
    d = len(bounds)
    lo = [b[0] for b in bounds]
    span = [b[1] - b[0] for b in bounds]
    unit = [[(x[j] - lo[j]) / span[j] for j in range(d)] for x in X]

    # Per-dimension location/coverage of the sampled set.
    per_dim_mean = [statistics.fmean(u[j] for u in unit) for j in range(d)]
    per_dim_std = [statistics.pstdev(u[j] for u in unit) for j in range(d)]

    best_i = max(range(len(Y)), key=lambda i: Y[i])
    best_unit = unit[best_i]

    # Pairwise-distance quantiles (unit cube, d-invariant statistics).
    dists = [
        sum((unit[i][j] - unit[k][j]) ** 2 for j in range(d)) ** 0.5
        for i in range(len(unit)) for k in range(i + 1, len(unit))
    ] or [0.0]
    dists_sorted = sorted(dists)

    def _q(p):
        return dists_sorted[min(len(dists_sorted) - 1, int(p * len(dists_sorted)))]

    # Objective-curve statistics.
    y_sorted = sorted(Y, reverse=True)
    top_gap = (y_sorted[0] - y_sorted[1]) if len(Y) > 1 else 0.0
    y_std = statistics.pstdev(Y)
    best_pos = best_i / max(1, len(Y) - 1)  # relative time of the incumbent
    last3_trend = (
        statistics.fmean(Y[-3:]) - statistics.fmean(Y[-6:-3] or Y[-3:])
        if len(Y) >= 4 else 0.0
    )

    return [
        *per_dim_mean, *per_dim_std, *best_unit,
        _q(0.1), _q(0.5), _q(0.9),
        top_gap, y_std, best_pos, last3_trend,
    ]


def _gold_utilities(record) -> dict[str, float] | None:
    gold = {
        aid: res["utility"]
        for aid, res in record.rollout_results.items()
        if res["status"] == "valid"
    }
    return gold if len(gold) >= 2 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=Path("runs/phase8_6d_h3_20seed_5family_s100_119.jsonl"))
    parser.add_argument("--json-out", type=Path,
                        default=Path("runs/exp_lenz_gate2_probe.json"))
    args = parser.parse_args()

    records = [r for r in iter_jsonl(args.dataset)]
    out = {"dataset": str(args.dataset), "families": {}}

    for fam in FAMILIES:
        rows = [r for r in records if r.problem["family"] == fam]
        seeds = sorted({int(r.problem["seed"]) for r in rows})

        def feats(r, which):
            phase = phase_of_ratio(
                float(r.state["resource"]["remaining_budget_ratio"]))
            frozen = extract_features(r.state, phase)
            traj = _trajectory_features(r.metadata)
            if which == "frozen15":
                return frozen
            if which == "traj_ext":
                return frozen + traj
            return traj

        acc = {w: {"correct": 0, "n": 0, "regrets": []} for w in
               ("frozen15", "traj_ext", "traj_only")}

        from sklearn.ensemble import RandomForestRegressor

        for held_seed in seeds:
            train = [r for r in rows if int(r.problem["seed"]) != held_seed]
            test = [r for r in rows if int(r.problem["seed"]) == held_seed]
            for which in acc:
                Xtr, Ytr, valid_train = [], [], []
                for r in train:
                    gold = _gold_utilities(r)
                    if gold is None or set(gold) != set(ACTIONS):
                        continue
                    Xtr.append(feats(r, which))
                    Ytr.append([gold[a] for a in ACTIONS])
                    valid_train.append(r)
                if not Xtr or not test:
                    continue
                rf = RandomForestRegressor(
                    n_estimators=200, max_depth=8, min_samples_leaf=5,
                    random_state=0, n_jobs=8,
                )
                rf.fit(Xtr, Ytr)
                for r in test:
                    gold = _gold_utilities(r)
                    if gold is None:
                        continue
                    pred = dict(zip(ACTIONS, rf.predict([feats(r, which)])[0]))
                    # Restrict the predicted argmax to the record's valid
                    # gold keys (partial-width records: invalid actions are
                    # not in gold; width-agnostic like the RQ2.1 protocol).
                    pred_top = max(gold, key=lambda a: pred[a])
                    gold_top = max(gold, key=lambda a: gold[a])
                    acc[which]["correct"] += int(pred_top == gold_top)
                    acc[which]["n"] += 1
                    u_star = gold[gold_top]
                    acc[which]["regrets"].append(u_star - gold[pred_top])

        fam_out = {}
        for which, a in acc.items():
            if a["n"] == 0:
                continue
            fam_out[which] = {
                "top1_agreement": round(a["correct"] / a["n"], 4),
                "mean_regret": round(statistics.fmean(a["regrets"]), 4),
                "n": a["n"],
            }
        # Paired improvement (traj_ext − frozen15) on top-1, clustered by
        # the shared fold/test structure: simple per-record bootstrap.
        out["families"][fam] = fam_out
        print(fam, json.dumps(fam_out))

    args.json_out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

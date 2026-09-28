"""Gate-4 multiplicity sensitivity (2026-09-28).

Workshop-paper outline limitation #12: per-family Gate-4 CIs are unadjusted
for the 5 comparisons per baseline type. This script recomputes each
per-family seed-clustered bootstrap CI at the Bonferroni-adjusted level
(1 - alpha/5 = 99%) from the stored per-seed gains
(runs/exp_rq1_2_h3_20seed_report.json gate4.rows) and reports which
family-level claims survive.

Method matches validate_phase8_6b._bootstrap_ci: resample seeds with
replacement (cluster = seed), mean of resampled gains, percentile CI;
repetitions=4000, rng seeded per cell.

Usage:
    .venv/bin/python scripts/exp_gate4_multiplicity_sensitivity.py
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FAMILIES = ("ackley", "branin", "hartmann", "rastrigin", "sixhumpcamel")
BASELINES = ("stop", "uniform", "behavior")
N_FAMILIES = len(FAMILIES)


def bootstrap_ci(values: list[float], level: float, repetitions: int, seed: int):
    rng = random.Random(seed)
    n = len(values)
    lo_q = (1.0 - level) / 2.0
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(repetitions)
    )
    return (means[int(lo_q * repetitions)], means[int((1.0 - lo_q) * repetitions)])


def main() -> int:
    src = json.loads((ROOT / "runs/exp_rq1_2_h3_20seed_report.json").read_text())
    rows = src["gate4"]["rows"]
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        for b in BASELINES:
            by_cell[(r["family"], b)].append(float(r["gains"][b]))

    out = {"method": "seed-clustered bootstrap, Bonferroni alpha/5 -> 99% CI",
           "cells": {}, "survivors_95": {}, "survivors_99": {}}
    for b in BASELINES:
        out["survivors_95"][b] = []
        out["survivors_99"][b] = []
        for i, fam in enumerate(FAMILIES):
            gains = by_cell[(fam, b)]
            ci95 = bootstrap_ci(gains, 0.95, 4000, seed=i)
            ci99 = bootstrap_ci(gains, 0.99, 4000, seed=i)
            out["cells"][f"{fam}/{b}"] = {
                "n_seeds": len(gains),
                "mean_gain": round(statistics.fmean(gains), 4),
                "ci95": [round(x, 4) for x in ci95],
                "ci99_bonferroni": [round(x, 4) for x in ci99],
                "lo_gt_0_95": ci95[0] > 0,
                "lo_gt_0_99": ci99[0] > 0,
            }
            if ci95[0] > 0:
                out["survivors_95"][b].append(fam)
            if ci99[0] > 0:
                out["survivors_99"][b].append(fam)

    dst = ROOT / "runs/exp_gate4_multiplicity_sensitivity.json"
    dst.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {dst}")
    for b in BASELINES:
        print(f"{b}: 95% CI-lo>0 on {len(out['survivors_95'][b])}/5 "
              f"({', '.join(out['survivors_95'][b]) or '-'}); "
              f"99% (Bonferroni) on {len(out['survivors_99'][b])}/5 "
              f"({', '.join(out['survivors_99'][b]) or '-'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

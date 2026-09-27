"""Hierarchical paired analysis of the crossed IOH closed-loop experiment.

Instances are sampled first, then seeds within each sampled instance. The
three-instance panel remains exploratory; intervals are shown as uncertainty
descriptions, not as multiplicity-adjusted hypothesis tests.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def hierarchical_ci(by_instance: dict[int, list[float]], *, repetitions: int = 4000,
                    seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    instances = sorted(by_instance)
    boot = []
    for _ in range(repetitions):
        sample = []
        for _ in instances:
            instance = instances[rng.randrange(len(instances))]
            values = by_instance[instance]
            sample.extend(values[rng.randrange(len(values))] for _ in values)
        boot.append(statistics.fmean(sample))
    boot.sort()
    return [boot[100], boot[3899]]


def analyze(payload: dict) -> dict:
    config = payload["config"]
    runs = payload["runs"]
    expected = (len(config["fids"]) * len(config["dims"])
                * len(config["instances"]) * len(config["seeds"])
                * len(config["policies"]))
    if len(runs) != expected:
        raise ValueError(f"incomplete panel: {len(runs)} of {expected} runs")
    by_key = {}
    for row in runs:
        key = (row["fid"], row["dim"], row["instance"], row["seed"], row["policy"])
        if key in by_key:
            raise ValueError(f"duplicate run: {key}")
        by_key[key] = row
    cells = {}
    for fid in config["fids"]:
        for dim in config["dims"]:
            cell = {"n_instances": len(config["instances"]),
                    "n_seeds_per_instance": len(config["seeds"]),
                    "policies": {}, "paired": {}}
            for policy in config["policies"]:
                values = [by_key[fid, dim, instance, seed, policy]["normalized_regret"]
                          for instance in config["instances"] for seed in config["seeds"]]
                if any(v is None for v in values):
                    raise ValueError("normalized regret missing")
                cell["policies"][policy] = statistics.fmean(values)
            for baseline in [p for p in config["policies"] if p != "rf_all"]:
                by_instance = defaultdict(list)
                for instance in config["instances"]:
                    for seed in config["seeds"]:
                        rf = by_key[fid, dim, instance, seed, "rf_all"]
                        ref = by_key[fid, dim, instance, seed, baseline]
                        if rf["initial_values"] != ref["initial_values"]:
                            raise ValueError("paired initial designs differ")
                        by_instance[instance].append(
                            rf["normalized_regret"] - ref["normalized_regret"]
                        )
                all_diffs = [v for values in by_instance.values() for v in values]
                cell["paired"][baseline] = {
                    "mean_rf_minus_baseline_regret": statistics.fmean(all_diffs),
                    "ci95_instance_then_seed_bootstrap": hierarchical_ci(by_instance),
                    "wins_ties_losses": [sum(v < 0 for v in all_diffs),
                                          sum(v == 0 for v in all_diffs),
                                          sum(v > 0 for v in all_diffs)],
                }
            cells[f"f{fid}/d{dim}"] = cell
    return {
        "config": config,
        "n_runs": len(runs),
        "cells": cells,
        "interpretation": "Exploratory external-transfer stress test. Lower regret is better. "
                          "Within-cell intervals resample instances then seeds; "
                          "only three instances per function/dimension and no "
                          "multiple-comparison correction.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("runs/exp_rq2_ioh_closed_loop_crossed.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/exp_rq2_ioh_crossed_analysis.json"))
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for key, cell in result["cells"].items():
        ref = cell["paired"]["surrogate"]
        print(key, "RF", f"{cell['policies']['rf_all']:.3f}",
              "Surrogate", f"{cell['policies']['surrogate']:.3f}",
              "RF−Surrogate", f"{ref['mean_rf_minus_baseline_regret']:+.3f}",
              "CI", [round(x, 3) for x in ref["ci95_instance_then_seed_bootstrap"]])


if __name__ == "__main__":
    main()

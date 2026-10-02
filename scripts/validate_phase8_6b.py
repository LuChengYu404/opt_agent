"""Report replayable decision-label gates on a DecisionDataset JSONL.

One early decision point per (problem, seed) enters the paired terminal test,
so extra records from a long trajectory do not count as independent trials.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.dataset import (  # noqa: E402
    aggregate_mass,
    gate2_stability,
    iter_jsonl,
    oracle_test_from_record,
    phase_of,
)


def _bootstrap_ci(values: list[float], *, repetitions: int = 2000) -> tuple[float, float]:
    rng = random.Random(0)
    n = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(repetitions)
    )
    return means[int(0.025 * repetitions)], means[int(0.975 * repetitions)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-replicates", type=int, default=5)
    parser.add_argument("--max-actions", type=int, default=30)
    args = parser.parse_args()

    records = list(iter_jsonl(args.dataset))
    if not records:
        parser.error("dataset is empty")
    gate2 = gate2_stability(records)
    gate3 = aggregate_mass(records)

    selected = {}
    for record in records:
        if phase_of(record) == "early":
            key = (record.problem["family"], int(record.problem["seed"]))
            selected.setdefault(key, record)

    oracle_rows = []
    for (family, seed), record in sorted(selected.items()):
        result = oracle_test_from_record(
            record,
            random_replicates=args.random_replicates,
            max_actions=args.max_actions,
            # Archived phase8.6b artifacts predate the h2 AF executor;
            # replayed for analysis with the semantics mismatch acknowledged
            # (guide 4.5 frozen-artifact replay discipline).
            allow_legacy_harness=True,
        )
        oracle_rows.append({
            "family": family,
            "seed": seed,
            "record_id": record.id,
            # Execution provenance (codex round-2 MINOR): these rows replay
            # legacy-harness records through the current executor.
            "replay_execution": "current-harness executor on legacy records (migration analysis)",
            "gold_action": result.gold_action,
            "terminal": result.terminal,
            "gains": result.gains,
        })

    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in oracle_rows:
        by_family[row["family"]].append(row)
    oracle_summary = {}
    for family, rows in sorted(by_family.items()):
        comparisons = {}
        for baseline in ("stop", "uniform", "behavior"):
            values = [row["gains"][baseline] for row in rows]
            lo, hi = _bootstrap_ci(values)
            comparisons[baseline] = {
                "mean_gain": statistics.fmean(values),
                "ci95": [lo, hi],
                "positive_seeds": sum(value > 0 for value in values),
            }
        oracle_summary[family] = {
            "n_seeds": len(rows),
            "comparisons": comparisons,
        }

    gate4_pass = bool(oracle_summary) and all(
        family["n_seeds"] >= 10
        and family["comparisons"]["behavior"]["ci95"][0] > 0
        and family["comparisons"]["uniform"]["ci95"][0] > 0
        and family["comparisons"]["stop"]["ci95"][0] > 0
        for family in oracle_summary.values()
    )
    report = {
        "dataset": str(args.dataset),
        "n_records": len(records),
        "gate2": {
            "pass": gate2.pass_,
            "min_problem_seeds": gate2.total_seeds,
            "min_phase_seeds": gate2.min_phase_seeds,
            "problems": gate2.problems,
        },
        "gate3": {
            "pass": gate3.pass_,
            "adjusted_lift_mean": gate3.adjusted_lift_mean,
            "seed_clustered_ci95": (
                list(gate3.seed_clustered_ci95)
                if gate3.seed_clustered_ci95 is not None else None
            ),
            "seed_rows": gate3.seed_rows,
        },
        "gate4": {
            "pass": gate4_pass,
            "summary": oracle_summary,
            "rows": oracle_rows,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "gate2": report["gate2"]["pass"],
        "gate3": report["gate3"]["pass"],
        "gate4": report["gate4"]["pass"],
        "oracle_summary": oracle_summary,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""EXP-RQ3 mechanism check: deterministic global-then-local schedule baseline.

The per-step action-distribution analysis (`runs/exp_rq0_5_action_distribution.md`)
attributes the LLM's Rastrigin advantage to a global-to-local temporal schedule,
and the workshop outline explicitly notes a deterministic schedule baseline has
not been compared. This script runs the predeclared `GlobalThenLocal` control
(10 `global_exploration` decisions, then `local_exploitation`) on the exact
EXP-RQ3 panel — 5 BOCoDe families x seeds 100-109, budget 30, n_init 5 — and
verifies each trajectory's initial Sobol values match the recorded sweep rows,
so the results pair exactly against the existing `rf` and `llm` arms in
`runs/exp_rq3_router_sweep*.json` without re-running any LLM calls.

Example:
  uv run python scripts/exp_rq3_schedule_baseline.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _shared_rq_features import FAMILIES  # noqa: E402
from exp_rq1_rq2_closed_loop import PROBLEM_IDS, GlobalThenLocal  # noqa: E402
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend  # noqa: E402
from opt_agent.loop import DecisionLoop  # noqa: E402

SWEEP_FILES = (
    ROOT / "runs/exp_rq3_router_sweep.json",
    ROOT / "runs/exp_rq3_router_sweep_seeds103_109.json",
)


def run_one(family: str, seed: int, budget: int) -> dict:
    torch.set_num_threads(1)
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(problem_id=PROBLEM_IDS[family],
                                     seed=seed, n_init=5))
    initial = backend.get_history().objectives[:, 0].tolist()
    policy = GlobalThenLocal(seed)
    start = time.monotonic()
    trajectory = DecisionLoop(backend, total_budget=budget).run(
        policy, max_rounds=budget + 5,
    )
    elapsed = time.monotonic() - start
    history = backend.get_history()
    values = history.objectives[:, 0].tolist()
    if len(values) > budget:
        raise ValueError("objective budget exceeded")
    return {
        "family": family, "seed": seed, "budget": budget,
        "policy": "global_then_local", "system2_mode": "none",
        "initial_values": initial,
        "terminal_best": max(values), "n_evaluations": len(values),
        "n_steps": len(trajectory.steps),
        "termination": trajectory.termination_reason.value,
        "actions": [step.action.type.value for step in trajectory.steps],
        "routes": ["schedule"] * len(trajectory.steps),
        "llm_calls": 0, "llm_tokens": 0, "llm_fallbacks": 0,
        "elapsed_s": elapsed,
    }


def paired_bootstrap(diffs: list[float], n_boot: int = 10_000,
                     seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(diffs[rng.randrange(len(diffs))]
                                    for _ in range(len(diffs)))
                   for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[min(int(0.975 * n_boot), n_boot - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(100, 110)))
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/exp_rq3_schedule_baseline.json")
    args = parser.parse_args()
    if args.budget <= 5 or args.workers < 1:
        parser.error("budget must exceed five initial points and workers must be positive")

    sweeps = []
    for path in SWEEP_FILES:
        sweeps.extend(json.loads(path.read_text())["runs"])
    reference: dict[tuple, dict] = {}
    for row in sweeps:
        for arm in ("rf", "llm"):
            if row["policy"] == arm:
                reference.setdefault((row["family"], row["seed"], arm), row)
    if len(reference) != len(FAMILIES) * len(args.seeds) * 2:
        parser.error(f"expected {len(FAMILIES) * len(args.seeds) * 2} reference "
                     f"rows, found {len(reference)}; run the router sweep first")

    rows = []
    if args.output.exists():
        rows = json.loads(args.output.read_text())["runs"]
    done = {(r["family"], r["seed"]) for r in rows}
    tasks = [(family, seed, args.budget)
             for family in FAMILIES for seed in args.seeds
             if (family, seed) not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save_row(row):
        expected = reference[(row["family"], row["seed"], "rf")]["initial_values"]
        if row["initial_values"] != expected:
            raise ValueError(f"paired-start mismatch for {row['family']} "
                             f"s{row['seed']}: Sobol init differs from sweep")
        rows.append(row)
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps({"runs": rows}, allow_nan=False) + "\n")
        temp.replace(args.output)
        print(f"{row['family']} s{row['seed']} global_then_local "
              f"best={row['terminal_best']:.3f}", flush=True)

    if args.workers == 1 or not tasks:
        for task in tasks:
            save_row(run_one(*task))
    else:
        with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=multiprocessing.get_context("fork")) as pool:
            futures = [pool.submit(run_one, *task) for task in tasks]
            for future in as_completed(futures):
                save_row(future.result())

    by_arm: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for (family, seed, arm), row in reference.items():
        by_arm[(family, arm)][seed] = row
    own: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        own[(row["family"], "schedule")][row["seed"]] = row
    comparison = {}
    for family in FAMILIES:
        for arm in ("rf", "llm"):
            common = sorted(set(own[(family, "schedule")]) & set(by_arm[(family, arm)]))
            if not common:
                continue
            diffs = [own[(family, "schedule")][s]["terminal_best"]
                     - by_arm[(family, arm)][s]["terminal_best"] for s in common]
            lo, hi = paired_bootstrap(diffs)
            wins = sum(1 for d in diffs if d > 0)
            ties = sum(1 for d in diffs if d == 0)
            comparison[f"{family}/schedule_vs_{arm}"] = {
                "n_paired_seeds": len(common),
                "mean_delta": statistics.fmean(diffs),
                "ci95": [lo, hi],
                "wtl_schedule_vs_arm": [wins, ties, len(diffs) - wins - ties],
                "ci_excludes_zero_positive": lo > 0,
                "ci_excludes_zero_negative": hi < 0,
            }
    report = {
        "policy": "global_then_local (10 global decisions, then local)",
        "n_runs": len(rows), "seeds": args.seeds, "budget": args.budget,
        "paired_against": [str(p) for p in SWEEP_FILES],
        "comparisons": comparison,
        "runs": rows,
    }
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output)
    for key, comp in comparison.items():
        lo, hi = comp["ci95"]
        w, t, l = comp["wtl_schedule_vs_arm"]
        flag = ("+" if comp["ci_excludes_zero_positive"]
                else "-" if comp["ci_excludes_zero_negative"] else " ")
        print(f"{flag} {key:36s} n={comp['n_paired_seeds']:2d} "
              f"meanΔ={comp['mean_delta']:+8.3f} CI[{lo:+8.3f},{hi:+8.3f}] "
              f"W/T/L={w}/{t}/{l}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

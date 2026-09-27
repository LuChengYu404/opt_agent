"""EXP-RQ2.2 — Within-function dimension transfer (clean dim×family design).

The previous EXP-RQ2.1 cross-family transfer was confounded by dimension:
each family has a fixed dimension (Branin=2, Hartmann=6, Ackley=10, ...).
This experiment fixes the function and tests whether an RF trained on
two source dimensions transfers to a held-out target dimension of the
same function.

Protocol:
  Train dims: d_train ∈ {2, 10} per function (instances 0, 1).
  Test  dim : d_test = 5 (instance 2, untouched by training).
  Functions: f6 AttractiveSector, f8 Rosenbrock, f15 RotatedRastrigin.
  Seeds    : 100..104 (5 seeds × 3 functions × 4 train cells × 2 train
             dims = 120 trajectories; test: 5 × 3 × 1 = 15 test cells).

  Each training trajectory runs RuleFullBudget to completion; we record
  (state_features → action_id) pairs as supervised labels. The trained
  RF is deployed on d_test against Surrogate, Local, and RuleFullBudget.

Primary endpoint: paired normalized regret at budget 30, RF vs Surrogate,
RF vs Local, RF vs RuleFullBudget, by (function, test dim). Lower is
better.

A non-trivial `rf_vs_RuleFullBudget` win/loss pattern across functions
would imply the learned state-conditional policy is *not* just a noisy
copy of RuleFullBudget.

Example:
  uv run python scripts/exp_rq2_2_ioh_within_function_dim_transfer.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import ioh
from exp_rq1_rq2_closed_loop import (  # noqa: E402
    FixedPolicy, RFPolicy, RuleFullBudget, fit_models, paired_ci,
)
from ioh_backend import IOHBackend  # noqa: E402
from opt_agent.actions.factory import candidate_actions_for_state  # noqa: E402
from opt_agent.dataset.validation import phase_of_ratio  # noqa: E402
from opt_agent.loop import DecisionLoop  # noqa: E402

from _shared_rq_features import ACTIONS, extract_features  # noqa: E402

# Multi-output regression label: 1.0 for the action actually chosen, 0 elsewhere.
POLICIES = ("rf", "rule_full", "surrogate", "local")
TRAIN_DIMS = (2, 10)
TEST_DIM = 5
FIDS = (6, 8, 15)
TRAIN_INSTANCES = (0, 1)
TEST_INSTANCE = 2
_WORKER_MODELS: dict = {}


def _feature_row(state, phase: str) -> list[float]:
    return extract_features(asdict(state), phase)


def _label_row(actions: list[str], chosen: str) -> list[float]:
    return [1.0 if action == chosen else 0.0 for action in ACTIONS]


def run_train_one(fid: int, dim: int, instance: int, seed: int,
                  budget: int) -> dict:
    """Run RuleFullBudget to completion, save per-step (state, action) pairs."""
    torch.set_num_threads(1)
    problem = ioh.get_problem(fid=fid, dimension=dim, instance=instance,
                              problem_class=ioh.ProblemClass.BBOB)
    backend = IOHBackend()
    backend.initialize(problem, n_init=5, seed=seed)
    initial = backend.get_history().objectives[:, 0].tolist()

    captured: list[dict] = []
    policy = RuleFullBudget(seed=seed)
    original_select = policy.select_action

    def wrapped_select(state):
        action = original_select(state)
        phase = phase_of_ratio(state.resource.remaining_budget_ratio)
        captured.append({
            "features": _feature_row(state, phase),
            "phase": phase,
            "action": action.type.value,
        })
        return action

    policy.select_action = wrapped_select  # type: ignore[assignment]
    DecisionLoop(backend, total_budget=budget).run(policy, max_rounds=budget + 5)
    return {
        "fid": fid, "dim": dim, "instance": instance, "seed": seed,
        "budget": budget, "policy": "rule_full",
        "initial_values": initial,
        "captured_steps": captured,
        "n_steps": len(captured),
    }


def run_test_one(fid: int, instance: int, seed: int, budget: int,
                 policy_name: str, models: dict) -> dict:
    """Deploy the trained RF on the held-out (function, test dim, test instance)."""
    torch.set_num_threads(1)
    problem = ioh.get_problem(fid=fid, dimension=TEST_DIM, instance=instance,
                              problem_class=ioh.ProblemClass.BBOB)
    backend = IOHBackend()
    backend.initialize(problem, n_init=5, seed=seed)
    initial = backend.get_history().objectives[:, 0].tolist()
    if policy_name == "rf":
        policy = RFPolicy(models, seed)
    elif policy_name == "rule_full":
        policy = RuleFullBudget(seed=seed)
    else:
        action_id = {"surrogate": "surrogate_acquisition",
                     "local": "local_exploitation"}[policy_name]
        policy = FixedPolicy(action_id, seed)
    DecisionLoop(backend, total_budget=budget).run(policy, max_rounds=budget + 5)
    values = backend.get_history().objectives[:, 0].tolist()
    initial_best = max(initial)
    best = max(values)
    optimum = -float(problem.optimum.y) if problem.optimum is not None else None
    denom = optimum - initial_best if optimum and optimum != initial_best else None
    regret = ((optimum - best) / denom if denom and denom > 0 else None)
    return {
        "fid": fid, "dim": TEST_DIM, "instance": instance, "seed": seed,
        "budget": budget, "policy": policy_name,
        "initial_values": initial,
        "terminal_best_max": best,
        "optimum_max": optimum,
        "normalized_regret": regret,
        "n_evaluations": len(values),
    }


def _train_worker(task):
    fid, dim, instance, seed, budget = task
    return run_train_one(fid, dim, instance, seed, budget)


def _test_worker(task):
    fid, instance, seed, budget, policy, models = task
    return run_test_one(fid, instance, seed, budget, policy, models)


def fit_ioh_models(rows: list[dict]):
    X, Y = [], []
    for row in rows:
        for step in row["captured_steps"]:
            X.append(step["features"])
            Y.append(_label_row(ACTIONS, step["action"]))
    if not X:
        raise ValueError("no captured training steps")
    family_indices = [0] * len(X)
    return fit_models(X, Y, family_indices, None)


def summarize_test(rows: list[dict]) -> dict:
    lookup = {(r["fid"], r["dim"], r["instance"], r["seed"], r["policy"]): r
              for r in rows}
    cells = {}
    for fid in sorted({r["fid"] for r in rows}):
        cell = {"fids": fid, "policies": {}, "paired": {}}
        for policy in POLICIES:
            values = [lookup[fid, TEST_DIM, TEST_INSTANCE, seed, policy]["normalized_regret"]
                      for seed in sorted({r["seed"] for r in rows
                                          if r["fid"] == fid and r["policy"] == policy})]
            cell["policies"][policy] = statistics.fmean(values)
        for baseline in [p for p in POLICIES if p != "rf"]:
            diffs = []
            for seed in sorted({r["seed"] for r in rows if r["fid"] == fid}):
                rf = lookup[fid, TEST_DIM, TEST_INSTANCE, seed, "rf"]
                ref = lookup[fid, TEST_DIM, TEST_INSTANCE, seed, baseline]
                if rf["initial_values"] != ref["initial_values"]:
                    raise ValueError("paired starts differ")
                diffs.append(rf["normalized_regret"] - ref["normalized_regret"])
            cell["paired"][baseline] = {
                "mean_rf_minus_baseline_regret": statistics.fmean(diffs),
                "ci95_paired_bootstrap": paired_ci(diffs),
                "wins_ties_losses": [sum(v < 0 for v in diffs),
                                      sum(v == 0 for v in diffs),
                                      sum(v > 0 for v in diffs)],
            }
        cells[f"f{fid}/d{TEST_DIM}"] = cell
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/exp_rq2_2_ioh_within_function_dim_transfer.json")
    args = parser.parse_args()
    if args.budget <= 5 or args.workers < 1:
        parser.error("budget must exceed 5 and workers must be positive")
    torch.set_num_threads(1)
    np.random.seed(0)
    train_tasks = [(fid, dim, instance, seed, args.budget)
                   for fid in FIDS for dim in TRAIN_DIMS
                   for instance in TRAIN_INSTANCES for seed in args.seeds]
    print(f"Training rollouts: {len(train_tasks)} trajectories "
          f"(3 functions × {len(TRAIN_DIMS)} dims × {len(TRAIN_INSTANCES)} "
          f"instances × {len(args.seeds)} seeds)")
    train_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("fork")) as pool:
        futures = [pool.submit(_train_worker, t) for t in train_tasks]
        for future in as_completed(futures):
            row = future.result()
            train_rows.append(row)
            print(f"train f{row['fid']} d{row['dim']} i{row['instance']} "
                  f"s{row['seed']}: {row['n_steps']} captured steps",
                  flush=True)
    models, n_train = fit_ioh_models(train_rows)
    print(f"Trained RF on {n_train} state-action pairs from "
          f"{len(train_rows)} trajectories")
    test_tasks = [(fid, TEST_INSTANCE, seed, args.budget, policy, models)
                  for fid in FIDS for seed in args.seeds for policy in POLICIES]
    print(f"Test rollouts: {len(test_tasks)} trajectories "
          f"(3 functions × {len(args.seeds)} seeds × {len(POLICIES)} policies)")
    test_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("fork")) as pool:
        futures = [pool.submit(_test_worker, t) for t in test_tasks]
        for future in as_completed(futures):
            row = future.result()
            test_rows.append(row)
            print(f"test f{row['fid']} d{TEST_DIM} s{row['seed']} "
                  f"{row['policy']}: regret={row['normalized_regret']:.4f}",
                  flush=True)
    config = {
        "fids": list(FIDS),
        "train_dims": list(TRAIN_DIMS),
        "train_instances": list(TRAIN_INSTANCES),
        "test_dim": TEST_DIM,
        "test_instance": TEST_INSTANCE,
        "seeds": args.seeds,
        "budget": args.budget,
        "n_train_trajectories": len(train_rows),
        "n_train_steps": n_train,
        "policies": list(POLICIES),
        "model": "RF 200 trees, depth 8, leaf 5; EXP-RQ2.1 frozen features/targets",
    }
    summary = summarize_test(test_rows)
    args.output.write_text(json.dumps({
        "config": config, "train_rollouts": train_rows, "test_runs": test_rows,
        "summary": summary,
    }, indent=2, allow_nan=False) + "\n")
    print(f"\n=== Per-function test cells (test_dim={TEST_DIM}, "
          f"test_instance={TEST_INSTANCE}, n={len(args.seeds)}) ===")
    for key, cell in summary.items():
        print(f"\n{key}:")
        for policy, mean in cell["policies"].items():
            print(f"  {policy:10s} mean regret = {mean:.4f}")
        for baseline, paired in cell["paired"].items():
            ci = paired["ci95_paired_bootstrap"]
            wtl = paired["wins_ties_losses"]
            print(f"  rf − {baseline:10s}: {paired['mean_rf_minus_baseline_regret']:+.4f} "
                  f"CI [{ci[0]:+.4f}, {ci[1]:+.4f}] W/T/L {wtl}")
    print(f"\nSaved {len(test_rows)} test runs to {args.output}")


if __name__ == "__main__":
    main()
"""Frozen BOCoDe-trained RF on unseen IOH functions, dimensions, instances.

This is a closed-loop transfer test. All RF training uses the BOCoDe 20-seed
dataset; no IOH outcomes or labels enter model fitting. Results are paired by
(function, dimension, instance, seed) and include anytime performance.

Example:
  uv run python scripts/exp_rq2_ioh_closed_loop.py \
      --fids 6 8 15 --dims 2 5 10 --instances 2 3 --seeds 100 101
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
from pathlib import Path

import ioh
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from exp_rq1_rq2_closed_loop import (  # noqa: E402
    RFPolicy, FixedPolicy, RuleFullBudget, GlobalThenLocal,
    fit_models, paired_ci,
)
from exp_rq2_1_cross_family_transfer import build_dataset  # noqa: E402
from ioh_backend import IOHBackend  # noqa: E402
from opt_agent.dataset import read_jsonl  # noqa: E402
from opt_agent.loop import DecisionLoop  # noqa: E402


POLICIES = ("rf_all", "rule_full", "surrogate", "local", "global_then_local")
_WORKER_MODELS: dict = {}


def _run_worker(task: tuple[int, int, int, int, int, str]) -> dict:
    fid, dim, instance, seed, budget, policy = task
    torch.set_num_threads(1)
    return run_one(fid, dim, instance, seed, budget, policy, _WORKER_MODELS)


def run_one(fid: int, dim: int, instance: int, seed: int, budget: int,
            policy_name: str, models: dict) -> dict:
    problem = ioh.get_problem(fid=fid, dimension=dim, instance=instance,
                              problem_class=ioh.ProblemClass.BBOB)
    backend = IOHBackend()
    backend.initialize(problem, n_init=5, seed=seed)
    initial = backend.get_history().objectives[:, 0].tolist()
    if policy_name == "rf_all":
        policy = RFPolicy(models, seed)
    elif policy_name == "rule_full":
        policy = RuleFullBudget(seed=seed)
    elif policy_name == "global_then_local":
        policy = GlobalThenLocal(seed)
    else:
        policy = FixedPolicy({"surrogate": "surrogate_acquisition",
                              "local": "local_exploitation"}[policy_name], seed)
    trajectory = DecisionLoop(backend, total_budget=budget).run(
        policy, max_rounds=budget + 5,
    )
    values = backend.get_history().objectives[:, 0].tolist()
    initial_best = max(initial)
    best = max(values)
    optimum = -float(problem.optimum.y) if problem.optimum is not None else None
    denom = optimum - initial_best if optimum is not None else None
    regret = ((optimum - best) / denom if denom and denom > 0 else None)
    anytime = []
    incumbent = initial_best
    for value in values[5:]:
        incumbent = max(incumbent, value)
        anytime.append(incumbent)
    anytime.extend([incumbent] * (budget - 5 - len(anytime)))
    return {
        "fid": fid, "function": problem.meta_data.name, "dim": dim,
        "instance": instance, "seed": seed, "budget": budget,
        "policy": policy_name, "initial_values": initial,
        "initial_best_max": initial_best, "terminal_best_max": best,
        "optimum_max": optimum, "normalized_regret": regret,
        "anytime_auc_gain": statistics.fmean(anytime) - initial_best,
        "n_evaluations": len(values), "n_steps": len(trajectory.steps),
        "termination": trajectory.termination_reason.value,
        "actions": [step.action.type.value for step in trajectory.steps],
    }


def summarize(rows: list[dict]) -> dict:
    lookup = {(r["fid"], r["dim"], r["instance"], r["seed"], r["policy"]): r
              for r in rows}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["fid"], row["dim"], row["policy"])].append(row)
    out = {}
    for (fid, dim, policy), group in sorted(groups.items()):
        result = {
            "n_trajectories": len(group),
            "mean_normalized_regret": statistics.fmean(
                r["normalized_regret"] for r in group
                if r["normalized_regret"] is not None
            ),
            "mean_evaluations": statistics.fmean(r["n_evaluations"] for r in group),
            "paired": {},
        }
        if policy == "rf_all":
            for baseline in POLICIES[1:]:
                references = [lookup.get((fid, dim, r["instance"], r["seed"], baseline))
                              for r in group]
                if any(ref is None for ref in references):
                    continue
                if any(r["initial_values"] != ref["initial_values"]
                       for r, ref in zip(group, references)):
                    raise ValueError("IOH paired starts differ")
                diffs = [r["normalized_regret"] - ref["normalized_regret"]
                         for r, ref in zip(group, references)]
                result["paired"][baseline] = {
                    "mean_regret_difference": statistics.fmean(diffs),
                    "ci95_paired_observation_bootstrap": paired_ci(diffs),
                    "wins_ties_losses": [sum(v < 0 for v in diffs),
                                          sum(v == 0 for v in diffs),
                                          sum(v > 0 for v in diffs)],
                }
        out[f"f{fid}/d{dim}/{policy}"] = result
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "runs/phase8_6d_h3_20seed_5family.jsonl")
    parser.add_argument("--fids", nargs="+", type=int, default=[6, 8, 15])
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 5, 10])
    parser.add_argument("--instances", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--policies", nargs="+", choices=POLICIES,
                        default=list(POLICIES))
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/exp_rq2_ioh_closed_loop.json")
    args = parser.parse_args()
    if args.budget <= 5 or not all((args.fids, args.dims, args.instances, args.seeds)):
        parser.error("nonempty panel and budget > 5 required")
    if args.workers < 1:
        parser.error("workers must be positive")
    torch.set_num_threads(1)
    records = [r.__dict__ for r in read_jsonl(args.dataset, version_policy="warn")]
    X, Y, family_indices, _, _ = build_dataset(records)
    models, n_train = fit_models(X, Y, family_indices, None)
    config = {
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "training_families": sorted({r["problem"]["family"] for r in records}),
        "n_train": n_train, "fids": args.fids, "dims": args.dims,
        "instances": args.instances, "seeds": args.seeds, "budget": args.budget,
        "policies": args.policies,
        "model": "RF 200 trees, depth 8, leaf 5; EXP-RQ2.1 frozen features/targets",
    }
    rows = []
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous["config"] != config:
            parser.error("existing output has different configuration")
        rows = previous["runs"]
    done = {(r["fid"], r["dim"], r["instance"], r["seed"], r["policy"])
            for r in rows}
    tasks = [(fid, dim, instance, seed, args.budget, policy)
             for fid in args.fids for dim in args.dims
             for instance in args.instances for seed in args.seeds
             for policy in args.policies
             if (fid, dim, instance, seed, policy) not in done]

    def save_row(row: dict) -> None:
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = args.output.with_suffix(args.output.suffix + ".tmp")
        checkpoint.write_text(json.dumps({
            "config": config, "runs": rows,
        }, allow_nan=False) + "\n")
        checkpoint.replace(args.output)
        print(f"f{row['fid']} d{row['dim']} i{row['instance']} "
              f"s{row['seed']} {row['policy']}: "
              f"regret={row['normalized_regret']:.4f}", flush=True)

    if args.workers == 1:
        for fid, dim, instance, seed, budget, policy in tasks:
            save_row(run_one(fid, dim, instance, seed, budget, policy, models))
    else:
        if "fork" not in multiprocessing.get_all_start_methods():
            parser.error("parallel mode requires fork for copy-on-write RF models")
        global _WORKER_MODELS
        _WORKER_MODELS = models
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("fork")) as pool:
            futures = [pool.submit(_run_worker, task) for task in tasks]
            for future in as_completed(futures):
                save_row(future.result())
    policy_order = {policy: i for i, policy in enumerate(args.policies)}
    rows.sort(key=lambda r: (r["fid"], r["dim"], r["instance"], r["seed"],
                             policy_order[r["policy"]]))
    args.output.write_text(json.dumps({
        "config": config, "runs": rows, "summary": summarize(rows),
    }, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()

"""Frozen RF policies in complete BO loops on seeds absent from the labels.

The model and features match EXP-RQ2.1. ``rf_all`` trains on all five
development families; ``rf_transfer`` leaves the evaluated family out.
Neither policy observes rollout utilities during deployment. The primary
endpoint is paired terminal best value at a fixed evaluation budget.

Example:
  uv run python scripts/exp_rq1_rq2_closed_loop.py --seeds 100 101 102
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _shared_rq_features import ACTIONS, FAMILIES, extract_features  # noqa: E402
from exp_rq2_1_cross_family_transfer import build_dataset  # noqa: E402
from opt_agent.actions.factory import candidate_actions_for_state  # noqa: E402
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend  # noqa: E402
from opt_agent.dataset import read_jsonl  # noqa: E402
from opt_agent.dataset.validation import phase_of_ratio  # noqa: E402
from opt_agent.loop import DecisionLoop  # noqa: E402
from opt_agent.loop.policies import RulePolicy  # noqa: E402


POLICIES = (
    "rf_all", "rf_transfer", "rf_all_no_retrain", "rf_transfer_no_retrain",
    "rule_full", "surrogate", "local", "global_then_local",
)
PROBLEM_IDS = {family: family.title().replace("Sixhumpcamel", "SixHumpCamel")
               for family in FAMILIES}
_WORKER_MODELS: dict = {}


def _run_worker(task: tuple[str, int, int, str]) -> dict:
    family, seed, budget, name = task
    torch.set_num_threads(1)
    return run_one(family, seed, budget, name, _WORKER_MODELS[family])


class RuleFullBudget(RulePolicy):
    LOW_BUDGET_THRESHOLD = 0


class FixedPolicy:
    def __init__(self, action_id: str, seed: int) -> None:
        self.action_id, self.seed = action_id, seed

    def select_action(self, state):
        return candidate_actions_for_state(state)[self.action_id]


class GlobalThenLocal:
    """Predeclared control: 10 global decisions, then local decisions."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.decisions = 0

    def select_action(self, state):
        choices = candidate_actions_for_state(state)
        action_id = "global_exploration" if self.decisions < 10 else "local_exploitation"
        self.decisions += 1
        return choices[action_id]


class RFPolicy:
    def __init__(self, models: dict, seed: int, *, allow_retrain: bool = True) -> None:
        self.models, self.seed = models, seed
        self.allow_retrain = allow_retrain

    def select_action(self, state):
        choices = candidate_actions_for_state(state)
        if not self.allow_retrain and len(choices) > 1:
            choices.pop("retrain_surrogate", None)
        phase = phase_of_ratio(state.resource.remaining_budget_ratio)
        features = extract_features(asdict(state), phase)
        predictions = {a: float(self.models[a].predict([features])[0]) for a in choices}
        return choices[max(predictions, key=predictions.get)]


def fit_models(X: list, Y: list, family_indices: list, excluded_family: str | None):
    from sklearn.ensemble import RandomForestRegressor

    indices = [i for i, fi in enumerate(family_indices)
               if excluded_family is None or FAMILIES[fi] != excluded_family]
    if not indices:
        raise ValueError("empty RF training set")
    models = {}
    for j, action in enumerate(ACTIONS):
        model = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=1,
        )
        model.fit([X[i] for i in indices], [Y[i][j] for i in indices])
        models[action] = model
    return models, len(indices)


def policy_for(name: str, seed: int, models: dict):
    if name == "rf_all":
        return RFPolicy(models["all"], seed)
    if name == "rf_transfer":
        return RFPolicy(models["transfer"], seed)
    if name == "rf_all_no_retrain":
        return RFPolicy(models["all"], seed, allow_retrain=False)
    if name == "rf_transfer_no_retrain":
        return RFPolicy(models["transfer"], seed, allow_retrain=False)
    if name == "rule_full":
        return RuleFullBudget(seed=seed)
    if name == "surrogate":
        return FixedPolicy("surrogate_acquisition", seed)
    if name == "local":
        return FixedPolicy("local_exploitation", seed)
    if name == "global_then_local":
        return GlobalThenLocal(seed)
    raise ValueError(name)


def run_one(family: str, seed: int, budget: int, name: str, models: dict) -> dict:
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(problem_id=PROBLEM_IDS[family], seed=seed, n_init=5))
    initial = backend.get_history().objectives[:, 0].tolist()
    initial_best = max(initial)
    initial_scale = statistics.pstdev(initial) or 1.0
    trajectory = DecisionLoop(backend, total_budget=budget).run(
        policy_for(name, seed, models), max_rounds=budget + 5,
    )
    history = backend.get_history()
    values = history.objectives[:, 0].tolist()
    if len(values) > budget:
        raise ValueError("policy exceeded objective-evaluation budget")
    incumbent = initial_best
    anytime = []
    for value in values[5:]:
        incumbent = max(incumbent, value)
        anytime.append(incumbent)
    # Carry the terminal incumbent forward if a policy stops early. This
    # makes anytime AUC comparable at the same objective-evaluation budget.
    anytime.extend([incumbent] * (budget - 5 - len(anytime)))
    return {
        "family": family, "seed": seed, "budget": budget, "policy": name,
        "initial_values": initial, "initial_best": initial_best,
        "initial_scale": initial_scale, "terminal_best": max(values),
        "normalized_terminal_gain": (max(values) - initial_best) / initial_scale,
        "normalized_anytime_auc": statistics.fmean(
            (best - initial_best) / initial_scale for best in anytime
        ),
        "n_evaluations": len(values), "n_steps": len(trajectory.steps),
        "termination": trajectory.termination_reason.value,
        "actions": [step.action.type.value for step in trajectory.steps],
        "best_by_evaluation": anytime,
    }


def paired_ci(differences: list[float], seed: int = 0) -> list[float]:
    if not differences:
        raise ValueError("no paired observations")
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(statistics.fmean(differences[rng.randrange(n)] for _ in range(n))
                   for _ in range(4000))
    return [means[100], means[3899]]


def summarize(rows: list[dict]) -> dict:
    by_key = {(r["family"], r["seed"], r["budget"], r["policy"]): r for r in rows}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["budget"], row["policy"])].append(row)
    out = {}
    for (family, budget, policy), group in sorted(groups.items()):
        group.sort(key=lambda r: r["seed"])
        paired = {}
        for baseline in ("rule_full", "surrogate", "local", "global_then_local"):
            if policy == baseline:
                continue
            references = [by_key.get((family, r["seed"], budget, baseline)) for r in group]
            if any(r is None for r in references):
                continue
            for row, reference in zip(group, references):
                if row["initial_values"] != reference["initial_values"]:
                    raise ValueError("paired policies did not share initial observations")
            differences = [(row["terminal_best"] - ref["terminal_best"])
                           / row["initial_scale"] for row, ref in zip(group, references)]
            raw_differences = [row["terminal_best"] - ref["terminal_best"]
                               for row, ref in zip(group, references)]
            paired[baseline] = {
                "mean_raw_terminal_gain": statistics.fmean(raw_differences),
                "ci95_raw_seed_bootstrap": paired_ci(raw_differences),
                "mean_normalized_terminal_gain": statistics.fmean(differences),
                "ci95_seed_bootstrap": paired_ci(differences),
                "wins_ties_losses": [sum(v > 0 for v in differences),
                                      sum(v == 0 for v in differences),
                                      sum(v < 0 for v in differences)],
            }
        out[f"{family}/budget{budget}/{policy}"] = {
            "n_seeds": len(group),
            "mean_terminal_best": statistics.fmean(r["terminal_best"] for r in group),
            "mean_normalized_anytime_auc": statistics.fmean(
                r["normalized_anytime_auc"] for r in group),
            "mean_evaluations": statistics.fmean(r["n_evaluations"] for r in group),
            "first_actions": dict(Counter(r["actions"][0] for r in group)),
            "paired": paired,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "runs/phase8_6d_h3_20seed_5family.jsonl")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 105)))
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="independent trajectories to run in parallel")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/exp_rq1_rq2_closed_loop.json")
    args = parser.parse_args()
    if args.budget <= 5 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("budget must exceed five initial points; seeds must be unique")
    if args.workers < 1:
        parser.error("workers must be positive")
    torch.set_num_threads(1)
    records = [r.__dict__ for r in read_jsonl(args.dataset, version_policy="warn")]
    training_seeds = {int(r["problem"]["seed"]) for r in records}
    uses_rf = any(policy.startswith("rf_") for policy in args.policies)
    if uses_rf and training_seeds.intersection(args.seeds):
        parser.error("test seeds overlap with the training dataset")
    X, Y, family_indices, _, _ = build_dataset(records)
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    models = {}
    if "rf_all" in args.policies or "rf_all_no_retrain" in args.policies:
        models["all"], n_all = fit_models(X, Y, family_indices, None)
    else:
        n_all = 0
    results = []
    config = {
        "dataset": str(args.dataset), "dataset_sha256": dataset_hash,
        "train_seeds": sorted(training_seeds), "test_seeds": args.seeds,
        "families": args.families, "budget": args.budget,
        "policies": args.policies, "n_train_all": n_all,
        "model": "RF 200 trees, depth 8, leaf 5; EXP-RQ2.1 frozen features/targets",
        "global_then_local": "first 10 decisions global_exploration, then local_exploitation",
    }
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous["config"] != config:
            parser.error("existing output has a different configuration; choose another path")
        results = previous["runs"]
    done = {(r["family"], r["seed"], r["policy"]) for r in results}
    models_by_family = {}
    for family in args.families:
        family_models = {"all": models.get("all")}
        if "rf_transfer" in args.policies or "rf_transfer_no_retrain" in args.policies:
            family_models["transfer"], _ = fit_models(X, Y, family_indices, family)
        models_by_family[family] = family_models
    tasks = [(family, seed, args.budget, policy)
             for family in args.families for seed in args.seeds
             for policy in args.policies if (family, seed, policy) not in done]

    def save_row(row: dict) -> None:
        results.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Only the parent writes. This is safe even if workers complete out
        # of order, and the atomic replace leaves a valid resume checkpoint.
        checkpoint = args.output.with_suffix(args.output.suffix + ".tmp")
        checkpoint.write_text(json.dumps({
            "config": config, "runs": results,
        }, allow_nan=False) + "\n")
        checkpoint.replace(args.output)
        print(f"{row['family']} seed={row['seed']} {row['policy']}: "
              f"best={row['terminal_best']:.4f} "
              f"evals={row['n_evaluations']} "
              f"first={row['actions'][0]}", flush=True)

    if args.workers == 1:
        for family, seed, budget, policy in tasks:
            save_row(run_one(family, seed, budget, policy, models_by_family[family]))
    else:
        if "fork" not in multiprocessing.get_all_start_methods():
            parser.error("parallel mode requires fork for copy-on-write RF models")
        global _WORKER_MODELS
        _WORKER_MODELS = models_by_family
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("fork")) as pool:
            futures = [pool.submit(_run_worker, task) for task in tasks]
            for future in as_completed(futures):
                save_row(future.result())
    order = {family: i for i, family in enumerate(args.families)}
    policy_order = {policy: i for i, policy in enumerate(args.policies)}
    results.sort(key=lambda r: (order[r["family"]], r["seed"], policy_order[r["policy"]]))
    args.output.write_text(json.dumps({
        "config": config, "runs": results, "summary": summarize(results),
    }, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(results)} runs to {args.output}")


if __name__ == "__main__":
    main()

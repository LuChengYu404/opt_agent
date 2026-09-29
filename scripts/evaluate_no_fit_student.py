"""Train state-only students on Phase 8.6D utility labels and evaluate loops.

Training seeds are 0-7, validation seeds are 8-9. Benchmark seeds should be
new (10+). Counterfactual outcomes are used only to construct training labels;
the evaluated policies receive the observable OptimizationState alone.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.ablate_phase8_6d_utility import _rank  # noqa: E402
from scripts.phase9_baselines import _features  # noqa: E402
from opt_agent.actions.factory import (  # noqa: E402
    ALL_CANDIDATE_SPECS, candidate_actions_for_state,
)
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend  # noqa: E402
from opt_agent.dataset.decision_record import read_jsonl  # noqa: E402
from opt_agent.loop import DecisionLoop  # noqa: E402
from opt_agent.loop.policies import RulePolicy  # noqa: E402
from opt_agent.rollout.utility import UtilityWeights  # noqa: E402


ACTIONS = [spec[0] for spec in ALL_CANDIDATE_SPECS]
WEIGHTS = {
    "current_student": UtilityWeights(),
    "no_fit_student": UtilityWeights(w_fit=0),
}


class RuleFullBudget(RulePolicy):
    """RulePolicy with its early STOP threshold removed."""

    LOW_BUDGET_THRESHOLD = 0


def _bootstrap_ci(values: list[float]) -> list[float]:
    rng = random.Random(0)
    n = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(2000)
    )
    return [means[50], means[1950]]


class StateStudent:
    def __init__(self, model: nn.Module, mean: torch.Tensor,
                 std: torch.Tensor, seed: int) -> None:
        self.model = model.eval()
        self.mean = mean
        self.std = std
        self.seed = seed

    def reset(self) -> None:
        pass

    def select_action(self, state, backend=None):
        candidates = candidate_actions_for_state(state, seed=self.seed)
        features = torch.tensor(
            _features(SimpleNamespace(state=asdict(state)), ACTIONS),
            dtype=torch.float32,
        )
        with torch.no_grad():
            logits = self.model(((features - self.mean) / self.std).unsqueeze(0))[0]
        action_id = max(candidates, key=lambda a: float(logits[ACTIONS.index(a)]))
        return candidates[action_id]


def _train(records, weights: UtilityWeights, seed: int):
    train = [r for r in records if int(r.problem["seed"]) < 8]
    valid = [r for r in records if int(r.problem["seed"]) in (8, 9)]
    if not train or not valid:
        raise ValueError("need training seeds 0-7 and validation seeds 8-9")
    x_train = torch.tensor([_features(r, ACTIONS) for r in train], dtype=torch.float32)
    x_valid = torch.tensor([_features(r, ACTIONS) for r in valid], dtype=torch.float32)
    y_train = torch.tensor([ACTIONS.index(_rank(r, weights)) for r in train])
    y_valid = torch.tensor([ACTIONS.index(_rank(r, weights)) for r in valid])
    mean = x_train.mean(0)
    std = x_train.std(0, unbiased=False).clamp_min(1e-6)
    x_train, x_valid = (x_train - mean) / std, (x_valid - mean) / std
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x_train.shape[1], 64), nn.ReLU(), nn.Linear(64, len(ACTIONS)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    best_loss = math.inf
    best_state = None
    stale = 0
    best_epoch = 0
    for epoch in range(300):
        model.train()
        loss = nn.functional.cross_entropy(model(x_train), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_logits = model(x_valid)
            valid_loss = float(nn.functional.cross_entropy(valid_logits, y_valid))
        if valid_loss < best_loss - 1e-4:
            best_loss = valid_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= 30:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    with torch.no_grad():
        train_top1 = float((model(x_train).argmax(1) == y_train).float().mean())
        valid_top1 = float((model(x_valid).argmax(1) == y_valid).float().mean())
    return model, mean, std, {
        "n_train": len(train), "n_validation": len(valid),
        "train_top1": train_top1, "validation_top1": valid_top1,
        "validation_cross_entropy": best_loss, "best_epoch": best_epoch,
        "train_label_counts": dict(Counter(ACTIONS[i] for i in y_train.tolist())),
        "validation_label_counts": dict(Counter(ACTIONS[i] for i in y_valid.tolist())),
    }


def _run_one(problem: str, seed: int, budget: int, policy_name: str,
             models: dict) -> dict:
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(problem_id=problem, seed=seed, n_init=5))
    initial_y = backend.get_history().objectives[:, 0]
    initial_best = float(initial_y.max())
    objective_scale = float(initial_y.std(correction=0)) or 1.0
    if policy_name == "rule":
        policy = RulePolicy(seed=seed)
    elif policy_name == "rule_full_budget":
        policy = RuleFullBudget(seed=seed)
    else:
        model, mean, std = models[policy_name]
        policy = StateStudent(model, mean, std, seed)
    trajectory = DecisionLoop(backend, total_budget=budget).run(policy)
    history = backend.get_history()
    return {
        "problem": problem,
        "seed": seed,
        "budget": budget,
        "policy": policy_name,
        "initial_best": initial_best,
        "objective_scale": objective_scale,
        "terminal_best": float(history.objectives.max()),
        "total_evaluations": len(history),
        "new_evaluations": len(history) - 5,
        "n_steps": len(trajectory.steps),
        "termination": trajectory.termination_reason.value,
        "actions": [step.action.type.value for step in trajectory.steps],
    }


def _summarize(rows: list[dict]) -> dict:
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["problem"], row["budget"], row["policy"])].append(row)
    summary = {}
    for (problem, budget, policy), group in sorted(by_key.items()):
        references = {
            baseline: {
                r["seed"]: r for r in by_key[(problem, budget, baseline)]
            }
            for baseline in ("rule", "rule_full_budget")
        }
        if any(len(ref) != len(group) for ref in references.values()):
            raise AssertionError("missing paired baseline run")
        differences = {
            baseline: [
                (r["terminal_best"] - reference[r["seed"]]["terminal_best"])
                / r["objective_scale"]
                for r in group
            ]
            for baseline, reference in references.items()
        }
        summary[f"{problem}/budget{budget}/{policy}"] = {
            "n_seeds": len(group),
            "mean_terminal_best": statistics.fmean(r["terminal_best"] for r in group),
            "mean_new_evaluations": statistics.fmean(r["new_evaluations"] for r in group),
            "first_stop_count": sum(r["actions"][0] == "stop" for r in group),
            "paired": {
                baseline: {
                    "mean_normalized_gain": statistics.fmean(diff),
                    "ci95": _bootstrap_ci(diff),
                    "positive_seeds": sum(x > 0 for x in diff),
                    "negative_seeds": sum(x < 0 for x in diff),
                }
                for baseline, diff in differences.items()
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("runs/phase8_6d_hfull_10seed.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase8_6d_heldout_student.json"))
    parser.add_argument("--problems", nargs="+", default=[
        "Branin", "Ackley", "Hartmann", "SixHumpCamel", "Rastrigin",
    ])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10, 15)))
    parser.add_argument("--budgets", nargs="+", type=int, default=[15, 30])
    args = parser.parse_args()
    if min(args.seeds) < 10:
        parser.error("benchmark seeds must be >= 10 to remain held out")
    if min(args.budgets) <= 5:
        parser.error("budgets must exceed the five initial evaluations")
    torch.set_num_threads(1)
    records = read_jsonl(args.dataset, version_policy="warn")
    models = {}
    training = {}
    for name, weights in WEIGHTS.items():
        model, mean, std, metrics = _train(records, weights, seed=20260926)
        models[name] = (model, mean, std)
        training[name] = metrics
    print("training", json.dumps(training, sort_keys=True), flush=True)

    rows = []
    policies = ("rule", "rule_full_budget", "current_student", "no_fit_student")
    for problem in args.problems:
        for budget in args.budgets:
            for seed in args.seeds:
                for policy in policies:
                    row = _run_one(problem, seed, budget, policy, models)
                    rows.append(row)
                    print(
                        f"{problem} budget={budget} seed={seed} {policy}: "
                        f"best={row['terminal_best']:.5g}, eval={row['new_evaluations']}, "
                        f"termination={row['termination']}", flush=True,
                    )

    result = {
        "dataset": str(args.dataset),
        "train_seeds": list(range(8)),
        "validation_seeds": [8, 9],
        "benchmark_seeds": args.seeds,
        "problems": args.problems,
        "budgets": args.budgets,
        "training": training,
        "summary": _summarize(rows),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

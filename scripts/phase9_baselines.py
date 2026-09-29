"""Phase 9 entry point: held-out-seed offline decision baselines.

These scores describe agreement with provisional rollout labels. They do not
establish closed-loop optimization benefit; Phase 8.6B oracle gates do that.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.dataset import read_jsonl  # noqa: E402


def _rule_action(record) -> str:
    state = record.state
    remaining = int(state["resource"]["remaining_budget"])
    if remaining <= 4:
        return "stop"
    if int(state["optimization"]["stagnation_steps"]) >= 3 and remaining >= 2:
        return "global_exploration"
    return "surrogate_acquisition"


def _evaluate(records, prior: dict[str, float]) -> dict:
    totals = defaultdict(list)
    for record in records:
        valid = [
            action for action, result in record.rollout_results.items()
            if result["status"] == "valid"
        ]
        gold = max(record.gold_probs, key=record.gold_probs.get)
        prior_action = max(valid, key=lambda action: prior.get(action, 0.0))
        rule_action = _rule_action(record)
        if rule_action not in valid:
            rule_action = "stop"
        choices = {"prior": prior_action, "rule": rule_action, "gold": gold}
        utilities = {a: record.rollout_results[a]["utility"] for a in valid}
        for name, action in choices.items():
            totals[f"{name}_top1"].append(float(action == gold))
            totals[f"{name}_utility"].append(utilities[action])
        totals["uniform_top1"].append(1.0 / len(valid))
        totals["uniform_utility"].append(statistics.fmean(utilities.values()))
        uniform_p = 1.0 / len(valid)
        totals["uniform_cross_entropy"].append(
            -sum(record.gold_probs[a] * math.log(uniform_p) for a in valid)
        )
        prior_total = sum(prior.get(a, 0.0) for a in valid)
        totals["prior_cross_entropy"].append(
            -sum(
                record.gold_probs[a] * math.log(max(prior.get(a, 0.0) / prior_total, 1e-12))
                for a in valid
            )
        )
    return {key: statistics.fmean(values) for key, values in sorted(totals.items())}


def _features(record, actions: list[str]) -> list[float]:
    """Use only the observable optimization state, never rollout outcomes."""
    state = record.state
    sampling = state["sampling"]
    surrogate = state["surrogate"]
    progress = state["optimization"]
    resource = state["resource"]

    def signed_log(value) -> float:
        x = float(value) if value is not None and math.isfinite(float(value)) else 0.0
        return math.copysign(math.log1p(abs(x)), x)

    budget = max(int(resource["total_budget"]), 1)
    values = [
        float(sampling["n_samples"]) / budget,
        float(sampling["dimension"]),
        signed_log(sampling["min_pairwise_distance"]),
        float(sampling["boundary_coverage"]),
        float(bool(surrogate["is_current"])),
        signed_log(surrogate["validation_rmse"]),
        signed_log(surrogate["mean_uncertainty"]),
        signed_log(surrogate["max_uncertainty"]),
        signed_log(progress["best_value"]),
        signed_log(progress["last_value"]),
        signed_log(progress["recent_improvement"]),
        float(progress["stagnation_steps"]) / budget,
        float(resource["remaining_budget_ratio"]),
    ]
    last = state["history"]["last_action"]
    values.extend(float(last == action) for action in actions)
    values.append(float(last is None))
    return values


def _mlp_baseline(train, validation, test, actions: list[str]) -> dict:
    torch.manual_seed(0)

    def tensors(records):
        x = torch.tensor([_features(r, actions) for r in records], dtype=torch.float32)
        y = torch.tensor(
            [[r.gold_probs[action] for action in actions] for r in records],
            dtype=torch.float32,
        )
        return x, y

    x_train, y_train = tensors(train)
    x_valid, y_valid = tensors(validation)
    x_test, y_test = tensors(test)
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_train, x_valid, x_test = ((x - mean) / std for x in (x_train, x_valid, x_test))

    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, len(actions)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.001)
    best_loss = math.inf
    best_state = None
    stale = 0
    for _ in range(300):
        model.train()
        logits = model(x_train)
        loss = -(y_train * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x_valid)
            val_loss = float(
                -(y_valid * torch.log_softmax(val_logits, dim=-1)).sum(dim=-1).mean()
            )
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 25:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x_test), dim=-1)
        cross_entropy = float(
            -(y_test * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
        )
        predictions = probs.argmax(dim=-1).tolist()
    chosen = [actions[index] for index in predictions]
    gold = [max(r.gold_probs, key=r.gold_probs.get) for r in test]
    return {
        "validation_cross_entropy": best_loss,
        "test_cross_entropy": cross_entropy,
        "test_top1": statistics.fmean(float(a == b) for a, b in zip(chosen, gold)),
        "test_rollout_utility": statistics.fmean(
            r.rollout_results[action]["utility"]
            for r, action in zip(test, chosen)
        ),
        "predicted_actions": dict(
            (action, chosen.count(action)) for action in actions if action in chosen
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--holdout-seeds", type=int, nargs="+", default=[8, 9])
    args = parser.parse_args()
    records = read_jsonl(args.dataset, version_policy="warn")
    heldout = set(args.holdout_seeds)
    train = [r for r in records if int(r.problem["seed"]) not in heldout]
    test = [r for r in records if int(r.problem["seed"]) in heldout]
    if not train or not test:
        parser.error("need both training and held-out seed records")

    # A constant System-1 probability prior is the simplest learned baseline.
    actions = sorted(train[0].gold_probs)
    prior = {
        action: statistics.fmean(r.gold_probs[action] for r in train)
        for action in actions
    }
    mlp_train = [r for r in train if int(r.problem["seed"]) != 7]
    mlp_valid = [r for r in train if int(r.problem["seed"]) == 7]
    mlp = (
        _mlp_baseline(mlp_train, mlp_valid, test, actions)
        if mlp_train and mlp_valid else None
    )
    by_family: dict[str, list] = defaultdict(list)
    for record in test:
        by_family[record.problem["family"]].append(record)
    report = {
        "dataset": str(args.dataset),
        "train_records": len(train),
        "test_records": len(test),
        "holdout_seeds": sorted(heldout),
        "prior": prior,
        "mlp": mlp,
        "overall": _evaluate(test, prior),
        "by_problem": {
            family: _evaluate(rows, prior)
            for family, rows in sorted(by_family.items())
        },
        "interpretation": "Offline agreement with provisional labels only; not an optimization-performance claim.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

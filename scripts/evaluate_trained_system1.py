"""Held-out closed-loop audit for real typed-decision System-1 checkpoints.

Uses a forced System-1 route to expose policy quality before router tuning.
The fallback is local RulePolicy; no external LLM calls are made.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend  # noqa: E402
from opt_agent.loop import DecisionLoop  # noqa: E402
from opt_agent.loop.policies import RulePolicy  # noqa: E402
from opt_agent.router import ConfidenceRouter  # noqa: E402
from opt_agent.system1 import HybridPolicy, TypedDecisionPolicy  # noqa: E402
from opt_agent.system2 import FallbackSystem2  # noqa: E402


class RuleFullBudget(RulePolicy):
    LOW_BUDGET_THRESHOLD = 0


def run(problem, seed, budget, name, model):
    if model is None:
        policy = RuleFullBudget(seed=seed)
    else:
        model.seed = seed
        policy = HybridPolicy(
            system1=model, router=ConfidenceRouter(tau_conf=0.0),
            system2=FallbackSystem2(baseline=RuleFullBudget(seed=seed)),
        )
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(problem_id=problem, seed=seed, n_init=5))
    initial = backend.get_history().objectives.flatten()
    scale = float(initial.std(correction=0)) or 1.0
    initial_best = float(initial.max())
    trajectory = DecisionLoop(backend, total_budget=budget).run(policy, max_rounds=40)
    history = backend.get_history()
    actions = [step.action.type.value for step in trajectory.steps]
    return {
        "problem": problem, "seed": seed, "budget": budget, "policy": name,
        "initial_best": initial_best, "scale": scale,
        "terminal_best": float(history.objectives.max()),
        "new_evaluations": len(history) - 5,
        "termination": trajectory.termination_reason.value,
        "actions": actions, "action_counts": dict(Counter(actions)),
        "routes": (dict(Counter(step.route.name for step in policy.history))
                   if model is not None else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="NAME=PATH; repeat for multiple checkpoints")
    parser.add_argument("--problems", nargs="+", default=["Branin", "Ackley", "Hartmann", "SixHumpCamel", "Rastrigin"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10, 15)))
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.seeds) < 10:
        parser.error("test seeds must be >= 10")
    torch.set_num_threads(1)
    models = {}
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        if not name or name == "rule_full_budget" or name in models:
            parser.error(f"invalid or duplicate checkpoint name: {name!r}")
        models[name] = TypedDecisionPolicy.from_checkpoint(path)
    rows = []
    for problem in args.problems:
        for seed in args.seeds:
            for name, model in [("rule_full_budget", None), *models.items()]:
                row = run(problem, seed, args.budget, name, model)
                rows.append(row)
                print(f"{problem} seed={seed} {name}: best={row['terminal_best']:.4f} eval={row['new_evaluations']} {row['termination']}", flush=True)
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["problem"], row["policy"])].append(row)
    summary = {}
    for (problem, name), group in sorted(by_group.items()):
        baseline = {r["seed"]: r for r in by_group[(problem, "rule_full_budget")]}
        diffs = [(r["terminal_best"] - baseline[r["seed"]]["terminal_best"]) / r["scale"] for r in group]
        summary[f"{problem}/{name}"] = {
            "n": len(group), "mean_best": statistics.fmean(r["terminal_best"] for r in group),
            "mean_new_evaluations": statistics.fmean(r["new_evaluations"] for r in group),
            "mean_normalized_gain_vs_rule_full": statistics.fmean(diffs),
            "first_stop": sum(bool(r["actions"]) and r["actions"][0] == "stop" for r in group),
            "max_rounds": sum(r["termination"] == "max_rounds" for r in group),
            "action_counts": dict(sum((Counter(r["actions"]) for r in group), Counter())),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checkpoints": args.checkpoint, "rows": rows, "summary": summary}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

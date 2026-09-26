"""Run real Jev zero-shot BO trajectories paired with full-budget RulePolicy.

Loads JEV_API from .env, but never writes the key or raw HTTP response to disk.
Each completed pair is saved immediately so an interrupted run is auditable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend  # noqa: E402
from opt_agent.loop import DecisionLoop  # noqa: E402
from opt_agent.loop.policies import RulePolicy  # noqa: E402
from opt_agent.system1 import JevPolicy  # noqa: E402


class RuleFullBudget(RulePolicy):
    LOW_BUDGET_THRESHOLD = 0


def run_one(problem: str, seed: int, budget: int, policy, name: str, max_rounds: int) -> dict:
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(problem_id=problem, seed=seed, n_init=5))
    initial = backend.get_history().objectives.flatten()
    trajectory = DecisionLoop(backend, total_budget=budget).run(policy, max_rounds=max_rounds)
    rows = []
    for i, step in enumerate(trajectory.steps):
        item = {
            "round": i, "action": step.action.type.value,
            "n_new_samples": step.result.n_new_samples,
            "best_value": step.next_state.optimization.best_value,
        }
        # After the Laya/Jev interface unification, both policies return
        # JevDecision and carry the same metadata fields, so any policy
        # name accepted here gets the full row payload.
        if name in ("jev", "laya"):
            decision = policy.decisions[i]
            item.update({
                "confidence": decision.confidence,
                "probabilities": decision.probabilities,
                "latency_s": decision.latency_s,
                "input_tokens": decision.input_tokens,
                "output_tokens": decision.output_tokens,
                "model": decision.model,
            })
        rows.append(item)
    return {
        "problem": problem, "seed": seed, "budget": budget, "policy": name,
        "initial_best": float(initial.max()),
        "objective_scale": float(initial.std(correction=0)) or 1.0,
        "terminal_best": float(backend.get_history().objectives.max()),
        "new_evaluations": len(backend.get_history()) - 5,
        "termination": trajectory.termination_reason.value,
        "actions": rows,
    }


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["problem"], row["budget"], row["policy"])].append(row)
    summary = {}
    for (problem, budget, name), group in sorted(groups.items()):
        reference = {r["seed"]: r for r in groups[(problem, budget, "rule_full_budget")]}
        paired = [(r["terminal_best"] - reference[r["seed"]]["terminal_best"]) / r["objective_scale"] for r in group]
        decisions = [a for row in group for a in row["actions"]]
        summary[f"{problem}/budget{budget}/{name}"] = {
            "n": len(group),
            "mean_terminal_best": statistics.fmean(r["terminal_best"] for r in group),
            "mean_new_evaluations": statistics.fmean(r["new_evaluations"] for r in group),
            "mean_normalized_gain_vs_rule_full": statistics.fmean(paired),
            "first_stop": sum(bool(r["actions"]) and r["actions"][0]["action"] == "stop" for r in group),
            "action_counts": dict(Counter(a["action"] for a in decisions)),
            "api_calls": len(decisions) if name == "jev" else 0,
            "input_tokens": sum(a.get("input_tokens", 0) for a in decisions),
            "mean_latency_s": statistics.fmean(a.get("latency_s", 0) for a in decisions) if name == "jev" and decisions else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", default=["Branin", "Ackley", "Hartmann", "SixHumpCamel", "Rastrigin"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20, 25)))
    parser.add_argument("--budgets", nargs="+", type=int, default=[30])
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("runs/phase9_5_jev_zero_shot.json"))
    parser.add_argument("--model", default="jev-1.13.0",
                        help="Pin the hosted model version for reproducible research runs")
    parser.add_argument("--state-format", choices=("structured", "narrative"),
                        default="structured")
    parser.add_argument("--criteria-format", choices=("legacy", "executable"),
                        default="executable",
                        help="Show exact action parameters and evaluation costs; legacy reproduces prior prompts")
    parser.add_argument("--direct", action="store_true",
                        help="Ignore HTTP(S)_PROXY variables for the Jev API connection")
    parser.add_argument("--resume", action="store_true", help="Skip completed pairs in the output file")
    args = parser.parse_args()
    if min(args.budgets) <= 5 or min(args.seeds) < 0:
        parser.error("budgets must exceed five initial samples and seeds must be nonnegative")
    torch.set_num_threads(1)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get("model_requested") != args.model:
            parser.error("--resume model differs from the saved report")
        if previous.get("torch_num_threads") != 1:
            parser.error("--resume report was not generated with one torch thread")
        if previous.get("state_format", "structured") != args.state_format:
            parser.error("--resume state format differs from the saved report")
        if previous.get("criteria_format", "legacy") != args.criteria_format:
            parser.error("--resume criteria format differs from the saved report")
        if previous.get("direct", False) != args.direct:
            parser.error("--resume proxy mode differs from the saved report")
        rows = previous["rows"]
    completed = {(r["problem"], r["budget"], r["seed"])
                 for r in rows if r["policy"] == "jev"}
    for problem in args.problems:
        for budget in args.budgets:
            for seed in args.seeds:
                if (problem, budget, seed) in completed:
                    continue
                rule = run_one(problem, seed, budget, RuleFullBudget(seed=seed), "rule_full_budget", args.max_rounds)
                jev = JevPolicy(seed=seed, model=args.model,
                                state_format=args.state_format,
                                criteria_format=args.criteria_format,
                                trust_env=not args.direct)
                try:
                    model = run_one(problem, seed, budget, jev, "jev", args.max_rounds)
                finally:
                    jev.close()
                if rule["initial_best"] != model["initial_best"]:
                    raise AssertionError("paired runs have different initial states")
                rows.extend((rule, model))
                args.output.write_text(json.dumps({
                    "model_requested": jev.model, "state_format": args.state_format,
                    "criteria_format": args.criteria_format,
                    "direct": args.direct, "torch_num_threads": 1, "rows": rows,
                    "summary": summarize(rows),
                }, indent=2, sort_keys=True) + "\n")
                print(f"{problem} budget={budget} seed={seed}: rule={rule['terminal_best']:.4f}, jev={model['terminal_best']:.4f}, jev_eval={model['new_evaluations']}, steps={len(model['actions'])}", flush=True)
    print(json.dumps(summarize(rows), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

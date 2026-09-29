"""Separate first-action and continuation effects for the no-fit student.

Four arms on each paired problem/seed/budget:
  rule -> rule, student -> rule, rule -> student, student -> student.
The Rule arm uses its full-budget variant so early STOP does not confound
the continuation comparison.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_no_fit_student import (  # noqa: E402
    RuleFullBudget, StateStudent, WEIGHTS, _bootstrap_ci, _train,
)
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend  # noqa: E402
from opt_agent.dataset.decision_record import read_jsonl  # noqa: E402
from opt_agent.loop import DecisionLoop  # noqa: E402


class SwitchPolicy:
    def __init__(self, first, later) -> None:
        self.first = first
        self.later = later
        self.decisions = 0

    def select_action(self, state, backend=None):
        policy = self.first if self.decisions == 0 else self.later
        self.decisions += 1
        return policy.select_action(state, backend)


def _run(problem, seed, budget, first_name, later_name, model_data):
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(problem_id=problem, seed=seed, n_init=5))
    initial_y = backend.get_history().objectives[:, 0]
    scale = float(initial_y.std(correction=0)) or 1.0

    def policy(name):
        if name == "rule":
            return RuleFullBudget(seed=seed)
        model, mean, std = model_data
        return StateStudent(model, mean, std, seed)

    trajectory = DecisionLoop(backend, total_budget=budget).run(
        SwitchPolicy(policy(first_name), policy(later_name))
    )
    return {
        "problem": problem, "seed": seed, "budget": budget,
        "first": first_name, "later": later_name,
        "objective_scale": scale,
        "terminal_best": float(backend.get_history().objectives.max()),
        "new_evaluations": len(backend.get_history()) - 5,
        "actions": [x.action.type.value for x in trajectory.steps],
    }


def _summary(rows):
    by_scenario = defaultdict(list)
    for r in rows:
        by_scenario[(r["problem"], r["budget"])].append(r)
    result = {}
    for (problem, budget), group in sorted(by_scenario.items()):
        by_seed = defaultdict(dict)
        for r in group:
            by_seed[r["seed"]][f"{r['first']}->{r['later']}"] = r
        if any(len(v) != 4 for v in by_seed.values()):
            raise AssertionError("incomplete 2x2 paired scenario")
        cells = {}
        for arm in ("rule->rule", "student->rule", "rule->student", "student->student"):
            runs = [v[arm] for v in by_seed.values()]
            cells[arm] = {
                "mean_terminal_best": statistics.fmean(r["terminal_best"] for r in runs),
                "mean_new_evaluations": statistics.fmean(r["new_evaluations"] for r in runs),
                "first_actions": dict(Counter(r["actions"][0] for r in runs)),
                "action_counts": dict(Counter(a for r in runs for a in r["actions"])),
            }

        def contrast(left, right):
            values = [
                (v[left]["terminal_best"] - v[right]["terminal_best"])
                / v[left]["objective_scale"]
                for v in by_seed.values()
            ]
            return {
                "mean_normalized_gain": statistics.fmean(values),
                "ci95": _bootstrap_ci(values),
                "positive_seeds": sum(x > 0 for x in values),
                "negative_seeds": sum(x < 0 for x in values),
            }

        result[f"{problem}/budget{budget}"] = {
            "n_seeds": len(by_seed),
            "arms": cells,
            "student_first_with_rule_continuation": contrast("student->rule", "rule->rule"),
            "student_continuation_after_rule_first": contrast("rule->student", "rule->rule"),
            "student_continuation_after_student_first": contrast("student->student", "student->rule"),
            "full_student_vs_full_rule": contrast("student->student", "rule->rule"),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("runs/phase8_6d_hfull_10seed.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase8_6d_continuation_effect.json"))
    parser.add_argument("--problems", nargs="+", default=[
        "Branin", "Ackley", "Hartmann", "SixHumpCamel", "Rastrigin"
    ])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10, 20)))
    parser.add_argument("--budget", type=int, default=30)
    args = parser.parse_args()
    if min(args.seeds) < 10:
        parser.error("evaluation seeds must be >= 10")
    torch.set_num_threads(1)
    records = read_jsonl(args.dataset, version_policy="warn")
    model_data = _train(records, WEIGHTS["no_fit_student"], seed=20260926)[:3]
    rows = []
    for problem in args.problems:
        for seed in args.seeds:
            for first, later in (
                ("rule", "rule"), ("student", "rule"),
                ("rule", "student"), ("student", "student"),
            ):
                row = _run(problem, seed, args.budget, first, later, model_data)
                rows.append(row)
                print(
                    f"{problem} seed={seed} {first}->{later}: "
                    f"best={row['terminal_best']:.5g} eval={row['new_evaluations']}",
                    flush=True,
                )
    result = {
        "dataset": str(args.dataset),
        "budget": args.budget,
        "seeds": args.seeds,
        "summary": _summary(rows),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

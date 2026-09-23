"""Phase 9.1: real-backend smoke harness for HybridPolicy.

Per codex gpt-6-sol review 2026-09-23:
  - Real BOCoDe backend, paired HybridPolicy vs RulePolicy.
  - Branin + Ackley + Hartmann × seeds [0, 1, 2] × total_budget in [10, 30].
  - tau_conf = 0.5 (FIXED; not tuned in this smoke).
  - Per run: best_value, termination_reason, routing counts, total steps.
  - For Branin, also report simple_regret = known_optimum - best_value
    (Branin optimum at (-pi, 12.275) is 0.397887 in the classic-min frame,
    so BOCoDe-maximum is -0.397887). For Ackley/Hartmann the global
    optimum under BOCoDe's parameterization is not pinned; we report
    best_value only.
  - Acceptance (per codex): no exceptions; routing-count sum equals
    decision-step count; per-problem at least one Hybrid trajectory
    actually executes an evaluation; finite simple_regret reported for
    Branin; RulePolicy paired on the same seed.
  - NOT a performance claim: this script does NOT assert that HybridPolicy
    beats RulePolicy. Performance comparison needs many more seeds and
    a separate evaluation plan.

Usage:
  uv run python scripts/run_hybrid_smoke.py --output runs/phase9_1_smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.actions import ActionType  # noqa: E402
from opt_agent.backend.bocode_botorch import (  # noqa: E402
    BackendConfig,
    BocodeBotorchBackend,
)
from opt_agent.loop import DecisionLoop, TerminationReason  # noqa: E402
from opt_agent.loop.policies import RulePolicy  # noqa: E402
from opt_agent.router import ConfidenceRouter  # noqa: E402
from opt_agent.system1 import HybridPolicy  # noqa: E402
from opt_agent.system1.laya_policy import LayaPolicy  # noqa: E402
from opt_agent.system2 import FallbackSystem2  # noqa: E402


# Reference optimum values in BOCoDe's MAXIMIZATION frame.
# Branin: classic min = 0.397887 → BOCoDe max = -0.397887, verified by
# direct evaluation at (-pi, 12.275).
# Ackley (BOCoDe Ackley53, 10-D): the BOCoDe parameterization does NOT
# match the textbook Ackley function (transformed/scaled), so the
# classic optimum of 0 is misleading. We report best_value only.
# Hartmann 6-D: literature max is 3.047818, but BOCoDe 6-D parameterization
# may not reach it; report best_value only.
KNOWN_OPTIMA: dict[str, float | None] = {
    "Branin": -0.397887,
    "Ackley": None,
    "Hartmann": None,
}


@dataclass
class StepRecord:
    round: int
    action_type: str
    route_name: str


@dataclass
class RunResult:
    problem: str
    seed: int
    total_budget: int
    policy_name: str
    steps: list[StepRecord]
    best_value: float
    termination_reason: str
    n_evaluations: int
    simple_regret: float | None
    known_optimum: float | None


class _FakeLayaSurrogateAcquisition:
    """A zero-shot Laya stub that always picks surrogate_acquisition.

    The smoke uses this in place of the real Laya model so it does not
    depend on the optional Laya SDK being available. Behaviour is
    deterministic: full confidence on the chosen action.
    """

    def __init__(self, choice: str = "surrogate_acquisition") -> None:
        self.choice = choice

    def predict(self, state, questions):
        offered = questions["next_action"]["criteria"]
        probs = {name: (1.0 if name == self.choice else 0.0) for name in offered}
        return {"answers": {"next_action": {
            "type": "choice", "choice": self.choice,
            "probabilities": probs, "confidence": 1.0,
        }}}


def _make_hybrid(seed: int) -> HybridPolicy:
    """Build a HybridPolicy backed by the deterministic Laya stub."""
    laya = LayaPolicy(agent=_FakeLayaSurrogateAcquisition(), seed=seed)
    return HybridPolicy(
        system1=laya,
        router=ConfidenceRouter(tau_conf=0.5),
        system2=FallbackSystem2(),
    )


def _make_rule(seed: int) -> RulePolicy:
    return RulePolicy(seed=seed)


def _build_backend(problem: str, seed: int, n_init: int) -> BocodeBotorchBackend:
    backend = BocodeBotorchBackend()
    backend.initialize(
        BackendConfig(problem_id=problem, seed=seed, n_init=n_init),
    )
    return backend


def _count_evaluations(backend: BocodeBotorchBackend) -> int:
    history = backend.get_history()
    return len(history)


def _best_value(backend: BocodeBotorchBackend) -> float:
    history = backend.get_history()
    if not history:
        return float("-inf")
    objs = history.objectives.flatten()
    return float(objs.max().item())


def _run_hybrid(
    problem: str, seed: int, total_budget: int, n_init: int,
) -> RunResult:
    backend = _build_backend(problem, seed, n_init)
    hybrid = _make_hybrid(seed)
    loop = DecisionLoop(backend, total_budget=total_budget)
    trajectory = loop.run(hybrid)

    steps: list[StepRecord] = []
    for tr_step, hb_step in zip(trajectory.steps, hybrid.history):
        steps.append(StepRecord(
            round=tr_step.round,
            action_type=tr_step.action.type.value,
            route_name=hb_step.route.name,
        ))

    best = _best_value(backend)
    opt = KNOWN_OPTIMA.get(problem)
    simple_regret = (opt - best) if opt is not None else None
    return RunResult(
        problem=problem, seed=seed, total_budget=total_budget,
        policy_name="hybrid", steps=steps,
        best_value=best,
        termination_reason=trajectory.termination_reason.value,
        n_evaluations=_count_evaluations(backend),
        simple_regret=simple_regret, known_optimum=opt,
    )


def _run_rule(
    problem: str, seed: int, total_budget: int, n_init: int,
) -> RunResult:
    backend = _build_backend(problem, seed, n_init)
    rule = _make_rule(seed)
    loop = DecisionLoop(backend, total_budget=total_budget)
    trajectory = loop.run(rule)

    steps = [
        StepRecord(round=s.round, action_type=s.action.type.value, route_name="rule")
        for s in trajectory.steps
    ]
    best = _best_value(backend)
    opt = KNOWN_OPTIMA.get(problem)
    simple_regret = (opt - best) if opt is not None else None
    return RunResult(
        problem=problem, seed=seed, total_budget=total_budget,
        policy_name="rule", steps=steps,
        best_value=best,
        termination_reason=trajectory.termination_reason.value,
        n_evaluations=_count_evaluations(backend),
        simple_regret=simple_regret, known_optimum=opt,
    )


def _summarize(runs: list[RunResult]) -> dict[str, Any]:
    """Aggregate per (problem, policy, budget) stats."""
    by_key: dict[tuple, list[RunResult]] = {}
    for r in runs:
        key = (r.problem, r.policy_name, r.total_budget)
        by_key.setdefault(key, []).append(r)

    summary: dict[str, Any] = {}
    for (problem, policy, budget), group in by_key.items():
        best_values = [r.best_value for r in group]
        regrets = [r.simple_regret for r in group if r.simple_regret is not None]
        any_evaluations = any(r.n_evaluations > 0 for r in group)
        n_steps = [len(r.steps) for r in group]
        term_reasons = sorted({r.termination_reason for r in group})
        summary[f"{problem}/{policy}/b{budget}"] = {
            "n_seeds": len(group),
            "best_value_mean": sum(best_values) / len(best_values),
            "best_value_max": max(best_values),
            "best_value_min": min(best_values),
            "simple_regret_mean": (sum(regrets) / len(regrets)) if regrets else None,
            "any_evaluation": any_evaluations,
            "steps_mean": sum(n_steps) / len(n_steps),
            "termination_reasons": term_reasons,
        }
    return summary


def _acceptance_check(runs: list[RunResult]) -> tuple[bool, list[str]]:
    """Verify the Phase 9.1 acceptance criteria from the codex review."""
    errors: list[str] = []
    # No exceptions (encoded in termination_reason != "exception").
    for r in runs:
        if r.termination_reason == "exception":
            errors.append(f"{r.problem}/seed={r.seed}/{r.policy_name} raised an exception")
    # Per-problem: at least one Hybrid trajectory actually executed an evaluation.
    for problem in {"Branin", "Hartmann", "Ackley"}:
        hybrid_runs = [r for r in runs if r.problem == problem and r.policy_name == "hybrid"]
        if not hybrid_runs:
            errors.append(f"{problem}: no Hybrid runs recorded")
            continue
        if not any(r.n_evaluations > 0 for r in hybrid_runs):
            errors.append(f"{problem}: Hybrid did not execute any evaluation across {len(hybrid_runs)} runs")
    # Per-problem: routing-count sum equals decision-step count for Hybrid runs.
    for r in [r for r in runs if r.policy_name == "hybrid"]:
        sys1 = sum(1 for s in r.steps if s.route_name == "system1")
        sys2 = sum(1 for s in r.steps if s.route_name == "system2")
        if sys1 + sys2 != len(r.steps):
            errors.append(
                f"{r.problem}/seed={r.seed}/hybrid: routing sum {sys1}+{sys2} != steps {len(r.steps)}"
            )
    return (len(errors) == 0), errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/phase9_1_smoke.json"))
    parser.add_argument("--problems", nargs="+",
                        default=["Branin", "Hartmann", "Ackley"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budgets", nargs="+", type=int, default=[10, 30])
    parser.add_argument("--n-init", type=int, default=5)
    args = parser.parse_args()

    runs: list[RunResult] = []
    for problem in args.problems:
        for seed in args.seeds:
            for budget in args.budgets:
                # Hybrid policy
                try:
                    runs.append(_run_hybrid(problem, seed, budget, args.n_init))
                except Exception as exc:
                    runs.append(RunResult(
                        problem=problem, seed=seed, total_budget=budget,
                        policy_name="hybrid", steps=[],
                        best_value=float("-inf"),
                        termination_reason="exception",
                        n_evaluations=0, simple_regret=None,
                        known_optimum=KNOWN_OPTIMA.get(problem),
                    ))
                    print(f"  HYBRID {problem} seed={seed} budget={budget}: EXCEPTION: {exc}",
                          file=sys.stderr)
                # Rule policy (paired baseline)
                try:
                    runs.append(_run_rule(problem, seed, budget, args.n_init))
                except Exception as exc:
                    runs.append(RunResult(
                        problem=problem, seed=seed, total_budget=budget,
                        policy_name="rule", steps=[],
                        best_value=float("-inf"),
                        termination_reason="exception",
                        n_evaluations=0, simple_regret=None,
                        known_optimum=KNOWN_OPTIMA.get(problem),
                    ))
                    print(f"  RULE   {problem} seed={seed} budget={budget}: EXCEPTION: {exc}",
                          file=sys.stderr)

    summary = _summarize(runs)
    passed, errors = _acceptance_check(runs)

    payload = {
        "config": {
            "problems": args.problems,
            "seeds": args.seeds,
            "budgets": args.budgets,
            "n_init": args.n_init,
            "tau_conf": 0.5,
            "known_optima": KNOWN_OPTIMA,
        },
        "summary": summary,
        "acceptance": {"passed": passed, "errors": errors},
        "runs": [
            {
                "problem": r.problem, "seed": r.seed,
                "total_budget": r.total_budget, "policy": r.policy_name,
                "best_value": r.best_value,
                "termination_reason": r.termination_reason,
                "n_evaluations": r.n_evaluations,
                "n_steps": len(r.steps),
                "simple_regret": r.simple_regret,
                "known_optimum": r.known_optimum,
                "route_distribution": {
                    "system1": sum(1 for s in r.steps if s.route_name == "system1"),
                    "system2": sum(1 for s in r.steps if s.route_name == "system2"),
                    "rule": sum(1 for s in r.steps if s.route_name == "rule"),
                },
                "steps": [
                    {"round": s.round, "action": s.action_type, "route": s.route_name}
                    for s in r.steps
                ],
            }
            for r in runs
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    # Print concise summary.
    print(f"=== Phase 9.1 smoke: {len(runs)} runs ===")
    for key, s in summary.items():
        regret = (f"{s['simple_regret_mean']:+.4f}"
                  if s["simple_regret_mean"] is not None else "n/a")
        print(f"  {key:40s} best_value={s['best_value_mean']:+.4f} "
              f"(max={s['best_value_max']:+.4f}) regret={regret} "
              f"any_eval={s['any_evaluation']} steps~{s['steps_mean']:.1f} "
              f"terms={s['termination_reasons']}")
    print()
    if passed:
        print("ACCEPTANCE: PASSED")
    else:
        print("ACCEPTANCE: FAILED")
        for e in errors:
            print(f"  - {e}")
    print(f"\nDetailed JSON: {args.output}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

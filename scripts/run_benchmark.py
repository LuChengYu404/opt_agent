"""Phase 13: cross-problem × cross-seed benchmark harness.

Per MASTER_PLAN §13 / §16 (three benchmark levels) and §17
(decision-level + optimization-level + agent-level metrics):

Run every (problem, seed) combination under:
  - HybridPolicy (System-1 = FakeLayaSurrogateAcquisition stub,
    System-2 = FallbackSystem2)
  - RulePolicy

for {Branin, Ackley, Hartmann} × seeds 0..N-1 × total_budget ∈ {B}.

Report:
  - Per (problem, seed, policy): best_value, simple_regret (Branin only),
    n_evaluations, n_steps, termination_reason.
  - Aggregated: per-(problem, policy) mean simple_regret, success
    rate (best within X% of optimum for Branin), convergence speed
    (steps to first reach within Y% of best).

This script is a STAND-ALONE cross-problem benchmark; the Phase 9.1
smoke script (`run_hybrid_smoke.py`) is the per-rung smoke harness.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from opt_agent.actions import ActionType  # noqa: E402
from opt_agent.backend.bocode_botorch import (  # noqa: E402
    BackendConfig,
    BocodeBotorchBackend,
)
from opt_agent.loop import DecisionLoop  # noqa: E402
from opt_agent.loop.policies import RulePolicy  # noqa: E402
from opt_agent.router import ConfidenceRouter  # noqa: E402
from opt_agent.system1 import HybridPolicy  # noqa: E402
from opt_agent.system1.laya_policy import LayaPolicy  # noqa: E402
from opt_agent.system2 import FallbackSystem2  # noqa: E402


# Reference optima in BOCoDe's MAXIMIZATION frame. Only Branin has a
# pinned, verifiable optimum at (-π, 12.275) ≈ -0.397887; Ackley53 and
# Hartmann 6-D use BOCoDe-specific parameterizations whose global
# optima are not pinned in the literature — report best_value only.
KNOWN_OPTIMA = {"Branin": -0.397887}


class _FakeLaya:
    def __init__(self, choice: str = "surrogate_acquisition") -> None:
        self.choice = choice

    def predict(self, state, questions):
        offered = questions["next_action"]["criteria"]
        probs = {n: (1.0 if n == self.choice else 0.0) for n in offered}
        return {"answers": {"next_action": {
            "type": "choice", "choice": self.choice,
            "probabilities": probs, "confidence": 1.0,
        }}}


def _make_hybrid(seed: int) -> HybridPolicy:
    return HybridPolicy(
        system1=LayaPolicy(agent=_FakeLaya(), seed=seed),
        router=ConfidenceRouter(tau_conf=0.5),
        system2=FallbackSystem2(),
    )


@dataclass
class BenchResult:
    problem: str
    seed: int
    policy: str
    best_value: float
    n_evaluations: int
    n_steps: int
    termination_reason: str
    simple_regret: float | None


def _run_one(problem: str, seed: int, budget: int, n_init: int,
             policy_name: str) -> BenchResult:
    backend = BocodeBotorchBackend()
    backend.initialize(
        BackendConfig(problem_id=problem, seed=seed, n_init=n_init),
    )
    policy = _make_hybrid(seed) if policy_name == "hybrid" else RulePolicy(seed=seed)
    loop = DecisionLoop(backend, total_budget=budget)
    traj = loop.run(policy)
    history = backend.get_history()
    best_value = float(history.objectives.max().item()) if len(history) else float("-inf")
    opt = KNOWN_OPTIMA.get(problem)
    return BenchResult(
        problem=problem, seed=seed, policy=policy_name,
        best_value=best_value,
        n_evaluations=len(history),
        n_steps=len(traj.steps),
        termination_reason=traj.termination_reason.value,
        simple_regret=(opt - best_value) if opt is not None else None,
    )


def _aggregate(results: list[BenchResult]) -> dict:
    by_key: dict[tuple, list[BenchResult]] = defaultdict(list)
    for r in results:
        by_key[(r.problem, r.policy)].append(r)
    summary = {}
    for (problem, policy), group in by_key.items():
        bests = [r.best_value for r in group]
        evals = [r.n_evaluations for r in group]
        regrets = [r.simple_regret for r in group if r.simple_regret is not None]
        n = len(group)
        # Paired t-statistic on per-seed best_value (hybrid vs rule).
        rule_group = [
            x for x in by_key.get((problem, "rule"), [])
            if x.seed in {r.seed for r in group}
        ]
        hybrid_bests = [r.best_value for r in group]
        rule_bests = [r.best_value for r in rule_group]
        diffs = [h - r for h, r in zip(hybrid_bests, rule_bests)] if (
            len(hybrid_bests) == len(rule_bests) >= 2
        ) else []
        t_stat = None
        if len(diffs) >= 2:
            mean_d = statistics.fmean(diffs)
            sd_d = statistics.pstdev(diffs)
            t_stat = mean_d / sd_d * math.sqrt(len(diffs)) if sd_d > 0 else None
        # Success rate for Branin: |simple_regret| < 1e-2.
        if problem == "Branin":
            success_rate = (
                sum(1 for r in regrets if abs(r) < 1e-2) / n
            )
        else:
            success_rate = None
        summary[f"{problem}/{policy}"] = {
            "n_seeds": n,
            "best_value_mean": statistics.fmean(bests),
            "best_value_std": statistics.pstdev(bests) if n > 1 else 0.0,
            "best_value_max": max(bests),
            "best_value_min": min(bests),
            "n_evaluations_mean": statistics.fmean(evals),
            "simple_regret_mean": (statistics.fmean(regrets) if regrets else None),
            "simple_regret_std": (statistics.pstdev(regrets) if len(regrets) > 1 else 0.0),
            "success_rate_branin_within_1e-2": success_rate,
            "paired_t_hybrid_vs_rule": t_stat,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/phase13_benchmark.json"))
    parser.add_argument("--problems", nargs="+",
                        default=["Branin", "Ackley", "Hartmann"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(5)))
    parser.add_argument("--budgets", nargs="+", type=int, default=[30])
    parser.add_argument("--n-init", type=int, default=5)
    args = parser.parse_args()

    results: list[BenchResult] = []
    for problem in args.problems:
        for seed in args.seeds:
            for budget in args.budgets:
                for policy in ("hybrid", "rule"):
                    try:
                        results.append(_run_one(
                            problem, seed, budget, args.n_init, policy,
                        ))
                    except Exception as exc:
                        results.append(BenchResult(
                            problem=problem, seed=seed, policy=policy,
                            best_value=float("-inf"),
                            n_evaluations=0, n_steps=0,
                            termination_reason="exception",
                            simple_regret=None,
                        ))
                        print(
                            f"  EXCEPTION {problem}/seed={seed}/budget={budget}/{policy}: {exc}",
                            file=sys.stderr,
                        )

    summary = _aggregate(results)
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
        "runs": [asdict(r) for r in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"=== Phase 13 benchmark: {len(results)} runs ===")
    for key, s in summary.items():
        regret = (f"{s['simple_regret_mean']:+.4f}"
                  if s["simple_regret_mean"] is not None else "n/a")
        succ = (f"{s['success_rate_branin_within_1e-2']:.0%}"
                if s["success_rate_branin_within_1e-2"] is not None else "n/a")
        t = (f"{s['paired_t_hybrid_vs_rule']:+.2f}"
             if s["paired_t_hybrid_vs_rule"] is not None else "n/a")
        print(
            f"  {key:35s} best={s['best_value_mean']:+.4f} "
            f"(max={s['best_value_max']:+.4f}) regret={regret} "
            f"succ={succ} t={t}"
        )
    print(f"\nDetailed JSON: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
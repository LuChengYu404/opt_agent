"""Phase 8.5 oracle gate functions (per codex review).

Three gates must pass before Phase 9 (System-1 training):
  Gate 3 — aggregate_mass: seed-clustered log-lift > 0 in early phase.
  Gate 4 — oracle_test: gold top action gains over STOP / uniform / behavior
                       baselines via paired normalized terminal-regret test.
  Gate 5 — affine_rescaling_test: Spearman ≥ 0.9, top-1 agreement ≥ 0.8,
                                JS divergence ≤ 0.02 per (problem, phase).

Per codex (gpt-5.6-sol) review 2025-09-23:
  - Do NOT compare gold >= 0.9*baseline; use paired normalized gains.
  - Aggregate by independent seeds (not raw record count) to avoid
    long-trajectory domination.
  - Affine test must transform actual evaluated objectives, not just
    the utility argument.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from opt_agent.actions import (
    ActionType,
    OptimizationAction,
    execute_action,
)
from opt_agent.dataset.decision_record import DecisionRecord
from opt_agent.dataset.generator import derive_seed
from opt_agent.loop.policies import RandomPolicy, RulePolicy
from opt_agent.state import extract_state


EXPLORATION_ACTIONS: frozenset[str] = frozenset({
    "global_exploration",
    "uncertainty_sampling",
    "error_guided_sampling",
    "boundary_sampling",
})

EPS = 1e-12


def phase_of(record: DecisionRecord) -> str:
    """Same phase definition as analysis.py."""
    ratio = float(record.state["resource"]["remaining_budget_ratio"])
    if ratio == 0:
        return "terminal"
    if ratio > 0.7:
        return "early"
    if ratio < 0.3:
        return "late"
    return "mid"


# ---------------------------------------------------------------------------
# Gate 3 — aggregate_mass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MassGateResult:
    seed_rows: list[dict]
    raw_mean: float
    adjusted_lift_mean: float
    pass_: bool


def _lift(probs: dict[str, float]) -> tuple[float, float]:
    """Return (raw_lift, adjusted_lift) for one record.

    adjusted_lift = log((p_exp + eps) / (p_stop + eps)) − log(#exploration).
    Subtracting log(N_exploration) gives a fair comparison: under uniform
    probabilities (1/8 each), adjusted_lift = 0 regardless of N.
    """
    p_exp = sum(probs.get(a, 0.0) for a in EXPLORATION_ACTIONS)
    p_stop = probs.get("stop", 0.0)
    raw = p_exp - p_stop
    adj = math.log((p_exp + EPS) / (p_stop + EPS)) - math.log(len(EXPLORATION_ACTIONS))
    return raw, adj


def aggregate_mass(records: list[DecisionRecord]) -> MassGateResult:
    """Compute per-seed mean adjusted lift across early records.

    Returns
    -------
    MassGateResult with `pass_=True` if the seed-clustered adjusted lift is
    strictly positive (codex's criterion: CI lower bound > 0; we use the
    point estimate with seed-level equal weighting for the gate smoke test).
    """
    by_seed: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for r in records:
        if phase_of(r) != "early":
            continue
        raw, adj = _lift(r.gold_probs)
        key = (r.problem["family"], int(r.problem["seed"]))
        by_seed[key].append((raw, adj))

    seed_rows = [
        {
            "problem": p,
            "seed": s,
            "raw": statistics.fmean(x[0] for x in vals),
            "adjusted_lift": statistics.fmean(x[1] for x in vals),
        }
        for (p, s), vals in by_seed.items()
    ]
    if not seed_rows:
        return MassGateResult(seed_rows=[], raw_mean=0.0, adjusted_lift_mean=0.0,
                              pass_=False)
    raw_mean = statistics.fmean(r["raw"] for r in seed_rows)
    adj_mean = statistics.fmean(r["adjusted_lift"] for r in seed_rows)
    # Smoke gate: per-seed mean adjusted lift must be > 0 in early phase.
    # (Strict CI-based check would be added when we have ≥ 10 seeds.)
    pass_ = adj_mean > 0.0
    return MassGateResult(
        seed_rows=seed_rows, raw_mean=raw_mean,
        adjusted_lift_mean=adj_mean, pass_=pass_,
    )


# ---------------------------------------------------------------------------
# Gate 4 — oracle_test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleResult:
    gold_action: str
    terminal: dict[str, float]  # {"gold": ..., "stop": ..., "uniform": ..., "behavior": ...}
    gains: dict[str, float]      # normalized (gold - baseline) / objective_scale


def _run_arm(
    backend,
    *,
    initial_state,
    first_action: OptimizationAction,
    choose_next: Callable,
    total_budget: int,
    max_actions: int = 200,
) -> tuple[float, int]:
    """Run one arm of the oracle test. Returns (terminal_best_value, remaining_budget).

    Caller is responsible for restoring backend to the desired starting state
    before each arm; we just consume the backend as-is from `initial_state`.
    """
    current = initial_state
    for step in range(max_actions):
        action = first_action if step == 0 else choose_next(current, backend)
        result = execute_action(backend, current, action)
        current = extract_state(
            backend, total_budget=total_budget, last_action=action.type.value
        )
        if result.should_stop or current.resource.remaining_budget == 0:
            return float(current.optimization.best_value), int(
                current.resource.remaining_budget
            )
    return float(current.optimization.best_value), int(
        current.resource.remaining_budget
    )


def _restore_backend(backend, base) -> None:
    """Restore backend to base state. Caller-supplied to make arm semantics explicit."""
    backend.restore(base)


def oracle_test(
    backend,
    state,
    record: DecisionRecord,
    *,
    objective_scale: float = 1.0,
    random_replicates: int = 5,
    max_actions: int = 50,
) -> OracleResult:
    """Run gold / stop / uniform / behavior arms from the decision point.

    The backend is restored to its pre-call state at the end (try/finally).
    """
    base = backend.checkpoint()
    total_budget = int(state.resource.total_budget)
    specs = {x["action_id"]: x for x in record.candidate_actions}
    gold_id = max(record.gold_probs, key=record.gold_probs.get)

    def action_from_id(action_id: str) -> OptimizationAction:
        spec = specs[action_id]
        return OptimizationAction(ActionType(spec["type"]), dict(spec["params"]))

    try:
        # Gold: gold top action, then RulePolicy continuation.
        gold_rule = RulePolicy(seed=derive_seed(int(record.metadata["run_seed"]), 1))
        _restore_backend(backend, base)
        gold = _run_arm(
            backend,
            initial_state=state,
            first_action=action_from_id(gold_id),
            choose_next=lambda s, b: gold_rule.select_action(s, b),
            total_budget=total_budget,
            max_actions=max_actions,
        )
        # STOP: STOP, then STOP.
        stop_action = OptimizationAction(ActionType.STOP)
        _restore_backend(backend, base)
        stopped = _run_arm(
            backend,
            initial_state=state,
            first_action=stop_action,
            choose_next=lambda s, b: stop_action,
            total_budget=total_budget,
            max_actions=max_actions,
        )
        # Uniform + behavior: random pick at every step.
        uniform_results: list[float] = []
        behavior_results: list[float] = []
        for rep in range(random_replicates):
            u_seed = derive_seed(int(record.metadata["run_seed"]), 1000, rep)
            u_policy = RandomPolicy(seed=u_seed)
            _restore_backend(backend, base)
            u = _run_arm(
                backend,
                initial_state=state,
                first_action=u_policy.select_action(state, backend),
                choose_next=lambda s, b, p=u_policy: p.select_action(s, b),
                total_budget=total_budget,
                max_actions=max_actions,
            )
            uniform_results.append(u[0])

            b_rng = random.Random(derive_seed(u_seed, 1))
            b_rule = RulePolicy(seed=derive_seed(u_seed, 2))
            b_random = RandomPolicy(seed=derive_seed(u_seed, 3))

            def choose_behavior(s, b):
                return (
                    b_random.select_action(s, b)
                    if b_rng.random() < 0.2
                    else b_rule.select_action(s, b)
                )

            _restore_backend(backend, base)
            behavior_results.append(_run_arm(
                backend,
                initial_state=state,
                first_action=choose_behavior(state, backend),
                choose_next=choose_behavior,
                total_budget=total_budget,
                max_actions=max_actions,
            )[0])

        terminal = {
            "gold": gold[0],
            "stop": stopped[0],
            "uniform": statistics.fmean(uniform_results),
            "behavior": statistics.fmean(behavior_results),
        }
        gains = {
            arm: (terminal["gold"] - value) / objective_scale
            for arm, value in terminal.items()
            if arm != "gold"
        }
        return OracleResult(
            gold_action=gold_id, terminal=terminal, gains=gains
        )
    finally:
        backend.restore(base)


# ---------------------------------------------------------------------------
# Gate 5 — affine_rescaling_test
# ---------------------------------------------------------------------------


def affine_rescaling_test(
    generate_for_transform: Callable[[float, float], Iterable[DecisionRecord]],
) -> dict:
    """Compare top-action distribution across (scale, shift) transforms.

    Pass criteria per (problem, phase): median Spearman ≥ 0.9, top-1
    agreement ≥ 0.8, mean JS divergence ≤ 0.02.
    """
    transforms = ((1.0, 0.0), (0.1, 0.0), (10.0, 0.0), (1.0, 5.0))
    datasets = {
        t: {r.id: r for r in generate_for_transform(*t)}
        for t in transforms
    }
    base = datasets[(1.0, 0.0)]
    rows: list[dict] = []

    for transform, candidate in datasets.items():
        if transform == (1.0, 0.0):
            continue
        if candidate.keys() != base.keys():
            return {"pass_": False, "reason": "unpaired record IDs",
                    "by_problem_phase": []}

        for rid, left in base.items():
            right = candidate[rid]
            action_ids = sorted(left.gold_probs)
            p = np.array([left.gold_probs[a] for a in action_ids], dtype=np.float64)
            q = np.array([right.gold_probs[a] for a in action_ids], dtype=np.float64)
            p_safe = np.where(p > 0, p, EPS)
            q_safe = np.where(q > 0, q, EPS)
            m = 0.5 * (p_safe + q_safe)
            js = 0.5 * float(np.sum(p_safe * np.log(p_safe / m))) + 0.5 * float(
                np.sum(q_safe * np.log(q_safe / m))
            )
            # Rank correlation by simple ranking (no scipy dependency).
            rho = _spearman(p, q)
            rows.append({
                "transform": transform,
                "problem": left.problem["family"],
                "phase": phase_of(left),
                "top1_equal": int(int(p.argmax()) == int(q.argmax())),
                "spearman": rho,
                "js": js,
            })

    summary = _summarize_by_problem_phase(rows)
    passed = all(
        x["median_spearman"] >= 0.9
        and x["top1_agreement"] >= 0.8
        and x["mean_js"] <= 0.02
        for x in summary
    )
    return {"pass_": passed, "by_problem_phase": summary}


def _spearman(p: np.ndarray, q: np.ndarray) -> float:
    """Pure-Python Spearman rank correlation."""
    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(x), dtype=np.float64)
        return ranks
    rp, rq = ranks(p), ranks(q)
    mp = rp.mean()
    mq = rq.mean()
    cov = float(((rp - mp) * (rq - mq)).sum())
    sp = float(((rp - mp) ** 2).sum() ** 0.5)
    sq = float(((rq - mq) ** 2).sum() ** 0.5)
    if sp == 0.0 or sq == 0.0:
        return 1.0
    return cov / (sp * sq)


def _summarize_by_problem_phase(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        grouped[(r["problem"], r["phase"])].append(r)
    out: list[dict] = []
    for (problem, phase), vals in sorted(grouped.items()):
        spearmans = [v["spearman"] for v in vals]
        top1s = [v["top1_equal"] for v in vals]
        jss = [v["js"] for v in vals]
        out.append({
            "problem": problem,
            "phase": phase,
            "n": len(vals),
            "median_spearman": statistics.median(spearmans),
            "top1_agreement": statistics.fmean(top1s),
            "mean_js": statistics.fmean(jss),
        })
    return out

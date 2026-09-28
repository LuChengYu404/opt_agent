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

Sprint 0 addition (per codex Q-T2 2026-09-25):
  - `replay_with_config` re-runs any record with a different config
    (continuation_spec / utility_weights / horizon / temperature) without
    re-running the trajectory that produced the record.
"""

from __future__ import annotations

import base64
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch

from opt_agent.actions import (
    ActionType,
    OptimizationAction,
    execute_action,
)
from opt_agent.backend.checkpoint import Checkpoint
from opt_agent.dataset.decision_record import DecisionRecord
from opt_agent.dataset.generator import derive_seed
from opt_agent.loop.policies import RandomPolicy, RulePolicy
from opt_agent.rollout.rollout import RolloutOutput
from opt_agent.rollout.specs import ContinuationPolicySpec, spec_from_metadata
from opt_agent.rollout.utility import UtilityWeights
from opt_agent.state import extract_state


EXPLORATION_ACTIONS: frozenset[str] = frozenset({
    "global_exploration",
    "uncertainty_sampling",
    "error_guided_sampling",
    "boundary_sampling",
})

EPS = 1e-12


def phase_of_ratio(ratio: float) -> str:
    """Classify a BO trajectory point by remaining-budget ratio.

    Boundaries (single source of truth for both ``phase_of(record)`` and
    the dataset analyzer):
      - ratio == 0       → terminal
      - ratio >  0.7     → early
      - ratio <  0.3     → late
      - else             → mid
    """
    if ratio == 0:
        return "terminal"
    if ratio > 0.7:
        return "early"
    if ratio < 0.3:
        return "late"
    return "mid"


def phase_of(record: DecisionRecord) -> str:
    """Classify a record's trajectory phase from its ``remaining_budget_ratio``."""
    return phase_of_ratio(float(record.state["resource"]["remaining_budget_ratio"]))


# ---------------------------------------------------------------------------
# Gate 3 — aggregate_mass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MassGateResult:
    seed_rows: list[dict]
    raw_mean: float
    adjusted_lift_mean: float
    pass_: bool
    seed_clustered_ci95: tuple[float, float] | None = (
        None  # bootstrap 95% CI over per-seed adjusted_lift; None if < 5 seeds
    )


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
    strictly positive: with >= 5 seeds the criterion is the seed-clustered
    bootstrap CI lower bound > 0; fewer seeds fall back to the point
    estimate (single-seed smoke compatibility).
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

    # Seed-clustered bootstrap CI (per codex review 2026-09-26): when ≥ 5
    # seeds are available, gate criterion is the CI lower bound > 0 instead
    # of the point estimate. Fewer seeds fall back to point estimate (kept
    # for backward compat with single-seed smoke tests).
    seed_clustered_ci95: tuple[float, float] | None = None
    n_seeds = len(seed_rows)
    if n_seeds >= 5:
        seed_lifts = [r["adjusted_lift"] for r in seed_rows]
        seed_clustered_ci95 = _bootstrap_ci_mean(seed_lifts, repetitions=2000, seed=0)
        pass_ = seed_clustered_ci95[0] > 0.0
    else:
        pass_ = adj_mean > 0.0

    return MassGateResult(
        seed_rows=seed_rows, raw_mean=raw_mean,
        adjusted_lift_mean=adj_mean, seed_clustered_ci95=seed_clustered_ci95,
        pass_=pass_,
    )


def _bootstrap_ci_mean(
    values: list[float], *, repetitions: int = 2000, seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean of `values`. Pure-Python; no scipy."""
    import random
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(repetitions)
    )
    lo = means[int(0.025 * repetitions)]
    hi = means[int(0.975 * repetitions)]
    return (lo, hi)


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
    if current.resource.remaining_budget <= 0:
        return float(current.optimization.best_value), 0
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
    max_actions: int = 200,
) -> OracleResult:
    """Compare first actions with the same RulePolicy continuation.

    The backend is restored to its pre-call state at the end (try/finally).
    max_actions defaults to 200, matching oracle_test_from_record/_run_arm —
    the same record must get the same action cap from either entry point.
    """
    if not math.isfinite(objective_scale) or objective_scale <= 0:
        raise ValueError("objective_scale must be finite and positive")
    if random_replicates < 1:
        raise ValueError("random_replicates must be positive")
    base = backend.checkpoint()
    total_budget = int(state.resource.total_budget)
    specs = {x["action_id"]: x for x in record.candidate_actions}
    gold_id = max(record.gold_probs, key=record.gold_probs.get)

    def action_from_id(action_id: str) -> OptimizationAction:
        spec = specs[action_id]
        return OptimizationAction(ActionType(spec["type"]), dict(spec["params"]))

    continuation_seed = derive_seed(int(record.metadata["run_seed"]), 1)
    continuation_policy = RulePolicy(seed=continuation_seed)
    def choose_continuation(s, b):
        return continuation_policy.select_action(s, b)

    try:
        # Gold: gold top action, then RulePolicy continuation.
        _restore_backend(backend, base)
        gold = _run_arm(
            backend,
            initial_state=state,
            first_action=action_from_id(gold_id),
            choose_next=choose_continuation,
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
        # Baselines differ only in their first action; all non-STOP arms then
        # use the same continuation, isolating the decision being evaluated.
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
                choose_next=choose_continuation,
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
                choose_next=choose_continuation,
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
# Gate 4 — replayable oracle test (Phase 8.6B)
# ---------------------------------------------------------------------------


def oracle_test_from_record(
    record: DecisionRecord,
    *,
    random_replicates: int = 5,
    max_actions: int = 200,
    backend_factory: Callable | None = None,
) -> OracleResult:
    """Replay a DecisionRecord's backend state and run the oracle test.

    The record must contain a complete backend checkpoint. A fresh backend
    restores that checkpoint before the paired oracle arms are run.

    Parameters
    ----------
    backend_factory : Callable | None
        Optional factory returning a fresh backend. Defaults to a
        BocodeBotorchBackend reconstructed with the problem id from the record.
    """
    from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend

    cp = _reconstruct_checkpoint(record)
    if backend_factory is None:
        # Phase 9 Gate 5 — thread the affine objective transform from record
        # metadata so that future evaluate() calls append transformed values
        # to the (already-transformed) history loaded from the checkpoint.
        affine = _affine_kwargs_from_metadata(record.metadata)
        backend = BocodeBotorchBackend(**affine)
        backend.initialize(BackendConfig(
            problem_id=cp.problem_id,
            seed=int(record.problem["seed"]),
            n_init=1,
        ))
    else:
        backend = backend_factory()
    backend.restore(cp)

    state = extract_state(
        backend,
        total_budget=int(record.state["resource"]["total_budget"]),
        last_action=record.state["history"].get("last_action"),
    )
    if state.sampling.n_samples != int(record.state["sampling"]["n_samples"]):
        raise ValueError(f"record {record.id!r} replay sample count differs")
    if not math.isclose(
        state.optimization.best_value,
        float(record.state["optimization"]["best_value"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(f"record {record.id!r} replay best value differs")
    if state.surrogate.is_current != bool(record.state["surrogate"]["is_current"]):
        raise ValueError(f"record {record.id!r} replay surrogate status differs")
    objective_scale = float(record.metadata.get("optimization_scale", 1.0))
    return oracle_test(
        backend,
        state,
        record,
        objective_scale=objective_scale,
        random_replicates=random_replicates,
        max_actions=max_actions,
    )


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


# ---------------------------------------------------------------------------
# Gate 2 — multi-seed stability (Phase 8.6B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate2Result:
    """Per-problem, per-phase stability report.

    Per the codex Phase 9 gate:
      - ≥ 10 independent seeds
      - per-seed stability of top-action distribution
      - point estimate or seed-clustered CI > 0 for adjusted_lift
    """
    problems: dict[str, dict[str, dict]]  # problem → phase → stats
    per_seed_lift: dict[tuple[str, str, int], float]  # (problem, phase, seed)
    total_seeds: int  # minimum independent seed count across problems
    min_phase_seeds: int
    pass_: bool

    def summary(self) -> str:
        lines = [
            f"Gate 2 — multi-seed stability ({self.total_seeds} seeds/problem; "
            f"minimum {self.min_phase_seeds} seeds/phase)"
        ]
        for problem, phases in self.problems.items():
            for phase, stats in phases.items():
                lines.append(
                    f"  {problem} {phase}: n_records={stats['n_records']}, "
                    f"top_action={stats['top_action']!r}, "
                    f"top_action_share={stats['top_action_share']:.2f}, "
                    f"mean_adj_lift={stats['mean_adjusted_lift']:.3f}"
                )
        lines.append(f"  pass = {self.pass_}")
        return "\n".join(lines)


def gate2_stability(records: list[DecisionRecord]) -> Gate2Result:
    """Compute Gate 2 stability metrics from a multi-seed dataset.

    Per (problem, phase):
      - n_records: count of records
      - top_action: most-frequent gold argmax
      - top_action_share: fraction of records with that argmax
      - mean_adjusted_lift: average of log-lift across records

    Pass criterion: at least 10 independent seeds per problem, at least 5
    seeds in every observed nonterminal phase, and the same top action in
    at least 60% of seed-level mean probability distributions. Multiple
    trajectories from one seed count as one independent observation.
    """
    by_problem_phase_seed: dict[tuple[str, str, int], list[DecisionRecord]] = (
        defaultdict(list)
    )
    for r in records:
        phase = phase_of(r)
        key = (r.problem["family"], phase, int(r.problem["seed"]))
        by_problem_phase_seed[key].append(r)

    per_seed_lift: dict[tuple[str, str, int], float] = {}
    by_problem_phase: dict[tuple[str, str], list[tuple[int, list[DecisionRecord]]]] = defaultdict(list)
    for (problem, phase, seed), recs in by_problem_phase_seed.items():
        if phase != "terminal":
            by_problem_phase[(problem, phase)].append((seed, recs))

    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for (problem, phase), seed_groups in sorted(by_problem_phase.items()):
        top_counts: dict[str, int] = defaultdict(int)
        seed_lifts: list[float] = []
        for seed, recs in seed_groups:
            # Partial-rollout records (long-horizon quota datasets) cover a
            # k-of-8 subset per record. Per-action means are taken only over
            # records where the action was actually rolled out; the seed's
            # top action is the argmax over actions covered by its records.
            covered = {
                a for r in recs for a in r.gold_probs
            }
            mean_probs = {}
            for action in sorted(covered):
                vals = [r.gold_probs[action] for r in recs if action in r.gold_probs]
                if vals:
                    mean_probs[action] = statistics.fmean(vals)
            top_counts[max(mean_probs, key=mean_probs.get)] += 1
            lift = statistics.fmean(_lift(r.gold_probs)[1] for r in recs)
            per_seed_lift[(problem, phase, seed)] = lift
            seed_lifts.append(lift)
        top_action, top_count = max(top_counts.items(), key=lambda kv: kv[1])
        grouped[problem][phase] = {
            "n_records": sum(len(recs) for _, recs in seed_groups),
            "n_seeds": len(seed_groups),
            "top_action": top_action,
            "top_action_share": top_count / len(seed_groups),
            "mean_adjusted_lift": statistics.fmean(seed_lifts),
        }

    min_phase_seeds = min(
        (stats["n_seeds"] for phases in grouped.values() for stats in phases.values()),
        default=0,
    )
    seeds_by_problem: dict[str, set[int]] = defaultdict(set)
    for (problem, phase, seed) in by_problem_phase_seed:
        if phase != "terminal":
            seeds_by_problem[problem].add(seed)
    min_seeds = min((len(seeds) for seeds in seeds_by_problem.values()), default=0)
    concentration_ok = all(
        stats["top_action_share"] >= 0.6
        for phases in grouped.values() for stats in phases.values()
    )
    pass_ = min_seeds >= 10 and min_phase_seeds >= 5 and concentration_ok

    return Gate2Result(
        problems=dict(grouped),
        per_seed_lift=per_seed_lift,
        total_seeds=min_seeds,
        min_phase_seeds=min_phase_seeds,
        pass_=pass_,
    )


# ---------------------------------------------------------------------------
# Sprint 0 — replay_with_config (per codex Q-T2 2026-09-25)
# ---------------------------------------------------------------------------


def _reconstruct_checkpoint(record: DecisionRecord) -> Checkpoint:
    """Build a `Checkpoint` from a record's replay metadata.

    Shared between `replay_with_config` and any future replay-based gate.
    Raises `ValueError` if required keys are missing or the model / RNG state
    is absent when needed.
    """
    from opt_agent.backend.checkpoint import Checkpoint

    meta = record.metadata
    required = (
        "backend_X", "backend_objectives", "backend_constraints",
        "backend_id", "backend_problem_id", "backend_problem_bounds",
        "backend_fitted", "backend_model_state_b64", "backend_rng_state_b64",
    )
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(
            f"record {record.id!r} missing replay keys {missing}. "
            "Re-generate dataset with Sprint 0 / Phase 8.6B replay support."
        )

    X = torch.tensor(meta["backend_X"], dtype=torch.double)
    Y = torch.tensor(meta["backend_objectives"], dtype=torch.double)
    C = torch.tensor(meta["backend_constraints"], dtype=torch.double)

    fitted = bool(meta["backend_fitted"])
    model_bytes = (
        base64.b64decode(meta["backend_model_state_b64"], validate=True)
        if meta["backend_model_state_b64"] is not None else None
    )
    rng_bytes = (
        base64.b64decode(meta["backend_rng_state_b64"], validate=True)
        if meta["backend_rng_state_b64"] is not None else None
    )
    if fitted and model_bytes is None:
        raise ValueError(f"record {record.id!r} has fitted GP but no model state")
    if rng_bytes is None:
        raise ValueError(f"record {record.id!r} has no RNG state")
    return Checkpoint(
        backend_id=str(meta["backend_id"]),
        problem_id=str(meta["backend_problem_id"]),
        problem_bounds=[tuple(b) for b in meta["backend_problem_bounds"]],
        X=X, objectives=Y, constraints=C,
        fitted=fitted,
        model_state_dict=model_bytes,
        torch_rng_state=rng_bytes,
    )


def _weights_from_metadata(meta: dict) -> UtilityWeights:
    """Reconstruct `UtilityWeights` from `metadata["utility_weights"]`.

    Falls back to Phase 8.6B single-key format (`w_stop_opportunity` only)
    for backward compatibility with records generated before Sprint 0.
    """
    uw = meta.get("utility_weights")
    if isinstance(uw, dict) and uw:
        return UtilityWeights.from_dict(uw)
    # Legacy fallback (Phase 8.6B and earlier): only w_stop_opportunity
    # was persisted; everything else uses the dataclass defaults.
    return UtilityWeights(w_stop_opportunity=float(meta.get("w_stop_opportunity", 0.75)))


def _affine_kwargs_from_metadata(meta: dict) -> dict:
    """Return `objective_scale` / `objective_shift` kwargs for backend ctor.

    Phase 9 Gate 5 — records generated by `DatasetGenerator` with a non-identity
    affine transform carry the two values in `metadata["objective_scale"]` and
    `metadata["objective_shift"]`. Records from before Track B (no fields) and
    from identity-transform runs both fall back to (1.0, 0.0).

    Used by `oracle_test_from_record` and `replay_with_config` so that future
    `evaluate()` calls append transformed values to the (already-transformed)
    checkpoint history. Without this thread, a replayed trajectory would mix
    raw and transformed objectives — silent corruption.

    The function validates `scale > 0` and finite `shift` so that a corrupted
    record raises immediately rather than silently corrupting the replay.
    """
    scale = float(meta.get("objective_scale", 1.0))
    shift = float(meta.get("objective_shift", 0.0))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"record metadata has invalid objective_scale={scale!r} "
            "(must be finite and > 0); record may be corrupted"
        )
    if not math.isfinite(shift):
        raise ValueError(
            f"record metadata has invalid objective_shift={shift!r} "
            "(must be finite); record may be corrupted"
        )
    return {"objective_scale": scale, "objective_shift": shift}


def _candidates_from_record(record: DecisionRecord) -> list:
    """Reconstruct the candidate list from `record.candidate_actions`."""
    from opt_agent.rollout.rollout import Candidate
    return [
        Candidate(
            action_id=ca["action_id"],
            action=OptimizationAction(ActionType(ca["type"]), dict(ca["params"])),
        )
        for ca in record.candidate_actions
    ]


def replay_with_config(
    record: DecisionRecord,
    *,
    continuation_spec: ContinuationPolicySpec | None = None,
    utility_weights: UtilityWeights | None = None,
    horizon: int | None = None,
    temperature: float | None = None,
    backend_factory: Callable | None = None,
) -> RolloutOutput:
    """Replay a `DecisionRecord` with a new (or original) rollout config.

    Loads the record's base checkpoint into a fresh backend, then runs
    `CounterfactualRollout` with the supplied (or original) config on the
    same candidate list. This is the Sprint 0 mechanism for the Q-S5
    diagnostic matrix: re-run any record with different
    {continuation_spec × utility_weights × horizon} without re-running
    the trajectory that produced the record.

    Parameters
    ----------
    record : DecisionRecord
        Must contain Sprint 0 / Phase 8.6B replay metadata
        (`backend_X`, ..., `backend_model_state_b64`, `backend_rng_state_b64`).
    continuation_spec : ContinuationPolicySpec | None
        Override for the continuation policy. Defaults to the spec recorded
        in `record.metadata["continuation_policy"]`.
    utility_weights : UtilityWeights | None
        Override for utility weights. Defaults to the weights recorded in
        `record.metadata["utility_weights"]` (or Phase 8.6B legacy fallback).
    horizon : int | None
        Override for rollout horizon. Defaults to `record.metadata["horizon"]`.
    temperature : float | None
        Recorded alongside the output for convenience (not used in this
        function — `gold_probs` are computed by the caller via `to_gold_probs`
        from the returned `RolloutOutput`).
    backend_factory : Callable | None
        Optional factory returning a fresh `UnifiedBackend`. Defaults to a
        `BocodeBotorchBackend` initialized with the problem from the record.

    Returns
    -------
    RolloutOutput
        New rollout results from the supplied config. The original record is
        not mutated. The backend is restored to its pre-call state.
    """
    from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend
    from opt_agent.rollout.rollout import CounterfactualRollout

    meta = record.metadata
    # Validate replay metadata FIRST so the user sees the missing-keys error
    # before any other metadata access (e.g. `meta["horizon"]`).
    base = _reconstruct_checkpoint(record)

    spec = continuation_spec if continuation_spec is not None else spec_from_metadata(
        meta.get("continuation_policy")
    )
    weights = utility_weights if utility_weights is not None else _weights_from_metadata(meta)
    h = horizon if horizon is not None else int(meta["horizon"])

    # Reconstruct backend.
    problem_id = str(meta["backend_problem_id"])
    if backend_factory is None:
        # Phase 9 Gate 5 — same as oracle_test_from_record: read affine
        # transform from metadata so evaluate() appends transformed values
        # to the (already-transformed) checkpoint history.
        affine = _affine_kwargs_from_metadata(meta)
        backend = BocodeBotorchBackend(**affine)
        backend.initialize(BackendConfig(
            problem_id=problem_id,
            seed=int(record.problem["seed"]),
            n_init=1,
        ))
    else:
        backend = backend_factory()
    try:
        backend.restore(base)
        state = extract_state(
            backend,
            total_budget=int(record.state["resource"]["total_budget"]),
            last_action=record.state["history"].get("last_action"),
        )
        if state.sampling.n_samples != int(record.state["sampling"]["n_samples"]):
            raise ValueError(
                f"record {record.id!r} replay sample count differs "
                f"({state.sampling.n_samples} vs "
                f"{record.state['sampling']['n_samples']})"
            )
        candidates = _candidates_from_record(record)
        rollout = CounterfactualRollout(
            backend,
            horizon=h,
            utility_weights=weights,
            continuation_spec=spec,
            optimization_scale=float(meta.get("optimization_scale", 1.0)),
            evaluation_scale=float(meta.get("evaluation_scale", 1.0)),
            fit_scale=float(meta.get("fit_scale", 1.0)),
        )
        return rollout.rollout(state, candidates)
    finally:
        # Always leave the backend at the original base state.
        backend.restore(base)

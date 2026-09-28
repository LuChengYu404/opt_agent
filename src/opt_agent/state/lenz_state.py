"""Lenz-style rich state: the (D_t, c_t, H_t) view of a BO run (2026-09-28).

Motivated by the agentic-BO design spectrum
(`docs/ref_papers/state_action_design_survey.md`): Sara/lenz
(Brunzema et al. 2026) condition the decision agent on the FULL state
tuple s_t = (D_t observed trials, c_t backend configuration, K_t exogenous
context, H_t deliberation history), while this project's frozen
``OptimizationState`` is a compressed 15-scalar projection of (D_t, c_t).

This module materializes the rich view WITHOUT touching the frozen design:

- ``LenzState`` carries D_t (every evaluated (x, y)), c_t (problem, bounds,
  surrogate/acquisition type, objective transform), H_t (the action history)
  — plus ``summary``, the standard ``OptimizationState`` projection, so any
  existing consumer (executable criteria, phase gating, RF features) keeps
  working unchanged.
- ``extract_lenz_state`` is a pure read over the backend, mirroring
  ``extract_state``'s no-mutation contract.
- K_t (exogenous context: problem description + standing instructions) is
  rendered at prompt time from ``config.problem`` — see
  ``opt_agent.system1.lenz_prompt``.

Branch ``exp/lenz-state-jev-presentation``; the frozen design on
``exp/rlcd-laya-finetune`` is the submission artifact and stays untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from opt_agent.backend.bocode_botorch import UnifiedBackend
from opt_agent.state.extractor import extract_state
from opt_agent.state.models import (
    HistoryState,
    OptimizationProgress,
    OptimizationState,
    ResourceState,
    SamplingState,
    SurrogateState,
)


@dataclass(frozen=True)
class LenzTrial:
    """One entry of D_t: an evaluated point and its (max-frame) objective."""

    x: tuple[float, ...]
    y: float


@dataclass(frozen=True)
class LenzConfig:
    """c_t: the backend configuration the agent may reason about.

    Mirrors what lenz exposes (problem + model/acquisition type + region +
    objective transform). In this project surrogate and acquisition are
    FIXED (SingleTaskGP + LogExpectedImprovement), so c_t is constant
    across a run except for the affine objective transform — recording it
    keeps the tuple shape honest for future reconfigure-style actions.
    """

    problem: str
    dimension: int
    bounds: tuple[tuple[float, float], ...]
    surrogate: str
    acquisition: str
    objective_transform: tuple[float, float]  # (scale, shift)
    surrogate_fitted: bool


@dataclass(frozen=True)
class LenzState:
    """Rich (lenz-style) view of a BO run plus the compressed projection.

    Attributes
    ----------
    trials : D_t — every evaluated (x, y), in evaluation order.
    config : c_t — backend configuration (see ``LenzConfig``).
    decision_history : H_t — high-level action ids taken so far, in order;
        ``None`` entries mark steps produced by the initial design (no
        high-level action). This is the memory the frozen design discards
        beyond ``last_action``.
    summary : the standard ``OptimizationState`` (15-scalar projection of
        (D_t, c_t)) — kept so executable criteria, phase gating, and the
        RF feature path work unchanged on top of a LenzState.
    """

    trials: tuple[LenzTrial, ...]
    config: LenzConfig
    decision_history: tuple[Optional[str], ...]
    summary: OptimizationState

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def remaining_budget(self) -> int:
        return self.summary.resource.remaining_budget

    def best_trial(self) -> LenzTrial:
        """Trial with the max objective (BOCoDe MAX frame)."""
        return max(self.trials, key=lambda t: t.y)


def extract_lenz_state(
    backend: UnifiedBackend,
    *,
    total_budget: int,
    decision_history: tuple[Optional[str], ...] = (),
) -> LenzState:
    """Build the rich state from a backend. Pure read (no mutation).

    ``decision_history`` is supplied by the caller (the DecisionLoop knows
    which high-level actions produced which samples); its length need not
    equal n_trials — the initial design contributes samples with no
    high-level action (``None`` entries).
    """
    if total_budget <= 0:
        raise ValueError(f"total_budget must be > 0; got {total_budget}")

    history = backend.get_history()
    n = len(history)
    if n and decision_history and len(decision_history) > n:
        raise ValueError(
            f"decision_history ({len(decision_history)} entries) is longer "
            f"than the backend history ({n} samples)"
        )

    bounds = backend.bounds
    if bounds is None:
        raise ValueError("backend must expose bounds for the lenz state")
    lo, hi = bounds[0].tolist(), bounds[1].tolist()

    config = backend.config
    trials = tuple(
        LenzTrial(x=tuple(float(v) for v in row), y=float(y))
        for row, y in zip(history.X.tolist(), history.objectives[:, 0].tolist())
    )

    lenz_config = LenzConfig(
        problem=config.problem_id if config is not None else "unknown",
        dimension=int(history.X.shape[1]),
        bounds=tuple(zip(lo, hi)),
        surrogate="SingleTaskGP",
        acquisition="LogExpectedImprovement",
        # Affine objective transform lives on the backend instance
        # (Gate 5 wiring), not on BackendConfig.
        objective_transform=(
            float(getattr(backend, "objective_scale", 1.0)),
            float(getattr(backend, "objective_shift", 0.0)),
        ),
        surrogate_fitted=bool(backend.is_fitted),
    )

    summary = extract_state(backend, total_budget=total_budget,
                            last_action=decision_history[-1] if decision_history else None)

    return LenzState(
        trials=trials,
        config=lenz_config,
        decision_history=tuple(decision_history),
        summary=summary,
    )


def _summary_from_record_state(d: dict) -> OptimizationState:
    """Rebuild the frozen OptimizationState from a record's serialized
    ``state`` dict (JSONL round-trip: NaN surrogate fields appear as
    ``None`` and are restored to NaN)."""
    import math

    def _f(v):
        return float("nan") if v is None else float(v)

    s, g, o, r, h = (
        d["sampling"], d["surrogate"], d["optimization"],
        d["resource"], d.get("history", {}),
    )
    return OptimizationState(
        sampling=SamplingState(
            n_samples=int(s["n_samples"]), dimension=int(s["dimension"]),
            min_pairwise_distance=_f(s["min_pairwise_distance"]),
            boundary_coverage=_f(s["boundary_coverage"]),
        ),
        surrogate=SurrogateState(
            is_current=bool(g["is_current"]),
            validation_rmse=_f(g["validation_rmse"]),
            mean_uncertainty=_f(g["mean_uncertainty"]),
            max_uncertainty=_f(g["max_uncertainty"]),
        ),
        optimization=OptimizationProgress(
            best_value=_f(o["best_value"]), last_value=_f(o["last_value"]),
            recent_improvement=_f(o["recent_improvement"]),
            stagnation_steps=int(o["stagnation_steps"]),
        ),
        resource=ResourceState(
            used_budget=int(r["used_budget"]), total_budget=int(r["total_budget"]),
            remaining_budget=int(r["remaining_budget"]),
            remaining_budget_ratio=_f(r["remaining_budget_ratio"]),
        ),
        history=HistoryState(last_action=h.get("last_action")),
    )


def lenz_state_from_record(record) -> LenzState:
    """Rebuild a LenzState from a serialized DecisionRecord (JSONL row).

    D_t and c_t are reconstructed from the record's replay metadata
    (``backend_X`` / ``backend_objectives`` / ``backend_problem_bounds`` /
    ``backend_problem_id`` / ``backend_fitted``); the compressed summary is
    rebuilt from the record's ``state`` block.

    H_t CAVEAT: the dataset records store only ``state.history.last_action``
    — the full decision sequence is NOT persisted per record. The rebuilt
    H_t is therefore truncated to that single entry (empty when the record
    is at the initial design). Full H_t is available only on live runs via
    ``extract_lenz_state``.
    """
    meta = record.metadata if hasattr(record, "metadata") else record["metadata"]
    state_d = record.state if hasattr(record, "state") else record["state"]

    X = meta["backend_X"]
    Y = [row[0] for row in meta["backend_objectives"]]
    trials = tuple(
        LenzTrial(x=tuple(float(v) for v in row), y=float(y))
        for row, y in zip(X, Y)
    )
    bounds = tuple(
        (float(lo), float(hi)) for lo, hi in meta["backend_problem_bounds"]
    )
    config = LenzConfig(
        problem=str(meta["backend_problem_id"]),
        dimension=len(bounds),
        bounds=bounds,
        surrogate="SingleTaskGP",
        acquisition="LogExpectedImprovement",
        objective_transform=(1.0, 0.0),  # h3 datasets carry no affine transform
        surrogate_fitted=bool(meta["backend_fitted"]),
    )
    last_action = state_d.get("history", {}).get("last_action")
    return LenzState(
        trials=trials,
        config=config,
        decision_history=(last_action,) if last_action else (),
        summary=_summary_from_record_state(state_d),
    )

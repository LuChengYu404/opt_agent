"""Counterfactual Rollout (Phase 6).

Per MASTER_PLAN.md §9–§10 and codex (gpt-5.6-sol) review 2025-09-23.

For each candidate action a_k, starting from the SAME state s_t:
    1. restore(base)             — rewind to s_t
    2. execute_action(a_k)
    3. (if horizon > 1) loop:    — follow-up policy picks the next action
         - branch_state = extract_state
         - action' = policy.select_action(branch_state, backend)
         - execute_action(action')
         - if result.should_stop or budget exhausted: break
    4. compute utility:  U(a|s) = w_opt · (best_after − best_before)
                            − w_eval · n_evaluations − w_fit · n_fits

Phase 6 specifics (per codex review):
  - Invalid candidates (e.g. budget violation) become status="invalid" entries
    instead of raising — they don't poison other candidates.
  - Backend is always restored to base via try/finally.
  - Candidates carry an `action_id` (string) so identical action types with
    different params don't collide in the results dict.
  - `to_gold_probs()` uses softmax; rejects empty / invalid-only / non-finite
    temperature / utilities.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from opt_agent.actions.executor import execute_action
from opt_agent.actions.models import OptimizationAction
from opt_agent.backend.bocode_botorch import UnifiedBackend
from opt_agent.backend.checkpoint import Checkpoint
from opt_agent.rollout.utility import UtilityWeights, compute_utility
from opt_agent.state import OptimizationState, extract_state


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A candidate action tagged with a stable id (used as results dict key)."""
    action_id: str
    action: OptimizationAction


@dataclass(frozen=True)
class CounterfactualResult:
    action_id: str
    action: OptimizationAction
    status: str  # "valid" | "invalid"
    n_new_samples: int
    n_model_fits: int
    delta_optimization: float
    delta_surrogate: float       # NaN if no surrogate metric before/after
    delta_coverage: float        # NaN if no coverage delta computable
    utility: float
    metrics: dict[str, float] = field(default_factory=dict)
    trajectory_tail: tuple = field(default_factory=tuple)
    error: Optional[str] = None


@dataclass(frozen=True)
class RolloutOutput:
    s_t: OptimizationState
    results: tuple[CounterfactualResult, ...]


# ---------------------------------------------------------------------------
# Counterfactual rollout
# ---------------------------------------------------------------------------


ContinuationFactory = Callable[[], "PolicyLike"]


class PolicyLike:
    """Structural type — actual policies from opt_agent.loop.policies."""

    def select_action(self, state, backend=None) -> OptimizationAction: ...


class CounterfactualRollout:
    """Run counterfactual rollouts from a state for a list of candidate actions.

    Parameters
    ----------
    backend : UnifiedBackend
    horizon : int
        How many action steps to simulate per branch (≥1).
    utility_weights : UtilityWeights
    continuation_policy_factory : Callable[[], PolicyLike] | None
        Factory producing a fresh policy for H>1 follow-up steps. Defaults to
        RulePolicy() if None.
    max_rollout_steps : int
        Hard cap on total (candidates × horizon) evaluations to prevent
        runaway computation.
    """

    def __init__(
        self,
        backend: UnifiedBackend,
        *,
        horizon: int = 1,
        utility_weights: UtilityWeights | None = None,
        continuation_policy_factory: ContinuationFactory | None = None,
        max_rollout_steps: int = 200,
        optimization_scale: float = 1.0,
        evaluation_scale: float = 1.0,
        fit_scale: float = 1.0,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1; got {horizon}")
        scales = (optimization_scale, evaluation_scale, fit_scale)
        if not all(math.isfinite(s) and s > 0 for s in scales):
            raise ValueError(
                f"utility scales must be finite and positive; got {scales}"
            )
        self.backend = backend
        self.horizon = horizon
        self.weights = utility_weights or UtilityWeights()
        self.continuation_policy_factory = continuation_policy_factory or _default_factory
        self.max_rollout_steps = max_rollout_steps
        self.optimization_scale = optimization_scale
        self.evaluation_scale = evaluation_scale
        self.fit_scale = fit_scale

    def rollout(
        self,
        state: OptimizationState,
        candidates: list[Candidate],
    ) -> RolloutOutput:
        if not candidates:
            raise ValueError("candidates must not be empty")
        # Reject duplicate action_ids — they would collide in results dict.
        ids = [c.action_id for c in candidates]
        duplicates = [i for i, c in Counter(ids).items() if c > 1]
        if duplicates:
            raise ValueError(
                f"duplicate action_id in candidates: {duplicates}. "
                f"Each candidate must have a unique id."
            )
        if len(candidates) * self.horizon > self.max_rollout_steps:
            raise ValueError(
                f"rollout step cap exceeded: {len(candidates)} candidates × "
                f"{self.horizon} horizon = {len(candidates) * self.horizon} > "
                f"{self.max_rollout_steps}"
            )

        base = self.backend.checkpoint()
        observed = extract_state(
            self.backend,
            total_budget=state.resource.total_budget,
            last_action=state.history.last_action,
        )
        # State sanity check: extract_state from current backend must match the
        # passed state (otherwise caller passed stale state).
        if observed.sampling.n_samples != state.sampling.n_samples:
            raise ValueError(
                f"State does not describe the checkpointed backend: "
                f"state has n_samples={state.sampling.n_samples}, "
                f"backend has {observed.sampling.n_samples}."
            )

        results: list[CounterfactualResult] = []
        try:
            for candidate in candidates:
                results.append(self._run_one_branch(state, candidate, base))
        finally:
            self.backend.restore(base)

        return RolloutOutput(s_t=state, results=tuple(results))

    def _run_one_branch(
        self,
        state: OptimizationState,
        candidate: Candidate,
        base: Checkpoint,
    ) -> CounterfactualResult:
        self.backend.restore(base)
        # Re-extract branch's starting state from the restored backend.
        branch_state = extract_state(
            self.backend,
            total_budget=state.resource.total_budget,
            last_action=state.history.last_action,
        )
        steps: list = []
        n_new = 0
        n_fits = 0
        status = "valid"
        error: Optional[str] = None
        policy = (
            self.continuation_policy_factory() if self.horizon > 1 else None
        )
        try:
            action = candidate.action
            for step_index in range(self.horizon):
                if step_index > 0:
                    if branch_state.resource.remaining_budget <= 0:
                        break
                    assert policy is not None
                    action = policy.select_action(branch_state, self.backend)

                result = execute_action(self.backend, branch_state, action)
                n_new += result.n_new_samples
                n_fits += result.n_model_fits

                next_state = extract_state(
                    self.backend,
                    total_budget=state.resource.total_budget,
                    last_action=action.type.value,
                )
                steps.append((branch_state, action, result, next_state))
                branch_state = next_state

                if result.should_stop:
                    break
        except ValueError as exc:
            status, error = "invalid", str(exc)

        # Phase 7: Δopt = best_after − best_before (positive = improvement).
        delta_opt = branch_state.optimization.best_value - state.optimization.best_value

        # Phase 7: Δsurrogate = validation_rmse_before − validation_rmse_after
        # (positive = surrogate improved). NaN if either side is NaN.
        rs_before = state.surrogate.validation_rmse
        rs_after = branch_state.surrogate.validation_rmse
        if math.isfinite(rs_before) and math.isfinite(rs_after):
            delta_surrogate = float(rs_before - rs_after)
        else:
            delta_surrogate = float("nan")

        # Phase 7: Δcoverage = boundary_coverage_after − boundary_coverage_before
        # (positive = more boundary coverage).
        delta_coverage = float(
            branch_state.sampling.boundary_coverage
            - state.sampling.boundary_coverage
        )

        utility = compute_utility(
            delta_optimization=delta_opt,
            delta_surrogate=delta_surrogate,
            delta_coverage=delta_coverage,
            n_new_samples=n_new,
            n_model_fits=n_fits,
            weights=self.weights,
            optimization_scale=self.optimization_scale,
            evaluation_scale=self.evaluation_scale,
            fit_scale=self.fit_scale,
            is_stop=(candidate.action.type.name == "STOP"),
            remaining_budget_ratio=branch_state.resource.remaining_budget_ratio,
        )
        return CounterfactualResult(
            action_id=candidate.action_id,
            action=candidate.action,
            status=status,
            n_new_samples=n_new,
            n_model_fits=n_fits,
            delta_optimization=float(delta_opt),
            delta_surrogate=float(delta_surrogate),
            delta_coverage=float(delta_coverage),
            utility=float(utility),
            metrics={
                "best_before": float(state.optimization.best_value),
                "best_after": float(branch_state.optimization.best_value),
                "rmse_before": float(rs_before),
                "rmse_after": float(rs_after),
                "boundary_coverage_before": float(state.sampling.boundary_coverage),
                "boundary_coverage_after": float(branch_state.sampling.boundary_coverage),
            },
            trajectory_tail=tuple(steps),
            error=error,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_factory() -> PolicyLike:
    """Default continuation policy: a fresh RulePolicy."""
    from opt_agent.loop.policies import RulePolicy
    return RulePolicy()


def to_gold_probs(
    results: tuple[CounterfactualResult, ...],
    *,
    temperature: float = 1.0,
    include_invalid_as_zero: bool = False,
) -> dict[str, float]:
    """Convert CounterfactualResult utilities into softmax probabilities.

    Parameters
    ----------
    include_invalid_as_zero : bool
        If True, invalid candidates are included in the output dict with
        probability 0.0 (preserves aligned vocabulary). Default False: only
        valid candidates are returned (vocabulary may shrink).
    """
    if not results:
        raise ValueError("results must not be empty")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            f"temperature must be finite and > 0; got {temperature}"
        )
    valid = [r for r in results if r.status == "valid"]
    if not valid:
        if not include_invalid_as_zero:
            raise ValueError("no valid candidate actions")
        # All invalid; emit zeros so vocabulary is preserved.
        return {r.action_id: 0.0 for r in results}
    utilities = np.asarray([r.utility for r in valid], dtype=np.float64)
    if not np.isfinite(utilities).all():
        raise ValueError("valid utilities must be finite")
    z = utilities / temperature
    z -= z.max()
    probabilities = np.exp(z)
    probabilities /= probabilities.sum()
    valid_probs = {
        r.action_id: float(p)
        for r, p in zip(valid, probabilities, strict=True)
    }
    if include_invalid_as_zero:
        for r in results:
            if r.status != "valid":
                valid_probs[r.action_id] = 0.0
    return valid_probs

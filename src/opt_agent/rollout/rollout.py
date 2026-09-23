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
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1; got {horizon}")
        self.backend = backend
        self.horizon = horizon
        self.weights = utility_weights or UtilityWeights()
        self.continuation_policy_factory = continuation_policy_factory or _default_factory
        self.max_rollout_steps = max_rollout_steps

    def rollout(
        self,
        state: OptimizationState,
        candidates: list[Candidate],
    ) -> RolloutOutput:
        if not candidates:
            raise ValueError("candidates must not be empty")
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

        delta_opt = branch_state.optimization.best_value - state.optimization.best_value
        utility = compute_utility(
            delta_optimization=delta_opt,
            n_new_samples=n_new,
            n_model_fits=n_fits,
            weights=self.weights,
        )
        return CounterfactualResult(
            action_id=candidate.action_id,
            action=candidate.action,
            status=status,
            n_new_samples=n_new,
            n_model_fits=n_fits,
            delta_optimization=float(delta_opt),
            utility=float(utility),
            metrics={
                "best_before": float(state.optimization.best_value),
                "best_after": float(branch_state.optimization.best_value),
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
) -> dict[str, float]:
    """Convert CounterfactualResult utilities into softmax probabilities."""
    if not results:
        raise ValueError("results must not be empty")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            f"temperature must be finite and > 0; got {temperature}"
        )
    valid = [r for r in results if r.status == "valid"]
    if not valid:
        raise ValueError("no valid candidate actions")
    utilities = np.asarray([r.utility for r in valid], dtype=np.float64)
    if not np.isfinite(utilities).all():
        raise ValueError("valid utilities must be finite")
    z = utilities / temperature
    z -= z.max()
    probabilities = np.exp(z)
    probabilities /= probabilities.sum()
    return {
        r.action_id: float(p)
        for r, p in zip(valid, probabilities, strict=True)
    }

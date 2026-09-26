"""DecisionLoop — Phase 4 closure of state → action → state cycle.

Per MASTER_PLAN.md §32 acceptance criterion:
    state = env.reset()
    action = policy(state)
    next_state, result = env.step(action)

This module provides:
  - `DecisionLoop` class wrapping the cycle
  - `Trajectory` / `TrajectoryStep` dataclasses for logging
  - `TerminationReason` enum

Codex (gpt-5.6-sol) review 2025-09-23 informed the design:
  - `step(action)` takes an action directly (no internal policy call) — makes
    testing trivial: caller chooses the action, loop just enforces invariants.
  - `run(policy, ...)` is the higher-level convenience that combines
    extract_state + policy.select_action + step.
  - `TerminationReason` enum instead of free-form strings.
  - `reset()` clears bookkeeping but NOT backend history.
  - `step()` refuses external backend mutations outside the loop (best-effort
    via n_samples check).
  - Failed execution does not commit a step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from opt_agent.actions import ActionResult, OptimizationAction, execute_action
from opt_agent.backend.bocode_botorch import UnifiedBackend
from opt_agent.state import OptimizationState, extract_state


class TerminationReason(str, Enum):
    STOP = "stop"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MAX_ROUNDS = "max_rounds"


@dataclass(frozen=True)
class TrajectoryStep:
    round: int
    state: OptimizationState
    action: OptimizationAction
    result: ActionResult
    next_state: OptimizationState


@dataclass(frozen=True)
class Trajectory:
    steps: tuple[TrajectoryStep, ...]
    final_state: OptimizationState
    termination_reason: TerminationReason


class DecisionLoop:
    """Wraps the state → action → state cycle.

    Two modes:
      - Manual: caller calls `reset()` then `step(action)` repeatedly.
      - Automatic: caller calls `run(policy, max_rounds=...)`.
    """

    def __init__(
        self,
        backend: UnifiedBackend,
        *,
        total_budget: int,
    ) -> None:
        if total_budget <= 0:
            raise ValueError(f"total_budget must be > 0; got {total_budget}")
        self.backend = backend
        self.total_budget = total_budget
        self._round = 0
        self._steps: list[TrajectoryStep] = []
        self._state: OptimizationState | None = None
        self._termination_reason: TerminationReason | None = None
        self._last_seen_n: int | None = None

    # -- public read-only views --------------------------------------------

    @property
    def state(self) -> OptimizationState:
        if self._state is None:
            raise RuntimeError("DecisionLoop not reset; call reset() first.")
        return self._state

    @property
    def steps(self) -> tuple[TrajectoryStep, ...]:
        return tuple(self._steps)

    @property
    def termination_reason(self) -> TerminationReason | None:
        return self._termination_reason

    # -- core API ----------------------------------------------------------

    def reset(self) -> OptimizationState:
        """Snapshot initial state. Does NOT clear backend history."""
        self._round = 0
        self._steps = []
        self._termination_reason = None
        self._state = extract_state(self.backend, total_budget=self.total_budget)
        self._last_seen_n = len(self.backend.get_history())
        return self._state

    def step(self, action: OptimizationAction) -> tuple[OptimizationState, ActionResult]:
        """Execute one action and return (next_state, result).

        Refuses if the loop has not been reset, has already terminated, or if
        the backend has been mutated externally since reset/last step.
        """
        if self._state is None:
            raise RuntimeError(
                "DecisionLoop not reset; call reset() before step()."
            )
        if self._termination_reason is not None:
            raise RuntimeError(
                f"DecisionLoop already terminated: {self._termination_reason.value}"
            )

        # External-mutation detection.
        current_n = len(self.backend.get_history())
        if self._last_seen_n is not None and current_n != self._last_seen_n:
            raise ValueError(
                f"Backend was mutated outside DecisionLoop "
                f"(n went from {self._last_seen_n} to {current_n}). "
                f"Call reset() to refresh bookkeeping."
            )

        # Execute. If it raises, do NOT commit the step.
        next_state_obj, result = _execute_and_extract(
            self.backend, self._state, action, total_budget=self.total_budget
        )
        step = TrajectoryStep(
            round=self._round,
            state=self._state,
            action=action,
            result=result,
            next_state=next_state_obj,
        )
        self._steps.append(step)
        self._state = next_state_obj
        self._last_seen_n = len(self.backend.get_history())
        self._round += 1
        return next_state_obj, result

    def run(
        self,
        policy,
        *,
        max_rounds: int | None = None,
    ) -> Trajectory:
        """Run until STOP, budget exhausted, or max_rounds.

        `policy` must implement `select_action(state) -> OptimizationAction`.
        """
        if self._state is None:
            self.reset()

        while True:
            assert self._state is not None  # mypy hint

            # Stop conditions BEFORE picking an action.
            if max_rounds is not None and len(self._steps) >= max_rounds:
                self._termination_reason = TerminationReason.MAX_ROUNDS
                break
            if self._state.resource.remaining_budget <= 0:
                self._termination_reason = TerminationReason.BUDGET_EXHAUSTED
                break

            action = policy.select_action(self._state)
            _, result = self.step(action)

            if result.should_stop:
                self._termination_reason = TerminationReason.STOP
                break

        assert self._termination_reason is not None
        assert self._state is not None
        return Trajectory(
            steps=tuple(self._steps),
            final_state=self._state,
            termination_reason=self._termination_reason,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def execute_and_extract(
    backend: UnifiedBackend,
    state: OptimizationState,
    action: OptimizationAction,
    *,
    total_budget: int,
) -> tuple[OptimizationState, ActionResult]:
    """Run ``execute_action`` then re-extract state.

    Used by DecisionLoop, CounterfactualRollout, and oracle replay arms.
    The state-after-execute is whatever the backend has now plus the
    just-applied action's label as ``last_action``.
    """
    result = execute_action(backend, state, action)
    next_state = extract_state(
        backend, total_budget=total_budget, last_action=action.type.value,
    )
    return next_state, result


# Backwards-compatible alias for callers that imported the private name.
_execute_and_extract = execute_and_extract

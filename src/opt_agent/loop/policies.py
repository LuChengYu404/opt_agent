"""Phase 4 baseline policies: RandomPolicy and RulePolicy.

Per MASTER_PLAN.md §12 evaluation ladder: Random → Rule-based → LLM → Laya.
"""

from __future__ import annotations

import random
from typing import Protocol

from opt_agent.actions import ActionResult, ActionType, OptimizationAction
from opt_agent.backend.bocode_botorch import UnifiedBackend
from opt_agent.state import OptimizationState


class Policy(Protocol):
    """A decision policy: state → action."""

    def select_action(
        self, state: OptimizationState, backend: UnifiedBackend | None = None
    ) -> OptimizationAction:
        ...

    def reset(self) -> None:
        ...


class RandomPolicy:
    """Uniformly samples one of the 5 action types with valid params.

    GLOBAL_EXPLORATION's n_samples is bounded by `state.resource.remaining_budget`.
    RulePolicy mirrors the same logic (no point in proposing more than that).
    """

    _TYPES_WITH_PARAMS = (ActionType.GLOBAL_EXPLORATION, ActionType.LOCAL_EXPLOITATION)

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)
        self._initial_seed = seed

    def reset(self) -> None:
        self._rng = random.Random(self._initial_seed)

    def select_action(
        self, state: OptimizationState, backend: UnifiedBackend | None = None
    ) -> OptimizationAction:
        remaining = state.resource.remaining_budget
        if remaining <= 0:
            return OptimizationAction(ActionType.STOP)

        # Pick type, weighted equally.
        action_type = self._rng.choice(list(ActionType))

        if action_type == ActionType.GLOBAL_EXPLORATION:
            max_n = min(3, remaining)
            n_samples = self._rng.randint(1, max_n)
            return OptimizationAction(
                ActionType.GLOBAL_EXPLORATION,
                {"n_samples": n_samples, "seed": self._rng.randint(0, 10_000)},
            )
        if action_type == ActionType.LOCAL_EXPLOITATION:
            return OptimizationAction(
                ActionType.LOCAL_EXPLOITATION,
                {"step": round(self._rng.uniform(0.01, 0.2), 4),
                 "seed": self._rng.randint(0, 10_000)},
            )
        return OptimizationAction(action_type)


class RulePolicy:
    """Simple rule-based policy (Phase 4 baseline).

    Default action: SURROGATE_ACQUISITION.
    Switch to GLOBAL_EXPLORATION with n_samples=2 after 3 stagnant steps.
    Return STOP when remaining budget <= 4.
    """

    STAGNATION_THRESHOLD = 3
    LOW_BUDGET_THRESHOLD = 4

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def reset(self) -> None:
        # Stateless — no RNG.
        pass

    def select_action(
        self, state: OptimizationState, backend: UnifiedBackend | None = None
    ) -> OptimizationAction:
        if state.resource.remaining_budget <= self.LOW_BUDGET_THRESHOLD:
            return OptimizationAction(ActionType.STOP)
        if (
            state.optimization.stagnation_steps >= self.STAGNATION_THRESHOLD
            and state.resource.remaining_budget >= 2
        ):
            return OptimizationAction(
                ActionType.GLOBAL_EXPLORATION,
                {"n_samples": 2, "seed": self._seed + state.sampling.n_samples},
            )
        return OptimizationAction(ActionType.SURROGATE_ACQUISITION)

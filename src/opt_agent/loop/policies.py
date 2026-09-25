"""Phase 4 baseline policies: RandomPolicy and RulePolicy.

Per MASTER_PLAN.md §12 evaluation ladder: Random → Rule-based → LLM → Laya.

Sprint 0 additions:
  - UniformPolicy: samples uniformly over ALL 8 ActionTypes (vs RandomPolicy's 5).
  - OnPolicyMix:    RulePolicy with prob `rule_weight`, else RandomPolicy.
                    Mirrors DatasetGenerator's behavior mix (rule_weight = 1 −
                    random_action_rate = 0.8 by default).
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


class UniformPolicy:
    """Samples uniformly over ALL 8 ActionTypes (vs RandomPolicy's 5).

    Per codex (gpt-6-sol) Q-T2 review: a true state-aware uniform is preferred
    over RandomPolicy for ablation, because RandomPolicy currently samples only
    from 5 action types (excluding RETRAIN_SURROGATE / SURROGATE_ACQUISITION /
    STOP variants that already exist as actions) and the asymmetry can bias
    Gate 4 results. UniformPolicy also fills in valid params (n_samples,
    step, tol, seed) for the parameterized actions so the action is
    budget-feasible.

    Distinct from RandomPolicy in two ways:
      1. samples over all 8 ActionTypes (not 5).
      2. requires explicit valid-param construction for parameterized actions
         (GLOBAL_EXPLORATION, LOCAL_EXPLOITATION, UNCERTAINTY_SAMPLING,
         ERROR_GUIDED_SAMPLING, BOUNDARY_SAMPLING).
    """

    _PARAM_TYPES = (
        ActionType.GLOBAL_EXPLORATION,
        ActionType.LOCAL_EXPLOITATION,
        ActionType.UNCERTAINTY_SAMPLING,
        ActionType.ERROR_GUIDED_SAMPLING,
        ActionType.BOUNDARY_SAMPLING,
    )

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
        action_type = self._rng.choice(list(ActionType))
        return self._build(action_type, remaining)

    def _build(
        self, action_type: ActionType, remaining: int
    ) -> OptimizationAction:
        if action_type == ActionType.GLOBAL_EXPLORATION:
            n = self._rng.randint(1, min(3, remaining))
            return OptimizationAction(
                ActionType.GLOBAL_EXPLORATION,
                {"n_samples": n, "seed": self._rng.randint(0, 10_000)},
            )
        if action_type == ActionType.LOCAL_EXPLOITATION:
            return OptimizationAction(
                ActionType.LOCAL_EXPLOITATION,
                {
                    "step": round(self._rng.uniform(0.01, 0.2), 4),
                    "seed": self._rng.randint(0, 10_000),
                },
            )
        if action_type == ActionType.UNCERTAINTY_SAMPLING:
            n = self._rng.randint(1, min(3, remaining))
            return OptimizationAction(
                ActionType.UNCERTAINTY_SAMPLING,
                {"n_samples": n, "seed": self._rng.randint(0, 10_000)},
            )
        if action_type == ActionType.ERROR_GUIDED_SAMPLING:
            n = self._rng.randint(1, min(3, remaining))
            return OptimizationAction(
                ActionType.ERROR_GUIDED_SAMPLING,
                {"n_samples": n, "seed": self._rng.randint(0, 10_000)},
            )
        if action_type == ActionType.BOUNDARY_SAMPLING:
            n = self._rng.randint(1, min(3, remaining))
            return OptimizationAction(
                ActionType.BOUNDARY_SAMPLING,
                {
                    "n_samples": n,
                    "tol": round(self._rng.uniform(0.01, 0.10), 4),
                    "seed": self._rng.randint(0, 10_000),
                },
            )
        # Param-free actions.
        return OptimizationAction(action_type)


class OnPolicyMix:
    """The behavior policy mix used by `DatasetGenerator`.

    With probability `rule_weight` pick `RulePolicy`'s choice; otherwise pick
    `RandomPolicy`'s choice. Default `rule_weight=0.8` matches the generator's
    `random_action_rate=0.2` default.

    Per codex (gpt-6-sol) Q-T2 review: this is the "behavior" continuation
    used as a baseline in counterfactual rollout ablations — the original
    policy that generated the trajectory.
    """

    def __init__(self, seed: int = 0, rule_weight: float = 0.8) -> None:
        if not 0.0 <= rule_weight <= 1.0:
            raise ValueError(f"rule_weight must be in [0, 1]; got {rule_weight}")
        self._rng = random.Random(seed)
        self._rule = RulePolicy(seed=seed)
        self._random = RandomPolicy(seed=seed)
        self._rule_weight = float(rule_weight)
        self._initial_seed = seed

    def reset(self) -> None:
        self._rng = random.Random(self._initial_seed)
        self._rule.reset()
        self._random.reset()

    def select_action(
        self, state: OptimizationState, backend: UnifiedBackend | None = None
    ) -> OptimizationAction:
        if self._rng.random() < self._rule_weight:
            return self._rule.select_action(state, backend)
        return self._random.select_action(state, backend)

"""HybridPolicy — Phase 9 selective-reasoning glue.

Combines a System-1 (`decide(state) -> (action, probabilities, ...)`),
a `Router`, and a `System2Policy` behind the existing `Policy` protocol so
`DecisionLoop.run(policy)` can drive it unchanged.

Per codex gpt-6-sol review 2026-09-23 (major M4):
  - DecisionLoop is left intact. HybridPolicy implements `select_action`
    so it can be passed to `DecisionLoop(backend).run(policy)`.
  - Each step: ask system1 for an action + probability distribution;
    let the router decide whether to escalate; if so, delegate to
    system2. Routing metadata is recorded on `last_route` for analysis.

The system1 callable must return an object with `.action` (OptimizationAction)
and `.probabilities` (dict[str, float]). `LayaPolicy.decide()` already
satisfies this. For RulePolicy / RandomPolicy that lack `decide()`, use
`adapt_policy_as_system1(...)` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from opt_agent.actions import OptimizationAction, candidate_actions_for_state
from opt_agent.loop.policies import Policy
from opt_agent.router import Route, Router
from opt_agent.state import OptimizationState
from opt_agent.system2 import System2Policy


@runtime_checkable
class System1Decider(Protocol):
    """Anything with a `decide(state) -> (action, probabilities, ...)`.

    Implemented by `LayaPolicy` directly. For stub policies, use
    `adapt_policy_as_system1`.
    """

    def decide(self, state: OptimizationState) -> object: ...


@dataclass(frozen=True)
class HybridStep:
    """One step's worth of routing metadata.

    Attributes
    ----------
    route : Route
        The router's decision for this step.
    system1_action : OptimizationAction
        The action System-1 would have taken (recorded even on escalation
        so we can study the gap between system-1 and system-2 outputs).
    chosen_action : OptimizationAction
        The action actually returned to the DecisionLoop.
    """

    route: Route
    system1_action: OptimizationAction
    chosen_action: OptimizationAction


def adapt_policy_as_system1(policy: Policy) -> System1Decider:
    """Wrap a Policy-only system1 (RandomPolicy, RulePolicy, ...) as a
    System1Decider with delta probabilities (1.0 on the chosen action).

    The probabilities here are not calibrated; they simply reflect the
    policy's argmax. Calibration metrics should NOT be run on the
    resulting distribution.
    """

    class _Adapter:
        def __init__(self, p: Policy) -> None:
            self._policy = p
            self.last_probabilities: dict[str, float] = {}

        def decide(self, state: OptimizationState) -> object:
            action = self._policy.select_action(state)
            probs = {action.type.value: 1.0}
            self.last_probabilities = probs
            return _DecisionLike(action=action, probabilities=probs)

    return _Adapter(policy)


@dataclass(frozen=True)
class _DecisionLike:
    action: OptimizationAction
    probabilities: dict[str, float]


class HybridPolicy:
    """Selective-reasoning policy: System-1 + Router + System-2.

    Implements `Policy` so `DecisionLoop.run(policy)` is unchanged.
    """

    def __init__(
        self,
        system1: System1Decider,
        router: Router,
        system2: System2Policy,
    ) -> None:
        if not isinstance(system1, System1Decider):
            raise TypeError(
                "system1 must implement .decide(state) returning an "
                "object with .action and .probabilities; use "
                "adapt_policy_as_system1(policy) to wrap a Policy-only one."
            )
        self._system1 = system1
        self._router = router
        self._system2 = system2
        self.last_step: HybridStep | None = None
        self.history: list[HybridStep] = []

    def select_action(
        self, state: OptimizationState, backend=None
    ) -> OptimizationAction:
        decision = self._system1.decide(state)
        try:
            action = decision.action
            probs = dict(decision.probabilities)
        except AttributeError as exc:
            raise RuntimeError(
                "system1.decide() must return an object with .action and "
                ".probabilities"
            ) from exc
        if not probs:
            raise ValueError("system1 returned empty probabilities")
        # Verify the system-1 action is among the offered candidates.
        # Per codex M1: action_id must come from the shared candidate
        # factory, not be invented post hoc. We cross-check both:
        #   (a) the system-1 vocabulary (probs.keys()), AND
        #   (b) the state-filtered factory vocabulary
        #       (candidate_actions_for_state(state)).keys()
        # so a model that emits its own vocabulary cannot drift from
        # the factory's vocabulary.
        action_id = action.type.value
        if action_id not in probs:
            raise ValueError(
                f"system1 action {action_id!r} is not in its own probability "
                f"distribution {sorted(probs)}; check that the candidate "
                f"factory and the model output share the same vocabulary."
            )
        factory_candidates = candidate_actions_for_state(state, seed=0)
        if action_id not in factory_candidates:
            raise ValueError(
                f"system1 action {action_id!r} is not in the state-filtered "
                f"candidate factory's vocabulary "
                f"{sorted(factory_candidates)}; the model is selecting an "
                f"action that is not budget-valid for the current state."
            )
        # Per codex re-review: extra keys in `probs` would silently change
        # `max(probabilities.values())` and the router's escalation
        # decision. Reject any key not in the factory vocabulary.
        extra_keys = set(probs) - set(factory_candidates)
        if extra_keys:
            raise ValueError(
                f"system1 probability distribution contains keys not in "
                f"the candidate factory vocabulary: {sorted(extra_keys)}; "
                f"this would corrupt the router's confidence signal."
            )
        # Per codex re-review: also validate that the chosen action's
        # parameters are budget-valid (e.g. n_samples ≤ remaining_budget).
        # Without this, the ID check passes but execute_action raises
        # at runtime when an n_samples overshoot hits the budget.
        params = dict(action.params)
        if "n_samples" in params and params["n_samples"] > state.resource.remaining_budget:
            raise ValueError(
                f"system1 action {action_id!r} requests n_samples="
                f"{params['n_samples']} but only "
                f"{state.resource.remaining_budget} budget remains."
            )
        # Phase 12: register the current state with the router so an
        # OOD-aware router can consult the OOD detector. Routers that
        # accept a `state` parameter (per OODAwareRouter.update) are
        # notified; the base Router Protocol's update() is a 3-arg
        # online-learning hook and must NOT be called here.
        import inspect
        update_sig = inspect.signature(self._router.update)
        if "state" in update_sig.parameters:
            self._router.update(state=state)  # type: ignore[call-arg]
        route = self._router.route(probs)
        if route.name == "system1":
            chosen = action
        else:
            # Hand the System-2 policy the System-1 top action so its
            # fallback (if the LLM response is unusable) prefers the
            # most-confident available choice over an arbitrary pick.
            if hasattr(self._system2, "last_system1_top_action"):
                self._system2.last_system1_top_action = action  # type: ignore[attr-defined]
            chosen = self._system2.reason(state).action
            # Per codex re-review: System-2's chosen action must also be
            # budget-valid for the current state.
            chosen_id = chosen.type.value
            if chosen_id not in factory_candidates:
                raise RuntimeError(
                    f"system2 returned action {chosen_id!r} which is not "
                    f"in the candidate factory vocabulary "
                    f"{sorted(factory_candidates)}"
                )
            if "n_samples" in chosen.params and (
                chosen.params["n_samples"] > state.resource.remaining_budget
            ):
                raise RuntimeError(
                    f"system2 returned action {chosen_id!r} with "
                    f"n_samples={chosen.params['n_samples']} but only "
                    f"{state.resource.remaining_budget} budget remains."
                )
        self.last_step = HybridStep(
            route=route, system1_action=action, chosen_action=chosen,
        )
        self.history.append(self.last_step)
        return chosen

    def reset(self) -> None:
        self.last_step = None
        self.history.clear()
        # System-2 reset; System-1 and Router reset if they expose it.
        self._system2.reset()
        if hasattr(self._system1, "reset"):
            self._system1.reset()

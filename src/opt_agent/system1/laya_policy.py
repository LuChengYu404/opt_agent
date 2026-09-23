"""Zero-shot Laya reference policy for the existing optimization decision loop.

Laya is optional and loaded only when the first decision is requested. The
policy exposes the full choice distribution for later calibration and routing;
these probabilities are *model outputs*, not validated BO action probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from opt_agent.actions import ActionType, OptimizationAction
from opt_agent.backend.bocode_botorch import UnifiedBackend
from opt_agent.state import OptimizationState


class _LayaAgent(Protocol):
    def predict(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


_DESCRIPTIONS: dict[ActionType, str] = {
    ActionType.GLOBAL_EXPLORATION: "Evaluate new space-filling points across the search domain.",
    ActionType.LOCAL_EXPLOITATION: "Evaluate a small perturbation of the best observed point.",
    ActionType.UNCERTAINTY_SAMPLING: "Evaluate where the surrogate predicts high uncertainty.",
    ActionType.ERROR_GUIDED_SAMPLING: "Evaluate near a point with high surrogate residual.",
    ActionType.BOUNDARY_SAMPLING: "Evaluate points near the search-domain boundary.",
    ActionType.SURROGATE_ACQUISITION: "Evaluate the surrogate acquisition function's proposed point.",
    ActionType.RETRAIN_SURROGATE: "Refit the surrogate without using an evaluation.",
    ActionType.STOP: "Stop optimization and return the best observed point.",
}


def _finite(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _state_payload(state: OptimizationState) -> dict[str, Any]:
    """Keep the BO state compact enough for Laya's shared state/option context."""
    return {
        "objective": "maximize",
        "samples": state.sampling.n_samples,
        "dimension": state.sampling.dimension,
        "min_pairwise_distance": _finite(state.sampling.min_pairwise_distance),
        "boundary_coverage": _finite(state.sampling.boundary_coverage),
        "surrogate_current": state.surrogate.is_current,
        "validation_rmse": _finite(state.surrogate.validation_rmse),
        "mean_uncertainty": _finite(state.surrogate.mean_uncertainty),
        "max_uncertainty": _finite(state.surrogate.max_uncertainty),
        "best_value": _finite(state.optimization.best_value),
        "recent_improvement": _finite(state.optimization.recent_improvement),
        "stagnation_steps": state.optimization.stagnation_steps,
        "remaining_budget": state.resource.remaining_budget,
        "total_budget": state.resource.total_budget,
        "last_action": state.history.last_action,
    }


def _candidate_actions(state: OptimizationState, seed: int) -> dict[str, OptimizationAction]:
    remaining = state.resource.remaining_budget
    if remaining <= 0:
        return {ActionType.STOP.value: OptimizationAction(ActionType.STOP)}
    action_seed = seed + state.sampling.n_samples
    choices = {
        ActionType.GLOBAL_EXPLORATION.value: OptimizationAction(
            ActionType.GLOBAL_EXPLORATION,
            {"n_samples": min(2, remaining), "seed": action_seed},
        ),
        ActionType.UNCERTAINTY_SAMPLING.value: OptimizationAction(
            ActionType.UNCERTAINTY_SAMPLING, {"n_samples": 1, "seed": action_seed},
        ),
        ActionType.BOUNDARY_SAMPLING.value: OptimizationAction(
            ActionType.BOUNDARY_SAMPLING,
            {"n_samples": min(2, remaining), "tol": 0.05, "seed": action_seed},
        ),
        ActionType.SURROGATE_ACQUISITION.value: OptimizationAction(
            ActionType.SURROGATE_ACQUISITION
        ),
        ActionType.STOP.value: OptimizationAction(ActionType.STOP),
    }
    if state.sampling.n_samples:
        choices[ActionType.LOCAL_EXPLOITATION.value] = OptimizationAction(
            ActionType.LOCAL_EXPLOITATION, {"step": 0.1, "seed": action_seed}
        )
        choices[ActionType.ERROR_GUIDED_SAMPLING.value] = OptimizationAction(
            ActionType.ERROR_GUIDED_SAMPLING, {"n_samples": 1, "seed": action_seed}
        )
    if not state.surrogate.is_current:
        choices[ActionType.RETRAIN_SURROGATE.value] = OptimizationAction(
            ActionType.RETRAIN_SURROGATE
        )
    return choices


@dataclass(frozen=True)
class LayaDecision:
    action: OptimizationAction
    probabilities: dict[str, float]
    confidence: float


class LayaPolicy:
    """Wrap Laya's typed ``choice`` prediction as a DecisionLoop policy.

    ``agent`` may be injected for tests or a locally preloaded checkpoint. The
    default loads ``convaiinnovations/laya`` lazily via the optional SDK.
    """

    def __init__(
        self,
        *,
        model_id: str = "convaiinnovations/laya",
        subfolder: str | None = None,
        seed: int = 0,
        agent: _LayaAgent | None = None,
    ) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.model_id = model_id
        self.subfolder = subfolder
        self.seed = seed
        self._agent = agent
        self.last_decision: LayaDecision | None = None
        self.decisions: list[LayaDecision] = []

    def reset(self) -> None:
        self.last_decision = None
        self.decisions.clear()

    def _get_agent(self) -> _LayaAgent:
        if self._agent is None:
            try:
                import laya
            except ImportError as exc:
                raise RuntimeError(
                    "Laya is not installed. Install the optional dependency with "
                    "`uv sync --extra laya` before running LayaPolicy."
                ) from exc
            kwargs = {"subfolder": self.subfolder} if self.subfolder else {}
            self._agent = laya.load(self.model_id, **kwargs)
        return self._agent

    def decide(self, state: OptimizationState) -> LayaDecision:
        choices = _candidate_actions(state, self.seed)
        if state.resource.remaining_budget <= 0:
            decision = LayaDecision(choices[ActionType.STOP.value], {"stop": 1.0}, 1.0)
            self.last_decision = decision
            self.decisions.append(decision)
            return decision

        question = {
            "next_action": {
                "type": "choice",
                "instructions": (
                    "Choose the next action for maximizing the best observed objective "
                    "within the remaining evaluation budget. Consider exploration, "
                    "surrogate reliability, and whether stopping is justified."
                ),
                "criteria": {
                    action_id: _DESCRIPTIONS[action.type]
                    for action_id, action in choices.items()
                },
            }
        }
        result = self._get_agent().predict(_state_payload(state), question)
        try:
            answer = result["answers"]["next_action"]
            raw_probs = answer["probabilities"]
            selected = answer["choice"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Laya response lacks next_action choice/probabilities") from exc
        if selected not in choices:
            raise ValueError(f"Laya selected unavailable action: {selected!r}")
        if not isinstance(raw_probs, Mapping) or set(raw_probs) != set(choices):
            raise ValueError("Laya probabilities do not match the offered actions")
        probabilities = {name: float(raw_probs[name]) for name in choices}
        if any(not math.isfinite(p) or p < 0 for p in probabilities.values()):
            raise ValueError("Laya probabilities must be finite and non-negative")
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError("Laya probabilities must have positive total mass")
        probabilities = {name: value / total for name, value in probabilities.items()}
        confidence = float(answer.get("confidence", probabilities[selected]))
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Laya confidence must lie in [0, 1]")
        decision = LayaDecision(choices[selected], probabilities, confidence)
        self.last_decision = decision
        self.decisions.append(decision)
        return decision

    def select_action(
        self, state: OptimizationState, backend: UnifiedBackend | None = None
    ) -> OptimizationAction:
        return self.decide(state).action

"""Zero-shot Laya reference policy for the existing optimization decision loop.

Laya is optional and loaded only when the first decision is requested. The
policy exposes the full choice distribution for later calibration and routing;
these probabilities are *model outputs*, not validated BO action probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from opt_agent.actions import (
    ActionType,
    OptimizationAction,
    candidate_actions_for_state,
)
from opt_agent.backend.bocode_botorch import UnifiedBackend
from opt_agent.state import OptimizationState


class _LayaAgent(Protocol):
    def predict(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


_DESCRIPTIONS: dict[str, str] = {
    "global_exploration": "Evaluate new space-filling points across the search domain.",
    "local_exploitation": "Evaluate a small perturbation of the best observed point.",
    "uncertainty_sampling": "Evaluate where the surrogate predicts high uncertainty.",
    "error_guided_sampling": "Evaluate near a point with high surrogate residual.",
    "boundary_sampling": "Evaluate points near the search-domain boundary.",
    "surrogate_acquisition": "Evaluate the surrogate acquisition function's proposed point.",
    "retrain_surrogate": "Refit the surrogate without using an evaluation.",
    "stop": "Stop optimization and return the best observed point.",
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
    """Build budget-valid candidates via the shared factory.

    Per codex M1: a single factory owns candidate construction so the
    action_id vocabulary stays aligned with `full_candidates` and any
    other System-1 implementation.
    """
    return candidate_actions_for_state(state, seed=seed)


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
        self._mismatched_choice_count = 0

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
            decision = LayaDecision(choices["stop"], {"stop": 1.0}, 1.0)
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
                    action_id: _DESCRIPTIONS[action.type.value]
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
        # Per codex gpt-6-sol review 2026-09-24: the LLM's `choice` and the
        # `probabilities` dict are independent inputs — the agent might
        # emit a choice that doesn't match argmax(probabilities). Treat
        # that as a model-side inconsistency: trust the probabilities
        # (deterministic argmax) over the LLM's stated choice. This
        # keeps the System-1 → System-2 handoff well-defined: the
        # recorded `decision.action` is always the argmax action.
        argmax_action = max(probabilities, key=probabilities.get)
        if selected != argmax_action:
            self._mismatched_choice_count += 1
            selected = argmax_action
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

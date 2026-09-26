"""Shared prompt and state serialization for System-1 policies.

Both ``LayaPolicy`` and ``JevPolicy`` build their prompts from the same helpers
so the candidate vocabulary stays aligned. The four helpers below were moved
verbatim from ``laya_policy.py`` (``_DESCRIPTIONS``, ``_state_payload``) and
``jev_policy.py`` (``_state_narrative``, ``_executable_criterion``) — there is
no behavioural change. They are underscore-prefixed because they are
implementation details of the System-1 adapters; downstream code consumes
them only via ``LayaPolicy.decide()`` and ``JevPolicy.decide()``.

Pass 2 of the simplify review collapses the Laya/Jev duplication:

  - ``build_choice_request`` constructs the (state_arg, question) pair.
  - ``validate_choice_response`` parses the LLM answer.
  - Both policies use these helpers; their ``decide()`` methods shrink
    to the transport-specific code (HTTP retry vs SDK call) plus
    telemetry accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from opt_agent.actions import ActionType, OptimizationAction
from opt_agent.state import OptimizationState


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
    """Compact dict serialization of an OptimizationState for LLM prompts.
    Delegates to ``OptimizationState.to_payload`` so the field set lives
    in one place. Kept as a thin wrapper for backwards compatibility with
    any caller that already imports it from ``system1._prompt``."""
    return state.to_payload()


def _state_narrative(state: OptimizationState) -> str:
    """Explain the same observable state fields in plain English."""
    s = state.sampling
    o = state.optimization
    r = state.resource
    g = state.surrogate
    recent = (
        f"The latest evaluation improved the best value by {o.recent_improvement:.6g}."
        if o.recent_improvement > 0
        else "The latest evaluation did not improve the best value."
    )
    surrogate = (
        "A Gaussian-process surrogate is fitted. "
        f"Its training-history prediction RMSE is {g.validation_rmse:.6g}; "
        f"posterior standard deviation averages {g.mean_uncertainty:.6g} "
        f"and reaches {g.max_uncertainty:.6g} on a fixed search grid."
        if g.is_current and all(math.isfinite(x) for x in (
            g.validation_rmse, g.mean_uncertainty, g.max_uncertainty))
        else "No current surrogate diagnostics are available; an action that needs a surrogate may fit it first."
    )
    return (
        "We are maximizing an expensive black-box objective over "
        f"{s.dimension} continuous variables. An objective evaluation uses "
        "one unit of the finite evaluation budget; fitting the surrogate "
        "does not use an objective evaluation. "
        f"We have observed {s.n_samples} points out of a total budget of "
        f"{r.total_budget}, leaving {r.remaining_budget} "
        f"evaluation{'s' if r.remaining_budget != 1 else ''}. "
        f"The best objective value seen so far is {o.best_value:.6g}; "
        f"the most recent evaluated value is {o.last_value:.6g}. "
        f"{recent} It has been {o.stagnation_steps} evaluated points "
        "since the last improvement of the best value. "
        f"About {s.boundary_coverage:.1%} of evaluated points lie near "
        "a search-domain boundary; this is a fraction, not a measure of "
        "how well the whole domain is covered. "
        f"The smallest observed pairwise point distance is {s.min_pairwise_distance:.6g}. "
        f"{surrogate} The last high-level action was "
        f"{state.history.last_action or 'none (initial design)'}."
    )


def _executable_criterion(action: OptimizationAction, state: OptimizationState) -> str:
    """Describe the candidate that the executor will actually receive."""
    kind = action.type
    params = action.params
    description = _DESCRIPTIONS[kind.value]
    if kind == ActionType.STOP:
        return f"{description} Uses 0 objective evaluations and ends the run with any remaining budget unused."
    if kind == ActionType.RETRAIN_SURROGATE:
        return (
            f"{description} Uses 0 objective evaluations and fits the surrogate now. "
            "It gives a new diagnostic state before choosing again, but no new observation. "
            "Surrogate-based sampling actions already fit automatically when needed."
        )

    n_evaluations = int(params.get("n_samples", 1))
    details = []
    if "n_samples" in params:
        details.append(f"n_samples={n_evaluations}")
    if "step" in params:
        details.append(f"step={params['step']:g} of the domain width around the best point")
    if "tol" in params:
        details.append(f"tol={params['tol']:g} of the domain width from a boundary")
    if "seed" in params:
        details.append(f"seed={params['seed']} (fixed for reproducibility)")
    parameter_text = f" Fixed call parameters: {', '.join(details)}." if details else ""
    fit_text = (
        " Fits the surrogate first if it is not current."
        if kind in (ActionType.UNCERTAINTY_SAMPLING,
                    ActionType.ERROR_GUIDED_SAMPLING,
                    ActionType.SURROGATE_ACQUISITION) and not state.surrogate.is_current
        else " Uses the current surrogate without refitting."
        if kind in (ActionType.UNCERTAINTY_SAMPLING,
                    ActionType.ERROR_GUIDED_SAMPLING,
                    ActionType.SURROGATE_ACQUISITION)
        else " Does not fit the surrogate."
    )
    return (
        f"{description}{parameter_text} Uses {n_evaluations} objective "
        f"evaluation{'s' if n_evaluations != 1 else ''}.{fit_text}"
    )


@dataclass(frozen=True)
class DecisionOutput:
    """Single System-1 decision shape used by both ``LayaPolicy`` and
    ``JevPolicy`` (and any future System-1 adapter).

    Carries:
      - ``action``: the chosen ``OptimizationAction`` from the factory's
        budget-valid candidate pool.
      - ``probabilities``: full distribution over the offered action ids.
      - ``confidence``: scalar in [0, 1] (LLM-reported or sigmoid-derived).
      - ``latency_s``, ``input_tokens``, ``output_tokens``, ``model``:
        call telemetry. Local models set tokens to 0; the model field
        carries the local ``model_id`` or the API-reported model.
    """

    action: OptimizationAction
    probabilities: dict[str, float]
    confidence: float
    latency_s: float
    input_tokens: int
    output_tokens: int
    model: str


def build_choice_request(
    state: OptimizationState,
    choices: Mapping[str, OptimizationAction],
    *,
    state_format: Literal["structured", "narrative"] = "structured",
    criteria_format: Literal["legacy", "executable"] = "executable",
) -> tuple[Any, dict[str, Any]]:
    """Construct the (state_arg, inner_question) pair used by both
    LayaPolicy and JevPolicy.

    ``state_arg`` is the state payload passed as the predict call's
    ``state`` slot. ``inner_question`` is the dict that carries the
    choice primitive (top-level ``"next_action"``); callers wrap it
    with their own envelope (e.g. Jev nests under ``"questions"``).

    Centralising this here means a tweak to either prompt shape applies
    to both policies and there is no second copy of the instruction
    suffix or criteria block to drift out of sync.
    """
    state_arg = (
        _state_payload(state)
        if state_format == "structured"
        else _state_narrative(state)
    )
    criteria = {
        name: (
            _executable_criterion(action, state)
            if criteria_format == "executable"
            else _DESCRIPTIONS[name]
        )
        for name, action in choices.items()
    }
    instructions = (
        "Choose the next action for maximizing the best observed objective "
        "within the remaining evaluation budget. Consider exploration, "
        "surrogate reliability, and whether stopping is justified."
    ) + (
        " Each option is a complete executable call with fixed parameters; "
        "choose one option, without supplying parameters."
        if criteria_format == "executable" else ""
    )
    return state_arg, {
        "next_action": {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        }
    }


def validate_choice_response(
    raw: Mapping[str, Any],
    choices: Mapping[str, OptimizationAction],
) -> tuple[str, dict[str, float], float]:
    """Parse the LLM answer into ``(selected_id, probabilities, confidence)``.

    Validates that the answer shape matches the offered criteria, that the
    probabilities are finite and non-negative, and that their sum is
    within the published rounding tolerance. The reconcile-to-argmax
    policy (used by ``LayaPolicy`` because its SDK's ``choice`` field is
    independent of ``probabilities``) is left to the caller.
    """
    try:
        answer = raw["answers"]["next_action"]
        selected = answer["choice"]
        raw_probs = answer["probabilities"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Choice response missing 'answers.next_action' block") from exc
    if selected not in choices:
        raise ValueError("Choice response selected an unavailable action")
    if not isinstance(raw_probs, Mapping) or set(raw_probs) != set(choices):
        raise ValueError("Choice response probabilities do not match the offered actions")
    probs = {name: float(raw_probs[name]) for name in choices}
    if any(not math.isfinite(p) or p < 0 for p in probs.values()):
        raise ValueError("Choice probabilities must be finite and non-negative")
    total = sum(probs.values())
    if not 0.95 <= total <= 1.05:
        raise ValueError(f"Choice probability sum outside rounding tolerance: {total:.6f}")
    probs = {name: value / total for name, value in probs.items()}
    confidence = float(answer.get("confidence", probs[selected]))
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Choice confidence must lie in [0, 1]")
    return selected, probs, confidence
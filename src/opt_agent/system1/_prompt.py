"""Shared prompt and state serialization for System-1 policies.

Both ``LayaPolicy`` and ``JevPolicy`` build their prompts from the same helpers
so the candidate vocabulary stays aligned. The four helpers below were moved
verbatim from ``laya_policy.py`` (``_DESCRIPTIONS``, ``_state_payload``) and
``jev_policy.py`` (``_state_narrative``, ``_executable_criterion``) — there is
no behavioural change. They are underscore-prefixed because they are
implementation details of the System-1 adapters; downstream code consumes
them only via ``LayaPolicy.decide()`` and ``JevPolicy.decide()``.
"""

from __future__ import annotations

import math
from typing import Any

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
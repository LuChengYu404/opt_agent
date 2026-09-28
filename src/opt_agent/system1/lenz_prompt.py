"""Jev-style presentation of the lenz-style rich state (2026-09-28).

Renders a ``LenzState`` (full trial log + config + decision history, see
``opt_agent.state.lenz_state``) the way ``JevPolicy`` presents state and
actions: a narrative state text plus the executable-criteria action menu.
The output of ``build_lenz_choice_request`` has the same
``(state_arg, question)`` shape as ``system1._prompt.build_choice_request``
in (narrative, executable) mode, so the same transports (JevPolicy's
``{"model", "state", "questions"}`` envelope, LLMSystem1Policy) and the
same response parser (``validate_choice_response``) consume it unchanged.

Only the STATE is richer here; the action menu remains the frozen 8 typed
actions with executable criteria — the survey's option (c) without its
action-space half.
"""

from __future__ import annotations

from typing import Mapping

from opt_agent.actions import OptimizationAction
from opt_agent.state.lenz_state import LenzState
from opt_agent.system1._prompt import _DESCRIPTIONS, _executable_criterion


def lenz_state_to_narrative(state: LenzState) -> str:
    """Render the rich state as the Jev-style narrative state text.

    Structure: K_t context (what problem we are solving) → D_t trial log
    (every evaluated point, best marked) → c_t backend configuration →
    H_t decision history → the standard compressed-summary narrative
    (identical wording to the frozen design's narrative mode).
    """
    cfg = state.config
    best = state.best_trial()
    lines: list[str] = []

    # K_t — exogenous context.
    lines.append(
        f"We are maximizing an expensive black-box objective "
        f"('{cfg.problem}', {cfg.dimension} continuous variables) with a "
        f"fixed budget of {state.summary.resource.total_budget} objective "
        "evaluations."
    )

    # D_t — the full trial log.
    lines.append(f"Trial log ({state.n_trials} evaluations, best marked *):")
    for i, t in enumerate(state.trials):
        coords = ", ".join(f"{v:.6g}" for v in t.x)
        mark = " *" if t is best else ""
        lines.append(f"  [{i:>2}] f({coords}) = {t.y:.6g}{mark}")

    # c_t — backend configuration.
    region = "; ".join(f"[{lo:.4g}, {hi:.4g}]" for lo, hi in cfg.bounds)
    lines.append(
        f"Backend configuration: surrogate={cfg.surrogate} "
        f"(fitted={cfg.surrogate_fitted}), acquisition={cfg.acquisition}, "
        f"search region per dimension: {region}, "
        f"objective transform y' = {cfg.objective_transform[0]:g}*y "
        f"+ {cfg.objective_transform[1]:g}."
    )

    # H_t — decision history (the memory the frozen design discards).
    if state.decision_history:
        rendered = [
            a if a is not None else "initial-design" for a in state.decision_history
        ]
        lines.append("Decision history: " + " -> ".join(rendered))
    else:
        lines.append("Decision history: empty (initial design only).")

    # Compressed projection — same wording as the frozen narrative mode.
    from opt_agent.system1._prompt import _state_narrative

    lines.append(_state_narrative(state.summary))
    return "\n".join(lines)


def build_lenz_choice_request(
    state: LenzState,
    choices: Mapping[str, OptimizationAction],
) -> tuple[str, dict]:
    """Jev-style choice request over the lenz-style state.

    Returns ``(state_arg, question)`` with the exact shape of
    ``build_choice_request(..., state_format="narrative",
    criteria_format="executable")``: a narrative state string plus the
    ``next_action`` choice question whose criteria are the frozen menu's
    executable criteria (evaluated against the compressed summary, so the
    fit-text branch behaves exactly as in deployment).
    """
    state_arg = lenz_state_to_narrative(state)
    criteria = {
        name: _executable_criterion(action, state.summary)
        for name, action in choices.items()
    }
    instructions = (
        "Choose the next action for maximizing the best observed objective "
        "within the remaining evaluation budget. Consider exploration, "
        "surrogate reliability, and whether stopping is justified."
        " Each option is a complete executable call with fixed parameters; "
        "choose one option, without supplying parameters."
    )
    return state_arg, {
        "next_action": {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        }
    }


__all__ = [
    "LenzState",
    "build_lenz_choice_request",
    "lenz_state_to_narrative",
    "_DESCRIPTIONS",
]

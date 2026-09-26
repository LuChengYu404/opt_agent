"""Shared candidate-action factory used by System-1 and the dataset layer.

Per codex gpt-6-sol review 2026-09-23 (major M1):
  - Candidates come from a SINGLE source — `ALL_CANDIDATE_SPECS` below.
    `candidate_actions_for_state` filters that source by state; the
    dataset generator's `full_candidates` reads the same source for
    its unfiltered vocabulary. There is no second 8-action list.
  - action_id is the lowercase `ActionType.value` (matches the dataset's
    `full_candidates` vocabulary).
  - Factory filters by remaining_budget, sample count, and surrogate fit
    state so callers never need to validate budget feasibility separately.

The factory returns a `dict[action_id, OptimizationAction]` so the consumer
can:
  - Build the `criteria` payload for Laya's `choice` primitive.
  - Verify a chosen action_id is among the offered candidates.
  - Hand the OptimizationAction directly to DecisionLoop.step().
"""

from __future__ import annotations

from opt_agent.actions.models import ActionType, OptimizationAction
from opt_agent.state import OptimizationState


# Single source of truth for the candidate vocabulary. The first element
# of each tuple is the action_id (stable string), the second is the
# ActionType, and the third is the default parameter dict (without
# `seed`, which callers add per-trajectory).
#
# `full_candidates()` reads this list unfiltered; the runtime factory
# filters it by state.
#
# `BOUNDARY_TOL` is the default `tol` parameter for BOUNDARY_SAMPLING and
# the metric threshold used by `state.extractor._boundary_coverage` to
# decide whether a sampled point lies "near" a domain edge. Both must
# reference the same constant so the coverage metric stays aligned with
# the action's default search margin.
BOUNDARY_TOL: float = 0.05

ALL_CANDIDATE_SPECS: tuple[tuple[str, ActionType, dict], ...] = (
    ("global_exploration",  ActionType.GLOBAL_EXPLORATION,    {"n_samples": 2}),
    ("local_exploitation",  ActionType.LOCAL_EXPLOITATION,    {"step": 0.1}),
    ("uncertainty_sampling", ActionType.UNCERTAINTY_SAMPLING, {"n_samples": 1}),
    ("error_guided_sampling", ActionType.ERROR_GUIDED_SAMPLING, {"n_samples": 1}),
    ("boundary_sampling",   ActionType.BOUNDARY_SAMPLING,     {"n_samples": 2, "tol": BOUNDARY_TOL}),
    ("surrogate_acquisition", ActionType.SURROGATE_ACQUISITION, {}),
    ("retrain_surrogate",   ActionType.RETRAIN_SURROGATE,     {}),
    ("stop",                ActionType.STOP,                  {}),
)


def _make(action_type: ActionType, params: dict, seed: int) -> OptimizationAction:
    """Build an OptimizationAction, threading `seed` through if the
    action type accepts it. Mirrors the legacy `_ALLOWED_PARAMS` test
    in `actions.models`."""
    from opt_agent.actions.models import _ALLOWED_PARAMS
    merged = dict(params)
    if "seed" in _ALLOWED_PARAMS[action_type]:
        merged["seed"] = seed
    return OptimizationAction(action_type, merged)


def candidate_actions_for_state(
    state: OptimizationState, *, seed: int = 0,
) -> dict[str, OptimizationAction]:
    """Build budget-valid candidates as {action_id: OptimizationAction}.

    Filtering rules
    ---------------
    - remaining_budget <= 0  → only `stop` (forced termination).
    - remaining_budget > 0   → always include `stop` so the policy may
      terminate early. Emit at least one sampling action.
    - surrogate not yet fitted → include `retrain_surrogate`.
    - n_samples > 0           → include `local_exploitation` /
      `error_guided_sampling` (need at least one observed point).
    """
    remaining = state.resource.remaining_budget
    if remaining <= 0:
        return {"stop": OptimizationAction(ActionType.STOP)}

    action_seed = seed + state.sampling.n_samples
    out: dict[str, OptimizationAction] = {}

    for action_id, action_type, params in ALL_CANDIDATE_SPECS:
        if action_id == "stop":
            out["stop"] = OptimizationAction(ActionType.STOP)
            continue
        if action_id == "retrain_surrogate":
            if not state.surrogate.is_current:
                out[action_id] = _make(action_type, params, action_seed)
            continue
        if action_id in ("local_exploitation", "error_guided_sampling"):
            if state.sampling.n_samples:
                out[action_id] = _make(action_type, params, action_seed)
            continue
        # Sampling + surrogate_acquisition: cap n_samples at remaining budget.
        merged = dict(params)
        if "n_samples" in merged:
            merged["n_samples"] = min(merged["n_samples"], remaining)
        out[action_id] = _make(action_type, merged, action_seed)

    return out
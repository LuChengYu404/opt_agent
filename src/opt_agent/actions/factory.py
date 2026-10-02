"""Shared candidate-action factory used by System-1 and the dataset layer.

Per codex gpt-6-sol review 2026-09-23 (major M1):
  - Candidates come from a SINGLE source — `ALL_CANDIDATE_SPECS` below.
    `candidate_actions_for_state` filters that source by state; the
    dataset generator's `full_candidates` reads the same source for
    its unfiltered vocabulary. There is no second action list.
  - action_id is the lowercase `ActionType.value` (matches the dataset's
    `full_candidates` vocabulary).
  - Factory filters by remaining budget and sample count so callers never
    need to validate budget feasibility separately.

h3 (2026-10-01): heuristic samplers removed; no action carries a `seed`
(sampling randomness lives in the backend's checkpoint-restorable RNG
stream). Docs: docs/AF_SAMPLING_20261001.md.

The factory returns a `dict[action_id, OptimizationAction]` so the consumer
can:
  - Build the `criteria` payload for Laya's `choice` primitive.
  - Verify a chosen action_id is among the offered candidates.
  - Hand the OptimizationAction directly to DecisionLoop.step().
"""

from __future__ import annotations

from opt_agent.actions.models import ActionType, OptimizationAction
from opt_agent.state.models import OptimizationState


# Single source of truth for the candidate vocabulary. The first element
# of each tuple is the action_id (stable string), the second is the
# ActionType, and the third is the canonical parameter dict.
#
# `full_candidates()` reads this list unfiltered; the runtime factory
# filters it by state.
ALL_CANDIDATE_SPECS: tuple[tuple[str, ActionType, dict], ...] = (
    ("global_exploration",  ActionType.GLOBAL_EXPLORATION,    {"n_samples": 2}),
    ("local_exploitation",  ActionType.LOCAL_EXPLOITATION,    {"step": 0.1}),
    ("uncertainty_sampling", ActionType.UNCERTAINTY_SAMPLING, {"n_samples": 1}),
    ("surrogate_acquisition", ActionType.SURROGATE_ACQUISITION, {}),
    ("retrain_surrogate",   ActionType.RETRAIN_SURROGATE,     {}),
    ("stop",                ActionType.STOP,                  {}),
)


def sample_params(
    action_type: ActionType,
    remaining_budget: int,
    rng: "random.Random",
) -> dict:
    """Generate a random parameter dict for an action that requires one.

    Used by ``RandomPolicy`` and ``UniformPolicy`` (and any other
    ablation policy that needs a budget-feasible random action).
    """
    if action_type == ActionType.GLOBAL_EXPLORATION:
        return {"n_samples": rng.randint(1, min(3, remaining_budget))}
    if action_type == ActionType.LOCAL_EXPLOITATION:
        return {"step": round(rng.uniform(0.01, 0.2), 4)}
    if action_type == ActionType.UNCERTAINTY_SAMPLING:
        return {"n_samples": rng.randint(1, min(3, remaining_budget))}
    return {}


def candidate_actions_for_state(
    state: OptimizationState,
) -> dict[str, OptimizationAction]:
    """Build budget-valid candidates as {action_id: OptimizationAction}.

    Filtering rules
    ---------------
    - remaining_budget <= 0  → only `stop` (forced termination).
    - remaining_budget > 0   → always include `stop` so the policy may
      terminate early. Emit at least one sampling action.
    - surrogate not yet fitted → include `retrain_surrogate`.
    - n_samples > 0           → include `local_exploitation`
      (needs at least one observed point).
    """
    remaining = state.resource.remaining_budget
    if remaining <= 0:
        return {"stop": OptimizationAction(ActionType.STOP)}

    out: dict[str, OptimizationAction] = {}

    for action_id, action_type, params in ALL_CANDIDATE_SPECS:
        if action_id == "stop":
            out["stop"] = OptimizationAction(ActionType.STOP)
            continue
        if action_id == "retrain_surrogate":
            if not state.surrogate.is_current:
                out[action_id] = OptimizationAction(action_type, dict(params))
            continue
        if action_id == "local_exploitation":
            if state.sampling.n_samples:
                out[action_id] = OptimizationAction(action_type, dict(params))
            continue
        # Sampling + surrogate_acquisition: cap n_samples at remaining budget.
        merged = dict(params)
        if "n_samples" in merged:
            merged["n_samples"] = min(merged["n_samples"], remaining)
        out[action_id] = OptimizationAction(action_type, merged)

    return out

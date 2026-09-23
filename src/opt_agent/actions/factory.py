"""Shared candidate-action factory used by System-1 and the dataset layer.

Per codex gpt-6-sol review 2026-09-23 (major M1):
  - Candidates must come from a single factory so action_id is stable across
    LayaPolicy, the dataset generator, and any future System-1 implementations.
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


def candidate_actions_for_state(
    state: OptimizationState, *, seed: int = 0,
) -> dict[str, OptimizationAction]:
    """Build budget-valid candidates as {action_id: OptimizationAction}.

    Filtering rules
    ---------------
    - remaining_budget <= 0  → only `stop` (single candidate by necessity:
      the loop is forced to terminate).
    - remaining_budget > 0   → always include `stop` so the policy may
      terminate early when budget allows. Emit at least one sampling
      action when budget allows.
    - surrogate not yet fitted → include `retrain_surrogate`.
    - n_samples > 0           → include `local_exploitation` /
      `error_guided_sampling` (need at least one observed point).

    action_id is the lowercase ActionType value. The mapping from
    action_id to OptimizationAction is stable so LayaPolicy and any
    future System-1 agree on the candidate vocabulary.
    """
    remaining = state.resource.remaining_budget
    if remaining <= 0:
        return {"stop": OptimizationAction(ActionType.STOP)}

    out: dict[str, OptimizationAction] = {}
    action_seed = seed + state.sampling.n_samples

    # Sampling actions (at least one is always emitted when budget > 0).
    out["global_exploration"] = OptimizationAction(
        ActionType.GLOBAL_EXPLORATION,
        {"n_samples": min(2, remaining), "seed": action_seed},
    )
    out["uncertainty_sampling"] = OptimizationAction(
        ActionType.UNCERTAINTY_SAMPLING, {"n_samples": 1, "seed": action_seed},
    )
    out["boundary_sampling"] = OptimizationAction(
        ActionType.BOUNDARY_SAMPLING,
        {"n_samples": min(2, remaining), "tol": 0.05, "seed": action_seed},
    )
    # Surrogate exploitation
    out["surrogate_acquisition"] = OptimizationAction(ActionType.SURROGATE_ACQUISITION)

    # Local actions need at least one observed sample.
    if state.sampling.n_samples:
        out["local_exploitation"] = OptimizationAction(
            ActionType.LOCAL_EXPLOITATION, {"step": 0.1, "seed": action_seed},
        )
        out["error_guided_sampling"] = OptimizationAction(
            ActionType.ERROR_GUIDED_SAMPLING, {"n_samples": 1, "seed": action_seed},
        )

    # Retrain only when the surrogate is not current.
    if not state.surrogate.is_current:
        out["retrain_surrogate"] = OptimizationAction(ActionType.RETRAIN_SURROGATE)

    # Stop is always offered when budget allows: an early termination
    # is a valid policy choice even if not budget-forced.
    out["stop"] = OptimizationAction(ActionType.STOP)

    return out
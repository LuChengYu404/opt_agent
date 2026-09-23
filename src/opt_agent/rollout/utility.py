"""Utility function for counterfactual rollouts (Phase 6).

Per MASTER_PLAN.md §10 and codex (gpt-5.6-sol) review 2025-09-23.

U(a|s) = w_optimization * ΔQ_optimization
      + w_surrogate * ΔQ_surrogate   (Phase 6: 0 placeholder)
      + w_coverage * ΔQ_coverage     (Phase 6: 0 placeholder)
      − w_evaluation * cost_evaluation
      − w_fit * cost_model_fit
"""

from __future__ import annotations

from dataclasses import dataclass

from opt_agent.actions.models import ActionResult, ActionType


@dataclass(frozen=True)
class UtilityWeights:
    w_optimization: float = 1.0
    w_surrogate: float = 0.0   # placeholder
    w_coverage: float = 0.0    # placeholder
    w_evaluation: float = 1.0
    w_fit: float = 0.5


def compute_utility(
    *,
    delta_optimization: float,
    n_new_samples: int,
    n_model_fits: int,
    weights: UtilityWeights,
) -> float:
    """Compute the scalar utility of one counterfactual branch."""
    return (
        weights.w_optimization * delta_optimization
        - weights.w_evaluation * n_new_samples
        - weights.w_fit * n_model_fits
    )


def extract_outcome_metrics(result: ActionResult, action_type: ActionType) -> tuple[int, int]:
    """Return (n_new_samples, n_model_fits) from an ActionResult.

    Phase 6 only counts the action itself; multi-step rollouts sum across steps.
    """
    n_fit = 1 if action_type == ActionType.RETRAIN_SURROGATE else 0
    return result.n_new_samples, n_fit

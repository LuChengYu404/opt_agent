"""Utility function for counterfactual rollouts (Phase 8.5 fix).

Per MASTER_PLAN.md §10 and codex (gpt-5.6-sol) review 2025-09-23 + Phase 8.5
empirical finding (STOP dominance):

  U(a|s) = w_optimization * ΔQ_optimization / optimization_scale
        + w_surrogate * ΔQ_surrogate
        + w_coverage * ΔQ_coverage
        − w_evaluation * n_new_samples / evaluation_scale
        − w_fit * n_model_fits / fit_scale
        − w_stop_opportunity * remaining_budget_ratio   (if is_stop)

Scales (default 1.0) decouple problem units from weights: objective scale
estimated per problem family, evaluation/fit scale = total_budget. Defaults
preserve backward compatibility for tests that don't supply scales.

NaN / Inf handling: non-finite delta_surrogate / delta_coverage are treated
as 0; non-finite delta_optimization / scales raise ValueError (programmer
error, not data uncertainty).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class UtilityWeights:
    """Weights for the 5-term utility function.

    Phase 8.5 defaults (per codex review):
      - w_surrogate = 0: validation_rmse is in-sample training RMSE, not
        leave-one-out; cannot be trusted until surrogate metric is fixed.
      - w_coverage = 0: `boundary_coverage` is fraction near edge, not a
        proper coverage metric; needs a grid/fill-distance version.
      - w_stop_opportunity = 0: optional linear STOP penalty proportional
        to remaining_budget_ratio. Default 0 so backward-compatible.
    """
    w_optimization: float = 1.0
    w_surrogate: float = 0.0
    w_coverage: float = 0.0
    w_evaluation: float = 1.0
    w_fit: float = 0.5
    # Phase 8.5 default: 0.75 (codex-empirical) keeps STOP below non-STOP
    # actions in early phase; rises above them in late phase. Set to 0 to
    # disable the STOP opportunity penalty.
    w_stop_opportunity: float = 0.75


def compute_utility(
    *,
    delta_optimization: float,
    delta_surrogate: float = 0.0,
    delta_coverage: float = 0.0,
    n_new_samples: int,
    n_model_fits: int,
    weights: UtilityWeights,
    optimization_scale: float = 1.0,
    evaluation_scale: float = 1.0,
    fit_scale: float = 1.0,
    is_stop: bool = False,
    remaining_budget_ratio: float = 0.0,
) -> float:
    """Scalar utility of a counterfactual branch.

    Raises
    ------
    ValueError
        If any scale is non-positive, counts are negative, or
        delta_optimization is non-finite (programmer error).
    """
    if not math.isfinite(delta_optimization):
        raise ValueError("delta_optimization must be finite")
    if min(optimization_scale, evaluation_scale, fit_scale) <= 0:
        raise ValueError("utility scales must be positive")
    if n_new_samples < 0 or n_model_fits < 0:
        raise ValueError("resource counts must be non-negative")

    ds = 0.0 if not math.isfinite(delta_surrogate) else float(delta_surrogate)
    dc = 0.0 if not math.isfinite(delta_coverage) else float(delta_coverage)
    stop_penalty = (
        weights.w_stop_opportunity * float(remaining_budget_ratio)
        if is_stop else 0.0
    )
    return (
        weights.w_optimization * float(delta_optimization) / optimization_scale
        + weights.w_surrogate * ds
        + weights.w_coverage * dc
        - weights.w_evaluation * int(n_new_samples) / evaluation_scale
        - weights.w_fit * int(n_model_fits) / fit_scale
        - stop_penalty
    )

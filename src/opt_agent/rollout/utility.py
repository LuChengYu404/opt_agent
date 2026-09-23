"""Utility function for counterfactual rollouts (Phase 7 complete).

Per MASTER_PLAN.md §10 and codex (gpt-5.6-sol) review 2025-09-23:

  U(a|s) = w_optimization * ΔQ_optimization
        + w_surrogate * ΔQ_surrogate    (Phase 7: defaults to 0)
        + w_coverage * ΔQ_coverage      (Phase 7: defaults to 0)
        − w_evaluation * n_new_samples
        − w_fit * n_model_fits

NaN / Inf handling: non-finite delta_surrogate / delta_coverage are treated as 0
(uncertainty in the metric, not catastrophic error).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class UtilityWeights:
    """Weights for the 4-term utility function.

    Phase 7 default: w_surrogate = w_coverage = 0 (placeholder). Per codex,
    these should remain 0 until the Δsurrogate / Δcoverage metrics are
    verified correct, otherwise non-zero weights amplify metric bias.
    """
    w_optimization: float = 1.0
    w_surrogate: float = 0.0
    w_coverage: float = 0.0
    w_evaluation: float = 1.0
    w_fit: float = 0.5


def compute_utility(
    *,
    delta_optimization: float,
    delta_surrogate: float = 0.0,
    delta_coverage: float = 0.0,
    n_new_samples: int,
    n_model_fits: int,
    weights: UtilityWeights,
) -> float:
    """Scalar utility of a counterfactual branch.

    NaN / Inf deltas are coerced to 0 (uncertainty in the metric, not error).
    """
    ds = 0.0 if not math.isfinite(delta_surrogate) else float(delta_surrogate)
    dc = 0.0 if not math.isfinite(delta_coverage) else float(delta_coverage)
    return (
        weights.w_optimization * float(delta_optimization)
        + weights.w_surrogate * ds
        + weights.w_coverage * dc
        - weights.w_evaluation * int(n_new_samples)
        - weights.w_fit * int(n_model_fits)
    )

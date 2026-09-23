"""OptimizationState dataclass tree (Phase 2 + Phase 6 extension).

Per MASTER_PLAN.md §7 and codex (gpt-5.6-sol) review 2025-09-23 + Phase 6
correction (added coverage / uncertainty / error / boundary fields).

Conventions
-----------
- BOCoDe objectives are MAXIMIZED (see project-opt-agent-v3-bocode-direction).
  best_value = objectives.max(); recent_improvement > 0 means improving.
- All dataclasses are frozen=True so JSON serialization is trivial and
  extract_state() output is immutable across callers.
- SurrogateState posterior fields are populated when backend.is_fitted; when
  not fitted, validation_rmse / mean_uncertainty / max_uncertainty are NaN
  and is_current is False.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SamplingState:
    n_samples: int
    dimension: int
    min_pairwise_distance: float  # smallest distance between any two sampled points
    boundary_coverage: float     # fraction of samples within `tol` of any bounds edge


@dataclass(frozen=True)
class SurrogateState:
    is_current: bool            # True iff backend.is_fitted
    validation_rmse: float      # RMSE of one-step GP predictions on history (NaN if not fitted)
    mean_uncertainty: float     # mean posterior std on a Sobol grid (NaN if not fitted)
    max_uncertainty: float      # max posterior std on the same grid (NaN if not fitted)


@dataclass(frozen=True)
class OptimizationProgress:
    best_value: float
    last_value: float
    recent_improvement: float
    stagnation_steps: int


@dataclass(frozen=True)
class ResourceState:
    used_budget: int
    total_budget: int
    remaining_budget: int
    remaining_budget_ratio: float


@dataclass(frozen=True)
class HistoryState:
    last_action: Optional[str]


@dataclass(frozen=True)
class OptimizationState:
    sampling: SamplingState
    surrogate: SurrogateState
    optimization: OptimizationProgress
    resource: ResourceState
    history: HistoryState

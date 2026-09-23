"""OptimizationState dataclass tree (Phase 2).

Per MASTER_PLAN.md §7 and codex (gpt-5.6-sol) review 2025-09-23.

Conventions
-----------
- BOCoDe objectives are MAXIMIZED (see project-opt-agent-v3-bocode-direction).
  best_value = objectives.max(); recent_improvement > 0 means improving.
- All dataclasses are frozen=True so JSON serialization is trivial and
  extract_state() output is immutable across callers.
- Posterior-derived surrogate fields (predicted_best_mean, posterior_std_mean,
  posterior_std_max) are deferred until the backend exposes a public predict()
  API (Phase 5/6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SamplingState:
    n_samples: int
    dimension: int


@dataclass(frozen=True)
class SurrogateState:
    is_current: bool  # True iff backend.is_fitted


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
    # last_reward is intentionally omitted in Phase 2 — deferred to Phase 7
    # when reward semantics are defined for counterfactual rollouts.


@dataclass(frozen=True)
class OptimizationState:
    sampling: SamplingState
    surrogate: SurrogateState
    optimization: OptimizationProgress
    resource: ResourceState
    history: HistoryState

"""extract_state() — converts a backend's history into OptimizationState (Phase 2).

Pure read: must not mutate the backend or affect its RNG state.
"""

from __future__ import annotations

from typing import Optional

import torch

from opt_agent.backend.bocode_botorch import UnifiedBackend
from opt_agent.state.models import (
    HistoryState,
    OptimizationProgress,
    OptimizationState,
    ResourceState,
    SamplingState,
    SurrogateState,
)


def extract_state(
    backend: UnifiedBackend,
    *,
    total_budget: int,
    last_action: Optional[str] = None,
) -> OptimizationState:
    """Read backend history + optional action label, produce OptimizationState.

    Parameters
    ----------
    backend : UnifiedBackend
        Must be initialized (have history). is_fitted is read.
    total_budget : int
        Maximum number of evaluations allowed for this run. Must be > 0 and
        >= n_samples.
    last_action : Optional[str]
        Name of the action that produced the most recent sample. None means
        "no action recorded yet" (typically the initial design).

    Returns
    -------
    OptimizationState — frozen, JSON-serializable, tensor-free.
    """
    if total_budget <= 0:
        raise ValueError(f"total_budget must be > 0; got {total_budget}")

    history = backend.get_history()
    n_samples = len(history)
    if total_budget < n_samples:
        raise ValueError(
            f"total_budget ({total_budget}) is less than current n_samples "
            f"({n_samples}); cannot extract state for a budget that is "
            f"already exhausted."
        )

    values = history.objectives[:, 0].tolist()  # Phase 1: single-objective only
    dimension = int(history.X.shape[1])

    # best_value, last_value, recent_improvement, stagnation_steps
    # — derived from the entire history sequence.
    incumbent = values[0]
    best_value = incumbent
    stagnation_steps = 0
    recent_improvement = 0.0
    for value in values[1:]:
        if value > incumbent:
            recent_improvement = value - incumbent
            incumbent = value
            stagnation_steps = 0
        else:
            recent_improvement = 0.0
            stagnation_steps += 1
    best_value = incumbent
    last_value = values[-1]

    remaining_budget = total_budget - n_samples
    remaining_budget_ratio = remaining_budget / total_budget

    return OptimizationState(
        sampling=SamplingState(n_samples=n_samples, dimension=dimension),
        surrogate=SurrogateState(is_current=bool(backend.is_fitted)),
        optimization=OptimizationProgress(
            best_value=float(best_value),
            last_value=float(last_value),
            recent_improvement=float(recent_improvement),
            stagnation_steps=int(stagnation_steps),
        ),
        resource=ResourceState(
            used_budget=n_samples,
            total_budget=total_budget,
            remaining_budget=remaining_budget,
            remaining_budget_ratio=float(remaining_budget_ratio),
        ),
        history=HistoryState(last_action=last_action),
    )

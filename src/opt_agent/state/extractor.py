"""extract_state() — converts a backend's history into OptimizationState (Phase 2 + 6).

Pure read: must not mutate the backend or affect its RNG state.

Phase 6 additions: SamplingState includes min_pairwise_distance and
boundary_coverage; SurrogateState includes validation_rmse, mean_uncertainty,
max_uncertainty (populated only when backend.is_fitted; NaN otherwise).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.quasirandom import SobolEngine

from opt_agent.backend.bocode_botorch import UnifiedBackend
from opt_agent.state.models import (
    HistoryState,
    OptimizationProgress,
    OptimizationState,
    ResourceState,
    SamplingState,
    SurrogateState,
)

# Grid size for posterior uncertainty statistics.
_UNCERTAINTY_GRID_SIZE = 64
_BOUNDARY_TOL = 0.05  # fraction of range considered "near boundary"


def _min_pairwise_distance(X: torch.Tensor) -> float:
    if X.shape[0] < 2:
        return float("inf")
    diffs = X.unsqueeze(0) - X.unsqueeze(1)  # [n, n, d]
    dists = diffs.norm(dim=-1)  # [n, n]
    # Ignore self-distance (diagonal is 0).
    n = X.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=X.device)
    return float(dists[mask].min().item())


def _boundary_coverage(X: torch.Tensor, bounds: torch.Tensor, tol: float) -> float:
    """Fraction of samples within `tol` of any bound edge, normalized to bounds range."""
    if X.shape[0] == 0:
        return 0.0
    span = (bounds[1] - bounds[0]).clamp(min=1e-12)
    # Distance to nearest bound, normalized by span.
    dist_to_lo = (X - bounds[0]) / span
    dist_to_hi = (bounds[1] - X) / span
    nearest = torch.minimum(dist_to_lo, dist_to_hi).clamp(min=0.0)
    near = (nearest < tol).any(dim=-1)  # [n]
    return float(near.float().mean().item())


def _validation_rmse(backend: UnifiedBackend) -> float:
    """One-step RMSE of the GP on training history (predict at each X, compare to y)."""
    history = backend.get_history()
    if history.objectives.shape[0] < 2:
        return float("nan")
    mean, _ = backend.predict(history.X)
    residuals = mean - history.objectives
    return float(residuals.pow(2).mean().sqrt().item())


def _uncertainty_stats(backend: UnifiedBackend) -> tuple[float, float]:
    """Compute mean and max posterior std on a Sobol grid."""
    bounds = backend.bounds
    if bounds is None:
        return float("nan"), float("nan")
    d = bounds.shape[1]
    grid_unit = SobolEngine(dimension=d, scramble=True, seed=0).draw(
        _UNCERTAINTY_GRID_SIZE
    ).to(dtype=torch.double)
    grid = bounds[0] + grid_unit * (bounds[1] - bounds[0])
    _, std = backend.predict(grid)
    return float(std.mean().item()), float(std.max().item())


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

    # SamplingState coverage (Phase 6 extension).
    bounds = backend.bounds
    if bounds is not None and bounds.shape[1] == dimension:
        min_pd = _min_pairwise_distance(history.X)
        bcov = _boundary_coverage(history.X, bounds, tol=_BOUNDARY_TOL)
    else:
        min_pd = float("inf")
        bcov = 0.0

    # SurrogateState (Phase 6 extension).
    is_fitted = bool(backend.is_fitted)
    if is_fitted:
        try:
            val_rmse = _validation_rmse(backend)
            mean_u, max_u = _uncertainty_stats(backend)
        except Exception:
            val_rmse, mean_u, max_u = float("nan"), float("nan"), float("nan")
    else:
        val_rmse, mean_u, max_u = float("nan"), float("nan"), float("nan")

    remaining_budget = total_budget - n_samples
    remaining_budget_ratio = remaining_budget / total_budget

    return OptimizationState(
        sampling=SamplingState(
            n_samples=n_samples,
            dimension=dimension,
            min_pairwise_distance=float(min_pd),
            boundary_coverage=float(bcov),
        ),
        surrogate=SurrogateState(
            is_current=is_fitted,
            validation_rmse=float(val_rmse),
            mean_uncertainty=float(mean_u),
            max_uncertainty=float(max_u),
        ),
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

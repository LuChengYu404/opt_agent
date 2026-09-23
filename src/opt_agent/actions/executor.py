"""execute_action() — Phase 3.

Applies an OptimizationAction to a backend, returning an ActionResult.

Per codex (gpt-5.6-sol) review 2025-09-23:
  - Returns ActionResult(action, n_new_samples, should_stop), not bare action.
  - Budget pre-check raises ValueError WITHOUT mutating the backend.
  - Stale state detection: if backend.n_samples > state.sampling.n_samples, raise.
  - Backend exceptions propagate, with NO partial history mutation.
  - STOP does not require an initialized backend.
"""

from __future__ import annotations

import torch
from torch.quasirandom import SobolEngine

from opt_agent.actions.models import ActionResult, ActionType, OptimizationAction
from opt_agent.backend.bocode_botorch import UnifiedBackend
from opt_agent.state.models import OptimizationState


def execute_action(
    backend: UnifiedBackend,
    state: OptimizationState,
    action: OptimizationAction,
) -> ActionResult:
    """Apply `action` on `backend`, return an ActionResult.

    Raises
    ------
    ValueError
        If action requests more samples than the remaining budget, or if the
        passed `state` is stale (backend has been mutated since extract_state).
    RuntimeError
        If the backend reports an internal error (e.g. BOCoDe evaluation
        failure). The exception propagates without partially mutating history.
    """
    if action.type == ActionType.STOP:
        # STOP is allowed even on uninitialized backends.
        return ActionResult(action=action, n_new_samples=0, n_model_fits=0, should_stop=True)

    # Stale-state detection: if backend has grown past state, refuse.
    current_n = len(backend.get_history())
    if current_n != state.sampling.n_samples:
        raise ValueError(
            f"Stale state: state.sampling.n_samples={state.sampling.n_samples} "
            f"but backend has {current_n} samples. Re-extract state first."
        )

    # Budget pre-check + dispatch prepare.
    if action.type == ActionType.GLOBAL_EXPLORATION:
        n_samples = int(action.params.get("n_samples", 1))
        n_fits = 0
    elif action.type == ActionType.LOCAL_EXPLOITATION:
        n_samples = 1
        n_fits = 0
    elif action.type == ActionType.UNCERTAINTY_SAMPLING:
        n_samples = int(action.params.get("n_samples", 1))
        # UNCERTAINTY_SAMPLING fits GP if needed, then samples where std is high.
        n_fits = 0 if backend.is_fitted else 1
    elif action.type == ActionType.ERROR_GUIDED_SAMPLING:
        n_samples = int(action.params.get("n_samples", 1))
        n_fits = 0 if backend.is_fitted else 1
    elif action.type == ActionType.BOUNDARY_SAMPLING:
        n_samples = int(action.params.get("n_samples", 1))
        n_fits = 0  # BOUNDARY_SAMPLING does not require GP fit
    elif action.type == ActionType.SURROGATE_ACQUISITION:
        n_samples = 1
        n_fits = 0 if backend.is_fitted else 1
    elif action.type == ActionType.RETRAIN_SURROGATE:
        n_samples = 0
        n_fits = 1
    else:  # pragma: no cover
        raise ValueError(f"Unhandled action type: {action.type!r}")

    remaining = state.resource.remaining_budget
    if n_samples > remaining:
        raise ValueError(
            f"Action requests {n_samples} sample(s) but only {remaining} remain "
            f"in the budget. Rejecting without mutating backend."
        )

    # Dispatch.
    if action.type == ActionType.GLOBAL_EXPLORATION:
        _do_global_exploration(backend, action)
    elif action.type == ActionType.LOCAL_EXPLOITATION:
        _do_local_exploitation(backend, action)
    elif action.type == ActionType.UNCERTAINTY_SAMPLING:
        _do_uncertainty_sampling(backend, action)
    elif action.type == ActionType.ERROR_GUIDED_SAMPLING:
        _do_error_guided_sampling(backend, action)
    elif action.type == ActionType.BOUNDARY_SAMPLING:
        _do_boundary_sampling(backend, action)
    elif action.type == ActionType.SURROGATE_ACQUISITION:
        _do_surrogate_acquisition(backend)
    elif action.type == ActionType.RETRAIN_SURROGATE:
        backend.fit()
    else:  # pragma: no cover
        raise ValueError(f"Unhandled action type: {action.type!r}")

    return ActionResult(
        action=action,
        n_new_samples=n_samples,
        n_model_fits=n_fits,
        should_stop=False,
    )


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------


def _do_global_exploration(backend: UnifiedBackend, action: OptimizationAction) -> None:
    n_samples = int(action.params.get("n_samples", 1))
    seed = action.params.get("seed")
    bounds = backend.bounds
    assert bounds is not None, "Backend must expose bounds for GLOBAL_EXPLORATION"
    d = bounds.shape[1]
    sobol_seed = int(seed) if seed is not None else 0
    unit = SobolEngine(dimension=d, scramble=True, seed=sobol_seed).draw(n_samples).to(
        dtype=torch.double
    )
    X = bounds[0] + unit * (bounds[1] - bounds[0])
    backend.evaluate(X)


def _do_local_exploitation(backend: UnifiedBackend, action: OptimizationAction) -> None:
    step = float(action.params.get("step", 0.05))
    seed = action.params.get("seed")
    bounds = backend.bounds
    assert bounds is not None, "Backend must expose bounds for LOCAL_EXPLOITATION"
    history = backend.get_history()
    best_idx = int(history.objectives[:, 0].argmax().item())
    best_point = history.X[best_idx].clone()
    span = bounds[1] - bounds[0]
    rng = torch.Generator()
    rng.manual_seed(int(seed) if seed is not None else 0)
    delta = (torch.rand(best_point.shape, generator=rng, dtype=torch.double) * 2.0 - 1.0) * step * span
    candidate = (best_point + delta).clamp(min=bounds[0], max=bounds[1]).unsqueeze(0)
    backend.evaluate(candidate)


def _do_surrogate_acquisition(backend: UnifiedBackend) -> None:
    if not backend.is_fitted:
        backend.fit()
    X_new = backend.propose(n=1)
    backend.evaluate(X_new)


def _do_uncertainty_sampling(backend: UnifiedBackend, action: OptimizationAction) -> None:
    """Sample where the GP posterior std is highest.

    Fits GP if needed. Uses a Sobol grid to evaluate std, then picks the
    top-n_samples points.
    """
    from torch.quasirandom import SobolEngine

    if not backend.is_fitted:
        backend.fit()
    n_samples = int(action.params.get("n_samples", 1))
    seed = action.params.get("seed")
    bounds = backend.bounds
    assert bounds is not None, "Backend must expose bounds for UNCERTAINTY_SAMPLING"
    d = bounds.shape[1]
    grid_unit = SobolEngine(dimension=d, scramble=True, seed=int(seed) if seed is not None else 0).draw(
        max(n_samples * 16, 64)
    ).to(dtype=torch.double)
    grid = bounds[0] + grid_unit * (bounds[1] - bounds[0])  # [G, d]
    _, std = backend.predict(grid)  # [G, 1]
    std_flat = std.flatten()
    top_idx = std_flat.argsort(descending=True)[:n_samples]  # top-n
    selected = grid[top_idx]  # [n, d]
    backend.evaluate(selected)


def _do_error_guided_sampling(backend: UnifiedBackend, action: OptimizationAction) -> None:
    """Sample where the GP makes the largest prediction error on history.

    Phase 6 simplification: error = absolute residual (mean - observed objective)
    on training points. Sample near the top-error training point.
    """
    if not backend.is_fitted:
        backend.fit()
    n_samples = int(action.params.get("n_samples", 1))
    seed = action.params.get("seed")
    history = backend.get_history()
    mean, _ = backend.predict(history.X)
    err = (mean - history.objectives).abs().flatten()
    top_idx = int(err.argmax().item())
    top_point = history.X[top_idx]
    bounds = backend.bounds
    assert bounds is not None, "Backend must expose bounds for ERROR_GUIDED_SAMPLING"
    span = bounds[1] - bounds[0]
    rng = torch.Generator()
    rng.manual_seed(int(seed) if seed is not None else 0)
    delta = (torch.rand(top_point.shape, generator=rng, dtype=torch.double) * 2.0 - 1.0) * 0.05 * span
    candidates = (top_point + delta).clamp(min=bounds[0], max=bounds[1]).unsqueeze(0)
    if n_samples > 1:
        # repeat with different perturbations.
        candidates = candidates.repeat(n_samples, 1)
    backend.evaluate(candidates)


def _do_boundary_sampling(backend: UnifiedBackend, action: OptimizationAction) -> None:
    """Sample points within `tol` of any bounds edge (Sobol candidates near edges).

    Does not require GP fit. Provides explicit boundary coverage signal.
    """
    from torch.quasirandom import SobolEngine

    n_samples = int(action.params.get("n_samples", 1))
    tol = float(action.params.get("tol", 0.05))
    seed = action.params.get("seed")
    bounds = backend.bounds
    assert bounds is not None, "Backend must expose bounds for BOUNDARY_SAMPLING"
    d = bounds.shape[1]
    grid_unit = SobolEngine(dimension=d, scramble=True, seed=int(seed) if seed is not None else 0).draw(
        max(n_samples * 32, 128)
    ).to(dtype=torch.double)
    grid = bounds[0] + grid_unit * (bounds[1] - bounds[0])
    # Pick points closest to any bounds edge.
    span = (bounds[1] - bounds[0]).clamp(min=1e-12)
    dist_to_edge = torch.minimum(grid - bounds[0], bounds[1] - grid) / span
    nearest = dist_to_edge.min(dim=-1).values
    # Keep only those within tol (or top-n if fewer than n_samples qualify).
    mask = nearest < tol
    if mask.sum() < n_samples:
        # not enough near-boundary points — take the top-n closest.
        top_idx = nearest.argsort()[:n_samples]
    else:
        top_idx = nearest.argwhere()[:n_samples].flatten()
    selected = grid[top_idx]
    backend.evaluate(selected)

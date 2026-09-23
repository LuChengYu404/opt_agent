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
        return ActionResult(action=action, n_new_samples=0, should_stop=True)

    # Stale-state detection: if backend has grown past state, refuse.
    current_n = len(backend.get_history())
    if current_n != state.sampling.n_samples:
        raise ValueError(
            f"Stale state: state.sampling.n_samples={state.sampling.n_samples} "
            f"but backend has {current_n} samples. Re-extract state first."
        )

    # Budget pre-check.
    if action.type == ActionType.GLOBAL_EXPLORATION:
        n_samples = int(action.params.get("n_samples", 1))
    elif action.type == ActionType.LOCAL_EXPLOITATION:
        n_samples = 1
    elif action.type == ActionType.SURROGATE_ACQUISITION:
        n_samples = 1
    elif action.type == ActionType.RETRAIN_SURROGATE:
        n_samples = 0
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
    elif action.type == ActionType.SURROGATE_ACQUISITION:
        _do_surrogate_acquisition(backend)
    elif action.type == ActionType.RETRAIN_SURROGATE:
        backend.fit()
    else:  # pragma: no cover
        raise ValueError(f"Unhandled action type: {action.type!r}")

    return ActionResult(
        action=action,
        n_new_samples=n_samples,
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

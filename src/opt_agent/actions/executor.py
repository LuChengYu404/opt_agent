"""execute_action() — apply an OptimizationAction to a backend.

h3: every point-selection action maximizes a LMABO-portfolio AF
(backend/acquisitions.py; docs/AF_SAMPLING_20261001.md):

  GLOBAL_EXPLORATION   -> TS
  LOCAL_EXPLOITATION   -> PI inside the `step` trust region
  UNCERTAINTY_SAMPLING -> PosSTD
  SURROGATE_ACQUISITION-> `af` param (default LogEI)

Multi-point actions are sequential greedy (fit-if-stale -> propose ->
evaluate per point). Atomicity: a failing greedy action restores the
pre-action checkpoint (history + GP + RNG) before re-raising.
"""

from __future__ import annotations

import torch

from opt_agent.actions.models import ActionResult, ActionType, OptimizationAction
from opt_agent.backend.acquisitions import AF_DEFAULT
from opt_agent.backend.bocode_botorch import UnifiedBackend
from opt_agent.state.models import OptimizationState

# Action -> portfolio-AF mapping; one line per action if the protocol moves.
AF_GLOBAL_EXPLORATION = "TS"
AF_LOCAL_EXPLOITATION = "PI"
AF_UNCERTAINTY_SAMPLING = "PosSTD"


def execute_action(
    backend: UnifiedBackend,
    state: OptimizationState,
    action: OptimizationAction,
) -> ActionResult:
    if action.type is ActionType.STOP:
        return ActionResult(action, 0, 0, should_stop=True)

    if len(backend.get_history()) != state.sampling.n_samples:
        raise ValueError(
            f"Stale state: state.sampling.n_samples={state.sampling.n_samples} "
            f"but backend has {len(backend.get_history())} samples.")

    # RETRAIN consumes no evaluation budget; everything else samples >= 1.
    n_samples = 0 if action.type is ActionType.RETRAIN_SURROGATE \
        else int(action.params.get("n_samples", 1))
    remaining = state.resource.remaining_budget
    if n_samples > remaining:
        raise ValueError(
            f"Action requests {n_samples} sample(s) but only {remaining} "
            f"remain in the budget. Rejecting without mutating backend.")

    if action.type is ActionType.GLOBAL_EXPLORATION:
        n_fits = _greedy_af_samples(backend, n_samples, AF_GLOBAL_EXPLORATION)
    elif action.type is ActionType.LOCAL_EXPLOITATION:
        step = float(action.params.get("step", 0.1))
        n_fits = _greedy_af_samples(backend, 1, AF_LOCAL_EXPLOITATION,
                                    trust_region_step=step)
    elif action.type is ActionType.UNCERTAINTY_SAMPLING:
        n_fits = _greedy_af_samples(backend, n_samples, AF_UNCERTAINTY_SAMPLING)
    elif action.type is ActionType.SURROGATE_ACQUISITION:
        fits = 0
        if not backend.is_fitted:
            backend.fit()
            fits = 1
        backend.evaluate(backend.propose(
            n=1, af=str(action.params.get("af", AF_DEFAULT))))
        n_fits = fits
    elif action.type is ActionType.RETRAIN_SURROGATE:
        backend.fit()
        n_fits = 1
    else:  # pragma: no cover
        raise ValueError(f"Unhandled action type: {action.type!r}")

    return ActionResult(action, n_samples, n_fits, should_stop=False)


def _trust_region_bounds(backend: UnifiedBackend, step: float) -> torch.Tensor:
    """[incumbent - step*span, incumbent + step*span] ∩ domain, [2, d]."""
    history = backend.get_history()
    best = history.X[int(history.objectives[:, 0].argmax().item())]
    dom = backend.bounds
    span = dom[1] - dom[0]
    return torch.stack([
        torch.maximum(dom[0], best - step * span),
        torch.minimum(dom[1], best + step * span),
    ])


def _greedy_af_samples(
    backend: UnifiedBackend, n_samples: int, af: str,
    *, trust_region_step: float | None = None,
) -> int:
    """One fit-if-stale -> propose -> evaluate per point; returns fits."""
    cp = backend.checkpoint() if hasattr(backend, "checkpoint") else None
    try:
        fits = 0
        for _ in range(n_samples):
            if not backend.is_fitted:
                backend.fit()
                fits += 1
            bounds = (_trust_region_bounds(backend, trust_region_step)
                      if trust_region_step is not None else None)
            backend.evaluate(backend.propose(n=1, af=af, bounds=bounds))
    except Exception:
        if cp is not None:
            backend.restore(cp)
        raise
    return fits

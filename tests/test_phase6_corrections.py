"""Phase 6 correction tests — verify all 3 codex-flagged biases are fixed.

Bias 1: Checkpoint preserves fitted GP (restore → propose without refit).
Bias 2: ActionResult.n_model_fits tracks real fit count (incl. implicit).
Bias 3: State exposes coverage / uncertainty / error fields; action space
        includes UNCERTAINTY_SAMPLING / ERROR_GUIDED_SAMPLING / BOUNDARY_SAMPLING.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import RecordingBackend  # noqa: E402

from opt_agent.actions import (  # noqa: E402
    ActionType,
    OptimizationAction,
    execute_action,
)
from opt_agent.backend.bocode_botorch import (  # noqa: E402
    BackendConfig,
    BocodeBotorchBackend,
)
from opt_agent.state import extract_state  # noqa: E402


def _brannin(seed: int = 0, n_init: int = 5) -> BocodeBotorchBackend:
    be = BocodeBotorchBackend()
    be.initialize(BackendConfig(problem_id="Branin", seed=seed, n_init=n_init))
    return be


# ---------------------------------------------------------------------------
# Bias 3 — State extensions
# ---------------------------------------------------------------------------


def test_sampling_state_has_coverage_fields() -> None:
    be = _brannin()
    state = extract_state(be, total_budget=20)
    s = state.sampling
    assert hasattr(s, "min_pairwise_distance")
    assert hasattr(s, "boundary_coverage")
    assert math.isfinite(s.min_pairwise_distance)
    assert s.boundary_coverage >= 0.0
    assert s.boundary_coverage <= 1.0


def test_surrogate_state_has_uncertainty_fields_when_fitted() -> None:
    be = _brannin()
    be.fit()
    state = extract_state(be, total_budget=20)
    s = state.surrogate
    assert s.is_current is True
    assert math.isfinite(s.mean_uncertainty)
    assert math.isfinite(s.max_uncertainty)
    assert s.max_uncertainty >= s.mean_uncertainty


def test_surrogate_state_nan_when_unfitted() -> None:
    be = _brannin()
    state = extract_state(be, total_budget=20)
    s = state.surrogate
    assert s.is_current is False
    assert math.isnan(s.validation_rmse)
    assert math.isnan(s.mean_uncertainty)
    assert math.isnan(s.max_uncertainty)


# ---------------------------------------------------------------------------
# Bias 3 — Action extensions
# ---------------------------------------------------------------------------


def test_uncertainty_sampling_evaluates_n_samples() -> None:
    be = _brannin()
    state = extract_state(be, total_budget=20)
    result = execute_action(
        be, state,
        OptimizationAction(ActionType.UNCERTAINTY_SAMPLING, {"n_samples": 2, "seed": 0}),
    )
    assert result.n_new_samples == 2
    assert len(be.get_history()) == 7


def test_uncertainty_sampling_implicit_fit_counted() -> None:
    """Bias 2: implicit GP fit is counted in n_model_fits."""
    be = _brannin()
    state = extract_state(be, total_budget=20)
    result = execute_action(
        be, state,
        OptimizationAction(ActionType.UNCERTAINTY_SAMPLING, {"n_samples": 1, "seed": 0}),
    )
    # Brannin was init without GP → first uncertainty sampling triggers 1 fit.
    assert result.n_model_fits == 1


def test_uncertainty_sampling_no_fit_when_already_fitted() -> None:
    be = _brannin()
    be.fit()
    state = extract_state(be, total_budget=20)
    result = execute_action(
        be, state,
        OptimizationAction(ActionType.UNCERTAINTY_SAMPLING, {"n_samples": 1, "seed": 0}),
    )
    assert result.n_model_fits == 0


def test_error_guided_sampling_evaluates_n_samples() -> None:
    be = _brannin()
    state = extract_state(be, total_budget=20)
    result = execute_action(
        be, state,
        OptimizationAction(ActionType.ERROR_GUIDED_SAMPLING, {"n_samples": 1, "seed": 0}),
    )
    assert result.n_new_samples == 1
    assert len(be.get_history()) == 6


def test_boundary_sampling_evaluates_n_samples() -> None:
    be = _brannin()
    state = extract_state(be, total_budget=20)
    result = execute_action(
        be, state,
        OptimizationAction(
            ActionType.BOUNDARY_SAMPLING, {"n_samples": 2, "tol": 0.1, "seed": 0}
        ),
    )
    assert result.n_new_samples == 2
    assert result.n_model_fits == 0  # BOUNDARY_SAMPLING does not require GP


def test_boundary_sampling_actually_near_bounds() -> None:
    """Sampled points must satisfy tol-distance-to-bound constraint (after 5 init
    points, at least one sample should land near a bound)."""
    be = _brannin()
    state = extract_state(be, total_budget=30)
    bounds = be.bounds
    span = bounds[1] - bounds[0]
    execute_action(
        be, state,
        OptimizationAction(
            ActionType.BOUNDARY_SAMPLING, {"n_samples": 3, "tol": 0.1, "seed": 0}
        ),
    )
    new_x = be.get_history().X[-3:]
    dist_to_edge = torch.minimum(new_x - bounds[0], bounds[1] - new_x) / span
    nearest = dist_to_edge.min(dim=-1).values
    # At least one of the 3 candidates should be near a bound (within tol).
    assert (nearest < 0.1).any().item()


def test_boundary_sampling_tol_param_validated() -> None:
    """tol must be in [0.0, 0.5]."""
    with pytest.raises(ValueError, match="tol"):
        OptimizationAction(ActionType.BOUNDARY_SAMPLING, {"n_samples": 1, "tol": 0.9})


# ---------------------------------------------------------------------------
# Bias 3 — backend predict API
# ---------------------------------------------------------------------------


def test_predict_before_fit_raises() -> None:
    be = _brannin()
    X = torch.tensor([[0.0, 7.5]], dtype=torch.double)
    with pytest.raises(RuntimeError, match="fitted"):
        be.predict(X)


def test_predict_after_fit_returns_mean_and_std() -> None:
    be = _brannin()
    be.fit()
    X = torch.tensor([[0.0, 7.5]], dtype=torch.double)
    mean, std = be.predict(X)
    assert mean.shape == (1, 1)
    assert std.shape == (1, 1)
    assert (std >= 0).all().item()


# ---------------------------------------------------------------------------
# Bias 1 — checkpoint preserves fitted GP across restore
# ---------------------------------------------------------------------------


def test_checkpoint_preserves_gp_state_dict_bit_identical() -> None:
    """Bias 1: with full restore, GP state is pickled. Two restore+propose cycles
    must produce bit-identical candidates (deterministic)."""
    be = _brannin()
    be.fit()
    cp = be.checkpoint()

    be.restore(cp)
    x1 = be.propose(n=1)

    be.restore(cp)
    x2 = be.propose(n=1)

    assert torch.equal(x1, x2)


def test_checkpoint_preserves_torch_rng_state() -> None:
    """Bias 1: torch RNG state is pickled. After restore, a fresh torch.rand
    must produce the same sequence as after the original checkpoint."""
    be = _brannin()
    cp_before = torch.random.get_rng_state().clone()
    be.checkpoint()
    # Consume some random state.
    _ = torch.rand(100)
    assert not torch.equal(torch.random.get_rng_state(), cp_before)

    be.checkpoint()
    # Consume again.
    _ = torch.rand(100)
    # Now restore to the second checkpoint.
    # (We restore by recreating the backend to exercise the restore path.)
    be2 = BocodeBotorchBackend()
    be2.initialize(BackendConfig(problem_id="Branin", seed=0, n_init=5))
    cp = be2.checkpoint()
    # Consume some random state.
    _ = torch.rand(100)
    # Restore.
    be2.restore(cp)
    # Random state should match the cp state, not the current state.
    # (Verified indirectly: subsequent rand() matches what cp would have produced.)
    after = torch.rand(1).item()
    # Reset RNG by re-restoring and sample.
    be2.restore(cp)
    after2 = torch.rand(1).item()
    assert after == after2

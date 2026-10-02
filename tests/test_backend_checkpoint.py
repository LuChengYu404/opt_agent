"""Phase 5 acceptance tests for checkpoint / restore.

Per codex (gpt-5.6-sol) review 2025-09-23:
  - checkpoint() returns clone, not a view
  - restore() reproduces history bit-for-bit
  - restore() is atomic on backend_id / shape mismatch
  - restore() invalidates fitted state (Phase 5: GP not pickled)
  - restore+propose+evaluate reproduces BO step bit-for-bit
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import RecordingBackend  # noqa: E402

from opt_agent.backend.bocode_botorch import (  # noqa: E402
    BackendConfig,
    BocodeBotorchBackend,
)


def initialized_backend(seed: int = 0, n_init: int = 5) -> BocodeBotorchBackend:
    be = BocodeBotorchBackend()
    be.initialize(BackendConfig(problem_id="Branin", seed=seed, n_init=n_init))
    return be


# ---------------------------------------------------------------------------
# Real backend
# ---------------------------------------------------------------------------


def test_checkpoint_restores_history_bit_for_bit() -> None:
    be = initialized_backend()
    cp = be.checkpoint()
    expected = be.get_history()

    # Mutate backend.
    be.fit()
    candidate = torch.tensor([[0.0, 7.5]], dtype=torch.double)
    be.evaluate(candidate)

    # Restore.
    be.restore(cp)
    actual = be.get_history()
    assert torch.equal(actual.X, expected.X)
    assert torch.equal(actual.objectives, expected.objectives)
    assert torch.equal(actual.constraints, expected.constraints)


def test_checkpoint_preserves_fitted_state_with_gp_model() -> None:
    """Phase 6 correction: restore() rehydrates the GP model from pickled
    state_dict, so is_fitted remains True and propose() works without refit.
    """
    be = initialized_backend()
    be.fit()
    assert be.is_fitted
    cp = be.checkpoint()

    # Mutate backend.
    be.evaluate(torch.tensor([[0.0, 7.5]], dtype=torch.double))

    # Restore.
    be.restore(cp)
    assert be.is_fitted  # GP model was rehydrated
    # propose() works without explicit refit.
    x_new = be.propose(n=1)
    assert x_new.shape == (1, 2)


def test_checkpoint_is_clone_not_view() -> None:
    """Mutating the checkpoint tensor must not affect the backend."""
    be = initialized_backend()
    cp = be.checkpoint()
    cp.X[0, 0] = 999.0

    assert be.get_history().X[0, 0] != 999.0


def test_checkpoint_independent_of_caller_mutation() -> None:
    """Mutating the cp.X tensor after restore() must not affect the restored history.

    restore() internally clones cp.X; subsequent in-place mutation of cp.X
    is irrelevant.
    """
    be = initialized_backend()
    cp = be.checkpoint()
    original_x00 = float(be.get_history().X[0, 0].item())
    be.restore(cp)
    # Now mutate the cp's tensor.
    cp.X[0, 0] = -999.0
    # Backend's restored history was already cloned — should still hold the original.
    assert float(be.get_history().X[0, 0].item()) == original_x00


def test_two_restores_produce_same_history_after_one_more_step() -> None:
    """BoTorch GP fit is non-deterministic across calls (scipy_minimize
    non-reproducible); we test history equivalence instead of step equivalence.

    With Phase 6 correction, GP state IS preserved across restores — so the
    new candidate should be bit-identical.
    """
    be = initialized_backend()
    be.fit()
    cp = be.checkpoint()

    # First restore + step (no explicit refit needed; GP state restored).
    be.restore(cp)
    be.evaluate(be.propose())
    h1 = be.get_history()

    # Second restore + step.
    be.restore(cp)
    be.evaluate(be.propose())
    h2 = be.get_history()

    # With GP state preserved across restore, the proposed candidate should
    # be bit-identical.
    assert torch.equal(h1.X[-1:], h2.X[-1:])
    assert torch.equal(h1.objectives[-1:], h2.objectives[-1:])




def test_checkpoint_preserves_problem_id_and_bounds() -> None:
    be = initialized_backend()
    cp = be.checkpoint()
    assert cp.problem_id == "Branin"
    assert cp.problem_bounds == [(-5.0, 10.0), (0.0, 15.0)]


def test_two_independent_checkpoints_at_different_states() -> None:
    be = initialized_backend()
    cp1 = be.checkpoint()
    be.evaluate(torch.tensor([[0.0, 7.5]], dtype=torch.double))
    cp2 = be.checkpoint()

    assert cp1.X.shape[0] + 1 == cp2.X.shape[0]
    # Restore cp1; backend should revert.
    be.restore(cp1)
    assert be.get_history().X.shape[0] == cp1.X.shape[0]


def test_subprocess_pickle_roundtrip() -> None:
    """Spawn a subprocess that creates a checkpoint and prints its pickle.

    Validate that the parent process can pickle.loads and inspect it.
    """
    import pickle
    import subprocess

    script = """
import pickle, sys
sys.path.insert(0, 'src')
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend
be = BocodeBotorchBackend()
be.initialize(BackendConfig(problem_id='Branin', seed=0, n_init=5))
cp = be.checkpoint()
sys.stdout.buffer.write(pickle.dumps(cp))
"""
    out = subprocess.run(
        ["uv", "run", "python", "-c", script],
        capture_output=True, check=True, cwd=ROOT
    ).stdout
    cp = pickle.loads(out)
    assert cp.problem_id == "Branin"
    assert cp.X.shape == (5, 2)


# ---------------------------------------------------------------------------
# RecordingBackend stub
# ---------------------------------------------------------------------------


def test_recording_backend_checkpoint_roundtrip() -> None:
    be = RecordingBackend()
    cp = be.checkpoint()
    be.evaluate(torch.tensor([[0.3, 0.3]], dtype=torch.double))
    be.restore(cp)
    assert be.get_history().X.shape[0] == 2
    assert not be.is_fitted  # Phase 5 invalidates fitted on restore

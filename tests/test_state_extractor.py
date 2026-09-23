"""Phase 2 acceptance tests for OptimizationState extraction.

Per codex (gpt-5.6-sol) review 2025-09-23:
  - 4 core tests (initial, post-evaluate, JSON-serializable, pure/no-mutation)
  - 4 edge tests (total_budget <= 0, total_budget < n_samples,
                  improvement/tie/regression sequences, surrogate lifecycle)
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.backend.bocode_botorch import (  # noqa: E402
    BackendConfig,
    BocodeBotorchBackend,
    EvaluationBatch,
    UnifiedBackend,
)
from opt_agent.state import (  # noqa: E402
    extract_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_branin(seed: int = 42, n_init: int = 5) -> BocodeBotorchBackend:
    be = BocodeBotorchBackend()
    be.initialize(BackendConfig(problem_id="Branin", seed=seed, n_init=n_init))
    return be


def _stagnation_steps(values: list) -> int:
    incumbent = values[0]
    count = 0
    for value in values[1:]:
        if value > incumbent:
            incumbent = value
            count = 0
        else:
            count += 1
    return count


class _FakeBackend(UnifiedBackend):
    """Minimal UnifiedBackend stub for testing the extractor in isolation."""

    def __init__(self, values: list, fitted: bool = False) -> None:
        # Build an EvaluationBatch from values.
        self._values = values
        n = len(values)
        self._history = EvaluationBatch(
            X=torch.zeros((n, 2), dtype=torch.double),
            objectives=torch.tensor(values, dtype=torch.double).unsqueeze(-1),
            constraints=torch.zeros((n, 0), dtype=torch.double),
            feasible=torch.ones((n, 1), dtype=torch.bool),
        )
        self._fitted = fitted

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def initialize(self, config):
        raise NotImplementedError

    def evaluate(self, X):
        raise NotImplementedError

    def fit(self) -> None:
        raise NotImplementedError

    def propose(self, n: int = 1):
        raise NotImplementedError

    def get_history(self) -> EvaluationBatch:
        return self._history

    def checkpoint(self):
        from opt_agent.backend.checkpoint import Checkpoint
        return Checkpoint(
            backend_id=type(self).__module__ + "." + type(self).__name__,
            problem_id="FakeBackend",
            problem_bounds=[(0.0, 1.0), (0.0, 1.0)],
            X=self._history.X.detach().clone(),
            objectives=self._history.objectives.detach().clone(),
            constraints=self._history.constraints.detach().clone(),
            fitted=self._fitted,
            rng_state=None,
        )

    def restore(self, cp) -> None:
        from opt_agent.backend.checkpoint import Checkpoint
        if not isinstance(cp, Checkpoint):
            raise ValueError("restore expects Checkpoint")
        expected = type(self).__module__ + "." + type(self).__name__
        if cp.backend_id != expected:
            raise ValueError(f"backend_id mismatch: {cp.backend_id} vs {expected}")
        self._history = EvaluationBatch(
            X=cp.X.detach().clone(),
            objectives=cp.objectives.detach().clone(),
            constraints=cp.constraints.detach().clone(),
            feasible=torch.ones((cp.X.shape[0], 1), dtype=torch.bool),
        )
        self._fitted = False


# ---------------------------------------------------------------------------
# Core 4 acceptance tests (per codex)
# ---------------------------------------------------------------------------


def test_extract_initial_state() -> None:
    backend = _init_branin()
    history = backend.get_history()
    values = history.objectives[:, 0].tolist()

    state = extract_state(backend, total_budget=20)

    previous_best = max(values[:-1])
    expected_recent = max(0.0, values[-1] - previous_best)

    assert state.sampling.n_samples == 5
    assert state.sampling.dimension == 2
    assert state.surrogate.is_current is False
    assert state.optimization.best_value == pytest.approx(max(values))
    assert state.optimization.last_value == pytest.approx(values[-1])
    assert state.optimization.recent_improvement == pytest.approx(expected_recent)
    assert state.optimization.stagnation_steps == _stagnation_steps(values)
    assert state.resource.used_budget == 5
    assert state.resource.remaining_budget == 15
    assert state.resource.remaining_budget_ratio == pytest.approx(0.75)
    assert state.history.last_action is None


def test_extract_updates_after_evaluation() -> None:
    backend = _init_branin()

    backend.fit()
    assert extract_state(backend, total_budget=20).surrogate.is_current

    previous_best = backend.get_history().objectives.max().item()
    # Known Brannin optimum (BOCoDe-frame MAX): (-π, 12.275)
    optimum = torch.tensor([[-math.pi, 12.275]], dtype=torch.double)
    backend.evaluate(optimum)

    state = extract_state(
        backend,
        total_budget=20,
        last_action="evaluate_candidate",
    )
    history = backend.get_history()
    expected_best = history.objectives.max().item()

    assert state.sampling.n_samples == 6
    assert state.optimization.best_value == pytest.approx(expected_best)
    assert state.optimization.last_value == pytest.approx(
        history.objectives[-1, 0].item()
    )
    assert state.optimization.recent_improvement == pytest.approx(
        max(0.0, expected_best - previous_best)
    )
    assert state.surrogate.is_current is False
    assert state.resource.remaining_budget == 14
    assert state.history.last_action == "evaluate_candidate"


def test_state_is_json_serializable_and_tensor_free() -> None:
    state = extract_state(
        _init_branin(),
        total_budget=20,
        last_action="initial_design",
    )
    payload = asdict(state)

    # Must be JSON-serializable.
    json.dumps(payload)

    # Must contain no torch.Tensor instances.
    def contains_tensor(value) -> bool:
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, dict):
            return any(contains_tensor(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_tensor(v) for v in value)
        return False

    assert not contains_tensor(payload)


def test_extract_is_pure_and_fit_still_works() -> None:
    backend = _init_branin()
    before = backend.get_history()
    rng_before = torch.random.get_rng_state().clone()

    extract_state(backend, total_budget=20)

    after = backend.get_history()
    rng_after = torch.random.get_rng_state()

    assert torch.equal(before.X, after.X)
    assert torch.equal(before.objectives, after.objectives)
    assert torch.equal(before.constraints, after.constraints)
    assert torch.equal(before.feasible, after.feasible)
    assert torch.equal(rng_before, rng_after)

    backend.fit()
    assert backend.is_fitted


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_total_budget_zero_raises() -> None:
    with pytest.raises(ValueError, match="total_budget must be > 0"):
        extract_state(_init_branin(), total_budget=0)


def test_total_budget_negative_raises() -> None:
    with pytest.raises(ValueError, match="total_budget must be > 0"):
        extract_state(_init_branin(), total_budget=-5)


def test_total_budget_less_than_n_samples_raises() -> None:
    backend = _init_branin()  # 5 samples
    with pytest.raises(ValueError, match="less than current n_samples"):
        extract_state(backend, total_budget=3)


def test_improvement_tie_regression_sequence() -> None:
    """Synthetic sequence: [1.0, 1.5, 1.5, 1.2, 1.7, 1.7] → stagnation_steps=1 at end."""
    fake = _FakeBackend(values=[1.0, 1.5, 1.5, 1.2, 1.7, 1.7])
    state = extract_state(fake, total_budget=10, last_action="evaluate_candidate")
    assert state.optimization.best_value == pytest.approx(1.7)
    assert state.optimization.last_value == pytest.approx(1.7)
    # last step is a tie (1.7 == incumbent), so no improvement, stagnation=1.
    assert state.optimization.stagnation_steps == 1
    assert state.optimization.recent_improvement == pytest.approx(0.0)


def test_surrogate_lifecycle_false_to_fit_to_false() -> None:
    """is_current goes False → True after fit → False after evaluate."""
    backend = _init_branin()
    s0 = extract_state(backend, total_budget=20)
    assert s0.surrogate.is_current is False

    backend.fit()
    s1 = extract_state(backend, total_budget=20)
    assert s1.surrogate.is_current is True

    backend.evaluate(torch.tensor([[0.0, 0.0]], dtype=torch.double))
    s2 = extract_state(backend, total_budget=20)
    assert s2.surrogate.is_current is False

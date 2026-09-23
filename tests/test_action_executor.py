"""Phase 3 acceptance tests for execute_action().

Per codex (gpt-5.6-sol) review 2025-09-23:
  - RecordingBackend stub tracks fit/propose/evaluate call counts.
  - 15+ behavioral tests covering all 5 actions + validation + budget +
    stale-state + failure propagation.
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
    ActionResult,
    ActionType,
    OptimizationAction,
    execute_action,
)
from opt_agent.backend.bocode_botorch import (  # noqa: E402
    BackendConfig,
    BocodeBotorchBackend,
)
from opt_agent.state import extract_state  # noqa: E402


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def state_for(
    backend: UnifiedBackend, total_budget: int = 10, last_action: str | None = None
):
    return extract_state(backend, total_budget=total_budget, last_action=last_action)


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------


def test_global_exploration_adds_samples() -> None:
    backend = RecordingBackend()
    state = state_for(backend, total_budget=10)

    result = execute_action(
        backend,
        state,
        OptimizationAction(ActionType.GLOBAL_EXPLORATION, {"n_samples": 3}),
    )

    assert result.n_new_samples == 3
    assert result.n_model_fits == 0
    assert result.should_stop is False
    assert backend.evaluate_calls == 1
    assert len(backend.get_history()) == 5  # 2 initial + 3


def test_local_exploitation_perturbs_best() -> None:
    """LOCAL_EXPLOITATION perturbs the current best (max objective) point."""
    backend = RecordingBackend()
    history_before = backend.get_history()
    # Current best = index 0 (objective -0.5 > -1.0).
    best = history_before.X[history_before.objectives[:, 0].argmax()].clone()

    execute_action(
        backend,
        state_for(backend),
        OptimizationAction(
            ActionType.LOCAL_EXPLOITATION, {"step": 0.1, "seed": 4}
        ),
    )

    candidate = backend.last_evaluated[0]
    assert torch.all(torch.abs(candidate - best) <= 0.1 * 1.0 + 1e-12)
    assert torch.all(candidate >= backend.bounds[0])
    assert torch.all(candidate <= backend.bounds[1])
    assert not torch.equal(candidate, best)


def test_surrogate_acquisition_fits_proposes_and_evaluates() -> None:
    backend = RecordingBackend(fitted=False)

    result = execute_action(
        backend,
        state_for(backend),
        OptimizationAction(ActionType.SURROGATE_ACQUISITION),
    )

    assert backend.fit_calls == 1
    assert backend.propose_calls == 1
    assert backend.evaluate_calls == 1
    assert result.n_new_samples == 1
    assert result.n_model_fits == 1  # implicit fit counted
    assert torch.equal(
        backend.last_evaluated, torch.tensor([[0.25, 0.25]], dtype=torch.double)
    )


def test_surrogate_acquisition_skips_fit_when_current() -> None:
    backend = RecordingBackend(fitted=True)

    result = execute_action(
        backend,
        state_for(backend),
        OptimizationAction(ActionType.SURROGATE_ACQUISITION),
    )

    assert backend.fit_calls == 0
    assert backend.propose_calls == 1
    assert backend.evaluate_calls == 1
    assert result.n_model_fits == 0  # no fit needed, not counted


def test_retrain_does_not_evaluate() -> None:
    backend = RecordingBackend()

    result = execute_action(
        backend,
        state_for(backend),
        OptimizationAction(ActionType.RETRAIN_SURROGATE),
    )

    assert backend.fit_calls == 1
    assert backend.evaluate_calls == 0
    assert len(backend.get_history()) == 2
    assert result.n_new_samples == 0
    assert result.n_model_fits == 1


def test_stop_is_noop_and_signals_caller() -> None:
    backend = RecordingBackend()
    before = backend.get_history()

    result = execute_action(
        backend,
        state_for(backend),
        OptimizationAction(ActionType.STOP),
    )

    after = backend.get_history()
    assert result.should_stop
    assert result.n_new_samples == 0
    assert torch.equal(before.X, after.X)
    assert backend.fit_calls == 0
    assert backend.evaluate_calls == 0


def test_stop_does_not_require_initialized_backend() -> None:
    """STOP is special: it must work even when the backend would otherwise
    refuse (e.g. backend.get_history() raises)."""

    class Uninitialized(RecordingBackend):
        def get_history(self):
            raise RuntimeError("not initialized")

    backend = Uninitialized()
    # Pass a dummy state (STOP never reads it).
    dummy_state = state_for(RecordingBackend())
    result = execute_action(backend, dummy_state, OptimizationAction(ActionType.STOP))
    assert result.should_stop


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action_type", "params", "message"),
    [
        (ActionType.GLOBAL_EXPLORATION, {"n_samples": 0}, "n_samples"),
        (ActionType.GLOBAL_EXPLORATION, {"n_samples": True}, "n_samples"),
        (ActionType.LOCAL_EXPLOITATION, {"step": 0.0}, "step"),
        (ActionType.LOCAL_EXPLOITATION, {"step": float("nan")}, "step"),
        (ActionType.STOP, {"step": 0.1}, "Unsupported params"),
    ],
)
def test_invalid_params_rejected_at_construction(
    action_type, params, message
) -> None:
    with pytest.raises(ValueError, match=message):
        OptimizationAction(action_type, params)


def test_unknown_action_type_rejected_at_construction() -> None:
    with pytest.raises(TypeError, match="ActionType"):
        OptimizationAction("unknown")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Budget + stale-state safety
# ---------------------------------------------------------------------------


def test_insufficient_budget_rejected_without_mutation() -> None:
    backend = RecordingBackend()  # 2 samples
    state = state_for(backend, total_budget=2)  # remaining = 0

    with pytest.raises(ValueError, match="only 0 remain"):
        execute_action(
            backend,
            state,
            OptimizationAction(ActionType.GLOBAL_EXPLORATION, {"n_samples": 1}),
        )

    assert len(backend.get_history()) == 2
    assert backend.evaluate_calls == 0


def test_stale_state_rejected_without_additional_mutation() -> None:
    backend = RecordingBackend()
    stale = state_for(backend)
    # Mutate backend after extracting state.
    backend.evaluate(torch.tensor([[0.1, 0.1]], dtype=torch.double))
    count = len(backend.get_history())
    calls = backend.evaluate_calls

    with pytest.raises(ValueError, match="Stale state"):
        execute_action(
            backend,
            stale,
            OptimizationAction(ActionType.LOCAL_EXPLOITATION),
        )

    assert len(backend.get_history()) == count
    assert backend.evaluate_calls == calls


def test_backend_evaluation_error_propagates_without_mutation() -> None:
    backend = RecordingBackend(fail_evaluate=True)
    before = len(backend.get_history())

    with pytest.raises(RuntimeError, match="synthetic evaluation failure"):
        execute_action(
            backend,
            state_for(backend),
            OptimizationAction(ActionType.GLOBAL_EXPLORATION, {"n_samples": 1}),
        )

    assert len(backend.get_history()) == before


# ---------------------------------------------------------------------------
# State propagation
# ---------------------------------------------------------------------------


def test_extract_state_after_action_reflects_new_sample() -> None:
    backend = RecordingBackend()
    action = OptimizationAction(ActionType.GLOBAL_EXPLORATION, {"n_samples": 1})

    execute_action(backend, state_for(backend), action)
    state = extract_state(
        backend,
        total_budget=10,
        last_action=action.type.value,
    )

    assert state.sampling.n_samples == 3
    assert state.resource.used_budget == 3
    assert state.resource.remaining_budget == 7
    assert state.history.last_action == "global_exploration"
    assert not state.surrogate.is_current


def test_real_branin_global_exploration_works_end_to_end() -> None:
    """Integration: real Brannin backend, GLOBAL_EXPLORATION, no exceptions."""
    be = BocodeBotorchBackend()
    be.initialize(BackendConfig(problem_id="Branin", seed=0, n_init=3))
    state = extract_state(be, total_budget=20)
    result = execute_action(
        be,
        state,
        OptimizationAction(ActionType.GLOBAL_EXPLORATION, {"n_samples": 2}),
    )
    assert result.n_new_samples == 2
    assert len(be.get_history()) == 5

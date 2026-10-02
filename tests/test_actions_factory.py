"""Tests for opt_agent.actions.factory.candidate_actions_for_state."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.actions import ActionType, OptimizationAction, candidate_actions_for_state  # noqa: E402
from opt_agent.state import OptimizationState  # noqa: E402


def _state(
    *, remaining_budget: int = 10, n_samples: int = 0, is_current: bool = True,
) -> OptimizationState:
    """Build a minimal OptimizationState for factory tests."""
    state = OptimizationState(
        sampling=object.__new__(type("S", (), {"n_samples": n_samples})),  # placeholder
        surrogate=object.__new__(type("Su", (), {"is_current": is_current})),
        optimization=object.__new__(type("O", (), {})),
        resource=object.__new__(type("R", (), {"remaining_budget": remaining_budget})),
        history=object.__new__(type("H", (), {})),
    )
    # Patch in the attrs that factory reads.
    state.sampling.n_samples = n_samples
    state.surrogate.is_current = is_current
    state.resource.remaining_budget = remaining_budget
    return state


# ---------------------------------------------------------------------------
# Budget filtering
# ---------------------------------------------------------------------------


def test_factory_returns_only_stop_when_budget_exhausted() -> None:
    """Budget <= 0 → single STOP candidate (forced termination)."""
    cands = candidate_actions_for_state(_state(remaining_budget=0))
    assert set(cands) == {"stop"}
    assert cands["stop"].type == ActionType.STOP


def test_factory_includes_stop_when_budget_available() -> None:
    """Budget > 0 → STOP remains a valid choice (early termination)."""
    cands = candidate_actions_for_state(_state(remaining_budget=10))
    assert "stop" in cands


def test_factory_emits_at_least_one_sampling_action_when_budget_available() -> None:
    """Budget > 0 → at least one sampling action is in the candidate set."""
    cands = candidate_actions_for_state(_state(remaining_budget=10))
    sampling = {"global_exploration", "uncertainty_sampling", "boundary_sampling"}
    assert sampling & set(cands), f"no sampling action in {set(cands)}"


# ---------------------------------------------------------------------------
# Surrogate filtering
# ---------------------------------------------------------------------------


def test_factory_includes_retrain_when_surrogate_not_current() -> None:
    cands = candidate_actions_for_state(_state(remaining_budget=10, is_current=False))
    assert "retrain_surrogate" in cands


def test_factory_omits_retrain_when_surrogate_current() -> None:
    cands = candidate_actions_for_state(_state(remaining_budget=10, is_current=True))
    assert "retrain_surrogate" not in cands


# ---------------------------------------------------------------------------
# Local-action filtering (need at least one observed sample)
# ---------------------------------------------------------------------------


def test_factory_omits_local_actions_when_no_samples() -> None:
    cands = candidate_actions_for_state(_state(remaining_budget=10, n_samples=0))
    assert "local_exploitation" not in cands


def test_factory_includes_local_actions_when_samples_exist() -> None:
    cands = candidate_actions_for_state(_state(remaining_budget=10, n_samples=5))
    assert "local_exploitation" in cands


# ---------------------------------------------------------------------------
# Vocabulary alignment with full_candidates
# ---------------------------------------------------------------------------


def test_factory_action_ids_align_with_full_candidates() -> None:
    """The factory's action_id vocabulary must be a subset of full_candidates'
    vocabulary (per codex M1: a single source of truth)."""
    from opt_agent.dataset import full_candidates
    expected = {c.action_id for c in full_candidates()}
    cands = candidate_actions_for_state(_state(remaining_budget=10, n_samples=5, is_current=False))
    assert set(cands) <= expected, (
        f"factory emitted ids not in full_candidates: {set(cands) - expected}"
    )


def test_factory_values_are_optimization_actions() -> None:
    """Every candidate value must be an OptimizationAction of the right type."""
    cands = candidate_actions_for_state(_state(remaining_budget=10, n_samples=5, is_current=False))
    for action_id, action in cands.items():
        assert isinstance(action, OptimizationAction)
        assert action.type.value == action_id


# ---------------------------------------------------------------------------
# Params consistency
# ---------------------------------------------------------------------------


def test_factory_caps_n_samples_at_remaining_budget() -> None:
    """For remaining_budget < 2, sampling params must not exceed the budget."""
    cands = candidate_actions_for_state(_state(remaining_budget=1))
    ge = cands["global_exploration"]
    assert ge.params["n_samples"] <= 1




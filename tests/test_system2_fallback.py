"""Tests for FallbackSystem2 — escalation must not collapse into STOP."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.actions import ActionType  # noqa: E402
from opt_agent.loop import DecisionLoop, TerminationReason  # noqa: E402
from opt_agent.state import OptimizationState  # noqa: E402
from opt_agent.system2 import FallbackSystem2, System2Decision  # noqa: E402

from conftest import RecordingBackend  # noqa: E402


def _state_at(remaining_budget: int) -> OptimizationState:
    return OptimizationState(
        sampling=None, surrogate=None, optimization=None,
        resource=None, history=None,
    ) if False else _real_state(remaining_budget)


def _real_state(remaining_budget: int) -> OptimizationState:
    """Build a state with the given remaining_budget using RecordingBackend."""
    return _state_via_backend(remaining_budget)


def _state_via_backend(remaining_budget: int) -> OptimizationState:
    from opt_agent.state import extract_state
    backend = RecordingBackend()
    state = extract_state(backend, total_budget=remaining_budget)
    # Override remaining_budget if needed (extract_state uses 0 by default).
    object.__setattr__(state.resource, "remaining_budget", remaining_budget)
    return state


# ---------------------------------------------------------------------------
# FallbackSystem2 basic behavior
# ---------------------------------------------------------------------------


def test_fallback_returns_rule_policy_action() -> None:
    """With plenty of budget, RulePolicy returns SURROGATE_ACQUISITION."""
    state = _state_via_backend(remaining_budget=20)
    decision = FallbackSystem2().reason(state)
    assert isinstance(decision, System2Decision)
    assert decision.action.type == ActionType.SURROGATE_ACQUISITION
    assert decision.confidence == 1.0  # non-STOP → confident


def test_fallback_returns_stop_only_when_baseline_does() -> None:
    """At LOW_BUDGET_THRESHOLD=4, RulePolicy returns STOP. Fallback inherits."""
    state = _state_via_backend(remaining_budget=2)
    decision = FallbackSystem2().reason(state)
    assert decision.action.type == ActionType.STOP
    assert decision.confidence == 0.0  # STOP → not confident


def test_fallback_does_not_force_stop_on_high_budget() -> None:
    """Critical: even at exactly LOW_BUDGET_THRESHOLD+1, must NOT return STOP."""
    state = _state_via_backend(remaining_budget=5)
    decision = FallbackSystem2().reason(state)
    assert decision.action.type != ActionType.STOP
    assert decision.confidence == 1.0


def test_fallback_uses_stagnation_rule() -> None:
    """RulePolicy switches to GLOBAL_EXPLORATION when stagnation >= 3."""
    from opt_agent.state import OptimizationState
    state = _state_via_backend(remaining_budget=20)
    # Force stagnation_steps = 5 via object.__setattr__ (frozen dataclass).
    object.__setattr__(state.optimization, "stagnation_steps", 5)
    decision = FallbackSystem2().reason(state)
    assert decision.action.type == ActionType.GLOBAL_EXPLORATION


def test_hybrid_smoke_fallback_matches_same_seed_rule_policy() -> None:
    """Paired hybrid and RulePolicy runs must use the same exploration seed."""
    from scripts.run_hybrid_smoke import _make_system2
    from opt_agent.loop.policies import RulePolicy

    state = _state_via_backend(remaining_budget=20)
    object.__setattr__(state.optimization, "stagnation_steps", 5)
    fallback_action = _make_system2("fallback", seed=17).reason(state).action
    rule_action = RulePolicy(seed=17).select_action(state)
    assert fallback_action == rule_action
    assert fallback_action.params["seed"] == 17 + state.sampling.n_samples


# ---------------------------------------------------------------------------
# End-to-end: FallbackSystem2 as a DecisionLoop policy via a wrapper
# ---------------------------------------------------------------------------


def _fallback_as_policy(fb: FallbackSystem2):
    """Wrap FallbackSystem2 as a Policy-compatible object for DecisionLoop.run."""
    class _Policy:
        def __init__(self, fb: FallbackSystem2) -> None:
            self._fb = fb
        def select_action(self, state, backend=None):
            return self._fb.reason(state).action
        def reset(self):
            self._fb.reset()
    return _Policy(fb)


def test_fallback_drives_decision_loop_through_multiple_actions() -> None:
    """Regression for M2: Fallback must drive the loop through multiple
    non-STOP actions when budget allows, NOT terminate at the first step.
    With RecordingBackend, SURROGATE_ACQUISITION consumes budget fast, so
    budget=20 yields ~20 actions before RulePolicy switches to STOP at
    budget <= 4. We only assert that we get past 1 non-STOP step here —
    the strict invariant is that the loop is not poisoned by STOP-at-1.
    """
    backend = RecordingBackend()
    fb = FallbackSystem2()
    policy = _fallback_as_policy(fb)
    trajectory = DecisionLoop(backend, total_budget=20).run(policy)
    non_stop = [s for s in trajectory.steps if s.action.type != ActionType.STOP]
    assert len(non_stop) >= 2, (
        f"Fallback terminated too early; only {len(non_stop)} non-STOP "
        f"steps before {trajectory.termination_reason}"
    )
    # Termination may be STOP (budget hit LOW_BUDGET_THRESHOLD) or
    # BUDGET_EXHAUSTED, both are acceptable; the key is that we drove
    # multiple steps before stopping.
    assert trajectory.termination_reason in (
        TerminationReason.STOP,
        TerminationReason.BUDGET_EXHAUSTED,
    )


def test_fallback_reset_clears_baseline() -> None:
    """RulePolicy is stateless, so reset is a no-op — but Fallback should
    expose reset without error."""
    fb = FallbackSystem2()
    fb.reset()  # must not raise

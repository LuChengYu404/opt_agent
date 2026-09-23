"""Tests for HybridPolicy — System-1 + Router + System-2 selective glue.

End-to-end: HybridPolicy drives DecisionLoop.run(policy) on
RecordingBackend; routing stats + HybridStep history are inspected.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conftest import RecordingBackend  # noqa: E402

from opt_agent.actions import ActionType, OptimizationAction  # noqa: E402
from opt_agent.loop import DecisionLoop, TerminationReason  # noqa: E402
from opt_agent.router import ConfidenceRouter  # noqa: E402
from opt_agent.state import OptimizationState  # noqa: E402
from opt_agent.system1 import HybridPolicy, adapt_policy_as_system1  # noqa: E402
from opt_agent.system1.laya_policy import LayaPolicy  # noqa: E402
from opt_agent.system2 import FallbackSystem2  # noqa: E402


# ---------------------------------------------------------------------------
# Fake system1
# ---------------------------------------------------------------------------


@dataclass
class _FakeDecision:
    action: OptimizationAction
    probabilities: dict[str, float]


class _FakeSystem1:
    """A scripted system1: returns whatever probabilities/choices we ask."""

    def __init__(self, sequence: list[tuple[ActionType, dict[str, float]]]) -> None:
        # sequence[i] = (action_type, probabilities for step i)
        self._sequence = sequence
        self._idx = 0

    def decide(self, state: OptimizationState) -> _FakeDecision:
        action_type, probs = self._sequence[self._idx % len(self._sequence)]
        self._idx += 1
        return _FakeDecision(
            action=OptimizationAction(action_type),
            probabilities=probs,
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_hybrid_rejects_system1_without_decide() -> None:
    """A bare Policy (no decide()) must be wrapped via adapt_policy_as_system1."""
    bare = object()  # no decide
    with pytest.raises(TypeError, match="adapt_policy_as_system1"):
        HybridPolicy(system1=bare, router=ConfidenceRouter(),
                     system2=FallbackSystem2())


def test_adapt_policy_as_system1_wraps_a_policy() -> None:
    """RandomPolicy has no decide(); adapter must give it one."""
    from opt_agent.loop import RandomPolicy
    adapter = adapt_policy_as_system1(RandomPolicy(seed=0))
    state = _state_with_budget(20)
    decision = adapter.decide(state)
    assert hasattr(decision, "action")
    assert hasattr(decision, "probabilities")
    assert decision.probabilities == {decision.action.type.value: 1.0}


def _state_with_budget(remaining: int) -> OptimizationState:
    from opt_agent.state import extract_state
    backend = RecordingBackend()
    s = extract_state(backend, total_budget=remaining)
    object.__setattr__(s.resource, "remaining_budget", remaining)
    return s


# ---------------------------------------------------------------------------
# Routing decisions
# ---------------------------------------------------------------------------


def test_high_confidence_uses_system1_action() -> None:
    """ConfidenceRouter(tau=0.5) + probs {a: 0.9, b: 0.1} → use system1."""
    s1 = _FakeSystem1([(ActionType.SURROGATE_ACQUISITION,
                         {"surrogate_acquisition": 0.9, "stop": 0.1})])
    policy = HybridPolicy(
        system1=s1, router=ConfidenceRouter(tau_conf=0.5),
        system2=FallbackSystem2(),
    )
    action = policy.select_action(_state_with_budget(20))
    assert action.type == ActionType.SURROGATE_ACQUISITION
    assert policy.last_step.route.name == "system1"
    assert policy.last_step.chosen_action == policy.last_step.system1_action


def test_low_confidence_escalates_to_system2() -> None:
    """ConfidenceRouter(tau=0.5) + uniform probs → escalate to FallbackSystem2
    which delegates to RulePolicy (→ SURROGATE_ACQUISITION at budget=20).

    Note: the chosen action's TYPE may equal the system-1 action's type
    (both default to SURROGATE_ACQUISITION when budget allows). What
    matters for the routing contract is that the route was system2, and
    the chosen action came from the system2 path.
    """
    s1 = _FakeSystem1([(ActionType.SURROGATE_ACQUISITION,
                         {"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2, "e": 0.2})])
    policy = HybridPolicy(
        system1=s1, router=ConfidenceRouter(tau_conf=0.5),
        system2=FallbackSystem2(),
    )
    action = policy.select_action(_state_with_budget(20))
    assert action.type == ActionType.SURROGATE_ACQUISITION
    assert policy.last_step.route.name == "system2"
    # Route name must reflect escalation regardless of action-type match.
    assert policy.last_step.route.reason.startswith("below tau_conf")


def test_system1_low_confidence_does_not_collapse_to_stop() -> None:
    """Regression for M2: even when routing to System-2 with low remaining
    budget, the system2 stub must NOT default to STOP — it must delegate
    to RulePolicy, which itself stops only at LOW_BUDGET_THRESHOLD <= 4."""
    s1 = _FakeSystem1([(ActionType.SURROGATE_ACQUISITION,
                         {"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2, "e": 0.2})])
    policy = HybridPolicy(
        system1=s1, router=ConfidenceRouter(tau_conf=0.5),
        system2=FallbackSystem2(),
    )
    action = policy.select_action(_state_with_budget(20))
    assert action.type != ActionType.STOP


# ---------------------------------------------------------------------------
# End-to-end with DecisionLoop.run
# ---------------------------------------------------------------------------


def test_hybrid_drives_decision_loop_end_to_end() -> None:
    """End-to-end smoke: HybridPolicy drives DecisionLoop.run(policy) for
    one trajectory on RecordingBackend; verify trajectory shape and that
    we did not abort at step 1."""
    s1 = _FakeSystem1([
        (ActionType.SURROGATE_ACQUISITION,
         {"surrogate_acquisition": 0.9, "stop": 0.1}),
        (ActionType.SURROGATE_ACQUISITION,
         {"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2, "e": 0.2}),  # escalate
        (ActionType.GLOBAL_EXPLORATION,
         {"global_exploration": 0.95, "stop": 0.05}),
    ])
    hybrid = HybridPolicy(
        system1=s1, router=ConfidenceRouter(tau_conf=0.5),
        system2=FallbackSystem2(),
    )
    trajectory = DecisionLoop(RecordingBackend(), total_budget=8).run(hybrid)
    assert trajectory.termination_reason in (
        TerminationReason.STOP, TerminationReason.BUDGET_EXHAUSTED,
    )
    assert len(trajectory.steps) >= 2  # did not abort at step 1
    # Hybrid history aligns 1:1 with trajectory steps.
    assert len(hybrid.history) == len(trajectory.steps)


def test_hybrid_with_laya_policy_smoke() -> None:
    """Smoke: HybridPolicy wrapping the real LayaPolicy via a fake agent
    drives the loop. We inject FakeLaya so no real model is downloaded."""
    class _FakeLaya:
        def __init__(self, choice: str) -> None:
            self.choice = choice
        def predict(self, state, questions):
            offered = questions["next_action"]["criteria"]
            probs = {name: (1.0 if name == self.choice else 0.0) for name in offered}
            return {"answers": {"next_action": {
                "type": "choice", "choice": self.choice,
                "probabilities": probs, "confidence": 0.95,
            }}}

    laya = LayaPolicy(agent=_FakeLaya("surrogate_acquisition"))
    hybrid = HybridPolicy(
        system1=laya, router=ConfidenceRouter(tau_conf=0.5),
        system2=FallbackSystem2(),
    )
    trajectory = DecisionLoop(RecordingBackend(), total_budget=4).run(hybrid)
    assert trajectory.steps
    # Laya's chosen action (delta distribution) is fully confident → system1.
    for step in hybrid.history[:2]:
        assert step.route.name == "system1"
        assert step.chosen_action.type == step.system1_action.type

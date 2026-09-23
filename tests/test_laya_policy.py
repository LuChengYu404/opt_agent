"""The real Laya adapter must translate BO states to legal loop actions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conftest import RecordingBackend  # noqa: E402
from opt_agent.actions import ActionType  # noqa: E402
from opt_agent.loop import DecisionLoop, TerminationReason  # noqa: E402
from opt_agent.state import extract_state  # noqa: E402
from opt_agent.system1 import LayaPolicy  # noqa: E402


class FakeLaya:
    def __init__(self, choice: str) -> None:
        self.choice = choice
        self.requests = []

    def predict(self, state, questions):
        self.requests.append((state, questions))
        offered = questions["next_action"]["criteria"]
        probabilities = {name: float(name == self.choice) for name in offered}
        return {"answers": {"next_action": {
            "choice": self.choice,
            "probabilities": probabilities,
            "confidence": 1.0,
        }}}


def test_laya_choice_runs_through_decision_loop() -> None:
    agent = FakeLaya("surrogate_acquisition")
    backend = RecordingBackend()
    policy = LayaPolicy(agent=agent)
    trajectory = DecisionLoop(backend, total_budget=4).run(
        policy, max_rounds=5
    )
    assert trajectory.termination_reason == TerminationReason.BUDGET_EXHAUSTED
    assert [step.action.type for step in trajectory.steps] == [
        ActionType.SURROGATE_ACQUISITION,
        ActionType.SURROGATE_ACQUISITION,
    ]
    assert len(agent.requests) == 2
    assert len(policy.decisions) == len(trajectory.steps)
    assert policy.decisions[0].confidence == 1.0
    assert set(agent.requests[0][1]["next_action"]["criteria"]) == {
        action.value for action in ActionType
    }


def test_laya_candidates_respect_remaining_budget() -> None:
    agent = FakeLaya("global_exploration")
    state = extract_state(RecordingBackend(), total_budget=3)
    decision = LayaPolicy(agent=agent).decide(state)
    assert decision.action.params["n_samples"] == 1
    assert decision.probabilities["global_exploration"] == 1.0
    assert agent.requests[0][0]["remaining_budget"] == 1
    assert agent.requests[0][0]["validation_rmse"] is None


def test_exhausted_budget_stops_without_laya_call() -> None:
    agent = FakeLaya("global_exploration")
    state = extract_state(RecordingBackend(), total_budget=2)
    decision = LayaPolicy(agent=agent).decide(state)
    assert decision.action.type == ActionType.STOP
    assert decision.probabilities == {"stop": 1.0}
    assert agent.requests == []


def test_invalid_laya_choice_is_rejected_before_execution() -> None:
    agent = FakeLaya("not_a_real_action")
    state = extract_state(RecordingBackend(), total_budget=4)
    with pytest.raises(ValueError, match="unavailable action"):
        LayaPolicy(agent=agent).decide(state)


def test_retrain_removed_once_surrogate_is_current() -> None:
    agent = FakeLaya("retrain_surrogate")
    state = extract_state(RecordingBackend(fitted=True), total_budget=4)
    with pytest.raises(ValueError, match="unavailable action"):
        LayaPolicy(agent=agent).decide(state)

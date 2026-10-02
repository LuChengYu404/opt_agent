"""The real Laya adapter must translate BO states to legal loop actions.

After the Jev/Laya interface unification (2026-09-26), ``LayaPolicy.decide``
returns the same ``JevDecision`` dataclass that ``JevPolicy.decide`` returns.
These tests cover both the unchanged contract (action, probabilities,
confidence, budget-valid factory usage) and the new metadata fields
(latency_s, input_tokens, output_tokens, model) plus aggregate counters.
"""

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
from opt_agent.system1 import JevDecision, LayaPolicy  # noqa: E402
from opt_agent.system1._prompt import _DESCRIPTIONS  # noqa: E402


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
    # Unified interface: every decision is a JevDecision (same dataclass
    # as JevPolicy returns).
    assert all(isinstance(d, JevDecision) for d in policy.decisions)
    assert policy.decisions[0].confidence == 1.0
    assert set(agent.requests[0][1]["next_action"]["criteria"]) == {
        action.value for action in ActionType
    }


def test_laya_candidates_respect_remaining_budget() -> None:
    agent = FakeLaya("global_exploration")
    state = extract_state(RecordingBackend(), total_budget=3)
    decision = LayaPolicy(agent=agent).decide(state)
    assert isinstance(decision, JevDecision)
    assert decision.action.params["n_samples"] == 1
    assert decision.probabilities["global_exploration"] == 1.0
    assert agent.requests[0][0]["remaining_budget"] == 1
    assert agent.requests[0][0]["validation_rmse"] is None


def test_exhausted_budget_stops_without_laya_call() -> None:
    agent = FakeLaya("global_exploration")
    state = extract_state(RecordingBackend(), total_budget=2)
    decision = LayaPolicy(agent=agent).decide(state)
    assert isinstance(decision, JevDecision)
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


# --- New tests for the unified interface ---


def test_laya_decision_records_metadata_fields() -> None:
    """Each JevDecision returned by LayaPolicy carries the metadata fields
    JevPolicy exposes (latency_s, input_tokens, output_tokens, model)."""
    agent = FakeLaya("surrogate_acquisition")
    policy = LayaPolicy(agent=agent)
    state = extract_state(RecordingBackend(), total_budget=4)
    decision = policy.decide(state)
    assert isinstance(decision, JevDecision)
    assert decision.latency_s >= 0.0
    assert decision.input_tokens == 0
    assert decision.output_tokens == 0
    assert decision.model == "convaiinnovations/laya"
    assert decision.probabilities["surrogate_acquisition"] == 1.0


def test_laya_policy_aggregate_counters_track_calls() -> None:
    """call_count and total_latency_s grow with each decide() call;
    totals remain unchanged on the budget-exhausted early-return path."""
    agent = FakeLaya("surrogate_acquisition")
    policy = LayaPolicy(agent=agent)
    state = extract_state(RecordingBackend(), total_budget=2)
    d1 = policy.decide(state)
    d2 = policy.decide(state)
    assert policy.call_count == 2
    assert policy.total_input_tokens == 0
    assert policy.total_output_tokens == 0
    assert policy.total_latency_s >= 0.0
    # First decide() with remaining_budget > 0 actually timed predict().
    assert d1.latency_s >= 0.0
    # Second decide() with remaining_budget == 0 short-circuits to STOP
    # with latency_s == 0.
    assert d2.action.type == ActionType.STOP
    assert d2.latency_s == 0.0


def test_laya_default_criteria_uses_legacy_descriptions() -> None:
    """Lock the default ``criteria_format='legacy'`` so a future default flip
    to 'executable' would break this test instead of silently shifting the
    prompt distribution that produced
    ``runs/phase9_5_laya_zero_shot_5x5.json``."""
    agent = FakeLaya("global_exploration")
    policy = LayaPolicy(agent=agent)
    state = extract_state(RecordingBackend(), total_budget=4)
    policy.decide(state)
    criteria = agent.requests[0][1]["next_action"]["criteria"]
    assert criteria["global_exploration"] == _DESCRIPTIONS["global_exploration"]
    assert criteria["stop"] == _DESCRIPTIONS["stop"]
    # Executable-format hint must NOT appear with the legacy default.
    instructions = agent.requests[0][1]["next_action"]["instructions"]
    assert "without supplying parameters" not in instructions


def test_laya_criteria_format_executable_includes_fixed_call_params() -> None:
    """Mirrors test_jev_choice_uses_budget_valid_factory_action_and_records_usage:
    executable criteria must mention the action's fixed n_samples and its
    lazy per-point surrogate fitting (h3: no seed/tol params anymore)."""
    agent = FakeLaya("global_exploration")
    policy = LayaPolicy(agent=agent, criteria_format="executable")
    state = extract_state(RecordingBackend(), total_budget=4)
    policy.decide(state)
    criteria = agent.requests[0][1]["next_action"]["criteria"]
    assert "n_samples=" in criteria["global_exploration"]
    assert "seed=" not in criteria["global_exploration"]
    assert "objective evaluation" in criteria["global_exploration"]
    assert "Fits the surrogate lazily" in criteria["global_exploration"]
    assert (
        "without supplying parameters"
        in agent.requests[0][1]["next_action"]["instructions"]
    )


def test_laya_state_format_structured_vs_narrative() -> None:
    """``structured`` (default) sends a JSON dict; ``narrative`` sends a
    plain-English paragraph."""
    agent_struct = FakeLaya("global_exploration")
    LayaPolicy(agent=agent_struct).decide(
        extract_state(RecordingBackend(), total_budget=4)
    )
    assert isinstance(agent_struct.requests[0][0], dict)
    assert "remaining_budget" in agent_struct.requests[0][0]

    agent_narr = FakeLaya("global_exploration")
    LayaPolicy(agent=agent_narr, state_format="narrative").decide(
        extract_state(RecordingBackend(), total_budget=4)
    )
    assert isinstance(agent_narr.requests[0][0], str)
    # The narrative mentions budget headroom using "leaving N evaluation(s)".
    assert "leaving" in agent_narr.requests[0][0]


def test_laya_invalid_format_arguments_raise() -> None:
    """Constructor rejects unknown state_format / criteria_format values."""
    with pytest.raises(ValueError, match="state_format"):
        LayaPolicy(agent=FakeLaya("global_exploration"), state_format="bogus")
    with pytest.raises(ValueError, match="criteria_format"):
        LayaPolicy(agent=FakeLaya("global_exploration"), criteria_format="bogus")


def test_laya_reset_clears_aggregate_counters() -> None:
    """reset() mirrors decide()'s state including the new aggregate counters."""
    agent = FakeLaya("surrogate_acquisition")
    policy = LayaPolicy(agent=agent)
    state = extract_state(RecordingBackend(), total_budget=4)
    policy.decide(state)
    assert policy.call_count == 1
    policy.reset()
    assert policy.call_count == 0
    assert policy.total_input_tokens == 0
    assert policy.total_output_tokens == 0
    assert policy.total_latency_s == 0.0
    assert policy.decisions == []
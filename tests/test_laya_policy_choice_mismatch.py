"""Tests for LayaPolicy choice / argmax-probabilities consistency.

Per codex gpt-6-sol review 2026-09-24: the LLM's `choice` field and the
`probabilities` dict are independent. The agent might emit a `choice`
that does NOT equal argmax(probabilities). In that case we must trust
the deterministic argmax (over the LLM's stated choice) so the
System-1 → System-2 handoff records a well-defined action.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.actions import ActionType  # noqa: E402
from opt_agent.state import extract_state  # noqa: E402
from opt_agent.system1 import JevDecision, LayaPolicy  # noqa: E402

from conftest import RecordingBackend  # noqa: E402


class _FakeLayaChoiceNotArgmax:
    """Returns choice='global_exploration' but probabilities that put
    'surrogate_acquisition' as argmax. The LayaPolicy must trust the
    probabilities, not the stated choice."""

    def predict(self, state, questions):
        offered = questions["next_action"]["criteria"]
        # Distribute mass across all candidates offered by the factory.
        # surrogate_acquisition has the highest mass.
        probs = {name: 0.05 for name in offered}
        probs["surrogate_acquisition"] = 0.7
        # But the stated choice is global_exploration.
        return {"answers": {"next_action": {
            "type": "choice", "choice": "global_exploration",
            "probabilities": probs, "confidence": 0.9,
        }}}


class _FakeLayaChoiceMatchesArgmax:
    """Sanity check: when choice IS argmax, no override happens."""

    def predict(self, state, questions):
        offered = questions["next_action"]["criteria"]
        probs = {name: 0.05 for name in offered}
        probs["global_exploration"] = 0.7
        return {"answers": {"next_action": {
            "type": "choice", "choice": "global_exploration",
            "probabilities": probs, "confidence": 0.9,
        }}}


def test_laya_choice_mismatch_uses_argmax() -> None:
    """When the LLM's `choice` doesn't match argmax(probabilities),
    LayaPolicy uses the argmax and records the mismatch count."""
    agent = _FakeLayaChoiceNotArgmax()
    state = extract_state(RecordingBackend(), total_budget=20)
    policy = LayaPolicy(agent=agent, seed=0)
    decision = policy.decide(state)
    # Unified interface: the decision is a JevDecision.
    assert isinstance(decision, JevDecision)
    # The action type must be the argmax of probabilities.
    assert decision.action.type == ActionType.SURROGATE_ACQUISITION
    # The mismatch counter was incremented.
    assert policy._mismatched_choice_count == 1
    # Metadata fields populated even on the mismatch path.
    assert decision.latency_s >= 0.0
    assert decision.input_tokens == 0
    assert decision.output_tokens == 0
    assert decision.model == "convaiinnovations/laya"


def test_laya_choice_matches_argmax_no_override() -> None:
    """When the LLM's `choice` equals argmax(probabilities), no override
    happens and the mismatch counter stays at 0."""
    agent = _FakeLayaChoiceMatchesArgmax()
    state = extract_state(RecordingBackend(), total_budget=20)
    policy = LayaPolicy(agent=agent, seed=0)
    decision = policy.decide(state)
    assert isinstance(decision, JevDecision)
    assert decision.action.type == ActionType.GLOBAL_EXPLORATION
    assert policy._mismatched_choice_count == 0
"""Contract checks for the hosted Jev Choice adapter without API calls."""

from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.state import (  # noqa: E402
    HistoryState, OptimizationProgress, OptimizationState, ResourceState,
    SamplingState, SurrogateState,
)
from opt_agent.system1.jev_policy import JevPolicy  # noqa: E402


def _state() -> OptimizationState:
    return OptimizationState(
        sampling=SamplingState(5, 2, 0.1, 0.2),
        surrogate=SurrogateState(True, 0.1, 0.2, 0.3),
        optimization=OptimizationProgress(1.0, 1.0, 0.0, 0),
        resource=ResourceState(9, 10, 1, 0.1),
        history=HistoryState(None),
    )


def test_jev_choice_uses_budget_valid_factory_action_and_records_usage():
    seen = {}

    def responder(request):
        seen["payload"] = __import__("json").loads(request.content)
        seen["auth"] = request.headers["authorization"]
        choices = seen["payload"]["questions"]["next_action"]["criteria"]
        return httpx.Response(200, json={
            "model": "jev-test",
            "answers": {"next_action": {
                "type": "choice", "choice": "global_exploration",
                "probabilities": {name: (1.0 if name == "global_exploration" else 0.0) for name in choices},
                "confidence": 1.0,
            }},
            "usage": {"input_tokens": 123, "output_tokens": 4},
        })

    with httpx.Client(transport=httpx.MockTransport(responder)) as client:
        policy = JevPolicy(api_key="test-key", seed=17, client=client)
        result = policy.decide(_state())
    assert seen["auth"] == "Bearer test-key"
    assert seen["payload"]["model"] == "jev-latest"
    question = seen["payload"]["questions"]["next_action"]
    criteria = question["criteria"]
    assert "retrain_surrogate" not in criteria
    assert "without supplying parameters" in question["instructions"]
    assert "n_samples=1" in criteria["global_exploration"]  # capped at remaining=1
    assert "Uses 1 objective evaluation" in criteria["global_exploration"]
    assert "Fits the surrogate lazily" in criteria["global_exploration"]
    assert "step=0.1 of the domain width" in criteria["local_exploitation"]
    assert "Uses the current surrogate without refitting" in criteria["surrogate_acquisition"]
    assert "remaining budget unused" in criteria["stop"]
    assert result.action.params["n_samples"] == 1
    assert "seed" not in result.action.params  # h3: no seed params
    assert policy.call_count == 1
    assert policy.total_input_tokens == 123


def test_jev_describes_refit_and_automatic_fit_when_surrogate_is_stale():
    seen = {}

    def responder(request):
        criteria = __import__("json").loads(request.content)["questions"]["next_action"]["criteria"]
        seen["criteria"] = criteria
        return httpx.Response(200, json={
            "answers": {"next_action": {
                "type": "choice", "choice": "retrain_surrogate",
                "probabilities": {name: (1.0 if name == "retrain_surrogate" else 0.0) for name in criteria},
                "confidence": 1.0,
            }}
        })

    stale = replace(_state(), surrogate=SurrogateState(False, float("nan"),
                                                       float("nan"), float("nan")))
    with httpx.Client(transport=httpx.MockTransport(responder)) as client:
        result = JevPolicy(api_key="test-key", client=client).decide(stale)
    assert result.action.type.value == "retrain_surrogate"
    assert "Uses 0 objective evaluations" in seen["criteria"]["retrain_surrogate"]
    assert "no new observation" in seen["criteria"]["retrain_surrogate"]
    assert "already fit automatically" in seen["criteria"]["retrain_surrogate"]
    assert "Fits the surrogate first if it is not current" in seen["criteria"]["surrogate_acquisition"]


def test_jev_legacy_criteria_remain_available_for_prior_baselines():
    seen = {}

    def responder(request):
        question = __import__("json").loads(request.content)["questions"]["next_action"]
        seen["question"] = question
        criteria = question["criteria"]
        return httpx.Response(200, json={
            "answers": {"next_action": {
                "type": "choice", "choice": "global_exploration",
                "probabilities": {name: (1.0 if name == "global_exploration" else 0.0) for name in criteria},
                "confidence": 1.0,
            }}
        })

    with httpx.Client(transport=httpx.MockTransport(responder)) as client:
        JevPolicy(api_key="test-key", client=client,
                  criteria_format="legacy").decide(_state())
    assert seen["question"]["criteria"]["global_exploration"] == (
        "Thompson sampling: evaluate the maximizer of one posterior sample "
        "(global exploration, LMABO portfolio AF)."
    )
    assert "without supplying parameters" not in seen["question"]["instructions"]


def test_jev_rejects_unavailable_choice():
    def responder(_request):
        return httpx.Response(200, json={
            "answers": {"next_action": {
                "type": "choice", "choice": "retrain_surrogate",
                "probabilities": {"retrain_surrogate": 1.0}, "confidence": 1.0,
            }}
        })

    with httpx.Client(transport=httpx.MockTransport(responder)) as client:
        policy = JevPolicy(api_key="test-key", client=client)
        with pytest.raises(ValueError, match="unavailable"):
            policy.decide(_state())


def test_jev_normalizes_display_rounded_probabilities():
    def responder(request):
        choices = __import__("json").loads(request.content)["questions"]["next_action"]["criteria"]
        probabilities = {name: 0.0 for name in choices}
        probabilities["global_exploration"] = 0.50
        probabilities["uncertainty_sampling"] = 0.49
        return httpx.Response(200, json={
            "answers": {"next_action": {
                "type": "choice", "choice": "global_exploration",
                "probabilities": probabilities, "confidence": 0.5,
            }}
        })

    with httpx.Client(transport=httpx.MockTransport(responder)) as client:
        policy = JevPolicy(api_key="test-key", client=client)
        result = policy.decide(_state())
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_narrative_state_explains_budget_and_stagnation():
    seen = {}

    def responder(request):
        payload = __import__("json").loads(request.content)
        seen["state"] = payload["state"]
        choices = payload["questions"]["next_action"]["criteria"]
        return httpx.Response(200, json={
            "answers": {"next_action": {
                "type": "choice", "choice": "global_exploration",
                "probabilities": {name: (1.0 if name == "global_exploration" else 0.0) for name in choices},
                "confidence": 1.0,
            }}
        })

    with httpx.Client(transport=httpx.MockTransport(responder)) as client:
        policy = JevPolicy(api_key="test-key", client=client,
                           state_format="narrative")
        policy.decide(_state())
    assert isinstance(seen["state"], str)
    assert "leaving 1 evaluation" in seen["state"]
    assert "evaluated points since the last improvement" in seen["state"]
    assert "No current surrogate diagnostics" not in seen["state"]


def test_jev_retries_transient_transport_error(monkeypatch):
    attempts = 0

    def responder(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("temporary TLS failure")
        choices = __import__("json").loads(request.content)["questions"]["next_action"]["criteria"]
        return httpx.Response(200, json={
            "answers": {"next_action": {
                "type": "choice", "choice": "global_exploration",
                "probabilities": {name: (1.0 if name == "global_exploration" else 0.0) for name in choices},
                "confidence": 1.0,
            }}
        })

    monkeypatch.setattr("opt_agent.system1.jev_policy.time.sleep", lambda _delay: None)
    with httpx.Client(transport=httpx.MockTransport(responder)) as client:
        policy = JevPolicy(api_key="test-key", client=client)
        policy.decide(_state())
    assert attempts == 2
    assert policy.call_count == 1


def test_jev_prompt_helpers_live_in_shared_module() -> None:
    """Regression guard: prompt helpers were relocated from laya_policy.py
    to a shared _prompt module so neither policy file imports from the other.
    A future refactor that re-imports from laya_policy should fail this."""
    from opt_agent.system1._prompt import (  # noqa: E402
        _DESCRIPTIONS,
        _executable_criterion,
        _state_narrative,
        _state_payload,
    )
    # Sanity: each helper has a recognisable public shape.
    assert isinstance(_DESCRIPTIONS, dict)
    assert "global_exploration" in _DESCRIPTIONS
    assert callable(_state_payload)
    assert callable(_state_narrative)
    assert callable(_executable_criterion)
    # And jev_policy must import the helpers from the shared module, not
    # from laya_policy. Inspect the source text directly so this survives
    # any future module reload.
    import inspect  # noqa: E402
    from opt_agent.system1 import jev_policy  # noqa: E402
    src = inspect.getsource(jev_policy)
    assert "from opt_agent.system1._prompt import" in src
    assert "from opt_agent.system1.laya_policy import" not in src
    from opt_agent.system1 import laya_policy  # noqa: E402
    src_laya = inspect.getsource(laya_policy)
    assert "from opt_agent.system1._prompt import" in src_laya
    # Both policies import the shared DecisionOutput (and JevDecision
    # alias re-export) from _prompt, not from each other.
    assert "from opt_agent.system1.jev_policy import" not in src_laya

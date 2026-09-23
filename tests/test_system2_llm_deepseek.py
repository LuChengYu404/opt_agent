"""Tests for LLMSystem2 (DeepSeek-backed) — uses a fake httpx client."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conftest import RecordingBackend  # noqa: E402

from opt_agent.actions import ActionType  # noqa: E402
from opt_agent.state import extract_state  # noqa: E402
from opt_agent.system2 import LLMSystem2, System2Decision  # noqa: E402
from opt_agent.system2.llm_deepseek import DeepSeekCallStats  # noqa: E402


def _state_with_budget(remaining: int):
    backend = RecordingBackend()
    state = extract_state(backend, total_budget=remaining)
    return state


class _FakeResponse:
    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> dict:
        return self._body

    @property
    def text(self) -> str:
        return json.dumps(self._body)


class _FakeClient:
    def __init__(
        self,
        *,
        status_code: int = 200,
        body: dict | None = None,
        captured: dict | None = None,
    ) -> None:
        self.status_code = status_code
        self.body = body or _default_body()
        self.captured = captured if captured is not None else {}
        self.calls = 0

    def post(self, url, *, headers=None, json=None, timeout=None, **kwargs):
        self.calls += 1
        self.captured["url"] = url
        self.captured["json"] = json
        self.captured["headers"] = headers
        self.captured["timeout"] = timeout
        return _FakeResponse(self.status_code, self.body)


def _default_body() -> dict:
    return {
        "id": "fake-response-id",
        "choices": [{"message": {"content": json.dumps({
            "action_id": "global_exploration",
            "params": {"n_samples": 1, "seed": 7},
            "reasoning": "explore widely given large remaining budget",
        })}}],
        "usage": {"prompt_tokens": 200, "completion_tokens": 30, "total_tokens": 230},
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_llm_system2_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        LLMSystem2()


def test_llm_system2_uses_env_var(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-test-key")
    s = LLMSystem2()
    assert s.api_key == "env-test-key"


def test_llm_system2_override_via_constructor(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-test-key")
    s = LLMSystem2(api_key="ctor-key", base_url="http://x", model="m")
    assert s.api_key == "ctor-key"
    assert s.base_url == "http://x"
    assert s.model == "m"


# ---------------------------------------------------------------------------
# Reason flow with injected fake client
# ---------------------------------------------------------------------------


def test_llm_system2_parses_json_response() -> None:
    state = _state_with_budget(20)
    captured: dict = {}
    client = _FakeClient(captured=captured)
    s = LLMSystem2(client=client, _skip_key_check=True)
    decision = s.reason(state)
    assert isinstance(decision, System2Decision)
    assert decision.action.type == ActionType.GLOBAL_EXPLORATION
    assert decision.action.params["n_samples"] == 1
    assert decision.action.params["seed"] == 7
    assert decision.confidence == 1.0  # parsed cleanly, not fallback
    assert "explore widely" in decision.reasoning.lower() or "budget" in decision.reasoning.lower()
    # Stats recorded.
    assert s.last_stats is not None
    assert s.last_stats.total_tokens == 230
    assert s.call_count == 1
    assert s.total_tokens == 230
    # Request shape.
    assert captured["url"].endswith("/chat/completions")
    payload = captured["json"]
    assert payload["model"] == "deepseek-chat"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"


def test_llm_system2_unknown_action_id_falls_back() -> None:
    state = _state_with_budget(20)
    bad_body = {
        "id": "x",
        "choices": [{"message": {"content": json.dumps({
            "action_id": "fictional_action", "params": {},
            "reasoning": "I made a typo",
        })}}],
        "usage": {"total_tokens": 10},
    }
    s = LLMSystem2(client=_FakeClient(body=bad_body), _skip_key_check=True)
    decision = s.reason(state)
    assert decision.confidence == 0.0  # fallback marker
    assert decision.reasoning.startswith("fallback:")
    # The fallback picks the first non-stop candidate. RecordingBackend's
    # default state has surrogate not current and n_samples>=2, so the
    # first non-stop candidate is global_exploration.
    assert decision.action.type != ActionType.STOP


def test_llm_system2_unparseable_response_falls_back() -> None:
    state = _state_with_budget(20)
    bad_body = {
        "id": "x",
        "choices": [{"message": {"content": "I cannot help with that."}}],
        "usage": {"total_tokens": 10},
    }
    s = LLMSystem2(client=_FakeClient(body=bad_body), _skip_key_check=True)
    decision = s.reason(state)
    assert decision.confidence == 0.0
    assert decision.reasoning.startswith("fallback:")


def test_llm_system2_markdown_fence_is_stripped() -> None:
    state = _state_with_budget(20)
    body = {
        "id": "x",
        "choices": [{"message": {"content": (
            "```json\n"
            + json.dumps({
                "action_id": "surrogate_acquisition",
                "params": {},
                "reasoning": "use the GP",
            })
            + "\n```"
        )}}],
        "usage": {"total_tokens": 5},
    }
    s = LLMSystem2(client=_FakeClient(body=body), _skip_key_check=True)
    decision = s.reason(state)
    assert decision.action.type == ActionType.SURROGATE_ACQUISITION
    assert decision.confidence == 1.0


def test_llm_system2_forced_stop_when_budget_exhausted() -> None:
    """If only STOP is in the candidate set, we short-circuit without
    calling the API at all (no wasted tokens)."""
    state = _state_with_budget(4)  # total_budget must be > 0
    # Force remaining_budget=0 via object.__setattr__ (OptimizationState
    # is a frozen dataclass).
    object.__setattr__(state.resource, "remaining_budget", 0)
    captured: dict = {}
    client = _FakeClient(captured=captured)
    s = LLMSystem2(client=client, _skip_key_check=True)
    decision = s.reason(state)
    assert decision.action.type == ActionType.STOP
    assert decision.confidence == 1.0
    assert client.calls == 0  # never hit the API


def test_llm_system2_raises_on_api_error() -> None:
    state = _state_with_budget(20)
    client = _FakeClient(status_code=500, body={"error": "boom"})
    s = LLMSystem2(client=client, _skip_key_check=True)
    with pytest.raises(RuntimeError, match="DeepSeek API error"):
        s.reason(state)


def test_llm_system2_reset_is_noop() -> None:
    s = LLMSystem2(client=_FakeClient(), _skip_key_check=True)
    s.reset()  # must not raise
    assert s.call_count == 0


# ---------------------------------------------------------------------------
# Per codex gpt-6-sol re-review: params handling and fallback bookkeeping
# ---------------------------------------------------------------------------


def _body_with(action_id: str, params: dict, reasoning: str = "x") -> dict:
    return {
        "id": "x",
        "choices": [{"message": {"content": json.dumps({
            "action_id": action_id, "params": params, "reasoning": reasoning,
        })}}],
        "usage": {"total_tokens": 10},
    }


def test_llm_system2_preserves_candidate_defaults_when_params_empty() -> None:
    """If the LLM emits an empty params dict, the candidate's defaults
    (seed, n_samples, ...) must NOT be replaced by an empty set."""
    s = LLMSystem2(client=_FakeClient(body=_body_with(
        "global_exploration", {}, "trust the candidate defaults"
    )), _skip_key_check=True)
    state = _state_with_budget(20)
    decision = s.reason(state)
    assert decision.action.type == ActionType.GLOBAL_EXPLORATION
    # Candidate factory sets seed = 0 + n_samples=2. Both must survive.
    assert "seed" in decision.action.params
    assert decision.action.params["n_samples"] == 2
    assert "defaults" in decision.reasoning.lower() or s.fallback_count == 0


def test_llm_system2_drops_n_samples_overshoot() -> None:
    """An LLM-supplied n_samples that exceeds remaining_budget is
    dropped (with a fallback note), and the candidate's default wins."""
    s = LLMSystem2(client=_FakeClient(body=_body_with(
        "global_exploration", {"n_samples": 999},
        "I want lots of samples",
    )), _skip_key_check=True)
    state = _state_with_budget(3)  # tiny budget
    decision = s.reason(state)
    assert decision.action.type == ActionType.GLOBAL_EXPLORATION
    # n_samples must NOT be 999; candidate's n_samples is min(2, 3) = 2.
    assert decision.action.params["n_samples"] <= 3
    assert decision.reasoning.startswith("fallback:")
    assert s.fallback_count == 1


def test_llm_system2_drops_disallowed_param_keys() -> None:
    """LLM params outside the action's _ALLOWED_PARAMS are dropped."""
    s = LLMSystem2(client=_FakeClient(body=_body_with(
        "global_exploration", {"n_samples": 1, "fake_param": "ignored"},
        "trying to inject",
    )), _skip_key_check=True)
    state = _state_with_budget(20)
    decision = s.reason(state)
    assert decision.action.type == ActionType.GLOBAL_EXPLORATION
    assert "fake_param" not in decision.action.params
    # n_samples=1 is allowed; the candidate's n_samples=2 is overridden.
    assert decision.action.params["n_samples"] == 1
    assert s.fallback_count == 1  # fake_param was dropped


def test_llm_system2_drops_non_numeric_n_samples() -> None:
    """Per codex gpt-6-sol 2026-09-24: a non-numeric `n_samples`
    (e.g. the LLM emits "2" as a string) must NOT raise TypeError;
    it is dropped with a fallback note."""
    s = LLMSystem2(client=_FakeClient(body=_body_with(
        "global_exploration", {"n_samples": "2"}, "string n_samples",
    )), _skip_key_check=True)
    state = _state_with_budget(20)
    decision = s.reason(state)
    assert decision.action.type == ActionType.GLOBAL_EXPLORATION
    # The string "2" was dropped — candidate default n_samples=2 wins.
    assert decision.action.params["n_samples"] == 2
    assert s.fallback_count == 1
    assert "n_samples=not_numeric" in decision.reasoning


def test_llm_system2_drops_zero_n_samples() -> None:
    """n_samples <= 0 makes no semantic sense and is dropped."""
    s = LLMSystem2(client=_FakeClient(body=_body_with(
        "global_exploration", {"n_samples": 0}, "zero samples",
    )), _skip_key_check=True)
    state = _state_with_budget(20)
    decision = s.reason(state)
    assert decision.action.params["n_samples"] >= 1
    assert s.fallback_count == 1
    assert "n_samples=0<=0" in decision.reasoning or "n_samples=0.0<=0" in decision.reasoning


def test_llm_system2_drops_negative_seed() -> None:
    """Negative numeric seeds are dropped (seed must be non-negative)."""
    s = LLMSystem2(client=_FakeClient(body=_body_with(
        "global_exploration", {"seed": -1}, "negative seed",
    )), _skip_key_check=True)
    state = _state_with_budget(20)
    decision = s.reason(state)
    # Default seed (positive, from candidate) wins.
    assert decision.action.params["seed"] >= 0
    assert s.fallback_count == 1
    assert "seed=-1<0" in decision.reasoning or "seed=-1.0<0" in decision.reasoning


def test_llm_system2_drops_boolean_n_samples() -> None:
    """Booleans pass the isinstance(int,float) check; we exclude them
    explicitly because True/False as n_samples makes no sense."""
    s = LLMSystem2(client=_FakeClient(body=_body_with(
        "global_exploration", {"n_samples": True}, "bool n_samples",
    )), _skip_key_check=True)
    state = _state_with_budget(20)
    decision = s.reason(state)
    assert decision.action.params["n_samples"] != True
    assert decision.action.params["n_samples"] >= 1
    assert s.fallback_count == 1


def test_llm_system2_coerces_float_n_samples_to_int() -> None:
    """Per codex gpt-6-sol 2026-09-24: a float `n_samples: 2.0` from
    the LLM must be coerced to int (NOT raise ValueError inside
    OptimizationAction.__post_init__). The action's n_samples
    validator requires int; the parser coerces by key."""
    s = LLMSystem2(client=_FakeClient(body=_body_with(
        "global_exploration", {"n_samples": 2.0}, "float n_samples",
    )), _skip_key_check=True)
    state = _state_with_budget(20)
    decision = s.reason(state)
    assert isinstance(decision.action.params["n_samples"], int)
    assert decision.action.params["n_samples"] == 2
    # No fallback — the value is valid after coercion.
    assert s.fallback_count == 0


def test_llm_system2_coerces_float_seed_to_int() -> None:
    """Per codex gpt-6-sol 2026-09-24: a float `seed: 0.0` must be
    coerced to int (seed validator requires int)."""
    s = LLMSystem2(client=_FakeClient(body=_body_with(
        "global_exploration", {"seed": 0.0}, "float seed",
    )), _skip_key_check=True)
    state = _state_with_budget(20)
    decision = s.reason(state)
    assert isinstance(decision.action.params["seed"], int)
    assert decision.action.params["seed"] == 0


def test_llm_system2_coerces_step_and_tol_to_float() -> None:
    """Per-action float-typed params (step in local_exploitation,
    tol in boundary_sampling) must coerce int to float so the float
    validator accepts them."""
    s = LLMSystem2(client=_FakeClient(body=_body_with(
        "local_exploitation", {"step": 1}, "int step",
    )), _skip_key_check=True)
    state = _state_with_budget(20)
    decision = s.reason(state)
    assert isinstance(decision.action.params["step"], float)
    assert decision.action.params["step"] == 1.0


def test_llm_system2_coerces_tol_to_float_in_boundary_sampling() -> None:
    s = LLMSystem2(client=_FakeClient(body=_body_with(
        "boundary_sampling", {"tol": 0}, "int tol",
    )), _skip_key_check=True)
    state = _state_with_budget(20)
    decision = s.reason(state)
    assert isinstance(decision.action.params["tol"], float)


def test_llm_system2_drops_out_of_range_step() -> None:
    """step must be in [0.001, 1.0] per `_PARAM_SCHEMAS`. A float outside
    the range is dropped rather than crashing on construction."""
    s = LLMSystem2(client=_FakeClient(body=_body_with(
        "local_exploitation", {"step": 5.0}, "step too large",
    )), _skip_key_check=True)
    state = _state_with_budget(20)
    decision = s.reason(state)
    # The candidate's step (0.1) survives; the override is dropped.
    assert decision.action.params["step"] == pytest.approx(0.1)
    assert s.fallback_count == 1


def test_llm_system2_fallback_count_increments_on_unparseable() -> None:
    s = LLMSystem2(client=_FakeClient(body={
        "id": "x",
        "choices": [{"message": {"content": "garbage response"}}],
        "usage": {"total_tokens": 5},
    }), _skip_key_check=True)
    state = _state_with_budget(20)
    decision = s.reason(state)
    assert s.fallback_count == 1
    assert decision.reasoning.startswith("fallback:")


def test_llm_system2_fallback_count_increments_on_unknown_action() -> None:
    s = LLMSystem2(client=_FakeClient(body=_body_with(
        "fictional", {}, "made-up action",
    )), _skip_key_check=True)
    state = _state_with_budget(20)
    decision = s.reason(state)
    assert s.fallback_count == 1
    assert decision.reasoning.startswith("fallback:")


def test_llm_system2_prefers_system1_top_action_in_last_resort() -> None:
    """When the LLM response is unusable, the fallback should prefer
    the System-1 top action that the router recorded, not a random
    non-stop candidate."""
    s = LLMSystem2(client=_FakeClient(body={
        "id": "x",
        "choices": [{"message": {"content": "garbage"}}],
        "usage": {"total_tokens": 5},
    }), _skip_key_check=True)
    state = _state_with_budget(20)
    # Inject a System-1 top action as the router would.
    s.last_system1_top_action = state.sampling if False else None  # placeholder
    # Use a real action the candidate factory would produce.
    from opt_agent.actions import OptimizationAction as _OA
    s.last_system1_top_action = _OA(ActionType.UNCERTAINTY_SAMPLING, {"n_samples": 1})
    decision = s.reason(state)
    # The fallback should pick uncertainty_sampling (the System-1 top
    # action we recorded), not the arbitrary first non-stop candidate.
    assert decision.action.type == ActionType.UNCERTAINTY_SAMPLING
    assert "last System-1 top action" in decision.reasoning


def test_llm_system2_reset_clears_fallback_count() -> None:
    s = LLMSystem2(client=_FakeClient(body=_body_with(
        "fictional", {}, "x",
    )), _skip_key_check=True)
    state = _state_with_budget(20)
    s.reason(state)
    assert s.fallback_count == 1
    s.reset()
    assert s.fallback_count == 0
    assert s.last_system1_top_action is None

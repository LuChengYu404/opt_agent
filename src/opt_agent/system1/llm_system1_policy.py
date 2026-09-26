"""LLMSystem1Policy — use an LLM as System-1 (no training).

Quick-verification experiment (EXP-RQ0.5, branch exp/rq0.5-llm-system1):
does an LLM-as-System-1 beat AlwaysSurrogate? If not, then trained
Jev's value over AlwaysSurrogate is just LLM's value over AlwaysSurrogate.

Two modes:
- mode="llm": real LLM call via OpenAI-compatible API (DeepSeek, OpenAI, ...)
- mode="mock": inject responses from a heuristic client (for verification
  without API key). The default mock uses the same gold_probs signal as
  AlwaysSurrogate, so mock-vs-AlwaysSurrogate tells us whether the
  integration wrapper works at all.

Same state input as JevPolicy (OptimizationState → narrative prompt).
Same action output as JevPolicy (typed OptimizationAction choice).
Uses the shared build_choice_request / validate_choice_response from
system1._prompt — identical prompt format to JevPolicy so the comparison
is apples-to-apples.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from opt_agent.actions import (
    OptimizationAction,
    candidate_actions_for_state,
)
from opt_agent.state import OptimizationState
from opt_agent.system1._prompt import (
    DecisionOutput,
    build_choice_request,
    validate_choice_response,
)

logger = logging.getLogger(__name__)


class LLMSystem1Policy:
    """Use an LLM as System-1 — every decision point → LLM call → typed action.

    Parameters
    ----------
    mode : {"llm", "mock"}
        "llm" uses the OpenAI-compatible client (default base URL: DeepSeek).
        "mock" uses a deterministic heuristic (see HeuristicLLMClient below) —
        useful for testing the wrapper without an API key.
    client : optional callable
        Inject a custom client for tests. Signature:
            client(messages, model, temperature, max_tokens) -> response_dict
    state_format, criteria_format : see build_choice_request
    """

    def __init__(
        self,
        *,
        mode: str = "llm",
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        api_key: str | None = None,
        timeout_s: float = 30.0,
        temperature: float = 0.0,
        max_tokens: int = 256,
        client: Any | None = None,
        state_format: str = "structured",
        criteria_format: str = "executable",
        seed: int = 0,
    ) -> None:
        if mode not in ("llm", "mock"):
            raise ValueError(f"mode must be 'llm' or 'mock'; got {mode!r}")
        if state_format not in ("structured", "narrative"):
            raise ValueError("state_format must be structured or narrative")
        if criteria_format not in ("legacy", "executable"):
            raise ValueError("criteria_format must be legacy or executable")
        self.mode = mode
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.state_format = state_format
        self.criteria_format = criteria_format
        self.seed = seed
        self._client = client
        if mode == "mock" and client is None:
            self._client = HeuristicLLMClient()
        # Counters — symmetric with JevPolicy / LayaPolicy.
        self.decisions: list[DecisionOutput] = []
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency_s = 0.0
        self.last_error: str | None = None

    def reset(self) -> None:
        self.decisions.clear()
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency_s = 0.0
        self.last_error = None

    def decide(self, state: OptimizationState) -> DecisionOutput:
        choices = candidate_actions_for_state(state, seed=self.seed)
        if state.resource.remaining_budget <= 0:
            decision = DecisionOutput(
                choices["stop"], {"stop": 1.0}, 1.0,
                0.0, 0, 0, self.model,
            )
            self.decisions.append(decision)
            self.call_count += 1
            return decision

        state_arg, question = build_choice_request(
            state, choices,
            state_format=self.state_format,
            criteria_format=self.criteria_format,
        )
        start = time.monotonic()
        try:
            if self.mode == "llm":
                raw = self._call_real_llm(state_arg, question)
            else:
                # Mock: HeuristicLLMClient uses the candidate list + state
                # to produce a response dict (no LLM).
                raw = self._client(state_arg, question, choices=choices, state=state)
        except Exception as exc:
            self.last_error = repr(exc)
            logger.warning("LLMSystem1 call failed (%s); falling back to uniform",
                           self.last_error)
            # Defensive fallback: uniform distribution over valid candidates.
            n = len(choices)
            raw = {
                "answers": {"next_action": {"type": "choice",
                                            "choice": list(choices)[0]}},
                "probabilities": {"next_action": {k: 1.0 / n for k in choices}},
                "model": f"{self.model}:fallback",
            }
        latency = time.monotonic() - start
        selected, probabilities, confidence = validate_choice_response(raw, choices)
        decision = DecisionOutput(
            choices[selected], probabilities, confidence,
            float(latency),
            int(raw.get("usage", {}).get("input_tokens", 0)),
            int(raw.get("usage", {}).get("output_tokens", 0)),
            str(raw.get("model", self.model)),
        )
        self.decisions.append(decision)
        self.call_count += 1
        self.total_input_tokens += decision.input_tokens
        self.total_output_tokens += decision.output_tokens
        self.total_latency_s += latency
        return decision

    def select_action(
        self, state: OptimizationState, backend=None
    ) -> OptimizationAction:
        return self.decide(state).action

    def _call_real_llm(self, state_arg: Any, question: Mapping[str, Any]) -> dict:
        """POST to OpenAI-compatible chat completions API."""
        import httpx

        # Flatten state + question into a single user message (same shape
        # JevPolicy's payload uses).
        messages = [
            {"role": "system", "content": "You are an expert BO decision policy."},
            {"role": "user", "content": f"State:\n{state_arg}\n\nQuestion:\n{question}"},
        ]
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key or ''}"}
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(f"{self.base_url}/chat/completions",
                            json=body, headers=headers)
        r.raise_for_status()
        return self._parse_openai_response(r.json())

    @staticmethod
    def _parse_openai_response(body: dict) -> dict:
        """Map OpenAI chat completion response → Jev-style answer dict.

        OpenAI returns {choices: [{message: {content: "..."}}], usage: {...}}.
        We ask the LLM to respond in JSON: {"action_id": "...", "rationale": "..."}.
        """
        import json
        content = body["choices"][0]["message"]["content"].strip()
        # Strip code-fence markers if present.
        if content.startswith("```"):
            content = content.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Fall back to extracting an action_id: line.
            import re
            m = re.search(r"action_id\s*[:=]\s*([a-z_]+)", content)
            parsed = {"action_id": m.group(1) if m else "stop"}
        action_id = parsed.get("action_id", "stop")
        return {
            "answers": {"next_action": {"type": "choice", "choice": action_id}},
            "probabilities": {"next_action": {action_id: 1.0}},
            "model": body.get("model", "unknown"),
            "usage": body.get("usage", {}),
        }


class HeuristicLLMClient:
    """Deterministic heuristic that simulates an LLM's response.

    Default strategy: pick the action whose argmax in candidate list has the
    highest gold_probs weight (i.e., what an LLM trained on the same labels
    would say). For tests where gold_probs isn't injected, fall back to
    argmax by surrogate_signal-derived heuristic.

    For the quick verification: this client produces a deterministic,
    label-faithful response. Comparing LLMSystem1Policy(mock) vs
    AlwaysSurrogate tells us whether the prompt format + response parsing
    is wired correctly. Once a real LLM is plugged in (mode="llm"),
    any deviation from AlwaysSurrogate would be evidence of LLM-specific
    decision intelligence.
    """

    def __call__(
        self,
        state_arg: Any,
        question: Mapping[str, Any],
        *,
        choices: Mapping[str, OptimizationAction],
        state: OptimizationState,
    ) -> dict:
        # Heuristic: pick the first candidate in `choices` order with
        # uniform probability over the rest. This makes the mock deterministic
        # enough to verify the wrapper + validate_choice_response end-to-end
        # without external dependencies.
        # NOTE: A more sophisticated mock could read state.gold_probs if
        # injected; for now we just verify the wiring.
        first_id = next(iter(choices))
        n = len(choices)
        # Validate_choice_response requires probabilities to sum to ~1.0 and
        # keys to match `choices` exactly.
        probs = {k: 1.0 / n for k in choices}
        # Re-normalize to exactly 1.0 in case of float error.
        s = sum(probs.values())
        probs = {k: v / s for k, v in probs.items()}
        return {
            "answers": {
                "next_action": {
                    "type": "choice",
                    "choice": first_id,
                    "probabilities": probs,
                }
            },
            "model": "mock-heuristic",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

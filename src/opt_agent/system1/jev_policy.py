"""Hosted Jev zero-shot Choice policy for the BO decision loop.

Uses TypeSafe's documented POST /v1/systemone contract. The legacy state and
option descriptions remain available for reproducing earlier baselines.

The transport (HTTP retry / token accounting) lives here; the prompt
shape and Choice response validation are shared with ``LayaPolicy`` in
``system1._prompt``. ``JevDecision`` is kept as an alias of the shared
``DecisionOutput`` for callers that imported the older name.
"""

from __future__ import annotations

import os
import time
from typing import Any, Literal

import httpx

from opt_agent.actions import OptimizationAction, candidate_actions_for_state
from opt_agent.state import OptimizationState
from opt_agent.system1._prompt import (
    DecisionOutput,
    build_choice_request,
    validate_choice_response,
)

# Backwards-compatible alias: callers and tests import JevDecision.
JevDecision = DecisionOutput


class JevPolicy:
    """Call real Jev once per BO decision; record its full Choice output."""

    def __init__(
        self, *, seed: int = 0, api_key: str | None = None,
        model: str | None = None, base_url: str | None = None,
        timeout_s: float = 30.0, client: Any = None,
        state_format: Literal["structured", "narrative"] = "structured",
        criteria_format: Literal["legacy", "executable"] = "executable",
        trust_env: bool = True,
    ) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        key = api_key or os.environ.get("JEV_API") or os.environ.get("TYPESAFE_API_KEY")
        if not key and client is None:
            raise RuntimeError("JEV_API or TYPESAFE_API_KEY is required")
        self.api_key = key or ""
        self.model = model or os.environ.get("JEV_MODEL", "jev-latest")
        self.base_url = (base_url or os.environ.get("JEV_BASE_URL", "https://api.typesafe.ai/v1/systemone")).rstrip("/")
        self.timeout_s = timeout_s
        self.seed = seed
        if state_format not in ("structured", "narrative"):
            raise ValueError("state_format must be structured or narrative")
        if criteria_format not in ("legacy", "executable"):
            raise ValueError("criteria_format must be legacy or executable")
        self.state_format = state_format
        self.criteria_format = criteria_format
        self._client = client or httpx.Client(timeout=timeout_s, trust_env=trust_env)
        self._owns_client = client is None
        self.decisions: list[DecisionOutput] = []
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency_s = 0.0

    def reset(self) -> None:
        self.decisions.clear()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def decide(self, state: OptimizationState) -> DecisionOutput:
        choices = candidate_actions_for_state(state, seed=self.seed)
        if state.resource.remaining_budget <= 0:
            result = DecisionOutput(
                choices["stop"], {"stop": 1.0}, 1.0, 0.0, 0, 0, self.model,
            )
            self.decisions.append(result)
            return result
        state_arg, inner = build_choice_request(
            state, choices,
            state_format=self.state_format,
            criteria_format=self.criteria_format,
        )
        payload = {"model": self.model, "state": state_arg, "questions": inner}
        start = time.monotonic()
        for attempt in range(3):
            try:
                response = self._client.post(
                    self.base_url, json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            except httpx.TransportError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
                continue
            if response.status_code not in (429, 529) or attempt == 2:
                break
            retry_after = response.headers.get("retry-after")
            try:
                delay = float(retry_after) if retry_after else 2 ** attempt
            except ValueError:
                delay = 2 ** attempt
            time.sleep(min(max(delay, 0.0), 10.0))
        latency = time.monotonic() - start
        response.raise_for_status()
        body = response.json()
        # Strict: Jev must agree with its own probabilities, so an
        # inconsistent answer is an error (unlike LayaPolicy's softer
        # reconcile-to-argmax behavior).
        selected, probs, confidence = validate_choice_response(body, choices)
        # Jev must answer the Choice primitive (its API guarantees ``type=choice``).
        if body.get("answers", {}).get("next_action", {}).get("type") != "choice":
            raise ValueError("Jev returned a non-Choice answer type")
        if probs[selected] + 0.021 < max(probs.values()):
            raise ValueError("Jev Choice is inconsistent with its probabilities")
        usage = body.get("usage", {})
        result = DecisionOutput(
            choices[selected], probs, confidence, latency,
            int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)),
            str(body.get("model", self.model)),
        )
        self.decisions.append(result)
        self.call_count += 1
        self.total_input_tokens += result.input_tokens
        self.total_output_tokens += result.output_tokens
        self.total_latency_s += latency
        return result

    def select_action(self, state: OptimizationState, backend=None) -> OptimizationAction:
        return self.decide(state).action
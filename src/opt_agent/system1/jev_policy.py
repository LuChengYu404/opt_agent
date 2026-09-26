"""Hosted Jev zero-shot Choice policy for the BO decision loop.

Uses TypeSafe's documented POST /v1/systemone contract. The legacy state and
option descriptions remain available for reproducing earlier baselines.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import httpx

from opt_agent.actions import OptimizationAction, candidate_actions_for_state
from opt_agent.state import OptimizationState
from opt_agent.system1._prompt import (
    _DESCRIPTIONS,
    _executable_criterion,
    _state_narrative,
    _state_payload,
)


@dataclass(frozen=True)
class JevDecision:
    action: OptimizationAction
    probabilities: dict[str, float]
    confidence: float
    latency_s: float
    input_tokens: int
    output_tokens: int
    model: str


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
        self.decisions: list[JevDecision] = []
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency_s = 0.0

    def reset(self) -> None:
        self.decisions.clear()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def decide(self, state: OptimizationState) -> JevDecision:
        choices = candidate_actions_for_state(state, seed=self.seed)
        if state.resource.remaining_budget <= 0:
            result = JevDecision(choices["stop"], {"stop": 1.0}, 1.0, 0.0, 0, 0, self.model)
            self.decisions.append(result)
            return result
        payload = {
            "model": self.model,
            "state": (_state_payload(state) if self.state_format == "structured"
                      else _state_narrative(state)),
            "questions": {
                "next_action": {
                    "type": "choice",
                    "instructions": (
                        "Choose the next action for maximizing the best observed objective "
                        "within the remaining evaluation budget. Consider exploration, "
                        "surrogate reliability, and whether stopping is justified."
                    ) + (
                        " Each option is a complete executable call with fixed parameters; "
                        "choose one option, without supplying parameters."
                        if self.criteria_format == "executable" else ""
                    ),
                    "criteria": {
                        name: (_executable_criterion(action, state)
                               if self.criteria_format == "executable"
                               else _DESCRIPTIONS[name])
                        for name, action in choices.items()
                    },
                }
            },
        }
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
        try:
            answer = body["answers"]["next_action"]
            selected = answer["choice"]
            raw_probs = answer["probabilities"]
            confidence = float(answer["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Jev response lacks a valid Choice answer") from exc
        if answer.get("type") != "choice" or selected not in choices:
            raise ValueError("Jev returned an unavailable or non-Choice action")
        if not isinstance(raw_probs, Mapping) or set(raw_probs) != set(choices):
            raise ValueError("Jev probabilities do not match the offered actions")
        probs = {name: float(raw_probs[name]) for name in choices}
        if any(not math.isfinite(p) or p < 0 for p in probs.values()):
            raise ValueError("Jev probabilities must be finite and nonnegative")
        total = sum(probs.values())
        # The hosted API sometimes rounds each of up to eight displayed
        # probabilities to two decimals, so the displayed sum can differ
        # slightly from one. Normalize for downstream routing and logging.
        if not 0.95 <= total <= 1.05:
            raise ValueError(f"Jev probability sum is outside rounding tolerance: {total:.6f}")
        probs = {name: value / total for name, value in probs.items()}
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Jev confidence must lie in [0, 1]")
        if probs[selected] + 0.021 < max(probs.values()):
            raise ValueError("Jev Choice is inconsistent with its probabilities")
        usage = body.get("usage", {})
        result = JevDecision(
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

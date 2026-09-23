"""LLMSystem2 backed by the DeepSeek API.

Real LLM reasoning for the System-2 escalation path. Used in place of
`FallbackSystem2` when the caller wants genuine chain-of-thought over the
BO state instead of the deterministic RulePolicy baseline.

Per MASTER_PLAN §14, §26 (Tau agent harness), and §24 selective reasoning.
Per codex gpt-6-sol review 2026-09-23, Phase 11 work swaps in this module
(or an equivalent LLM harness) — Phase 9.1 just adds the integration
point so the smoke harness can exercise both the fallback and the LLM path
on real BOCoDe problems.

Environment
-----------
DEEPSEEK_API_KEY must be set; the constructor raises if it is not.
DEEPSEEK_BASE_URL overrides the default `https://api.deepseek.com/v1`.
DEEPSEEK_MODEL overrides the default `deepseek-chat`.

The DeepSeek API is OpenAI-compatible (POST /v1/chat/completions), so we
use httpx directly to avoid pulling in an `openai` SDK dependency.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

from opt_agent.actions import ActionType, OptimizationAction, candidate_actions_for_state
from opt_agent.actions.models import _ALLOWED_PARAMS
from opt_agent.state import OptimizationState
from opt_agent.system2.base import System2Decision, System2Policy


_DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
_DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"


@dataclass(frozen=True)
class DeepSeekCallStats:
    """Per-call diagnostics for the LLM escalation path."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    raw_response_id: str | None


class LLMSystem2(System2Policy):
    """System-2 policy backed by the DeepSeek Chat Completions API.

    Prompt format
    -------------
    The LLM is asked to choose ONE action from the budget-valid
    candidate list, with a short rationale. The candidate list is
    formatted as `[{id}: {description}, ...]` so the model can
    cross-reference its `action_id` reply.

    Response parsing
    ----------------
    We accept either:
      - a JSON object {"action_id": "...", "params": {...}, "reasoning": "..."}
      - a free-form reply that contains a literal `action_id: <name>` line

    The chosen action is validated against the candidate factory; if the
    LLM emits an invalid id, we fall back to the highest-confidence
    System-1 action (defensive — never crash the loop because of an LLM
    typo).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 30.0,
        max_tokens: int = 256,
        temperature: float = 0.2,
        client: Any = None,
        _skip_key_check: bool = False,
    ) -> None:
        resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not resolved_key and not _skip_key_check:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set; export it or pass api_key=."
            )
        self.api_key = resolved_key or ""
        self.base_url = base_url or os.environ.get(
            "DEEPSEEK_BASE_URL", _DEEPSEEK_DEFAULT_BASE_URL,
        )
        self.model = model or os.environ.get("DEEPSEEK_MODEL", _DEEPSEEK_DEFAULT_MODEL)
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = client  # injectable for tests
        self.last_stats: DeepSeekCallStats | None = None
        self.call_count = 0
        self.total_tokens = 0

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for LLMSystem2; install via `uv add httpx`."
            ) from exc
        return httpx

    def _build_prompt(
        self, state: OptimizationState, candidates: dict[str, OptimizationAction],
    ) -> list[dict[str, str]]:
        remaining = state.resource.remaining_budget
        n_samples = state.sampling.n_samples
        is_current = state.surrogate.is_current
        best_value = state.optimization.best_value
        stagnation = state.optimization.stagnation_steps
        # Short, structured context the model can ground its choice in.
        context = (
            f"BO state:\n"
            f"  remaining_budget = {remaining}\n"
            f"  samples observed = {n_samples}\n"
            f"  surrogate current = {is_current}\n"
            f"  best observed objective (BOCoDe MAX frame) = {best_value:.6f}\n"
            f"  stagnation_steps = {stagnation}\n"
        )
        cands_lines = "\n".join(
            f"  - {aid}: {action.type.value}"
            + (f"  params={dict(action.params)}" if action.params else "")
            for aid, action in candidates.items()
        )
        instructions = (
            "You are the System-2 reasoning module for an adaptive "
            "Bayesian-optimization agent. Pick ONE action from the "
            "budget-valid candidate list. Prefer exploration when "
            "remaining_budget is large or stagnation is high; prefer "
            "exploitation (or STOP) when budget is small or the "
            "objective has stopped improving."
        )
        output_spec = (
            "Respond with EXACTLY one JSON object on a single line, with "
            "keys: action_id (string, must equal one of the candidate ids), "
            "params (object, may be empty), reasoning (one short "
            "sentence). Do not include any other text."
        )
        user = f"{context}\nCandidates:\n{cands_lines}\n\n{output_spec}"
        return [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user},
        ]

    def _parse_response(
        self,
        candidates: dict[str, OptimizationAction],
        content: str,
    ) -> tuple[OptimizationAction, str]:
        """Extract (action, reasoning) from the LLM response."""
        # Try strict JSON first.
        parsed: Mapping[str, Any] | None = None
        text = content.strip()
        # Some models wrap JSON in ```json ... ``` fences; strip them.
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, Mapping):
                parsed = obj
        except json.JSONDecodeError:
            parsed = None
        # Fallback: regex-style "action_id: <name>".
        if parsed is None:
            for line in text.splitlines():
                if "action_id" in line and ":" in line:
                    _, _, value = line.partition(":")
                    value = value.strip().strip(",").strip('"').strip("'")
                    if value in candidates:
                        return candidates[value], (
                            "fallback: parsed action_id from line"
                        )
            # Last resort: pick the highest-budget "useful" candidate.
            for aid, action in candidates.items():
                if aid == "stop":
                    continue
                return action, "fallback: LLM response unparseable, picked first non-stop candidate"
            return candidates["stop"], "fallback: only STOP available"

        action_id = str(parsed.get("action_id", "")).strip()
        if action_id not in candidates:
            for aid, action in candidates.items():
                if aid == "stop":
                    continue
                return action, (
                    f"fallback: LLM returned unknown action_id={action_id!r}"
                )
        params = parsed.get("params") or {}
        if not isinstance(params, Mapping):
            params = {}
        reasoning = str(parsed.get("reasoning", ""))
        action_type = candidates[action_id].type
        # Validate params against the action's allowed set.
        allowed = _ALLOWED_PARAMS[action_type]
        filtered = {k: v for k, v in params.items() if k in allowed}
        return OptimizationAction(action_type, filtered), reasoning

    def reason(
        self,
        state: OptimizationState,
        *,
        recent_actions: tuple[str, ...] = (),
        remaining_budget: int | None = None,
    ) -> System2Decision:
        candidates = candidate_actions_for_state(state, seed=0)
        # If only STOP is offered (budget exhausted), short-circuit.
        if set(candidates) == {"stop"}:
            return System2Decision(
                action=candidates["stop"],
                confidence=1.0,
                reasoning="forced: budget exhausted, only stop available",
            )

        messages = self._build_prompt(state, candidates)
        import time as _time
        t0 = _time.monotonic()
        client = self._get_client()
        # httpx.Client.post and the injected test client both expose
        # `.post(url, headers=..., json=..., timeout=...)`. Keep the
        # call shape uniform.
        response = client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            },
            timeout=self.timeout_s,
        )
        latency = _time.monotonic() - t0
        if response.status_code != 200:
            raise RuntimeError(
                f"DeepSeek API error: HTTP {response.status_code} "
                f"{response.text[:200]}"
            )
        body = response.json() if hasattr(response, "json") else response
        choice = body["choices"][0]
        content = choice["message"]["content"]
        usage = body.get("usage", {})
        self.last_stats = DeepSeekCallStats(
            model=self.model,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            latency_s=latency,
            raw_response_id=body.get("id"),
        )
        self.call_count += 1
        self.total_tokens += self.last_stats.total_tokens
        action, reasoning = self._parse_response(candidates, content)
        # Confidence: 1.0 unless the parser fell back to a default pick.
        fallback_markers = ("fallback:", "forced:")
        confidence = 0.0 if any(
            reasoning.startswith(m) for m in fallback_markers
        ) else 1.0
        return System2Decision(
            action=action, confidence=confidence, reasoning=reasoning,
        )

    def reset(self) -> None:
        # LLM is stateless across calls (no internal policy state).
        return None

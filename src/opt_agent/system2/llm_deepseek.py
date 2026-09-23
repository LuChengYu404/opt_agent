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
        self.fallback_count = 0
        # The router sets this to the System-1 top action just before
        # escalation so the fallback can prefer it over arbitrary picks.
        self.last_system1_top_action = None

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
        remaining_budget: int,
    ) -> tuple[OptimizationAction, str]:
        """Extract (action, reasoning) from the LLM response.

        Per codex gpt-6-sol review 2026-09-23:
        - The CANDIDATE action is the source of truth for defaults
          (seed, n_samples, tol, ...). LLM `params` are treated as
          overrides; only keys that are allowed AND budget-valid survive.
        - Empty / missing `params` ⇒ use the candidate verbatim.
        - Any override that violates the budget or the action's allowed
          key set is dropped (with a `fallback:` note) and the candidate
          defaults are kept.
        - Unparseable responses fall back to the highest-confidence
          System-1 action recorded on `self.last_system1_top_action`,
          set by the router just before escalation. If that is missing
          (shouldn't happen), we fall back to the candidate factory's
          first non-stop candidate as a last resort.

        Returns
        -------
        (action, reasoning)
            The chosen action and a free-form note. Reasoning starts
            with `fallback:` whenever the parser deviated from the
            clean interpretation path.
        """
        # Try strict JSON first.
        parsed: Mapping[str, Any] | None = None
        text = content.strip()
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
                        self.fallback_count += 1
                        return candidates[value], (
                            "fallback: parsed action_id from line"
                        )
            self.fallback_count += 1
            return self._last_resort(candidates)

        action_id = str(parsed.get("action_id", "")).strip()
        if action_id not in candidates:
            self.fallback_count += 1
            return self._last_resort(
                candidates,
                reason=f"LLM returned unknown action_id={action_id!r}",
            )
        # Use the candidate's params as the base; LLM `params` are
        # overrides filtered through `_ALLOWED_PARAMS`, a budget
        # check on `n_samples`, and a TYPE check on numeric values.
        # Per codex gpt-6-sol review 2026-09-24: a non-numeric value
        # (e.g. `n_samples: "2"`) must NOT raise TypeError mid-build;
        # it is dropped with a `fallback:` note so the loop stays
        # running.
        base = candidates[action_id]
        merged = dict(base.params)
        params = parsed.get("params") or {}
        if not isinstance(params, Mapping):
            params = {}
        allowed = _ALLOWED_PARAMS[base.type]
        # Per-action-type param schema: which keys must be int vs float.
        # Mirrors `_PARAM_SCHEMAS` in `actions.models` without importing
        # the private constant — kept in sync by convention.
        int_keys = {
            ActionType.GLOBAL_EXPLORATION: frozenset({"n_samples", "seed"}),
            ActionType.LOCAL_EXPLOITATION: frozenset({"seed"}),
            ActionType.UNCERTAINTY_SAMPLING: frozenset({"n_samples", "seed"}),
            ActionType.ERROR_GUIDED_SAMPLING: frozenset({"n_samples", "seed"}),
            ActionType.BOUNDARY_SAMPLING: frozenset({"n_samples", "seed"}),
        }.get(base.type, frozenset())
        overrides_applied: list[str] = []
        overrides_dropped: list[str] = []
        for k, v in params.items():
            if k not in allowed:
                overrides_dropped.append(f"{k}=not_allowed")
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                overrides_dropped.append(f"{k}=not_numeric({type(v).__name__})")
                continue
            # Per codex gpt-6-sol 2026-09-24: coerce the numeric override
            # to the type the action validator expects (int for
            # n_samples / seed, float for step / tol). Without this
            # coercion, `n_samples: 2.0` raises
            # `ValueError("'n_samples' must be int; got float")` inside
            # OptimizationAction.__post_init__ and the loop crashes.
            if k in int_keys:
                v_typed: int | float = int(v)
            else:
                v_typed = float(v)
            if k == "n_samples":
                if v_typed > remaining_budget:
                    overrides_dropped.append(
                        f"n_samples={v_typed}>budget={remaining_budget}"
                    )
                    continue
                if v_typed <= 0:
                    overrides_dropped.append(
                        f"n_samples={v_typed}<=0"
                    )
                    continue
            elif v_typed < 0:
                overrides_dropped.append(f"{k}={v_typed}<0")
                continue
            merged[k] = v_typed
            overrides_applied.append(k)
        reasoning = str(parsed.get("reasoning", ""))
        if overrides_dropped:
            self.fallback_count += 1
            note = "; ".join(overrides_dropped)
            reasoning = (
                f"fallback: dropped LLM overrides [{note}]; "
                + (reasoning or "kept candidate defaults")
            )
        # Defensive construction: OptimizationAction's range validators
        # (e.g. step in (0, 1.0], tol in (0, 0.1], n_samples > 0) may
        # still reject overrides that passed our type / budget checks.
        # If construction raises, drop offending overrides one by one
        # and retry until the action builds — per codex gpt-6-sol
        # review 2026-09-24.
        try:
            action = OptimizationAction(base.type, merged)
        except (ValueError, TypeError) as exc:
            dropped_for_construction: list[str] = []
            merged = dict(base.params)
            for k in overrides_applied:
                try:
                    action = OptimizationAction(base.type, merged)
                    break
                except (ValueError, TypeError):
                    dropped_for_construction.append(k)
                    merged.pop(k, None)
                    if k == "n_samples":
                        # Re-derive from the candidate factory's
                        # default so the action is budget-valid.
                        merged["n_samples"] = min(2, remaining_budget)
            else:
                # All overrides dropped; construct with base.
                action = OptimizationAction(base.type, merged)
            self.fallback_count += 1
            reasoning = (
                f"fallback: dropped overrides that failed action "
                f"validation [{','.join(dropped_for_construction)}]; "
                + reasoning
            )
        return action, reasoning

    def _last_resort(
        self,
        candidates: dict[str, OptimizationAction],
        reason: str = "LLM response unparseable",
    ) -> tuple[OptimizationAction, str]:
        """Last-resort pick when the LLM response is unusable.

        Prefers the System-1 top action recorded on this instance by
        the router (the most confident available choice). If that is
        missing (defensive — should never happen), falls back to the
        candidate factory's first non-stop candidate.
        """
        last = getattr(self, "last_system1_top_action", None)
        if last is not None and last.type.value in candidates:
            return candidates[last.type.value], (
                f"fallback: {reason}; using last System-1 top action"
            )
        for aid, action in candidates.items():
            if aid == "stop":
                continue
            return action, (
                f"fallback: {reason}; picked first non-stop candidate"
            )
        return candidates["stop"], "fallback: only STOP available"

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
        action, reasoning = self._parse_response(
            candidates, content, state.resource.remaining_budget,
        )
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
        self.last_system1_top_action = None
        self.fallback_count = 0

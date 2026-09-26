"""Zero-shot Laya reference policy for the existing optimization decision loop.

Laya is optional and loaded only when the first decision is requested. The
policy exposes the full choice distribution for later calibration and routing;
these probabilities are *model outputs*, not validated BO action probabilities.

The interface mirrors ``JevPolicy`` so a single ``JevDecision`` dataclass is
the standard System-1 return type. Latency is measured around
``agent.predict()`` only (not around the full ``decide()`` body), input and
output tokens are 0 (the local SDK does not expose usage), and ``model`` is
set to ``self.model_id`` verbatim.
"""

from __future__ import annotations

import math
import time
from typing import Any, Literal, Mapping, Protocol

from opt_agent.actions import (
    OptimizationAction,
    candidate_actions_for_state,
)
from opt_agent.backend.bocode_botorch import UnifiedBackend
from opt_agent.state import OptimizationState
from opt_agent.system1._prompt import (
    _DESCRIPTIONS,
    _executable_criterion,
    _state_narrative,
    _state_payload,
)
from opt_agent.system1.jev_policy import JevDecision


class _LayaAgent(Protocol):
    def predict(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


class LayaPolicy:
    """Wrap Laya's typed ``choice`` prediction as a DecisionLoop policy.

    ``agent`` may be injected for tests or a locally preloaded checkpoint. The
    default loads ``convaiinnovations/laya`` lazily via the optional SDK.

    The ``state_format`` and ``criteria_format`` parameters mirror those on
    ``JevPolicy`` so callers can compare prompts across the local SDK and the
    hosted API. The defaults ``structured`` and ``legacy`` preserve the prompt
    shape that produced ``runs/phase9_5_laya_zero_shot_5x5.json``; do not flip
    the ``criteria_format`` default without re-running that baseline.
    """

    def __init__(
        self,
        *,
        model_id: str = "convaiinnovations/laya",
        subfolder: str | None = None,
        seed: int = 0,
        agent: _LayaAgent | None = None,
        state_format: Literal["structured", "narrative"] = "structured",
        criteria_format: Literal["legacy", "executable"] = "legacy",
    ) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if state_format not in ("structured", "narrative"):
            raise ValueError("state_format must be structured or narrative")
        if criteria_format not in ("legacy", "executable"):
            raise ValueError("criteria_format must be legacy or executable")
        self.model_id = model_id
        self.subfolder = subfolder
        self.seed = seed
        self.state_format = state_format
        self.criteria_format = criteria_format
        self._agent = agent
        self.last_decision: JevDecision | None = None
        self.decisions: list[JevDecision] = []
        self._mismatched_choice_count = 0
        # Aggregate counters — symmetric with JevPolicy so both policies
        # expose the same shape for downstream JSON reports.
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency_s = 0.0

    def reset(self) -> None:
        self.last_decision = None
        self.decisions.clear()
        self._mismatched_choice_count = 0
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency_s = 0.0

    def _get_agent(self) -> _LayaAgent:
        if self._agent is None:
            try:
                import laya
            except ImportError as exc:
                raise RuntimeError(
                    "Laya is not installed. Install the optional dependency with "
                    "`uv sync --extra laya` before running LayaPolicy."
                ) from exc
            kwargs = {"subfolder": self.subfolder} if self.subfolder else {}
            self._agent = laya.load(self.model_id, **kwargs)
        return self._agent

    def decide(self, state: OptimizationState) -> JevDecision:
        choices = candidate_actions_for_state(state, seed=self.seed)
        if state.resource.remaining_budget <= 0:
            decision = JevDecision(
                choices["stop"], {"stop": 1.0}, 1.0,
                0.0, 0, 0, self.model_id,
            )
            self.last_decision = decision
            self.decisions.append(decision)
            self.call_count += 1
            # latency_s == 0; totals remain unchanged.
            return decision

        question = {
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
                    name: (
                        _executable_criterion(action, state)
                        if self.criteria_format == "executable"
                        else _DESCRIPTIONS[name]
                    )
                    for name, action in choices.items()
                },
            }
        }
        state_arg = (
            _state_payload(state)
            if self.state_format == "structured"
            else _state_narrative(state)
        )
        # Per JevPolicy: latency covers only the model call, not parsing.
        start = time.monotonic()
        result = self._get_agent().predict(state_arg, question)
        latency = time.monotonic() - start
        try:
            answer = result["answers"]["next_action"]
            raw_probs = answer["probabilities"]
            selected = answer["choice"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Laya response lacks next_action choice/probabilities") from exc
        if selected not in choices:
            raise ValueError(f"Laya selected unavailable action: {selected!r}")
        if not isinstance(raw_probs, Mapping) or set(raw_probs) != set(choices):
            raise ValueError("Laya probabilities do not match the offered actions")
        probabilities = {name: float(raw_probs[name]) for name in choices}
        if any(not math.isfinite(p) or p < 0 for p in probabilities.values()):
            raise ValueError("Laya probabilities must be finite and non-negative")
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError("Laya probabilities must have positive total mass")
        probabilities = {name: value / total for name, value in probabilities.items()}
        # Per codex gpt-6-sol review 2026-09-24: the LLM's `choice` and the
        # `probabilities` dict are independent inputs — the agent might
        # emit a choice that doesn't match argmax(probabilities). Treat
        # that as a model-side inconsistency: trust the probabilities
        # (deterministic argmax) over the LLM's stated choice. This
        # keeps the System-1 → System-2 handoff well-defined: the
        # recorded `decision.action` is always the argmax action.
        argmax_action = max(probabilities, key=probabilities.get)
        if selected != argmax_action:
            self._mismatched_choice_count += 1
            selected = argmax_action
        confidence = float(answer.get("confidence", probabilities[selected]))
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Laya confidence must lie in [0, 1]")
        decision = JevDecision(
            choices[selected], probabilities, confidence,
            float(latency), 0, 0, self.model_id,
        )
        self.last_decision = decision
        self.decisions.append(decision)
        self.call_count += 1
        self.total_latency_s += latency
        # input_tokens / output_tokens stay 0 (local SDK has no usage).
        return decision

    def select_action(
        self, state: OptimizationState, backend: UnifiedBackend | None = None
    ) -> OptimizationAction:
        return self.decide(state).action
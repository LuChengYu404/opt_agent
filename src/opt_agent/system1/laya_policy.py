"""Zero-shot Laya reference policy for the existing optimization decision loop.

Laya is optional and loaded only when the first decision is requested. The
policy exposes the full choice distribution for later calibration and routing;
these probabilities are *model outputs*, not validated BO action probabilities.

The interface mirrors ``JevPolicy`` so a single ``DecisionOutput`` dataclass
(the ``JevDecision`` alias) is the standard System-1 return type. Latency
is measured around ``agent.predict()`` only (not around the full
``decide()`` body), input and output tokens are 0 (the local SDK does
not expose usage), and ``model`` is set to ``self.model_id`` verbatim.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Mapping, Protocol

from opt_agent.actions import (
    OptimizationAction,
    candidate_actions_for_state,
)
from opt_agent.backend.bocode_botorch import UnifiedBackend
from opt_agent.state import OptimizationState
from opt_agent.system1._prompt import (
    DecisionOutput,
    build_choice_request,
    validate_choice_response,
)


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
        self.last_decision: DecisionOutput | None = None
        self.decisions: list[DecisionOutput] = []
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

    def decide(self, state: OptimizationState) -> DecisionOutput:
        choices = candidate_actions_for_state(state, seed=self.seed)
        if state.resource.remaining_budget <= 0:
            decision = DecisionOutput(
                choices["stop"], {"stop": 1.0}, 1.0,
                0.0, 0, 0, self.model_id,
            )
            self.last_decision = decision
            self.decisions.append(decision)
            self.call_count += 1
            return decision

        state_arg, question = build_choice_request(
            state, choices,
            state_format=self.state_format,
            criteria_format=self.criteria_format,
        )
        start = time.monotonic()
        raw = self._get_agent().predict(state_arg, question)
        latency = time.monotonic() - start
        selected, probabilities, confidence = validate_choice_response(raw, choices)
        # Per codex gpt-6-sol review 2026-09-24: Laya's ``choice`` field is
        # independent of its ``probabilities``. Trust the deterministic
        # argmax over the LLM's stated choice so the System-1 → System-2
        # handoff records a well-defined action.
        argmax_action = max(probabilities, key=probabilities.get)
        if selected != argmax_action:
            self._mismatched_choice_count += 1
            selected = argmax_action
        decision = DecisionOutput(
            choices[selected], probabilities, confidence,
            float(latency), 0, 0, self.model_id,
        )
        self.last_decision = decision
        self.decisions.append(decision)
        self.call_count += 1
        self.total_latency_s += latency
        return decision

    def select_action(
        self, state: OptimizationState, backend: UnifiedBackend | None = None
    ) -> OptimizationAction:
        return self.decide(state).action
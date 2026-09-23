"""FallbackSystem2 — delegates to RulePolicy so escalation does not collapse
into premature termination.

Per codex gpt-6-sol review 2026-09-23 (major M2):
  - System-2 must delegate to a real baseline (RulePolicy here), never
    default to STOP. STOP is only returned when the baseline itself would
    have chosen STOP (e.g. low remaining budget).
  - Confidence is a constant 1.0: a deterministic policy's output is its
    own ground truth, so "how sure are we?" reduces to "did the baseline
    produce a non-trivial action?" — encoded as 0.0 for STOP and 1.0
    otherwise.
"""

from __future__ import annotations

from opt_agent.actions import ActionType
from opt_agent.loop.policies import RulePolicy
from opt_agent.state import OptimizationState

from opt_agent.system2.base import System2Decision, System2Policy


class FallbackSystem2(System2Policy):
    """Wrap RulePolicy as a System2Policy.

    The router can escalate here when System-1's confidence is low. The
    wrapped RulePolicy keeps the loop honest: it still respects the
    remaining-budget contract and will return STOP only when its own
    thresholds say so.
    """

    def __init__(self, *, baseline: RulePolicy | None = None) -> None:
        self._baseline = baseline or RulePolicy()

    def reason(
        self,
        state: OptimizationState,
        *,
        recent_actions: tuple[str, ...] = (),
        remaining_budget: int | None = None,
    ) -> System2Decision:
        # RulePolicy reads budget from state.resource.remaining_budget; we
        # pass the same state through.
        action = self._baseline.select_action(state)
        confidence = 0.0 if action.type == ActionType.STOP else 1.0
        return System2Decision(
            action=action,
            confidence=confidence,
            reasoning="fallback: RulePolicy baseline",
        )

    def reset(self) -> None:
        self._baseline.reset()

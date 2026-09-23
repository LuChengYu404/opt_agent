"""System-2 policy Protocol — high-level reasoning fallback.

A System-2 policy takes the same view of state as System-1 but is allowed
to reason more expensively (e.g. LLM chain-of-thought). It must return a
complete `OptimizationAction` plus a confidence score for routing.

Phase 9 only ships stubs; Phase 11 wires a real Tau harness here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from opt_agent.actions import OptimizationAction
from opt_agent.state import OptimizationState


@dataclass(frozen=True)
class System2Decision:
    """Result of a System-2 reasoning call.

    Attributes
    ----------
    action : OptimizationAction
        The action the System-2 policy chose. Must be budget-valid for
        the current state (caller validates via the action factory).
    confidence : float
        Calibrated confidence in [0, 1]. The router uses this to decide
        whether to escalate further or trust the action.
    reasoning : str
        Optional free-form explanation (LLM chain-of-thought text, etc.).
        Empty for stub implementations.
    """

    action: OptimizationAction
    confidence: float
    reasoning: str = ""


class System2Policy(Protocol):
    """High-level reasoning policy: state → System2Decision."""

    def reason(
        self,
        state: OptimizationState,
        *,
        recent_actions: tuple[str, ...] = (),
        remaining_budget: int | None = None,
    ) -> System2Decision:
        """Return a System2Decision for the given state.

        Implementations MUST honor the remaining_budget contract: never
        propose an action whose n_samples exceeds the budget.
        """
        ...

    def reset(self) -> None:
        """Clear any internal state. Default no-op."""
        ...

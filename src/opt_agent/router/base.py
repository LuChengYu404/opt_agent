"""Router Protocol — Phase 9 selective-reasoning decision.

The router decides, per step, whether to trust System-1's choice
distribution or escalate to System-2 (more expensive reasoning). Phase 9
ships a confidence-only first version. Phase 12 may add OOD detection,
entropy thresholds, and learned meta-policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

RouteName = Literal["system1", "system2"]


@dataclass(frozen=True)
class Route:
    """A routing decision for a single step.

    Attributes
    ----------
    name : {"system1", "system2"}
        Which subsystem should produce the action this step.
    confidence : float
        The router's confidence in the System-1 distribution in [0, 1].
        Recorded for offline analysis even when the route is system1.
    reason : str
        Short human-readable justification ("below tau_conf", "high
        entropy", etc.). Useful for router ablation studies.
    """

    name: RouteName
    confidence: float
    reason: str = ""


class Router(Protocol):
    """A selective-reasoning router: System-1 output → Route."""

    def route(
        self,
        probabilities: dict[str, float],
    ) -> Route:
        """Decide between System-1 (trust the distribution) and System-2.

        Implementations MAY inspect additional state, but the simplest
        reliable signal is `max(probabilities.values())`.
        """
        ...

    def update(
        self,
        probabilities: dict[str, float],
        chosen: str,
        reward: float,
    ) -> None:
        """Optional online update hook. Default no-op for non-learned routers."""
        ...

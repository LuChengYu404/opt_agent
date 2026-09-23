"""OOD-aware router: escalate when EITHER confidence is low OR state OOD.

Per MASTER_PLAN §15 (router escalation conditions) and §13 (System-1
"knows when it is unreliable"), the router can combine two escalation
signals:

  - low confidence in the System-1 distribution (cheap, per-step)
  - high out-of-distribution score on the current BO state (more
    expensive, requires a state snapshot)

The OOD signal is the new piece — it lets the router refuse to act on
states the System-1 was never trained on, even when its confidence is
artificially high (e.g., a delta distribution on an unfamiliar state).
"""

from __future__ import annotations

from dataclasses import dataclass

from opt_agent.router.base import Route, Router
from opt_agent.system1.ood import StateOODDetector


@dataclass
class OODAwareRouter:
    """Wraps a confidence router and an OOD detector.

    Decision order:
      1. Ask the confidence router whether the System-1 distribution
         is confident enough. If it escalates, return that route
         immediately (cheap, no OOD work needed).
      2. Otherwise, if a state has been registered via `update()`
         and the OOD detector flags it, escalate with an `OOD: ...`
         reason.
      3. Otherwise, trust System-1.

    The OOD escalation cutoff is `ood_detector.cutoff` (single source
    of truth — there's no separate router-level threshold).
    """

    inner: Router
    ood_detector: StateOODDetector
    last_state: object = None  # OptimizationState or None
    n_escalations_by_confidence: int = 0
    n_escalations_by_ood: int = 0

    def update(self, state: object) -> None:
        """Register the current BO state for the next route() call.

        HybridPolicy should call this before invoking route().
        """
        self.last_state = state

    def route(self, probabilities: dict[str, float]) -> Route:
        # Step 1 — confidence gate.
        conf_route = self.inner.route(probabilities)
        if conf_route.name == "system2":
            self.n_escalations_by_confidence += 1
            return conf_route
        # Step 2 — OOD gate (uses the detector's own cutoff).
        if self.last_state is not None:
            ood = self.ood_detector.score(self.last_state)
            if ood.is_ood:
                self.n_escalations_by_ood += 1
                return Route(
                    name="system2",
                    confidence=conf_route.confidence,
                    reason=f"OOD: {ood.reason}",
                )
        return conf_route

    def stats(self) -> dict[str, int]:
        return {
            "system1": 0,  # symmetric with ConfidenceRouter; callers
                             # usually consult last_step.route_name
            "system2_by_confidence": self.n_escalations_by_confidence,
            "system2_by_ood": self.n_escalations_by_ood,
        }
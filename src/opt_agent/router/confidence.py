"""ConfidenceRouter — first-version router using only max(probabilities).

Per codex gpt-6-sol review 2026-09-23:
  - First-version router must use only `max(choice_probs)` against a
    single threshold `tau_conf`. Other signals (`predict_score`,
    `predict_noul`, OOD detection) are deferred until those heads have
    calibration targets.
  - `tau_conf` is a flow parameter — it should be tuned on the
    calibration split and reported on an independent test split, not
    hard-coded by call sites. Initial default is 0.5; pass an explicit
    value for any real run.
  - `update()` is exposed for symmetry with the Router protocol but is
    a no-op (no online learning in Phase 9).
"""

from __future__ import annotations

from opt_agent.router.base import Route


class ConfidenceRouter:
    """Single-threshold confidence router.

    `route(probabilities)`:
      - confidence = max(probabilities.values())
      - if confidence < tau_conf → escalate to system2
      - else → trust system1
    """

    def __init__(self, *, tau_conf: float = 0.5) -> None:
        if not (0.0 <= tau_conf <= 1.0):
            raise ValueError(f"tau_conf must lie in [0, 1]; got {tau_conf}")
        self.tau_conf = tau_conf
        self.n_system1 = 0
        self.n_system2 = 0

    def route(self, probabilities: dict[str, float]) -> Route:
        if not probabilities:
            raise ValueError("probabilities must be non-empty")
        confidence = max(probabilities.values())
        if confidence < self.tau_conf:
            self.n_system2 += 1
            return Route(
                name="system2",
                confidence=confidence,
                reason=f"below tau_conf={self.tau_conf:.2f}",
            )
        self.n_system1 += 1
        return Route(
            name="system1",
            confidence=confidence,
            reason=f"at or above tau_conf={self.tau_conf:.2f}",
        )

    def update(
        self,
        probabilities: dict[str, float],
        chosen: str,
        reward: float,
    ) -> None:
        """No-op in Phase 9. Provided for the Router protocol."""
        return None

    def stats(self) -> dict[str, int]:
        """Per-process routing counts. Useful for ablation summaries."""
        return {"system1": self.n_system1, "system2": self.n_system2}

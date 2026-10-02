"""Router — decides whether to use System-1 (fast) or escalate to System-2.

Per codex gpt-6-sol review 2026-09-23 (major M3):
  - First-version router uses only `max(choice_probs)` against a single
    threshold. `predict_score` and `predict_noul` are explicitly
    deferred — those heads have no calibration targets in the current
    models and must NOT be trusted for routing decisions yet.
  - Threshold is a flow parameter (`tau_conf`); it must be tuned on the
    calibration split and reported on an independent test split, not
    hard-coded by the call site.

(The Phase-12 OODAwareRouter escalation signal was removed in the
2026-10-02 cleanup: no experiment script ever instantiated it.)
"""

from opt_agent.router.base import Route, Router
from opt_agent.router.confidence import ConfidenceRouter

__all__ = ["Route", "Router", "ConfidenceRouter"]

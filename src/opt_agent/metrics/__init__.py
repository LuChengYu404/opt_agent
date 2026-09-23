"""Calibration and decision-quality metrics for System-1 probability outputs.

Per MASTER_PLAN §13 + codex gpt-6-sol review 2026-09-23:
- ECE / Brier / NLL / reliability curves
- Calibration here measures INTERNAL consistency between a model's
  predicted distribution and the model's own argmax outcome — NOT real-
  world accuracy (which requires a separate held-out label source).
- 15-bin default with optional override; per-bin counts reported so small
  samples don't masquerade as calibration evidence.

Decision-quality metrics (top-1, top-k, KL vs gold_probs) live in
`decision.py` to keep the calibration API focused.
"""

from opt_agent.metrics.calibration import (
    CalibrationResult,
    ReliabilityBin,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_diagram_data,
)
from opt_agent.metrics.decision import (
    DecisionResult,
    aggregate_decision_metrics,
    kl_divergence,
    negative_log_likelihood_vs_gold,
    top_k_agreement,
    top_one_agreement,
)

__all__ = [
    # calibration
    "CalibrationResult",
    "ReliabilityBin",
    "expected_calibration_error",
    "brier_score",
    "negative_log_likelihood",
    "reliability_diagram_data",
    # decision
    "DecisionResult",
    "top_one_agreement",
    "top_k_agreement",
    "kl_divergence",
    "negative_log_likelihood_vs_gold",
    "aggregate_decision_metrics",
]

"""Shared RF probe feature/target extraction for RQ1.3 / RQ2.1.

Both the original RQ1.3 RF probe (removed 2026-10-02; recoverable from git
history) and `exp_rq2_1_cross_family_transfer.py` (RQ2.1, leave-one-family-out) needed
the same feature extractor and utility recomputer under family_median
normalization. Factoring them here prevents silent divergence if the
schema changes.

Used by:
- scripts/exp_rq2_1_cross_family_transfer.py (RQ2 cross-family transfer)
- scripts/exp_rlcd_loo_transfer.py / exp_rlcd_loo_confirmatory.py (RF
  baseline arms, via local imports)

Note: `exp_rq1_3_rf_probe.py` was removed in the 2026-10-02 cleanup; its
frozen local copy of this logic is recoverable from git history.

Public API:
    ACTIONS          : 6-tuple of action IDs (canonical order, h3 menu)
    FAMILIES         : 5-tuple of family names (canonical order)
    PHASE_VALUES     : 3-tuple of phase values (early, mid, late)
    extract_features(state, phase) -> list[float]
    extract_utilities(record_dict, family_scales) -> dict[str, float] | None
    compute_family_scales(records) -> dict[str, dict[str, float]]
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Iterable

# h3 (2026-10-01): 6-action menu — error_guided/boundary removed.
# Archived h1 attribution artifacts (8-action) are no longer indexable by
# the current analysis scripts; check out the h1 commit to analyze them.
ACTIONS = (
    "global_exploration", "local_exploitation", "uncertainty_sampling",
    "surrogate_acquisition", "retrain_surrogate", "stop",
)

FAMILIES = ("branin", "hartmann", "ackley", "sixhumpcamel", "rastrigin")

PHASE_VALUES = ("early", "mid", "late")


def extract_features(state: dict, phase: str) -> list[float]:
    """15-dim state vector: 12 numeric + 3 phase one-hot.

    Mirrors the original `exp_rq1_3_rf_probe.extract_features` so the two
    scripts produce identical feature matrices for the same record.
    """
    sampling = state.get("sampling", {})
    surrogate = state.get("surrogate", {})
    optimization = state.get("optimization", {})
    resource = state.get("resource", {})
    feat = [
        float(sampling.get("n_samples", 0) or 0),
        float(sampling.get("dimension", 0) or 0),
        float(sampling.get("min_pairwise_distance", 0.0) or 0.0),
        float(sampling.get("boundary_coverage", 0.0) or 0.0),
        float(surrogate.get("validation_rmse", 0.0) or 0.0),
        float(surrogate.get("mean_uncertainty", 0.0) or 0.0),
        float(surrogate.get("max_uncertainty", 0.0) or 0.0),
        float(optimization.get("best_value", 0.0) or 0.0),
        float(optimization.get("recent_improvement", 0.0) or 0.0),
        float(optimization.get("stagnation_steps", 0) or 0),
        float(resource.get("remaining_budget_ratio", 0.0) or 0.0),
        1.0 if surrogate.get("is_current", False) else 0.0,
        1.0 if phase == "early" else 0.0,
        1.0 if phase == "mid" else 0.0,
        1.0 if phase == "late" else 0.0,
    ]
    return [0.0 if not math.isfinite(v) else v for v in feat]


def extract_utilities(
    record_dict: dict, family_scales: dict
) -> dict[str, float] | None:
    """Recompute U(s, a) under family_median scale normalization.

    Same math as the original `exp_rq1_3_rf_probe.extract_utilities`: rescale
    from the record's `optimization_scale` to the family median so utilities
    are comparable across families.

    Returns None if any scale is non-positive or fewer than 2 actions have
    finite utilities (record is unusable for training).
    """
    fam = record_dict["problem"]["family"]
    new_scale = family_scales[fam]["median"]
    md = record_dict.get("metadata", {}) or {}
    old_scale = float(md.get("optimization_scale", 1.0))
    w_opt = float(md.get("utility_weights", {}).get("w_optimization", 1.0))
    if new_scale <= 0 or old_scale <= 0:
        return None
    rr = record_dict.get("rollout_results") or {}
    out: dict[str, float] = {}
    for action in ACTIONS:
        cell = rr.get(action, {})
        if not isinstance(cell, dict):
            continue
        u_orig = cell.get("utility")
        d_opt = cell.get("delta_optimization")
        if u_orig is None or d_opt is None:
            continue
        try:
            u_orig = float(u_orig)
            d_opt = float(d_opt)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(u_orig) and math.isfinite(d_opt)):
            continue
        u_new = u_orig + w_opt * d_opt * (1.0 / new_scale - 1.0 / old_scale)
        out[action] = u_new
    return out if len(out) >= 2 else None


def compute_family_scales(records: Iterable[dict]) -> dict[str, dict[str, float]]:
    """Compute per-family median `optimization_scale` across records."""
    by_fam: dict[str, list[float]] = defaultdict(list)
    for r in records:
        s = float((r.get("metadata") or {}).get("optimization_scale", 0.0))
        if s > 0:
            by_fam[r["problem"]["family"]].append(s)
    return {fam: {"median": statistics.median(vals)} for fam, vals in by_fam.items()}
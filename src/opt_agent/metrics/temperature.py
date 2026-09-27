"""Temperature scaling for System-1 probability outputs.

Per MASTER_PLAN §13 calibration and codex gpt-6-sol review 2026-09-23
("predict_score / predict_noul deliberately NOT consulted; temperature
scaling is the proper scoring-rule way to align confidence"):

Given a calibration corpus of (probs, outcome) pairs, find the single
scalar temperature T that minimizes negative log-likelihood of the
outcomes under the rescaled distribution softmax(log(probs) / T).

  T -> 1.0   sharpens (overconfident)
  T -> inf  uniform (under-confident)
  T = 1.0   no rescaling

We optimize T in log-space via bounded search on [log(0.1), log(10.0)]
and report the optimal T plus before / after NLL and ECE.

This module is intentionally a small pure-function API: callers
construct `TemperatureCalibrator`, call `.fit(probs, outcomes)` on a
calibration split, then call `.transform(probs)` on held-out data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    """Result of `fit_temperature`."""

    temperature: float
    nll_before: float
    nll_after: float
    n_records: int


def _log(p: float, eps: float = 1e-12) -> float:
    p = max(p, eps)
    return math.log(p)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _nll(probs: np.ndarray, outcomes_idx: np.ndarray,
        candidate_keys: Sequence[str], outcomes: Sequence[str]) -> float:
    """Mean -log P(outcome_i) under probs; skipped records → finite values only."""
    n_total = 0
    s = 0.0
    for i, outcome in enumerate(outcomes):
        if outcome not in candidate_keys:
            continue
        idx = candidate_keys.index(outcome)
        p = float(probs[i, idx])
        if p <= 0.0:
            return math.inf
        s += -_log(p)
        n_total += 1
    return s / n_total if n_total else math.inf


def fit_temperature(
    probs: Sequence[Mapping[str, float]],
    outcomes: Sequence[str],
    *,
    candidates: Sequence[str] | None = None,
    n_bins: int = 15,
) -> CalibrationResult:
    """Find the optimal temperature that minimizes NLL.

    Parameters
    ----------
    probs : sequence of {candidate: prob} dicts
    outcomes : sequence of observed outcome strings (must align 1:1 with probs)
    candidates : optional fixed ordering of candidate names. If `None`,
        the union of keys across probs is used (sorted) to define a
        consistent alignment.
    """
    if len(probs) != len(outcomes):
        raise ValueError(f"length mismatch: {len(probs)} vs {len(outcomes)}")
    if not probs:
        raise ValueError("empty calibration corpus")

    # Build a stable candidate ordering.
    if candidates is None:
        keys: set[str] = set()
        for p in probs:
            keys.update(p.keys())
        candidate_list = sorted(keys)
    else:
        candidate_list = list(candidates)

    # Convert probs to numpy log-probabilities for vectorized search.
    P = np.zeros((len(probs), len(candidate_list)), dtype=np.float64)
    for i, p in enumerate(probs):
        for j, c in enumerate(candidate_list):
            P[i, j] = max(float(p.get(c, 0.0)), 1e-12)
    # log-probs; renormalize to a proper distribution per row (defensive).
    P = P / P.sum(axis=-1, keepdims=True)
    L = np.log(P)

    # NLL before rescaling.
    nll_before = _nll(P, None, candidate_list, outcomes)

    # 1-D search over log T.
    best_T = 1.0
    best_nll = nll_before
    for log_T in np.linspace(math.log(0.1), math.log(10.0), 81):
        T = float(math.exp(log_T))
        scaled = _softmax(L / T)
        cand = _nll(scaled, None, candidate_list, outcomes)
        if math.isfinite(cand) and cand < best_nll:
            best_nll = cand
            best_T = T

    return CalibrationResult(
        temperature=best_T,
        nll_before=nll_before,
        nll_after=best_nll,
        n_records=len(probs),
    )


class TemperatureCalibrator:
    """Wraps a fitted temperature and applies it on demand."""

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {temperature}")
        self.temperature = float(temperature)

    @classmethod
    def fit(
        cls,
        probs: Sequence[Mapping[str, float]],
        outcomes: Sequence[str],
        *,
        candidates: Sequence[str] | None = None,
    ) -> "TemperatureCalibrator":
        result = fit_temperature(probs, outcomes, candidates=candidates)
        return cls(temperature=result.temperature)

    def transform(self, probs: Mapping[str, float]) -> dict[str, float]:
        """Apply the calibrated temperature to a single distribution."""
        if not probs:
            raise ValueError("probs must be non-empty")
        candidates = sorted(probs.keys())
        L = np.array(
            [_log(float(probs[c])) for c in candidates],
            dtype=np.float64,
        ).reshape(1, -1)
        scaled = _softmax(L / self.temperature)[0]
        return {c: float(v) for c, v in zip(candidates, scaled)}
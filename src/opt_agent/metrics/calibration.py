"""Calibration metrics for System-1 probability outputs.

Per MASTER_PLAN §13 + codex gpt-6-sol review 2026-09-23:
- Calibration here measures INTERNAL consistency between a model's
  predicted distribution and the model's own argmax outcome. This is NOT
  real-world accuracy (which requires a separate held-out label source).
- 15-bin default with optional override; per-bin counts reported so small
  samples don't masquerade as calibration evidence.

The unit of analysis is (probs, outcome) where `outcome` is typically
`argmax(probs)` for self-consistency or a separately observed label for
agreement metrics. When you need the latter, use `decision.py` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ReliabilityBin:
    """One bin of a reliability diagram.

    Attributes
    ----------
    confidence : float
        Mean predicted confidence in this bin.
    accuracy : float
        Fraction of records in this bin whose argmax matches `outcome`.
    count : int
        Number of records in this bin (always shown to flag noisy bins).
    bin_lo, bin_hi : float
        Lower and upper edges of this bin's confidence range. The top bin
        is closed on the upper edge so confidence == 1.0 lands there.
    """

    confidence: float
    accuracy: float
    count: int
    bin_lo: float
    bin_hi: float


@dataclass(frozen=True)
class CalibrationResult:
    """Result of `expected_calibration_error`.

    Attributes
    ----------
    ece : float
        Weighted average of |confidence - accuracy| across non-empty bins.
    bins : tuple[ReliabilityBin, ...]
        Per-bin detail; empty bins are skipped. Always inspect `count`
        before treating a single bin as evidence.
    n_records : int
        Number of (probs, outcome) pairs the ECE was computed over.
    n_bins : int
        Number of bins requested (regardless of how many were non-empty).
    notes : str
        Free-form warning, e.g. "small-sample warning: n=12, n_bins=15".
    """

    ece: float
    bins: tuple[ReliabilityBin, ...]
    n_records: int
    n_bins: int
    notes: str = ""


def _validate_inputs(
    probs: Sequence[Mapping[str, float]],
    outcomes: Sequence[str],
    n_bins: int,
) -> None:
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if len(probs) != len(outcomes):
        raise ValueError(
            f"probs/outcomes length mismatch: {len(probs)} vs {len(outcomes)}"
        )
    for i, p in enumerate(probs):
        if not p:
            raise ValueError(f"probs[{i}] is empty")
        for k, v in p.items():
            if not isinstance(v, (int, float)):
                raise TypeError(
                    f"probs[{i}][{k!r}] must be numeric; got {type(v).__name__}"
                )


def expected_calibration_error(
    probs: Sequence[Mapping[str, float]],
    outcomes: Sequence[str],
    n_bins: int = 15,
) -> CalibrationResult:
    """Compute Expected Calibration Error of (probs, outcome) pairs.

    Per-record confidence = max_a probs[a].
    Per-record accuracy = 1.0 if argmax(probs) == outcome else 0.0.

    ECE = sum_bins (count_bin / N) * |conf_bin - acc_bin|.
    """
    _validate_inputs(probs, outcomes, n_bins)
    if not probs:
        return CalibrationResult(ece=0.0, bins=(), n_records=0, n_bins=n_bins)

    confidences: list[float] = []
    accuracies: list[float] = []
    for p, outcome in zip(probs, outcomes):
        chosen = max(p, key=p.get)
        conf = float(p[chosen])
        conf = max(0.0, min(1.0, conf))  # numeric safety
        confidences.append(conf)
        accuracies.append(1.0 if chosen == outcome else 0.0)

    n = len(probs)
    bins: list[ReliabilityBin] = []
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        # Top bin closed on the upper edge so confidence == 1.0 lands there.
        if b == n_bins - 1:
            in_bin = [
                (c, a) for c, a in zip(confidences, accuracies)
                if lo <= c <= hi
            ]
        else:
            in_bin = [
                (c, a) for c, a in zip(confidences, accuracies)
                if lo <= c < hi
            ]
        if not in_bin:
            continue
        bin_conf = sum(c for c, _ in in_bin) / len(in_bin)
        bin_acc = sum(a for _, a in in_bin) / len(in_bin)
        bins.append(
            ReliabilityBin(
                confidence=bin_conf, accuracy=bin_acc,
                count=len(in_bin), bin_lo=lo, bin_hi=hi,
            )
        )
        ece += (len(in_bin) / n) * abs(bin_conf - bin_acc)

    notes = ""
    if n < n_bins * 5:
        notes = f"small-sample warning: n={n}, n_bins={n_bins}"
    return CalibrationResult(
        ece=ece, bins=tuple(bins), n_records=n, n_bins=n_bins, notes=notes,
    )


def reliability_diagram_data(
    probs: Sequence[Mapping[str, float]],
    outcomes: Sequence[str],
    n_bins: int = 15,
) -> list[tuple[float, float, int]]:
    """Convenience wrapper returning flat tuples for plotting.

    Returns [(bin_confidence, bin_accuracy, bin_count), ...].
    """
    result = expected_calibration_error(probs, outcomes, n_bins=n_bins)
    return [(b.confidence, b.accuracy, b.count) for b in result.bins]


def brier_score(
    probs: Mapping[str, float],
    outcome: str,
    candidates: Sequence[str],
) -> float:
    """Multi-class Brier score: sum_i (p_i - 1[outcome == i])^2 over candidates.

    Lower is better; for a one-hot distribution matching the outcome, Brier = 0.
    Range: [0.0, 2.0] when `candidates` covers two classes with probability 1
    on the wrong one.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")
    total = 0.0
    for c in candidates:
        p = float(probs.get(c, 0.0))
        target = 1.0 if c == outcome else 0.0
        total += (p - target) ** 2
    return total


def negative_log_likelihood(
    probs: Mapping[str, float], outcome: str
) -> float:
    """NLL = -log P(outcome). Returns +inf if P(outcome) <= 0."""
    p = float(probs.get(outcome, 0.0))
    if p <= 0.0:
        return math.inf
    return -math.log(p)

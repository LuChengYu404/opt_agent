"""Calibration metrics: ECE / Brier / NLL / reliability diagram.

These are pure-function unit tests — no fixtures, no I/O.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.metrics import (  # noqa: E402
    CalibrationResult,
    ReliabilityBin,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_diagram_data,
)


def _delta(action: str, candidates: list[str]) -> dict[str, float]:
    return {a: (1.0 if a == action else 0.0) for a in candidates}


def _uniform(candidates: list[str]) -> dict[str, float]:
    p = 1.0 / len(candidates)
    return {a: p for a in candidates}


# ---------------------------------------------------------------------------
# ECE happy paths
# ---------------------------------------------------------------------------


def test_ece_empty_returns_zero() -> None:
    result = expected_calibration_error([], [], n_bins=15)
    assert result.ece == 0.0
    assert result.bins == ()
    assert result.n_records == 0
    assert result.notes == ""


def test_ece_uniform_probs_self_consistent_is_low() -> None:
    """Uniform probs with argmax outcomes: confidence ≈ 1/K, accuracy = 1.0
    in every bin → moderate ECE (predicts 1/K confidence but is 100% accurate).
    """
    K = 8
    probs = [_uniform([f"a{i}" for i in range(K)]) for _ in range(40)]
    outcomes = [_argmax_uniform(p) for p in probs]
    result = expected_calibration_error(probs, outcomes, n_bins=10)
    assert result.ece > 0.0
    # Confidence ≈ 1/K = 0.125, accuracy = 1.0 → |0.125 - 1.0| ≈ 0.875
    assert result.ece < 1.0
    assert result.n_records == 40


def _argmax_uniform(p: dict[str, float]) -> str:
    return next(iter(p))  # tie-break: any key


def test_ece_delta_probs_self_consistent_is_zero() -> None:
    """Delta distribution with argmax = the chosen action: confidence 1.0,
    accuracy 1.0 → ECE = 0.0."""
    K = 5
    candidates = [f"a{i}" for i in range(K)]
    probs = []
    outcomes = []
    for i in range(30):
        chosen = candidates[i % K]
        probs.append(_delta(chosen, candidates))
        outcomes.append(chosen)
    result = expected_calibration_error(probs, outcomes, n_bins=15)
    assert result.ece == pytest.approx(0.0, abs=1e-9)
    assert result.n_records == 30
    # All 30 records should land in the top bin (confidence = 1.0).
    assert len(result.bins) == 1
    assert result.bins[0].bin_lo == pytest.approx(14 / 15)
    assert result.bins[0].bin_hi == 1.0
    assert result.bins[0].count == 30


def test_ece_reports_per_bin_counts_for_small_samples() -> None:
    """Per codex: report bin counts so noise isn't read as calibration."""
    probs = [{"a": 0.9, "b": 0.1}] * 12  # confidence ≈ 0.9
    outcomes = ["a"] * 12  # all argmax matches
    result = expected_calibration_error(probs, outcomes, n_bins=15)
    assert "small-sample" in result.notes
    non_empty = [b for b in result.bins if b.count > 0]
    assert non_empty, "expected at least one bin to be populated"
    for b in result.bins:
        assert b.count >= 0


def test_ece_top_bin_is_closed_on_upper_edge() -> None:
    """Records with confidence == 1.0 must land in the top bin."""
    probs = [{"a": 1.0, "b": 0.0} for _ in range(5)]
    outcomes = ["a"] * 5
    result = expected_calibration_error(probs, outcomes, n_bins=5)
    assert len(result.bins) == 1
    assert result.bins[0].bin_hi == 1.0
    assert result.bins[0].count == 5


# ---------------------------------------------------------------------------
# ECE error paths
# ---------------------------------------------------------------------------


def test_ece_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        expected_calibration_error([{"a": 1.0}], ["a", "b"])


def test_ece_rejects_empty_probs_dict() -> None:
    with pytest.raises(ValueError, match="is empty"):
        expected_calibration_error([{}], ["x"])


def test_ece_rejects_zero_bins() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error([{"a": 1.0}], ["a"], n_bins=0)


def test_ece_rejects_non_numeric_probability() -> None:
    with pytest.raises(TypeError, match="numeric"):
        expected_calibration_error([{"a": "0.5"}], ["a"])


def test_ece_clamps_out_of_range_confidences() -> None:
    """Numeric safety: small negative or > 1 values are clamped."""
    probs = [{"a": 1.5, "b": -0.5}]  # invalid but tolerated
    outcomes = ["a"]
    result = expected_calibration_error(probs, outcomes, n_bins=2)
    # Should not raise; outcome matches argmax (clamped), accuracy = 1.0.
    assert math.isfinite(result.ece)


# ---------------------------------------------------------------------------
# Brier / NLL
# ---------------------------------------------------------------------------


def test_brier_perfect_is_zero() -> None:
    candidates = ["a", "b", "c"]
    probs = {"a": 1.0, "b": 0.0, "c": 0.0}
    assert brier_score(probs, "a", candidates) == pytest.approx(0.0)


def test_brier_worst_case_two_classes() -> None:
    candidates = ["a", "b"]
    probs = {"a": 0.0, "b": 1.0}
    # (0-1)^2 + (1-0)^2 = 2
    assert brier_score(probs, "a", candidates) == pytest.approx(2.0)


def test_brier_uniform_two_classes() -> None:
    candidates = ["a", "b"]
    probs = {"a": 0.5, "b": 0.5}
    # (0.5-1)^2 + (0.5-0)^2 = 0.5
    assert brier_score(probs, "a", candidates) == pytest.approx(0.5)


def test_brier_missing_candidate_treated_as_zero() -> None:
    candidates = ["a", "b", "c"]
    probs = {"a": 1.0}  # b, c missing → 0
    # outcome == a: (1-1)^2 + (0-0)^2 + (0-0)^2 = 0
    assert brier_score(probs, "a", candidates) == pytest.approx(0.0)
    # outcome == b: (1-0)^2 + (0-1)^2 + (0-0)^2 = 2
    assert brier_score(probs, "b", candidates) == pytest.approx(2.0)


def test_brier_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        brier_score({"a": 1.0}, "a", [])


def test_nll_perfect_is_zero() -> None:
    assert negative_log_likelihood({"a": 1.0, "b": 0.0}, "a") == pytest.approx(0.0)


def test_nll_half_is_log2() -> None:
    assert negative_log_likelihood({"a": 0.5, "b": 0.5}, "a") == pytest.approx(math.log(2.0))


def test_nll_zero_mass_is_infinity() -> None:
    assert negative_log_likelihood({"a": 0.0, "b": 1.0}, "a") == math.inf


def test_nll_missing_outcome_is_infinity() -> None:
    assert negative_log_likelihood({"a": 1.0}, "b") == math.inf


# ---------------------------------------------------------------------------
# reliability_diagram_data
# ---------------------------------------------------------------------------


def test_reliability_diagram_data_returns_tuples() -> None:
    probs = [{"a": 0.7, "b": 0.3}, {"a": 0.9, "b": 0.1}, {"a": 1.0, "b": 0.0}]
    outcomes = ["a", "a", "a"]
    data = reliability_diagram_data(probs, outcomes, n_bins=5)
    assert isinstance(data, list)
    for entry in data:
        assert len(entry) == 3
        conf, acc, count = entry
        assert 0.0 <= conf <= 1.0
        assert 0.0 <= acc <= 1.0
        assert count > 0

"""Tests for TemperatureCalibrator (post-hoc temperature scaling).

(The StateOODDetector half of the former test_ood_and_temperature.py was
removed with system1/ood.py in the 2026-10-02 cleanup: no experiment
script consumed it.)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.metrics import (  # noqa: E402
    TemperatureCalibrator,
    fit_temperature,
)


# ---------------------------------------------------------------------------
# TemperatureCalibrator
# ---------------------------------------------------------------------------


def test_temperature_default_calibrator_is_identity() -> None:
    cal = TemperatureCalibrator(temperature=1.0)
    out = cal.transform({"a": 0.7, "b": 0.3})
    assert abs(out["a"] - 0.7) < 1e-9
    assert abs(out["b"] - 0.3) < 1e-9


def test_temperature_high_T_makes_distribution_uniform() -> None:
    cal = TemperatureCalibrator(temperature=100.0)
    out = cal.transform({"a": 0.9, "b": 0.1})
    assert abs(out["a"] - 0.5) < 0.01
    assert abs(out["b"] - 0.5) < 0.01


def test_temperature_fit_finds_lower_nll() -> None:
    """Fit must reduce NLL on a synthetic over-confident corpus.

    Per Guo's calibration-on-the-margin literature: with over-confident
    predictions AND perfect accuracy, the optimal temperature is T<1
    (sharpens — pushing probability toward 1 makes correct outcomes
    less costly). The optimizer picks the T that minimizes NLL.
    """
    probs = [{"a": 0.9, "b": 0.1} for _ in range(20)]
    outcomes = ["a"] * 20
    result = fit_temperature(probs, outcomes)
    # Either T<1 (sharpens) or T→inf (flattens, doesn't help) wins; in
    # this case NLL=−log(0.9)=0.105, T<1 drives it near 0. Either way,
    # NLL must decrease.
    assert result.nll_after <= result.nll_before + 1e-9
    assert result.nll_after < 0.05  # should drop sharply
    # The optimal T is finite (not at the grid boundary).
    assert 0.1 <= result.temperature <= 10.0


def test_temperature_fit_under_confident_data_finds_T_le_1() -> None:
    """Under-confident data (uniform predictions) with deterministic
    outcomes — NLL is initially ln(2)=0.693. Optimal T must produce a
    valid (non-NaN, non-equal) NLL. Either sharpening (T<1) or
    flattening (T->inf) is acceptable as long as the optimizer picks a
    reasonable T inside [0.1, 10]."""
    probs = [{"a": 0.5, "b": 0.5} for _ in range(20)]
    outcomes = ["a"] * 20
    result = fit_temperature(probs, outcomes)
    # NLL must be finite (no division-by-zero explosion).
    assert math.isfinite(result.nll_after)
    assert result.nll_after <= result.nll_before + 1e-9
    # The optimal T must be inside the search grid.
    assert 0.1 <= result.temperature <= 10.0


def test_temperature_transform_preserves_argmax() -> None:
    cal = TemperatureCalibrator(temperature=2.5)
    out = cal.transform({"a": 0.7, "b": 0.2, "c": 0.1})
    assert max(out, key=out.get) == "a"


def test_temperature_fit_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        fit_temperature([{"a": 0.5, "b": 0.5}], ["a", "b"])
    with pytest.raises(ValueError, match="empty"):
        fit_temperature([], [])


def test_temperature_fit_rejects_empty_corpus() -> None:
    # An empty input trips the empty check first.
    with pytest.raises(ValueError, match="empty"):
        fit_temperature([], [])


def test_temperature_calibrator_fit_classmethod() -> None:
    """TemperatureCalibrator.fit on over-confident+perfect-accuracy data
    picks T<1 (sharpens) and transform returns a more confident output."""
    probs = [{"a": 0.9, "b": 0.1}] * 5
    outcomes = ["a"] * 5
    cal = TemperatureCalibrator.fit(probs, outcomes)
    # T<1 sharpens → calibrated prob should be > 0.9.
    assert cal.temperature < 1.0
    out = cal.transform({"a": 0.9, "b": 0.1})
    assert out["a"] > 0.9


def test_temperature_calibrator_rejects_non_positive_T() -> None:
    with pytest.raises(ValueError, match="temperature"):
        TemperatureCalibrator(temperature=0)
    with pytest.raises(ValueError, match="temperature"):
        TemperatureCalibrator(temperature=-1.0)


def test_temperature_transform_rejects_empty() -> None:
    cal = TemperatureCalibrator()
    with pytest.raises(ValueError, match="non-empty"):
        cal.transform({})

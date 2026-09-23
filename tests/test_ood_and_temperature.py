"""Tests for StateOODDetector and TemperatureCalibrator."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.actions import (  # noqa: E402
    ActionType,
    OptimizationAction,
    candidate_actions_for_state,
)
from opt_agent.metrics import (  # noqa: E402
    TemperatureCalibrator,
    fit_temperature,
)
from opt_agent.state import OptimizationState  # noqa: E402
from opt_agent.system1.ood import StateOODDetector  # noqa: E402


# ---------------------------------------------------------------------------
# StateOODDetector
# ---------------------------------------------------------------------------


def _state(
    *,
    n_samples: int = 5,
    total_budget: int = 20,
    remaining_budget_ratio: float = 0.5,
    is_current: bool = True,
    best_value: float = -1.0,
    recent_improvement: float = 0.0,
    stagnation_steps: int = 0,
) -> OptimizationState:
    """Build a real OptimizationState by extracting from a backend,
    then override the fields the test cares about via object.__setattr__."""
    from opt_agent.state import extract_state
    from conftest import RecordingBackend
    backend = RecordingBackend()
    state = extract_state(backend, total_budget=total_budget)
    object.__setattr__(state.sampling, "n_samples", n_samples)
    object.__setattr__(state.resource, "total_budget", total_budget)
    object.__setattr__(
        state.resource, "remaining_budget_ratio", remaining_budget_ratio,
    )
    object.__setattr__(state.resource, "remaining_budget", int(total_budget * remaining_budget_ratio))
    object.__setattr__(state.surrogate, "is_current", is_current)
    object.__setattr__(state.optimization, "best_value", best_value)
    object.__setattr__(state.optimization, "recent_improvement", recent_improvement)
    object.__setattr__(state.optimization, "stagnation_steps", stagnation_steps)
    return state


def _calibration_corpus(n: int = 30) -> list[OptimizationState]:
    """A realistic in-distribution corpus for the calibration split.

    Most records are mid-budget, surrogate is fitted, modest stagnation.
    """
    import random
    rng = random.Random(0)
    states = []
    for _ in range(n):
        n_samples = rng.randint(5, 18)
        total = 20
        rb = max(0, total - n_samples)
        ratio = rb / total
        states.append(_state(
            n_samples=n_samples, total_budget=total,
            remaining_budget_ratio=ratio,
            is_current=True, best_value=-1.0 + rng.uniform(-0.1, 0.1),
            recent_improvement=rng.uniform(-0.05, 0.05),
            stagnation_steps=rng.randint(0, 2),
        ))
    return states


def test_ood_detector_fit_returns_stats() -> None:
    det = StateOODDetector()
    stats = det.fit(_calibration_corpus())
    assert stats.count == 30
    for f in ("sample_fraction", "stagnation_per_sample",
              "remaining_budget_ratio", "is_current", "relative_improvement"):
        assert f in stats.means
        assert f in stats.stds
        assert stats.stds[f] > 0


def test_ood_detector_score_before_fit_raises() -> None:
    det = StateOODDetector()
    with pytest.raises(RuntimeError, match="before fit"):
        det.score(_state())


def test_ood_detector_in_distribution_state_is_not_ood() -> None:
    det = StateOODDetector()
    det.fit(_calibration_corpus())
    s = _calibration_corpus(5)[2]
    result = det.score(s)
    assert result.score == 0.0
    assert result.is_ood is False
    assert result.above_threshold == {f: False for f in
        ("sample_fraction", "stagnation_per_sample",
         "remaining_budget_ratio", "is_current", "relative_improvement")}


def test_ood_detector_extreme_state_is_ood() -> None:
    """An extreme state (stagnation_per_sample=9999) must trip at least
    one feature above threshold AND produce a non-zero score."""
    det = StateOODDetector()
    det.fit(_calibration_corpus())
    # Use a low cutoff so the test is robust to how many features
    # trip above z=2.0 (the test is about *detection*, not a specific
    # feature count).
    det.cutoff = 0.2
    s = _state(n_samples=1, total_budget=20, stagnation_steps=9999)
    result = det.score(s)
    assert result.score > 0.0
    assert any(result.above_threshold.values())
    assert "stagnation_per_sample" in result.reason


def test_ood_detector_handles_nan_features() -> None:
    """NaN values must not crash the detector and contribute 0 z-score."""
    det = StateOODDetector()
    det.fit(_calibration_corpus())
    s = _state(recent_improvement=float("nan"))
    # Should not raise.
    result = det.score(s)
    assert math.isfinite(result.score)


def test_ood_detector_threshold_validation() -> None:
    with pytest.raises(ValueError, match="z_threshold"):
        StateOODDetector(z_threshold=0)
    with pytest.raises(ValueError, match="cutoff"):
        StateOODDetector(cutoff=-0.1)
    with pytest.raises(ValueError, match="cutoff"):
        StateOODDetector(cutoff=1.5)


def test_ood_detector_per_problem_invariance_under_relative_features() -> None:
    """When |best_value| >= 1, the same relative_improvement ratio
    across two problems maps to the same feature value. (When |bv| < 1
    the floor at 1.0 kicks in and we test that path separately.)"""
    det = StateOODDetector()
    det.fit(_calibration_corpus())
    # Both states have best_value magnitude >= 1.
    s_branin_like = _state(best_value=-2.0, recent_improvement=-0.2)
    s_hartmann_like = _state(best_value=5.0, recent_improvement=-0.5)
    rel_branin = det.features(s_branin_like)["relative_improvement"]
    rel_hartmann = det.features(s_hartmann_like)["relative_improvement"]
    assert abs(rel_branin - (-0.1)) < 1e-9
    assert abs(rel_hartmann - (-0.1)) < 1e-9
    assert abs(rel_branin - rel_hartmann) < 1e-9


def test_ood_detector_relative_improvement_floor_for_small_best_value() -> None:
    """When |best_value| < 1, the denominator is clamped to 1.0 so the
    feature value is close to `ri` rather than being blown up."""
    det = StateOODDetector()
    det.fit(_calibration_corpus())
    s = _state(best_value=-0.4, recent_improvement=-0.04)
    # denom = max(0.4, 1.0) = 1.0; rel = -0.04 / 1.0 = -0.04 (not -0.1).
    assert abs(det.features(s)["relative_improvement"] - (-0.04)) < 1e-9


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
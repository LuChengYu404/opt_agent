"""Tests for OODAwareRouter — combines confidence gate and OOD gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.actions import ActionType, OptimizationAction  # noqa: E402
from opt_agent.router import (  # noqa: E402
    ConfidenceRouter,
    OODAwareRouter,
)
from opt_agent.system1.ood import StateOODDetector  # noqa: E402

# Reuse the state builder from the OOD tests.
from test_ood_and_temperature import _state, _calibration_corpus  # noqa: E402


@pytest.fixture
def permissive_detector() -> StateOODDetector:
    """Detector tuned so any non-zero OOD score escalates.

    The router reads `ood_detector.cutoff` (single source of truth).
    Setting it to 0.0 makes the test robust to how many features trip
    above the z-threshold.
    """
    det = StateOODDetector()
    det.fit(_calibration_corpus())
    det.cutoff = 0.0
    return det


@pytest.fixture
def strict_detector() -> StateOODDetector:
    """Detector with default cutoff (0.5)."""
    det = StateOODDetector()
    det.fit(_calibration_corpus())
    return det


# ---------------------------------------------------------------------------
# Confidence gate still wins
# ---------------------------------------------------------------------------


def test_low_confidence_escalates_without_ood_check(strict_detector) -> None:
    """When confidence < tau_conf, OOD is not consulted."""
    inner = ConfidenceRouter(tau_conf=0.5)
    router = OODAwareRouter(inner=inner, ood_detector=strict_detector)
    route = router.route({"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2, "e": 0.2})
    assert route.name == "system2"
    assert router.n_escalations_by_confidence == 1
    assert router.n_escalations_by_ood == 0


def test_high_confidence_no_state_no_escalation(strict_detector) -> None:
    """Without an update() call, OOD is not consulted — no escalation."""
    inner = ConfidenceRouter(tau_conf=0.5)
    router = OODAwareRouter(inner=inner, ood_detector=strict_detector)
    route = router.route({"a": 0.9, "b": 0.1})
    assert route.name == "system1"


# ---------------------------------------------------------------------------
# OOD gate fires when confidence is high but state is OOD
# ---------------------------------------------------------------------------


def test_high_confidence_but_ood_state_escalates(permissive_detector) -> None:
    """Confident delta distribution + extreme state → escalate via OOD."""
    inner = ConfidenceRouter(tau_conf=0.5)
    router = OODAwareRouter(
        inner=inner, ood_detector=permissive_detector,
    )
    # Extreme state: many samples + huge stagnation. remaining_budget
    # must be > 0 so the candidate factory emits more than `stop`.
    state = _state(
        n_samples=15, total_budget=20, remaining_budget_ratio=0.25,
        stagnation_steps=9999, recent_improvement=0.0,
    )
    router.update(state)
    route = router.route({"a": 1.0})
    assert route.name == "system2"
    assert route.reason.startswith("OOD:")
    assert router.n_escalations_by_confidence == 0
    assert router.n_escalations_by_ood == 1


def test_in_distribution_state_trusts_system1(permissive_detector) -> None:
    """Confident delta + in-distribution state → trust System-1."""
    inner = ConfidenceRouter(tau_conf=0.5)
    router = OODAwareRouter(
        inner=inner, ood_detector=permissive_detector,
    )
    state = _state(
        n_samples=8, total_budget=20, remaining_budget_ratio=0.6,
        stagnation_steps=0, recent_improvement=0.0,
    )
    router.update(state)
    route = router.route({"a": 1.0})
    assert route.name == "system1"


# ---------------------------------------------------------------------------
# update() replaces the last_state between calls
# ---------------------------------------------------------------------------


def test_update_replaces_last_state(permissive_detector) -> None:
    inner = ConfidenceRouter(tau_conf=0.5)
    router = OODAwareRouter(
        inner=inner, ood_detector=permissive_detector,
    )
    # First step: in-distribution state → trust.
    in_state = _state(
        n_samples=8, total_budget=20, remaining_budget_ratio=0.6,
        stagnation_steps=0,
    )
    router.update(in_state)
    assert router.route({"a": 1.0}).name == "system1"

    # Second step: extreme state registered → escalate via OOD.
    out_state = _state(
        n_samples=15, total_budget=20, remaining_budget_ratio=0.25,
        stagnation_steps=9999,
    )
    router.update(out_state)
    assert router.route({"a": 1.0}).name == "system2"


def test_stats_track_both_escalation_sources(permissive_detector) -> None:
    inner = ConfidenceRouter(tau_conf=0.5)
    router = OODAwareRouter(
        inner=inner, ood_detector=permissive_detector,
    )
    # First: low confidence
    router.route({"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2, "e": 0.2})
    # Second: high confidence + OOD state
    router.update(_state(
        n_samples=15, total_budget=20, remaining_budget_ratio=0.25,
        stagnation_steps=9999,
    ))
    router.route({"a": 1.0})
    # Third: high confidence + in-distribution → no escalation
    router.update(_state(
        n_samples=8, total_budget=20, remaining_budget_ratio=0.6,
        stagnation_steps=0,
    ))
    router.route({"a": 1.0})
    s = router.stats()
    assert s["system2_by_confidence"] == 1
    assert s["system2_by_ood"] == 1


# ---------------------------------------------------------------------------
# HybridPolicy integration
# ---------------------------------------------------------------------------


def test_hybrid_policy_registers_state_with_ood_router(permissive_detector) -> None:
    """HybridPolicy.select_action must call router.update(state) before
    consulting the OOD detector, otherwise the OOD gate can never fire."""
    from dataclasses import dataclass
    from opt_agent.system1 import HybridPolicy
    from opt_agent.system2 import FallbackSystem2

    @dataclass
    class _StubDecision:
        action: object
        probabilities: dict

    class _StubSystem1:
        def decide(self, state):
            return _StubDecision(
                action=OptimizationAction(ActionType.SURROGATE_ACQUISITION),
                probabilities={"surrogate_acquisition": 1.0},
            )

    inner = ConfidenceRouter(tau_conf=0.5)
    router = OODAwareRouter(
        inner=inner, ood_detector=permissive_detector,
    )
    s1 = _StubSystem1()
    policy = HybridPolicy(system1=s1, router=router, system2=FallbackSystem2())
    state = _state(
        n_samples=15, total_budget=20, remaining_budget_ratio=0.25,
        stagnation_steps=9999,
    )
    policy.select_action(state)
    assert router.n_escalations_by_ood == 1
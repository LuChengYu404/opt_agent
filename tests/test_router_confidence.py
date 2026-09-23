"""Tests for ConfidenceRouter — single-threshold max(probabilities) routing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.router import ConfidenceRouter, Route  # noqa: E402


def test_route_high_confidence_uses_system1() -> None:
    r = ConfidenceRouter(tau_conf=0.5)
    route = r.route({"a": 0.7, "b": 0.3})
    assert route.name == "system1"
    assert route.confidence == pytest.approx(0.7)


def test_route_low_confidence_escalates() -> None:
    r = ConfidenceRouter(tau_conf=0.5)
    route = r.route({"a": 0.4, "b": 0.3, "c": 0.3})
    assert route.name == "system2"
    assert route.confidence == pytest.approx(0.4)


def test_route_at_threshold_trusts_system1() -> None:
    """Boundary: confidence == tau_conf should not escalate (>=)."""
    r = ConfidenceRouter(tau_conf=0.5)
    route = r.route({"a": 0.5, "b": 0.5})
    assert route.name == "system1"


def test_route_delta_distribution_always_system1() -> None:
    """A delta distribution (one action with prob 1.0) is fully confident."""
    r = ConfidenceRouter(tau_conf=0.5)
    route = r.route({"a": 1.0, "b": 0.0, "c": 0.0})
    assert route.name == "system1"
    assert route.confidence == 1.0


def test_route_uniform_below_default_threshold() -> None:
    """Uniform K=5 → max = 0.2 < default 0.5 → escalate."""
    r = ConfidenceRouter()
    route = r.route({"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2, "e": 0.2})
    assert route.name == "system2"


def test_route_rejects_empty_probabilities() -> None:
    r = ConfidenceRouter()
    with pytest.raises(ValueError, match="non-empty"):
        r.route({})


def test_router_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        ConfidenceRouter(tau_conf=-0.1)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        ConfidenceRouter(tau_conf=1.5)


def test_router_accepts_endpoint_thresholds() -> None:
    ConfidenceRouter(tau_conf=0.0)
    ConfidenceRouter(tau_conf=1.0)


def test_router_stats_track_counts() -> None:
    r = ConfidenceRouter(tau_conf=0.5)
    r.route({"a": 0.9})  # system1
    r.route({"a": 0.6})  # system1
    r.route({"a": 0.3, "b": 0.3, "c": 0.4})  # system2 (max = 0.4)
    s = r.stats()
    assert s == {"system1": 2, "system2": 1}


def test_router_update_is_noop() -> None:
    r = ConfidenceRouter()
    r.update({"a": 0.7}, "a", 1.0)  # must not raise
    assert r.stats() == {"system1": 0, "system2": 0}


def test_router_different_thresholds_give_different_routes() -> None:
    """At confidence = 0.4, tau=0.3 routes system1, tau=0.5 routes system2."""
    probs = {"a": 0.4, "b": 0.4, "c": 0.2}  # max = 0.4
    r1 = ConfidenceRouter(tau_conf=0.3)
    r2 = ConfidenceRouter(tau_conf=0.5)
    assert r1.route(probs).name == "system1"
    assert r2.route(probs).name == "system2"

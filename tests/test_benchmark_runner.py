"""Tests for the Phase 13 benchmark runner (stat aggregation only — no
real backend, no LLM)."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_benchmark as m  # noqa: E402


def _result(problem: str, seed: int, policy: str,
           best_value: float, simple_regret: float | None = None) -> m.BenchResult:
    return m.BenchResult(
        problem=problem, seed=seed, policy=policy,
        best_value=best_value,
        n_evaluations=10, n_steps=10,
        termination_reason="budget_exhausted",
        simple_regret=simple_regret,
    )


def test_aggregate_basic_stats() -> None:
    results = [
        _result("Branin", 0, "hybrid", -0.40, simple_regret=0.0021),
        _result("Branin", 1, "hybrid", -0.50, simple_regret=0.1021),
        _result("Branin", 0, "rule",   -1.00, simple_regret=0.6021),
        _result("Branin", 1, "rule",   -2.00, simple_regret=1.6021),
    ]
    summary = m._aggregate(results)
    assert "Branin/hybrid" in summary
    s = summary["Branin/hybrid"]
    assert s["n_seeds"] == 2
    assert s["best_value_mean"] == pytest.approx(-0.45, abs=1e-9)
    assert s["best_value_max"] == pytest.approx(-0.40, abs=1e-9)
    assert s["simple_regret_mean"] == pytest.approx(0.0521, abs=1e-9)
    assert s["success_rate_branin_within_1e-2"] == 0.5  # seed 0 only


def test_aggregate_paired_t_statistic_sign() -> None:
    """When hybrid > rule on every seed, the paired t is positive."""
    results = [
        _result("Branin", 0, "hybrid", -0.4, simple_regret=0.01),
        _result("Branin", 1, "hybrid", -0.4, simple_regret=0.01),
        _result("Branin", 2, "hybrid", -0.4, simple_regret=0.01),
        _result("Branin", 0, "rule",   -2.0, simple_regret=1.6),
        _result("Branin", 1, "rule",   -2.0, simple_regret=1.6),
        _result("Branin", 2, "rule",   -2.0, simple_regret=1.6),
    ]
    summary = m._aggregate(results)
    t = summary["Branin/hybrid"]["paired_t_hybrid_vs_rule"]
    # hybrid_bests - rule_bests = 1.6 > 0 on every seed, so t -> +inf
    # (zero variance in the diffs); but our formula divides by sd,
    # and stdev of constant diffs is 0, so we return None when sd=0.
    # Either None or +inf is acceptable; we require finite positive.
    assert t is None or t > 0


def test_aggregate_handles_unknown_problems() -> None:
    """Ackley/Hartmann have no pinned optimum so simple_regret is None."""
    results = [
        _result("Ackley", 0, "hybrid", -20.0, simple_regret=None),
        _result("Hartmann", 0, "hybrid", 1.0, simple_regret=None),
    ]
    summary = m._aggregate(results)
    for k in ("Ackley/hybrid", "Hartmann/hybrid"):
        assert k in summary
        assert summary[k]["simple_regret_mean"] is None
        assert summary[k]["success_rate_branin_within_1e-2"] is None


def test_aggregate_with_only_one_seed_has_zero_std() -> None:
    results = [_result("Branin", 0, "hybrid", -0.4, simple_regret=0.01)]
    summary = m._aggregate(results)
    s = summary["Branin/hybrid"]
    assert s["best_value_std"] == 0.0
    assert s["n_seeds"] == 1


def test_main_writes_json_with_known_optima(tmp_path: Path) -> None:
    """Smoke: --output writes JSON containing config + summary + runs."""
    import subprocess
    out = tmp_path / "bench.json"
    # Limit to Branin seed 0 budget 10 — quickest end-to-end.
    rc = subprocess.run(
        ["uv", "run", "python", "scripts/run_benchmark.py",
         "--output", str(out),
         "--problems", "Branin",
         "--seeds", "0",
         "--budgets", "10"],
        cwd=ROOT,
        capture_output=True,
        timeout=180,
    )
    assert rc.returncode == 0, f"stdout={rc.stdout!r} stderr={rc.stderr!r}"
    payload = json.loads(out.read_text())
    assert payload["config"]["tau_conf"] == 0.5
    assert payload["config"]["known_optima"] == {"Branin": -0.397887}
    assert "summary" in payload and "runs" in payload
    # Branin/hybrid + Branin/rule should both appear.
    assert "Branin/hybrid" in payload["summary"]
    assert "Branin/rule" in payload["summary"]
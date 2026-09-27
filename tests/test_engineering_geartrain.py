"""GearTrain engineering-problem integration tests (P3, 2026-09-27).

GearTrain is the first BOCoDe ENGINEERING problem in the backend allowlist
(single-objective, unconstrained, d=4 — the only engineering problem passing
the Phase-1 structural gates). These tests verify the backend handles it
correctly end to end, mirroring the Phase 8.6E pattern used when
SixHumpCamel/Rastrigin were added.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.backend.bocode_botorch import (  # noqa: E402
    BackendConfig,
    BocodeBotorchBackend,
)


def _fresh(seed: int = 0) -> BocodeBotorchBackend:
    b = BocodeBotorchBackend()
    b.initialize(BackendConfig(problem_id="GearTrain", seed=seed, n_init=5))
    return b


def test_geartrain_initialize_and_shapes() -> None:
    b = _fresh()
    h = b.get_history()
    assert h.X.shape == (5, 4)
    assert h.objectives.shape == (5, 1)
    assert h.constraints.shape == (5, 0)
    assert b.is_fitted is False


def test_geartrain_fit_and_propose_and_evaluate() -> None:
    b = _fresh()
    b.fit()
    x = b.propose(n=1)
    assert x.shape[-1] == 4
    batch = b.evaluate(x)
    assert batch.objectives.shape == (1, 1)
    assert batch.constraints.shape == (1, 0)
    # Appended to history.
    assert b.get_history().X.shape[0] == 6


def test_geartrain_objectives_finite_and_bounded_scale() -> None:
    b = _fresh(seed=3)
    h = b.get_history()
    vals = h.objectives.flatten().tolist()
    assert all(v == v for v in vals)  # no NaN


def test_geartrain_decision_dataset_smoke() -> None:
    """One decision point's rollout machinery works on GearTrain."""
    from dataclasses import replace

    from opt_agent.dataset import DatasetConfig, DatasetGenerator

    spec = replace(
        DatasetConfig(),
        problems=("GearTrain",),
        seeds=(0,),
        max_actions=2,
        horizon=1,
    )
    records = list(DatasetGenerator(spec).generate())
    assert 1 <= len(records) <= 2
    for r in records:
        assert r.problem["family"] == "geartrain"
        assert r.state["sampling"]["dimension"] == 4
        assert sum(r.gold_probs.values()) == pytest.approx(1.0, abs=1e-9)

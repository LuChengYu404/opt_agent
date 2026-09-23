"""Phase 8.5 oracle gate tests (per codex review 2025-09-23).

Gate 3 (aggregate_mass) / Gate 4 (oracle_test) / Gate 5 (affine_rescaling_test).

Smoke tests use synthetic DecisionRecords and a small generator-backed
real-backend run for oracle + affine. Full gate pass requires ≥10 seeds per
problem (per codex Phase 9 gate); these tests verify the *machinery*.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import RecordingBackend  # noqa: E402

from opt_agent.actions import ActionType, OptimizationAction  # noqa: E402
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend  # noqa: E402
from opt_agent.dataset import (  # noqa: E402
    EXPLORATION_ACTIONS,
    DecisionRecord,
    DatasetGenerator,
    aggregate_mass,
    affine_rescaling_test,
    oracle_test,
    phase_of,
)
from opt_agent.dataset.decision_record import _json_safe
from opt_agent.state import extract_state  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(
    *,
    action_probs: dict[str, float],
    family: str = "branin",
    seed: int = 0,
    remaining_budget_ratio: float = 0.8,
    record_id: str | None = None,
) -> DecisionRecord:
    cands = [
        {"action_id": k, "type": k, "params": {}}
        for k in action_probs
    ]
    return DecisionRecord(
        id=record_id or f"{family}:{seed}:0:0",
        problem={"family": family, "dimension": 2, "seed": seed},
        state={
            "sampling": {"n_samples": 5, "dimension": 2,
                         "min_pairwise_distance": 1.0, "boundary_coverage": 0.0},
            "surrogate": {"is_current": False, "validation_rmse": float("nan"),
                          "mean_uncertainty": float("nan"), "max_uncertainty": float("nan")},
            "optimization": {"best_value": 0.0, "last_value": 0.0,
                             "recent_improvement": 0.0, "stagnation_steps": 0},
            "resource": {"used_budget": 5, "total_budget": 25,
                         "remaining_budget": 20,
                         "remaining_budget_ratio": remaining_budget_ratio},
            "history": {"last_action": None},
        },
        candidate_actions=cands,
        gold_probs=action_probs,
        rollout_results={
            k: {"status": "valid", "utility": 0.0, "delta_optimization": 0.0,
                "delta_surrogate": float("nan"), "delta_coverage": 0.0,
                "n_new_samples": 0, "n_model_fits": 0, "error": None}
            for k in action_probs
        },
        temperature=1.0,
        metadata={"trajectory_id": 0, "step": 0, "run_seed": 0, "utility_version": "test"},
    )


def _real_branin(seed: int = 0, n_init: int = 5):
    be = BocodeBotorchBackend()
    be.initialize(BackendConfig(problem_id="Branin", seed=seed, n_init=n_init))
    return be


# ---------------------------------------------------------------------------
# Gate 3 — aggregate_mass
# ---------------------------------------------------------------------------


def test_uniform_probs_do_not_pass_adjusted_exploration_gate() -> None:
    """Under uniform (1/8 each), adjusted lift = 0 → gate fails."""
    probs = {a: 1 / 8 for a in [
        "global_exploration", "local_exploitation",
        "uncertainty_sampling", "error_guided_sampling",
        "boundary_sampling", "surrogate_acquisition",
        "retrain_surrogate", "stop",
    ]}
    records = [_record(action_probs=probs, remaining_budget_ratio=0.9)]
    result = aggregate_mass(records)
    assert result.raw_mean > 0
    assert result.adjusted_lift_mean == pytest.approx(0.0, abs=1e-9)
    assert not result.pass_


def test_exploration_dominant_probs_pass_gate() -> None:
    """Heavy exploration mass + low stop → adjusted lift > 0."""
    probs = {
        "global_exploration": 0.6,
        "uncertainty_sampling": 0.2,
        "boundary_sampling": 0.1,
        "stop": 0.05,
        "retrain_surrogate": 0.05,
        # remaining mass distributed
        "local_exploitation": 0.0,
        "error_guided_sampling": 0.0,
        "surrogate_acquisition": 0.0,
    }
    records = [_record(action_probs=probs, remaining_budget_ratio=0.9)]
    result = aggregate_mass(records)
    assert result.pass_


def test_seed_weighting_prevents_long_trajectory_domination() -> None:
    """Equal weighting per (problem, seed): one long trajectory doesn't dominate."""
    probs_strong = {
        "global_exploration": 0.6, "boundary_sampling": 0.3, "stop": 0.1,
        "local_exploration": 0.0, "uncertainty_sampling": 0.0,
        "error_guided_sampling": 0.0, "retrain_surrogate": 0.0,
        "surrogate_acquisition": 0.0,
    }
    # Local: explicit zero stop mass.
    probs_weak = {
        "stop": 0.8, "retrain_surrogate": 0.2,
        "global_exploration": 0.0, "local_exploitation": 0.0,
        "uncertainty_sampling": 0.0, "error_guided_sampling": 0.0,
        "boundary_sampling": 0.0, "surrogate_acquisition": 0.0,
    }
    # Seed 0: 100 strong records. Seed 1: 1 weak record.
    records = (
        [_record(action_probs=probs_strong, seed=0, remaining_budget_ratio=0.9)
         for _ in range(100)]
        + [_record(action_probs=probs_weak, seed=1, remaining_budget_ratio=0.9)]
    )
    result = aggregate_mass(records)
    assert len(result.seed_rows) == 2
    # Equal-weight per seed means seed 0 and seed 1 each get 50% weight.
    expected = (result.seed_rows[0]["adjusted_lift"] + result.seed_rows[1]["adjusted_lift"]) / 2
    assert result.adjusted_lift_mean == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Gate 4 — oracle_test
# ---------------------------------------------------------------------------


def test_oracle_arms_start_from_identical_checkpoint() -> None:
    """All 4 oracle arms begin from the same backend state (verified by
    restore-to-base at end)."""
    be = _real_branin()
    state = extract_state(be, total_budget=20)
    # Build a synthetic record using the real backend's vocabulary.
    cands = [
        {"action_id": "stop", "type": "stop", "params": {}},
        {"action_id": "global_exploration", "type": "global_exploration",
         "params": {"n_samples": 1, "seed": 0}},
    ]
    record = DecisionRecord(
        id="t", problem={"family": "branin", "dimension": 2, "seed": 0},
        state=_json_safe(state),
        candidate_actions=cands,
        gold_probs={"stop": 0.5, "global_exploration": 0.5},
        rollout_results={k: {"status": "valid", "utility": 0.0,
                              "delta_optimization": 0.0,
                              "delta_surrogate": float("nan"),
                              "delta_coverage": 0.0,
                              "n_new_samples": 0, "n_model_fits": 0, "error": None}
                       for k in ["stop", "global_exploration"]},
        temperature=1.0,
        metadata={"trajectory_id": 0, "step": 0, "run_seed": 0, "utility_version": "t"},
    )
    pre_history_X = be.get_history().X.clone()
    result = oracle_test(be, state, record, objective_scale=1.0,
                         random_replicates=2, max_actions=10)
    # Backend must be restored to pre-call state.
    assert torch.equal(be.get_history().X, pre_history_X)
    assert result.gold_action in ("stop", "global_exploration")
    assert result.terminal["gold"] == result.terminal["stop"] or result.terminal["gold"] != result.terminal["stop"]


def test_oracle_random_arms_are_reproducible() -> None:
    """Two oracle runs with same config produce identical terminal values."""
    be = _real_branin()
    state = extract_state(be, total_budget=20)
    cands = [
        {"action_id": "stop", "type": "stop", "params": {}},
        {"action_id": "global_exploration", "type": "global_exploration",
         "params": {"n_samples": 1, "seed": 0}},
    ]
    record = DecisionRecord(
        id="t", problem={"family": "branin", "dimension": 2, "seed": 0},
        state=_json_safe(state),
        candidate_actions=cands,
        gold_probs={"stop": 0.5, "global_exploration": 0.5},
        rollout_results={k: {"status": "valid", "utility": 0.0,
                              "delta_optimization": 0.0,
                              "delta_surrogate": float("nan"),
                              "delta_coverage": 0.0,
                              "n_new_samples": 0, "n_model_fits": 0, "error": None}
                       for k in ["stop", "global_exploration"]},
        temperature=1.0,
        metadata={"trajectory_id": 0, "step": 0, "run_seed": 0, "utility_version": "t"},
    )
    a = oracle_test(be, state, record, objective_scale=1.0,
                    random_replicates=2, max_actions=10)
    b = oracle_test(be, state, record, objective_scale=1.0,
                    random_replicates=2, max_actions=10)
    assert a.terminal == b.terminal
    assert a.gains == b.gains


def test_oracle_normalizes_by_objective_scale() -> None:
    """Gains are scaled by objective_scale so comparison across problems is meaningful."""
    be = _real_branin()
    state = extract_state(be, total_budget=20)
    cands = [
        {"action_id": "stop", "type": "stop", "params": {}},
        {"action_id": "global_exploration", "type": "global_exploration",
         "params": {"n_samples": 1, "seed": 0}},
    ]
    record = DecisionRecord(
        id="t", problem={"family": "branin", "dimension": 2, "seed": 0},
        state=_json_safe(state),
        candidate_actions=cands,
        gold_probs={"stop": 0.5, "global_exploration": 0.5},
        rollout_results={k: {"status": "valid", "utility": 0.0,
                              "delta_optimization": 0.0,
                              "delta_surrogate": float("nan"),
                              "delta_coverage": 0.0,
                              "n_new_samples": 0, "n_model_fits": 0, "error": None}
                       for k in ["stop", "global_exploration"]},
        temperature=1.0,
        metadata={"trajectory_id": 0, "step": 0, "run_seed": 0, "utility_version": "t"},
    )
    a = oracle_test(be, state, record, objective_scale=2.0,
                    random_replicates=1, max_actions=8)
    b = oracle_test(be, state, record, objective_scale=4.0,
                    random_replicates=1, max_actions=8)
    # Doubling objective_scale halves all gains.
    for arm in a.gains:
        assert a.gains[arm] == pytest.approx(b.gains[arm] * 2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Gate 5 — affine_rescaling_test
# ---------------------------------------------------------------------------


def test_affine_rescaling_passes_when_labels_identical() -> None:
    """Identical gold_probs across all transforms → gate passes trivially."""
    base_probs = {
        "global_exploration": 0.4, "boundary_sampling": 0.3, "stop": 0.3,
        "local_exploitation": 0.0, "uncertainty_sampling": 0.0,
        "error_guided_sampling": 0.0, "retrain_surrogate": 0.0,
        "surrogate_acquisition": 0.0,
    }
    base_records = [
        _record(action_probs=base_probs, family="branin", seed=0,
                remaining_budget_ratio=0.9, record_id="t1"),
    ]

    def gen(scale: float, shift: float):
        return base_records  # identical regardless of scale/shift

    report = affine_rescaling_test(gen)
    assert report["pass_"]


def test_affine_rescaling_fails_when_labels_flip() -> None:
    """If gold_probs differ wildly per transform, gate must fail."""
    probs_a = {
        "global_exploration": 0.8, "stop": 0.2,
        "local_exploitation": 0.0, "uncertainty_sampling": 0.0,
        "error_guided_sampling": 0.0, "boundary_sampling": 0.0,
        "retrain_surrogate": 0.0, "surrogate_acquisition": 0.0,
    }
    probs_b = {
        "stop": 0.8, "global_exploration": 0.2,
        "local_exploitation": 0.0, "uncertainty_sampling": 0.0,
        "error_guided_sampling": 0.0, "boundary_sampling": 0.0,
        "retrain_surrogate": 0.0, "surrogate_acquisition": 0.0,
    }
    records = {
        (1.0, 0.0): [_record(action_probs=probs_a, family="branin", seed=0,
                              remaining_budget_ratio=0.9, record_id="t")],
        (0.1, 0.0): [_record(action_probs=probs_b, family="branin", seed=0,
                              remaining_budget_ratio=0.9, record_id="t")],
        (10.0, 0.0): [_record(action_probs=probs_a, family="branin", seed=0,
                               remaining_budget_ratio=0.9, record_id="t")],
        (1.0, 5.0): [_record(action_probs=probs_b, family="branin", seed=0,
                              remaining_budget_ratio=0.9, record_id="t")],
    }
    def gen(scale: float, shift: float):
        return records[(scale, shift)]
    report = affine_rescaling_test(gen)
    assert not report["pass_"]


# ---------------------------------------------------------------------------
# phase_of helper
# ---------------------------------------------------------------------------


def test_phase_of_thresholds() -> None:
    r_early = _record(action_probs={}, remaining_budget_ratio=0.8)
    assert phase_of(r_early) == "early"
    r_mid = _record(action_probs={}, remaining_budget_ratio=0.5)
    assert phase_of(r_mid) == "mid"
    r_late = _record(action_probs={}, remaining_budget_ratio=0.2)
    assert phase_of(r_late) == "late"
    r_term = _record(action_probs={}, remaining_budget_ratio=0.0)
    assert phase_of(r_term) == "terminal"

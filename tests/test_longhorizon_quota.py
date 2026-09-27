"""Long-horizon rollout-quota tests (2026-09-27).

Covers the two quota axes added to the dataset generator:

  1. `rollout_action_subsample=k` — deterministic k-of-8 candidate subsets
     per decision point, partial-rollout metadata, subset gold_probs.
  2. `horizon_mode="remaining"` — per-decision-point horizon equals the
     remaining budget (true long-horizon labels).

Regression guards: k=None and k=8 must reproduce the historical full-vocab
behaviour exactly (same gold_probs keys and metadata without partial_rollout).

Tests use Branin with tiny budgets (cf. test_objective_transform.py) to keep
wall-clock low.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.dataset import DatasetConfig, DatasetGenerator, subsample_candidates
from opt_agent.dataset.generator import (
    _SUBSAMPLE_SALT,
    derive_seed,
    full_candidates,
)


def _quick_spec(**overrides) -> DatasetConfig:
    base = dict(
        problems=("Branin",),
        seeds=(0,),
        max_actions=4,
        horizon=1,
    )
    base.update(overrides)
    return replace(DatasetConfig(), **base)


def _records(spec: DatasetConfig) -> list:
    return list(DatasetGenerator(spec).generate())


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_subsample_k_validation() -> None:
    with pytest.raises(ValueError, match="rollout_action_subsample"):
        _quick_spec(rollout_action_subsample=0)
    with pytest.raises(ValueError, match="rollout_action_subsample"):
        _quick_spec(rollout_action_subsample=9)


def test_horizon_mode_validation() -> None:
    with pytest.raises(ValueError, match="horizon_mode"):
        _quick_spec(horizon_mode="weird")
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        _quick_spec(horizon_mode="fixed", horizon=0)


# ---------------------------------------------------------------------------
# subsample_candidates unit behaviour
# ---------------------------------------------------------------------------


def test_subsample_candidates_deterministic_and_order_preserving() -> None:
    cands = full_candidates(seed=0)
    ids = [c.action_id for c in cands]
    a = subsample_candidates(cands, 3, seed=123)
    b = subsample_candidates(cands, 3, seed=123)
    assert [c.action_id for c in a] == [c.action_id for c in b]
    # Order preserved relative to the full vocabulary.
    assert [c.action_id for c in a] == [i for i in ids if i in {c.action_id for c in a}]


def test_subsample_candidates_full_k_is_identity() -> None:
    cands = full_candidates(seed=0)
    assert subsample_candidates(cands, len(cands), seed=5) == cands


def test_subsample_candidates_rejects_bad_k() -> None:
    cands = full_candidates(seed=0)
    with pytest.raises(ValueError):
        subsample_candidates(cands, 0, seed=1)
    with pytest.raises(ValueError):
        subsample_candidates(cands, 9, seed=1)


# ---------------------------------------------------------------------------
# Generator integration — partial records
# ---------------------------------------------------------------------------


def test_partial_record_metadata_and_subset_gold_probs() -> None:
    full = _records(_quick_spec(horizon=2))
    part = _records(_quick_spec(horizon=2, rollout_action_subsample=3))

    assert full and part
    full_by_id = {r.id: r for r in full}
    for rec in part:
        meta = rec.metadata["partial_rollout"]
        assert meta["k"] == 3
        rolled_ids = set(meta["action_ids"])
        assert len(rolled_ids) == 3
        # gold_probs / rollout_results cover exactly the rolled subset.
        assert set(rec.gold_probs) == rolled_ids
        assert set(rec.rollout_results) == rolled_ids
        # Subset distribution still normalizes.
        assert sum(rec.gold_probs.values()) == pytest.approx(1.0, abs=1e-9)
        # candidate_actions still describe the FULL vocabulary.
        assert len(rec.candidate_actions) == 8
        # Same record id produced without subsampling (trajectory untouched).
        twin = full_by_id[rec.id]
        assert twin.metadata.get("partial_rollout") is None
        assert set(twin.gold_probs) == set(twin.candidate_actions[i]["action_id"] for i in range(8))


def test_subsample_does_not_change_the_walked_trajectory() -> None:
    """Subsampling only affects rollouts, never the behaviour policy's walk."""
    a = _records(_quick_spec(horizon=1))
    b = _records(_quick_spec(horizon=1, rollout_action_subsample=2))
    states_a = [r.state for r in a]
    states_b = [r.state for r in b]
    assert len(states_a) == len(states_b)
    for sa, sb in zip(states_a, states_b):
        assert sa["sampling"]["n_samples"] == sb["sampling"]["n_samples"]
        assert sa["optimization"]["best_value"] == pytest.approx(
            sb["optimization"]["best_value"], rel=1e-12, abs=1e-12
        )


def test_k8_matches_no_subsample_records() -> None:
    none_recs = _records(_quick_spec(horizon=2))
    k8_recs = _records(_quick_spec(horizon=2, rollout_action_subsample=8))
    assert len(none_recs) == len(k8_recs)
    for x, y in zip(none_recs, k8_recs):
        assert x.id == y.id
        assert x.gold_probs == y.gold_probs
        assert y.metadata.get("partial_rollout") is None


def test_subset_choice_is_stable_across_records_and_transforms() -> None:
    """The subset depends only on (run_seed, step, salt) — the same record id
    produced under different objective transforms rolls out the same subset,
    keeping Gate-5-style paired comparisons valid."""
    cands = full_candidates(seed=0)
    s1 = subsample_candidates(cands, 3, seed=derive_seed(77, 0, _SUBSAMPLE_SALT))
    s2 = subsample_candidates(cands, 3, seed=derive_seed(77, 0, _SUBSAMPLE_SALT))
    assert [c.action_id for c in s1] == [c.action_id for c in s2]
    # Determinism per (run_seed, step) is the guard; different steps are free
    # to collide or differ.


# ---------------------------------------------------------------------------
# horizon_mode="remaining"
# ---------------------------------------------------------------------------


def test_remaining_horizon_mode_records_actual_horizon() -> None:
    recs = _records(_quick_spec(horizon_mode="remaining"))
    assert recs
    for rec in recs:
        remaining = int(rec.state["resource"]["remaining_budget"])
        assert rec.metadata["horizon"] == max(1, remaining)
        assert rec.metadata["horizon_mode"] == "remaining"


def test_remaining_mode_with_subsample_combines_quotas() -> None:
    recs = _records(_quick_spec(
        horizon_mode="remaining", rollout_action_subsample=2, max_actions=4,
    ))
    assert recs
    for rec in recs:
        meta = rec.metadata["partial_rollout"]
        assert meta["k"] == 2
        assert len(meta["action_ids"]) == 2
        assert set(rec.rollout_results) == set(meta["action_ids"])


# ---------------------------------------------------------------------------
# rollout_replicates (P1b denoising)
# ---------------------------------------------------------------------------


def test_replicate_averaging_and_metadata() -> None:
    one = _records(_quick_spec(horizon=1))
    rep = _records(_quick_spec(horizon=1, rollout_replicates=2))
    assert len(rep) == len(one)
    for r in rep:
        for cell in r.rollout_results.values():
            replicas = cell.get("utility_replicas")
            assert replicas is not None and len(replicas) == 2
            assert cell["utility"] == pytest.approx(
                sum(replicas) / len(replicas), abs=1e-9
            )
            assert cell["utility_std"] >= 0.0


def test_replicate_single_is_identity() -> None:
    a = _records(_quick_spec(horizon=1))
    b = _records(_quick_spec(horizon=1, rollout_replicates=1))
    assert [r.id for r in a] == [r.id for r in b]
    for x, y in zip(a, b):
        for k in x.rollout_results:
            assert "utility_replicas" not in y.rollout_results[k]
            assert x.rollout_results[k]["utility"] == y.rollout_results[k]["utility"]


def test_replicate_validation() -> None:
    with pytest.raises(ValueError, match="rollout_replicates"):
        _quick_spec(rollout_replicates=0)
    with pytest.raises(ValueError, match="rollout_replicates"):
        _quick_spec(rollout_replicates=9)

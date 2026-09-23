"""Tests for opt_agent.dataset.splits: 5-way split assignment + helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.dataset import (  # noqa: E402
    DecisionRecord,
    Split,
    assign_splits,
    family_seed_of,
    gold_probs_kind,
    physical_state_hash,
    split_records,
)


def _record(family: str, seed: int, step: int = 0, *,
             metadata: dict | None = None) -> DecisionRecord:
    return DecisionRecord(
        id=f"{family}:{seed}:0:{step}",
        problem={"family": family, "dimension": 2, "seed": seed},
        state={"sampling": {"n_samples": step + 1}},
        candidate_actions=[{"action_id": "global_exploration"}],
        gold_probs={"global_exploration": 1.0},
        rollout_results={"global_exploration": {"status": "valid"}},
        temperature=1.0,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# family_seed_of
# ---------------------------------------------------------------------------


def test_family_seed_extracts_from_id() -> None:
    assert family_seed_of(_record("branin", 3, step=5)) == ("branin", 3)


def test_family_seed_handles_short_id() -> None:
    rec = DecisionRecord(
        id="malformed",
        problem={"family": "x", "seed": 0},
        state={},
        candidate_actions=[],
        gold_probs={},
        rollout_results={},
        temperature=1.0,
    )
    assert family_seed_of(rec) == ("unknown", -1)


# ---------------------------------------------------------------------------
# assign_splits
# ---------------------------------------------------------------------------


def _ten_seed_corpus() -> list[DecisionRecord]:
    return [_record(fam, seed, step=s)
            for fam in ("branin", "ackley", "hartmann")
            for seed in range(10)
            for s in range(3)]


def test_assign_splits_default_60_10_10_20() -> None:
    """With 3 in-distribution families × 10 seeds = 30 (family, seed) pairs,
    expect ~18 train / 3 dev / 3 calibration / 6 test (±1 due to rounding).
    """
    corpus = _ten_seed_corpus()
    assignments = assign_splits(corpus)
    counts: dict[Split, int] = {s: 0 for s in Split}
    for fam, sd in assignments:
        counts[assignments[(fam, sd)]] += 1
    assert counts[Split.OOD] == 0
    assert counts[Split.TRAIN] >= 16 and counts[Split.TRAIN] <= 20
    assert counts[Split.DEV] >= 2 and counts[Split.DEV] <= 4
    assert counts[Split.CALIBRATION] >= 2 and counts[Split.CALIBRATION] <= 4
    assert counts[Split.TEST] >= 4 and counts[Split.TEST] <= 8
    # Total must equal all (family, seed) pairs.
    assert sum(counts.values()) == 30


def test_assign_splits_ood_families() -> None:
    corpus = _ten_seed_corpus()
    assignments = assign_splits(corpus, ood_families=("hartmann",))
    for fam, sd in assignments:
        if fam == "hartmann":
            assert assignments[(fam, sd)] == Split.OOD
        else:
            assert assignments[(fam, sd)] != Split.OOD


def test_assign_splits_deterministic_for_same_seed() -> None:
    corpus = _ten_seed_corpus()
    a = assign_splits(corpus, seed=42)
    b = assign_splits(corpus, seed=42)
    assert a == b


def test_assign_splits_different_seeds_can_differ() -> None:
    corpus = _ten_seed_corpus()
    a = assign_splits(corpus, seed=1)
    b = assign_splits(corpus, seed=2)
    # With 30 in-dist (family, seed) pairs and a non-trivial ratio mix, the
    # shuffles almost certainly produce different assignments.
    assert a != b


def test_assign_splits_trajectory_integrity() -> None:
    """All records of one (family, seed) pair must share the same split."""
    corpus = _ten_seed_corpus()
    assignments = assign_splits(corpus)
    by_split = split_records(corpus, assignments)
    # Pick a (family, seed) and verify all its steps land in one bucket.
    for (fam, sd), split in assignments.items():
        bucket = by_split[split]
        for r in bucket:
            if family_seed_of(r) == (fam, sd):
                # All records of (fam, sd) appear together — verified by count.
                pass
        # All records with this (fam, sd) must be in the same bucket.
        own = [r for r in corpus if family_seed_of(r) == (fam, sd)]
        assert all(r in bucket for r in own)


def test_assign_splits_rejects_bad_ratios() -> None:
    corpus = _ten_seed_corpus()
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_splits(
            corpus,
            ratios={"train": 0.5, "dev": 0.1,
                    "calibration": 0.1, "test": 0.1},  # sum = 0.8
        )


def test_assign_splits_rejects_unknown_split_name() -> None:
    corpus = _ten_seed_corpus()
    with pytest.raises(ValueError, match="missing required splits"):
        assign_splits(
            corpus,
            ratios={"train": 0.5, "made_up_split": 0.5},
        )


def test_assign_splits_empty_records() -> None:
    assert assign_splits([]) == {}


# ---------------------------------------------------------------------------
# split_records
# ---------------------------------------------------------------------------


def test_split_records_partitions_correctly() -> None:
    corpus = _ten_seed_corpus()
    assignments = assign_splits(corpus, ood_families=("hartmann",))
    buckets = split_records(corpus, assignments)
    assert sum(len(v) for v in buckets.values()) == len(corpus)
    assert len(buckets[Split.OOD]) > 0
    for r in buckets[Split.OOD]:
        assert r.id.startswith("hartmann")


# ---------------------------------------------------------------------------
# physical_state_hash
# ---------------------------------------------------------------------------


def test_physical_state_hash_is_stable() -> None:
    rec = _record("branin", 0, step=2)
    h1 = physical_state_hash(rec)
    h2 = physical_state_hash(rec)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_physical_state_hash_excludes_gold_probs() -> None:
    """Changing gold_probs MUST NOT change the hash (label leakage prevention)."""
    rec1 = _record("branin", 0, step=2)
    rec1 = DecisionRecord(
        id=rec1.id, problem=rec1.problem, state=rec1.state,
        candidate_actions=rec1.candidate_actions,
        gold_probs={"a": 0.5, "b": 0.5},
        rollout_results=rec1.rollout_results,
        temperature=rec1.temperature, metadata=rec1.metadata,
    )
    rec2 = DecisionRecord(
        id=rec1.id, problem=rec1.problem, state=rec1.state,
        candidate_actions=rec1.candidate_actions,
        gold_probs={"a": 1.0, "b": 0.0},
        rollout_results=rec1.rollout_results,
        temperature=rec1.temperature, metadata=rec1.metadata,
    )
    assert physical_state_hash(rec1) == physical_state_hash(rec2)


def test_physical_state_hash_excludes_rollout_results() -> None:
    rec1 = _record("branin", 0, step=2)
    rec2 = DecisionRecord(
        id=rec1.id, problem=rec1.problem, state=rec1.state,
        candidate_actions=rec1.candidate_actions,
        gold_probs=rec1.gold_probs,
        rollout_results={"a": {"status": "valid", "utility": 99.0}},
        temperature=rec1.temperature, metadata=rec1.metadata,
    )
    assert physical_state_hash(rec1) == physical_state_hash(rec2)


def test_physical_state_hash_changes_with_state() -> None:
    rec1 = _record("branin", 0, step=2)
    rec2 = DecisionRecord(
        id=rec1.id, problem=rec1.problem,
        state={"sampling": {"n_samples": 99}},  # different state
        candidate_actions=rec1.candidate_actions,
        gold_probs=rec1.gold_probs,
        rollout_results=rec1.rollout_results,
        temperature=rec1.temperature, metadata=rec1.metadata,
    )
    assert physical_state_hash(rec1) != physical_state_hash(rec2)


def test_physical_state_hash_excludes_problem_seed() -> None:
    """Different problem.seed must NOT change the hash (problem.seed is
    not part of the physical BO state)."""
    rec1 = _record("branin", 0, step=2)
    rec2 = DecisionRecord(
        id=rec1.id,
        problem={"family": "branin", "dimension": 2, "seed": 999},  # different
        state=rec1.state,
        candidate_actions=rec1.candidate_actions,
        gold_probs=rec1.gold_probs,
        rollout_results=rec1.rollout_results,
        temperature=rec1.temperature, metadata=rec1.metadata,
    )
    assert physical_state_hash(rec1) == physical_state_hash(rec2)


def test_physical_state_hash_excludes_candidate_actions() -> None:
    """Different candidate params must NOT change the hash (candidate
    factory output is a policy artifact, not physics)."""
    rec1 = _record("branin", 0, step=2)
    rec2 = DecisionRecord(
        id=rec1.id, problem=rec1.problem, state=rec1.state,
        candidate_actions=[{"action_id": "global_exploration",
                            "params": {"n_samples": 99, "seed": 0}}],
        gold_probs=rec1.gold_probs,
        rollout_results=rec1.rollout_results,
        temperature=rec1.temperature, metadata=rec1.metadata,
    )
    assert physical_state_hash(rec1) == physical_state_hash(rec2)


def test_assign_splits_uses_sorted_family_order() -> None:
    """Per codex: family iteration order must be SORTED so assignments are
    independent of `set` iteration order (which varies across Python
    processes). Same input + same seed must always produce the same output."""
    corpus = _ten_seed_corpus()
    # Run twice with identical inputs.
    a = assign_splits(corpus, seed=0)
    b = assign_splits(corpus, seed=0)
    assert a == b
    # Running with shuffled record order should give the same result.
    import random
    shuffled = list(corpus)
    random.Random(99).shuffle(shuffled)
    c = assign_splits(shuffled, seed=0)
    assert a == c


# ---------------------------------------------------------------------------
# gold_probs_kind
# ---------------------------------------------------------------------------


def test_gold_probs_kind_default_is_rollout() -> None:
    rec = _record("branin", 0)
    assert gold_probs_kind(rec) == "rollout_utility_softmax"


def test_gold_probs_kind_reads_explicit_metadata() -> None:
    rec = _record("branin", 0, metadata={"gold_probs_kind": "human_annotated"})
    assert gold_probs_kind(rec) == "human_annotated"


def test_gold_probs_kind_handles_empty_string() -> None:
    """Empty string in metadata should fall back to default."""
    rec = _record("branin", 0, metadata={"gold_probs_kind": ""})
    assert gold_probs_kind(rec) == "rollout_utility_softmax"

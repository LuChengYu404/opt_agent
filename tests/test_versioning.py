"""Version-stamp tests — enforce RESEARCH_BOUNDARY_GUIDE.md §4.5 + §6.6.

Covers: stamp_metadata wiring, status classification (ok/legacy/mismatch),
and that the dataset generator actually stamps newly produced records.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opt_agent.dataset.versioning import (
    HARNESS_VERSION,
    UTILITY_VERSION,
    VersionStatus,
    check_record_versions,
    stamp_metadata,
)


def test_stamp_metadata_adds_and_preserves():
    out = stamp_metadata({"trajectory_id": 3})
    assert out["trajectory_id"] == 3
    assert out["harness_version"] == HARNESS_VERSION
    assert out["utility_version"] == UTILITY_VERSION


def test_stamp_metadata_overwrites_stale_stamps_and_handles_none():
    out = stamp_metadata({"harness_version": "old", "utility_version": "x"})
    assert out["harness_version"] == HARNESS_VERSION
    assert out["utility_version"] == UTILITY_VERSION
    assert stamp_metadata(None) == {
        "harness_version": HARNESS_VERSION,
        "utility_version": UTILITY_VERSION,
    }


def test_check_current_stamps_ok():
    status = check_record_versions(stamp_metadata({}))
    assert status["harness"] is VersionStatus.OK
    assert status["utility"] is VersionStatus.OK


def test_check_legacy_missing_harness_stamp():
    """Pre-h1 records (utility stamped, harness absent) read as legacy."""
    status = check_record_versions({"utility_version": UTILITY_VERSION})
    assert status["harness"] is VersionStatus.LEGACY
    assert status["utility"] is VersionStatus.OK


def test_check_mismatch_future_stamp():
    status = check_record_versions(
        {"harness_version": "h99", "utility_version": UTILITY_VERSION}
    )
    assert status["harness"] is VersionStatus.MISMATCH
    status = check_record_versions(
        {"harness_version": HARNESS_VERSION, "utility_version": "phase9.z"}
    )
    assert status["utility"] is VersionStatus.MISMATCH


def test_generator_wiring_uses_stamp_metadata():
    """The generator module must route its metadata through stamp_metadata."""
    import inspect

    from opt_agent.dataset import generator

    src = inspect.getsource(generator)
    assert "stamp_metadata(" in src, (
        "dataset/generator.py no longer wires version stamps via "
        "stamp_metadata; restore the call or bump versions deliberately"
    )
    assert f'"utility_version":' not in src, (
        "generator hardcodes utility_version; use the versioning constants"
    )


# ---------------------------------------------------------------------------
# Loading-boundary enforcement (round-3, codex round-2 C3)
# ---------------------------------------------------------------------------

def test_enforce_warn_on_legacy(tmp_path, monkeypatch):
    import warnings as _w

    from opt_agent.dataset import decision_record as dr
    from opt_agent.dataset.versioning import (
        VersionPolicy, enforce_record_versions,
    )
    # Reset the once-per-source warning memo for a clean check.
    import opt_agent.dataset.versioning as ver
    monkeypatch.setattr(ver, "_legacy_warned", set())
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        enforce_record_versions({"utility_version": ver.UTILITY_VERSION},
                                VersionPolicy.WARN, record_id="r1",
                                source="legacy.jsonl")
    assert any("legacy" in str(w.message) for w in caught)


def test_enforce_strict_raises_on_mismatch():
    import pytest as _pytest

    from opt_agent.dataset.versioning import (
        VersionPolicyError, enforce_record_versions,
    )
    with _pytest.raises(VersionPolicyError):
        enforce_record_versions({"harness_version": "h99"},
                                "strict", record_id="r2", source="x.jsonl")


def test_enforce_strict_allows_legacy_with_warning(monkeypatch):
    import warnings as _w

    import opt_agent.dataset.versioning as ver
    from opt_agent.dataset.versioning import enforce_record_versions

    monkeypatch.setattr(ver, "_legacy_warned", set())
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        out = enforce_record_versions({}, "strict", record_id="r3",
                                      source="y.jsonl")
    assert out["harness"].name == "LEGACY"
    assert any("legacy" in str(w.message) for w in caught)


def _make_record_line(stamps: dict) -> str:
    import json

    meta = {"trajectory_id": 0, "step": 0, "run_seed": 0, **stamps}
    rec = {
        "id": "t:0:0:0",
        "problem": {"family": "t", "dimension": 1, "seed": 0},
        "state": {},
        "candidate_actions": [],
        "gold_probs": {},
        "rollout_results": {},
        "temperature": 1.0,
        "metadata": meta,
    }
    return json.dumps(rec)


def test_read_jsonl_version_policies(tmp_path, monkeypatch):
    """read_jsonl enforces the pooling policy at the loading boundary."""
    import json
    import warnings as _w

    import opt_agent.dataset.versioning as ver
    from opt_agent.dataset.decision_record import read_jsonl

    monkeypatch.setattr(ver, "_legacy_warned", set())
    path = tmp_path / "mixed.jsonl"
    lines = [
        # legacy: no harness stamp, current utility stamp
        _make_record_line({"utility_version": ver.UTILITY_VERSION}),
        # current: both stamps
        _make_record_line({"harness_version": ver.HARNESS_VERSION,
                           "utility_version": ver.UTILITY_VERSION}),
        # mismatch: future utility stamp
        _make_record_line({"harness_version": ver.HARNESS_VERSION,
                           "utility_version": "phase99.z"}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # allow: historical behavior, no enforcement
    assert len(read_jsonl(path)) == 3

    # warn: legacy summarized once, mismatch warned, still readable
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        records = read_jsonl(path, version_policy="warn")
    assert len(records) == 3
    assert any("legacy" in str(w.message) for w in caught)
    assert any("phase99.z" in str(w.message) for w in caught)

    # strict: mismatch raises
    import pytest as _pytest
    from opt_agent.dataset.versioning import VersionPolicyError
    with _pytest.raises(VersionPolicyError):
        read_jsonl(path, version_policy="strict")

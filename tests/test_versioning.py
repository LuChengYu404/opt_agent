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

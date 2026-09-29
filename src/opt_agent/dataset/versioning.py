"""Dataset / harness version stamps — code-side half of guide §4.5 + §6.6.

Discipline (RESEARCH_BOUNDARY_GUIDE.md):

- ``HARNESS_VERSION`` freezes the measurement system: observation format
  (state features), action menu, and label pipeline. Any change to state
  features, the ActionType menu, or any S-class supervision-defining
  component (§6.6) REQUIRES a bump.
- ``UTILITY_VERSION`` is the finer stamp: it tracks the S-class label
  definition only (utility weights/scales, continuation policy, rollout
  horizon, gold temperature, action-subset quota, replicate count).

Reading rule (:func:`check_record_versions`):

- ``OK``       — record carries the current stamps; poolable.
- ``LEGACY``   — stamp missing (records predate the mechanism) or holds a
  known-past value: readable, but must not be pooled with current-version
  records without an explicit same-data A/B note (§6.6 rule 1).
- ``MISMATCH`` — stamp is not a recognized value (e.g. a future version
  read by older code): refuse to pool.

Known-past values are listed in ``_KNOWN_PAST``; extend it when bumping
either constant so old data reads as LEGACY rather than MISMATCH.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

HARNESS_VERSION = "h1"
UTILITY_VERSION = "phase8.6b"

# Stamps that may appear in historical records. "phase8.6b" doubles as the
# current UTILITY_VERSION; harness_version was simply absent before h1.
_KNOWN_PAST: dict[str, set[str]] = {
    "harness": set(),  # h1 is the first stamped harness version
    "utility": set(),  # phase8.6b is the first formalized utility version
}


class VersionStatus(str, Enum):
    OK = "ok"
    LEGACY = "legacy"
    MISMATCH = "mismatch"


class VersionPolicy(str, Enum):
    """Pooling policy enforced at a dataset loading boundary (guide 4.5/6.6).

    ALLOW  - no check (back-compat; implicit pooling decision is "unchecked").
    WARN   - summarize LEGACY once per source (do-not-pool reminder) and warn
             on MISMATCH, but keep reading.
    STRICT - raise VersionPolicyError on the first MISMATCH; LEGACY still only
             warns (legacy records are the repo's history, the policy makes
             the pooling decision EXPLICIT rather than forbidding reads).
    """

    ALLOW = "allow"
    WARN = "warn"
    STRICT = "strict"


class VersionPolicyError(ValueError):
    """A record carries an unrecognized version stamp under STRICT policy."""


_legacy_warned: set[str] = set()


def enforce_record_versions(
    record_metadata: Mapping[str, Any],
    policy: str | VersionPolicy,
    *,
    record_id: str = "",
    source: str = "",
) -> dict[str, VersionStatus]:
    """Apply a pooling policy to one record's metadata.

    Returns the per-axis status (same shape as check_record_versions).
    LEGACY warns at most once per (source, axis) to avoid spam on large
    legacy datasets; MISMATCH raises under STRICT.
    """
    policy = VersionPolicy(policy)
    status = check_record_versions(record_metadata)
    if policy is VersionPolicy.ALLOW:
        return status
    import warnings

    for axis, st in status.items():
        key = f"{source}:{axis}"
        if st is VersionStatus.LEGACY and key not in _legacy_warned:
            _legacy_warned.add(key)
            warnings.warn(
                f"{source}: {axis} stamp is legacy/absent (record "
                f"{record_id!r} first seen); pooling legacy records with "
                f"current-version records requires an explicit same-data "
                f"A/B note (guide 6.6 rule 1)",
                stacklevel=2,
            )
        if st is VersionStatus.MISMATCH:
            stamp = record_metadata.get(f"{axis}_version")
            msg = (f"{source}: record {record_id!r} has unrecognized "
                   f"{axis} stamp {stamp!r} (current: "
                   f"{HARNESS_VERSION if axis == 'harness' else UTILITY_VERSION}); "
                   f"refusing to pool under STRICT policy")
            if policy is VersionPolicy.STRICT:
                raise VersionPolicyError(msg)
            warnings.warn(msg, stacklevel=2)
    return status


def stamp_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    harness_version: str = HARNESS_VERSION,
    utility_version: str = UTILITY_VERSION,
) -> dict[str, Any]:
    """Return a copy of ``metadata`` with both version stamps applied.

    Existing stamp keys are overwritten (the caller is generating records
    with the current code, whose stamps ARE the current constants).
    """
    out = dict(metadata) if metadata else {}
    out["harness_version"] = harness_version
    out["utility_version"] = utility_version
    return out


def check_record_versions(
    record_metadata: Mapping[str, Any],
) -> dict[str, VersionStatus]:
    """Classify a record's stamps against the current constants.

    ``record_metadata`` is the ``metadata`` dict of a DecisionRecord (the
    stamps live at its top level; see the generator).
    """
    return {
        "harness": _status(record_metadata.get("harness_version"),
                           HARNESS_VERSION, "harness"),
        "utility": _status(record_metadata.get("utility_version"),
                           UTILITY_VERSION, "utility"),
    }


def _status(stamp: Any, current: str, axis: str) -> VersionStatus:
    if stamp == current:
        return VersionStatus.OK
    if stamp is None or stamp in _KNOWN_PAST.get(axis, set()):
        return VersionStatus.LEGACY
    return VersionStatus.MISMATCH

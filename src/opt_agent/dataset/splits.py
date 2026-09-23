"""Dataset splits for the Optimization Decision Dataset.

Per codex gpt-6-sol review 2026-09-23:
  - Default 60/10/10/20 train/dev/calibration/test ratio.
  - One full problem family can be held out for OOD (configurable).
  - Group by (family, seed); all trajectories of one (family, seed) share
    the same split (independent-seed grouping → no intra-seed leakage).
  - physical_state_hash: SHA-256 over canonical state JSON, for dedup and
    group-by-physics analyses. MUST exclude rollout_results, gold_probs,
    and replay metadata to prevent label leakage into the hash.
  - gold_probs_kind: provenance tag for soft targets. In this codebase
    all gold_probs are derived from rollout utility softmax — a soft
    target, not an independently observed "correct action".
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

from opt_agent.dataset.decision_record import DecisionRecord


class Split(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    CALIBRATION = "calibration"
    TEST = "test"
    OOD = "ood"


@dataclass(frozen=True)
class SplitAssignment:
    family: str
    seed: int
    split: Split


def family_seed_of(record: DecisionRecord) -> tuple[str, int]:
    """Extract (family, seed) from a DecisionRecord.id.

    The generator emits ids of the form `family:seed:trajectory:step`,
    e.g. `"branin:0:0:3"`. We drop the trajectory / step suffix.
    """
    parts = record.id.split(":")
    if len(parts) >= 2:
        try:
            return parts[0], int(parts[1])
        except ValueError:
            pass
    return "unknown", -1


def _validate_ratios(ratios: Mapping[str, float]) -> None:
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0; got {total:.6f}")
    required = {Split.TRAIN.value, Split.DEV.value,
                Split.CALIBRATION.value, Split.TEST.value}
    missing = required - set(ratios)
    if missing:
        raise ValueError(
            f"ratios missing required splits: {sorted(missing)}"
        )
    extra = set(ratios) - required
    if extra:
        raise ValueError(
            f"ratios contains unknown split names: {sorted(extra)}"
        )


def assign_splits(
    records: Iterable[DecisionRecord],
    *,
    ood_families: Sequence[str] = (),
    ratios: Mapping[str, float] | None = None,
    seed: int = 0,
) -> dict[tuple[str, int], Split]:
    """Assign each (family, seed) pair to one split.

    Algorithm
    ---------
    1. Collect unique (family, seed) pairs from `records`.
    2. Pairs whose family is in `ood_families` → Split.OOD.
    3. Remaining pairs: within each family, sort seeds, shuffle with the
       caller's `seed`, and assign by cumulative mid-point ratios.

    Returns
    -------
    Mapping from (family, seed) → Split. The same key always maps to
    the same split for a given (records, ood_families, ratios, seed).
    """
    if ratios is None:
        ratios = {
            "train": 0.60,
            "dev": 0.10,
            "calibration": 0.10,
            "test": 0.20,
        }
    _validate_ratios(ratios)

    pairs: set[tuple[str, int]] = set()
    for r in records:
        pairs.add(family_seed_of(r))

    by_family: dict[str, list[int]] = defaultdict(list)
    for fam, sd in pairs:
        by_family[fam].append(sd)

    assignments: dict[tuple[str, int], Split] = {}

    # 1. OOD families first (deterministic; no shuffling).
    for fam in by_family:
        if fam in ood_families:
            for sd in by_family[fam]:
                assignments[(fam, sd)] = Split.OOD

    # 2. In-distribution families: shuffle and assign by ratio midpoints.
    split_names = list(ratios.keys())
    cumulative: list[float] = []
    running = 0.0
    for r in ratios.values():
        running += r
        cumulative.append(running)

    rng = random.Random(seed)
    for fam, seeds in by_family.items():
        if fam in ood_families:
            continue
        in_dist = sorted(s for s in seeds if (fam, s) not in assignments)
        rng.shuffle(in_dist)
        n = len(in_dist)
        if n == 0:
            continue
        for idx, sd in enumerate(in_dist):
            # Midpoint rule: spread seeds as evenly as possible.
            target_ratio = (idx + 0.5) / n
            chosen_idx = 0
            for i, cum in enumerate(cumulative):
                if target_ratio < cum:
                    chosen_idx = i
                    break
            else:
                chosen_idx = len(cumulative) - 1
            assignments[(fam, sd)] = Split(split_names[chosen_idx])

    return assignments


def split_records(
    records: Iterable[DecisionRecord],
    assignments: Mapping[tuple[str, int], Split],
) -> dict[Split, list[DecisionRecord]]:
    """Partition records into per-split lists via the assignment map."""
    out: dict[Split, list[DecisionRecord]] = {s: [] for s in Split}
    for r in records:
        out[assignments[family_seed_of(r)]].append(r)
    return out


# ---------------------------------------------------------------------------
# physical_state_hash
# ---------------------------------------------------------------------------


def physical_state_hash(record: DecisionRecord) -> str:
    """SHA-256 hex digest over the canonical (problem, state, candidates)
    representation.

    The hash intentionally EXCLUDES rollout_results, gold_probs, and
    replay checkpoint metadata. Those are labels / supervision artifacts
    and must not feed into physics-only grouping.
    """
    payload = {
        "problem": record.problem,
        "state": record.state,
        "candidate_actions": record.candidate_actions,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# gold_probs_kind
# ---------------------------------------------------------------------------


def gold_probs_kind(record: DecisionRecord) -> str:
    """Provenance tag for `record.gold_probs`.

    Returns the explicit metadata value if present, otherwise the default
    `"rollout_utility_softmax"`. Callers should use this to distinguish
    label sources when comparing across runs.
    """
    explicit = record.metadata.get("gold_probs_kind")
    if explicit:
        return str(explicit)
    return "rollout_utility_softmax"

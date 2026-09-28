"""Dataset analysis helpers (Phase 8 empirical validation).

Per codex (gpt-5.6-sol) review 2025-09-23:
  - `validate_record` enforces the aligned-vocabulary invariant.
  - Phase is defined by remaining_budget_ratio (>0.7 early, <0.3 late, 0 terminal).
  - Entropy is normalized to [0, 1] by log(len(valid)).
  - We do NOT yet assert "early = exploration-dominant" etc.; those are
    acceptance thresholds to add only after we observe the patterns.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from opt_agent.dataset.decision_record import DecisionRecord, iter_jsonl


def _phase(ratio: float) -> str:
    """Classify a BO trajectory point by remaining-budget ratio. The
    canonical version lives in ``dataset.validation.phase_of_ratio``;
    this stub is a deferred-import indirection so ``dataset.analysis``
    doesn't pull in the full validation module (which transitively
    imports the rollout machinery) at import time.
    """
    from opt_agent.dataset.validation import phase_of_ratio
    return phase_of_ratio(ratio)


def validate_record(record: DecisionRecord) -> None:
    """Enforce vocabulary and probability invariants.

    ``gold_probs`` and ``rollout_results`` must cover the same action ids,
    and those ids must be a subset of the candidate vocabulary: with
    ``rollout_action_subsample < 8`` (partial rollout) only a k-subset of
    the candidates is rolled out and labeled. Full datasets remain
    set-equal across all three fields.
    """
    ids = [a["action_id"] for a in record.candidate_actions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{record.id}: duplicate candidate IDs")
    if set(record.gold_probs) != set(record.rollout_results):
        raise ValueError(f"{record.id}: unaligned action vocabulary")
    if not set(record.gold_probs) <= set(ids):
        raise ValueError(f"{record.id}: labeled ids outside candidate vocabulary")

    probs = record.gold_probs
    if any(not math.isfinite(p) or p < 0 for p in probs.values()):
        raise ValueError(f"{record.id}: invalid probability")
    if not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9):
        raise ValueError(
            f"{record.id}: probabilities do not sum to one (got {sum(probs.values())})"
        )

    for action_id, result in record.rollout_results.items():
        if result["status"] == "invalid" and probs[action_id] != 0:
            raise ValueError(
                f"{record.id}: invalid action {action_id!r} has nonzero probability"
            )


@dataclass(frozen=True)
class DatasetStats:
    n_records: int
    problems: dict[str, int]
    phase_counts: dict[str, dict[str, int]]
    avg_entropy: dict[str, float]
    early_phase_top_action: dict[str, str]
    late_phase_top_action: dict[str, str]
    invalid_rate: dict[str, float]


def analyze_dataset(path: Path) -> DatasetStats:
    """Walk a JSONL dataset and compute summary stats.

    Raises
    ------
    ValueError
        If any record violates the aligned-vocabulary / probability invariants.
    """
    problem_counts: Counter = Counter()
    phase_counts: dict[str, Counter] = defaultdict(Counter)
    entropy_sum: Counter = Counter()
    entropy_count: Counter = Counter()
    invalid_sum: Counter = Counter()
    candidate_sum: Counter = Counter()
    phase_probs: dict[tuple[str, str], Counter] = defaultdict(Counter)
    n_records = 0

    for record in iter_jsonl(path):
        validate_record(record)
        n_records += 1
        family = record.problem["family"]
        ratio = float(record.state["resource"]["remaining_budget_ratio"])
        phase = _phase(ratio)

        problem_counts[family] += 1
        phase_counts[family][phase] += 1

        valid = [
            aid
            for aid, r in record.rollout_results.items()
            if r["status"] == "valid"
        ]
        if valid:
            entropy = -sum(
                p * math.log(p)
                for p in record.gold_probs.values()
                if p > 0
            )
            entropy /= math.log(len(valid))  # normalized to [0, 1]
        else:
            entropy = 0.0

        if phase != "terminal":
            entropy_sum[family] += entropy
            entropy_count[family] += 1

        invalid_sum[family] += len(record.gold_probs) - len(valid)
        candidate_sum[family] += len(record.gold_probs)

        phase_probs[(family, phase)].update(record.gold_probs)

    def top(family: str, phase: str) -> str:
        totals = phase_probs[(family, phase)]
        return max(totals, key=totals.get) if totals else ""

    families = sorted(problem_counts)
    return DatasetStats(
        n_records=n_records,
        problems=dict(problem_counts),
        phase_counts={f: dict(phase_counts[f]) for f in families},
        avg_entropy={
            f: entropy_sum[f] / entropy_count[f] if entropy_count[f] else math.nan
            for f in families
        },
        early_phase_top_action={f: top(f, "early") for f in families},
        late_phase_top_action={f: top(f, "late") for f in families},
        invalid_rate={
            f: invalid_sum[f] / candidate_sum[f] if candidate_sum[f] else 0.0
            for f in families
        },
    )

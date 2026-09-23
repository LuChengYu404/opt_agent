"""Public re-exports for opt_agent.dataset."""

from opt_agent.dataset.analysis import DatasetStats, _validate_record, analyze_dataset
from opt_agent.dataset.decision_record import (
    DecisionRecord,
    iter_jsonl,
    write_jsonl,
)
from opt_agent.dataset.generator import (
    DatasetConfig,
    DatasetGenerator,
    derive_run_seed,
    full_candidates,
)
from opt_agent.dataset.splits import (
    Split,
    SplitAssignment,
    assign_splits,
    family_seed_of,
    gold_probs_kind,
    physical_state_hash,
    split_records,
)
from opt_agent.dataset.validation import (
    EXPLORATION_ACTIONS,
    Gate2Result,
    MassGateResult,
    OracleResult,
    aggregate_mass,
    affine_rescaling_test,
    gate2_stability,
    oracle_test,
    oracle_test_from_record,
    phase_of,
)

__all__ = [
    "DatasetConfig",
    "DatasetGenerator",
    "DatasetStats",
    "DecisionRecord",
    "EXPLORATION_ACTIONS",
    "Gate2Result",
    "MassGateResult",
    "OracleResult",
    "Split",
    "SplitAssignment",
    "aggregate_mass",
    "affine_rescaling_test",
    "analyze_dataset",
    "assign_splits",
    "derive_run_seed",
    "family_seed_of",
    "full_candidates",
    "gate2_stability",
    "gold_probs_kind",
    "iter_jsonl",
    "oracle_test",
    "oracle_test_from_record",
    "phase_of",
    "physical_state_hash",
    "split_records",
    "validate_record",
    "write_jsonl",
]


# Re-export _validate_record under a public name for tests.
validate_record = _validate_record
del _validate_record

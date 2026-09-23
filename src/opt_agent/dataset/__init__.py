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
from opt_agent.dataset.validation import (
    EXPLORATION_ACTIONS,
    MassGateResult,
    OracleResult,
    aggregate_mass,
    affine_rescaling_test,
    oracle_test,
    phase_of,
)

__all__ = [
    "DatasetConfig",
    "DatasetGenerator",
    "DatasetStats",
    "DecisionRecord",
    "EXPLORATION_ACTIONS",
    "MassGateResult",
    "OracleResult",
    "aggregate_mass",
    "affine_rescaling_test",
    "analyze_dataset",
    "derive_run_seed",
    "full_candidates",
    "iter_jsonl",
    "oracle_test",
    "phase_of",
    "validate_record",
    "write_jsonl",
]


# Re-export _validate_record under a public name for tests.
validate_record = _validate_record
del _validate_record

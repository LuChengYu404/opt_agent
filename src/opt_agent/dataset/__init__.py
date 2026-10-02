"""Public re-exports for opt_agent.dataset."""

from opt_agent.dataset.decision_record import (
    DecisionRecord,
    iter_jsonl,
    read_jsonl,
    validate_record,
    write_jsonl,
)
from opt_agent.dataset.generator import (
    DatasetConfig,
    DatasetGenerator,
    derive_run_seed,
    full_candidates,
    subsample_candidates,
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
    phase_of_ratio,
)

__all__ = [
    "DatasetConfig",
    "DatasetGenerator",
    "DecisionRecord",
    "EXPLORATION_ACTIONS",
    "Gate2Result",
    "MassGateResult",
    "OracleResult",
    "aggregate_mass",
    "affine_rescaling_test",
    "derive_run_seed",
    "full_candidates",
    "gate2_stability",
    "iter_jsonl",
    "read_jsonl",
    "oracle_test",
    "oracle_test_from_record",
    "phase_of",
    "phase_of_ratio",
    "subsample_candidates",
    "validate_record",
    "write_jsonl",
]

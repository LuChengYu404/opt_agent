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

__all__ = [
    "DatasetConfig",
    "DatasetGenerator",
    "DatasetStats",
    "DecisionRecord",
    "analyze_dataset",
    "derive_run_seed",
    "full_candidates",
    "iter_jsonl",
    "validate_record",
    "write_jsonl",
]


# Re-export _validate_record under a public name for tests.
validate_record = _validate_record
del _validate_record

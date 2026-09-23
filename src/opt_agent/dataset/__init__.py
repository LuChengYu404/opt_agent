"""Public re-exports for opt_agent.dataset."""

from opt_agent.dataset.decision_record import (
    DecisionRecord,
    iter_jsonl,
    write_jsonl,
)
from opt_agent.dataset.generator import (
    DatasetConfig,
    DatasetGenerator,
    full_candidates,
)

__all__ = [
    "DatasetConfig",
    "DatasetGenerator",
    "DecisionRecord",
    "full_candidates",
    "iter_jsonl",
    "write_jsonl",
]

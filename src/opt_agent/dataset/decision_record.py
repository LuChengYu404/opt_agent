"""Optimization Decision Dataset (Phase 7).

Per MASTER_PLAN.md §11 and codex (gpt-5.6-sol) review 2025-09-23.

JSONL schema (one record per decision point):

    {
      "id": "branin:0:0:3",
      "problem": {"family": "branin", "dimension": 2, "seed": 0},
      "state": {...dataclasses.asdict(OptimizationState)...},
      "candidate_actions": [
          {"action_id": "global_exploration", "type": "global_exploration", "params": {...}},
          ...
      ],
      "gold_probs": {"action_id": probability, ...},
      "rollout_results": {
          "action_id": {
              "status": "valid|invalid",
              "utility": float,
              "delta_optimization": float,
              "delta_surrogate": float|null,
              "delta_coverage": float,
              "n_new_samples": int,
              "n_model_fits": int,
              "error": str|null
          }, ...
      },
      "temperature": float,
      "metadata": {...}
    }

Aligned vocabulary invariant: keys in candidate_actions, gold_probs, and
rollout_results are identical (set equality). This is enforced by the generator
and verified by tests.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


@dataclass(frozen=True)
class DecisionRecord:
    """A single (state, candidate_actions, gold_probs) record for System-1 training."""
    id: str
    problem: dict[str, Any]
    state: dict[str, Any]
    candidate_actions: list[dict[str, Any]]
    gold_probs: dict[str, float]
    rollout_results: dict[str, dict[str, Any]]
    temperature: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _json_safe(value: Any) -> Any:
    """Recursively convert dataclasses, Enums, NaN/Inf floats to JSON-safe types."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_jsonl(records: Iterable[DecisionRecord], path: Path) -> None:
    """Write DecisionRecords to JSONL. NaN/Inf are sanitized to None."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            payload = _json_safe(record)
            stream.write(
                json.dumps(
                    payload,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            stream.write("\n")


def iter_jsonl(path: Path) -> Iterator[DecisionRecord]:
    """Read DecisionRecords from JSONL. Skips blank lines."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield DecisionRecord(**json.loads(line))
            except Exception as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid DecisionRecord"
                ) from exc

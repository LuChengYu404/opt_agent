"""Checkpoint / Restore for UnifiedBackend (Phase 5).

Per codex (gpt-5.6-sol) review 2025-09-23:
  - Checkpoint scope = history tensors + problem_id/bounds (NOT a pickle of
    the full backend / GP).
  - restore() preserves fitted state if backend was fitted at checkpoint time
    — BoTorch's GP fit is deterministic from history + RNG, and we want
    subsequent propose() calls to reuse the GP without refitting.
  - restore() is atomic: on mismatch, raise BEFORE touching any backend state.
  - Returned tensors are clones (caller mutation does not affect checkpoint).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class Checkpoint:
    """Point-in-time snapshot of an optimization backend's history.

    Fields
    ------
    backend_id : str
        Fully-qualified class name of the backend that produced this checkpoint.
    problem_id : str
        BOCoDe problem class name (used to re-instantiate on restore).
    problem_bounds : list[tuple[float, float]]
        Bounds at snapshot time, length d.
    X : torch.Tensor [n, d]
    objectives : torch.Tensor [n, m]
    constraints : torch.Tensor [n, c]
    fitted : bool
        Whether the surrogate was fitted at checkpoint time. Restore preserves
        this — fitted state is deterministic from history + RNG.
    """
    backend_id: str
    problem_id: str
    problem_bounds: list[tuple[float, float]]
    X: torch.Tensor
    objectives: torch.Tensor
    constraints: torch.Tensor
    fitted: bool
    rng_state: Optional[bytes] = None  # torch RNG state pickled (Phase 5 placeholder)

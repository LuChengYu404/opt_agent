"""Checkpoint / Restore for UnifiedBackend (Phase 5).

Per codex (gpt-5.6-sol) review 2025-09-23 + Phase 6 correction:
  - Checkpoint scope = history tensors + problem_id/bounds + GP state_dict +
    torch RNG state. NOT a pickle of the full backend.
  - restore() preserves fitted state INCLUDING the fitted GP model — the GP
    state_dict is rehydrated so subsequent propose() calls see the same
    surrogate without a refit. This is critical for counterfactual rollout
    fairness: every candidate must start from the EXACT same backend state.
  - restore() is atomic: on mismatch, raise BEFORE touching any backend state.
  - Returned tensors are clones (caller mutation does not affect checkpoint).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class Checkpoint:
    """Point-in-time snapshot of an optimization backend.

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
        Whether the surrogate was fitted at checkpoint time.
    model_state_dict : Optional[bytes]
        Pickled BoTorch GP state_dict (when fitted=True). Allows restore to
        rehydrate the exact GP without refitting.
    torch_rng_state : Optional[bytes]
        Pickled torch RNG state. Restored before any subsequent random ops.
    """
    backend_id: str
    problem_id: str
    problem_bounds: list[tuple[float, float]]
    X: torch.Tensor
    objectives: torch.Tensor
    constraints: torch.Tensor
    fitted: bool
    model_state_dict: Optional[bytes] = None
    torch_rng_state: Optional[bytes] = None

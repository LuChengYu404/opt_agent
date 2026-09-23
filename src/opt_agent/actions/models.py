"""OptimizationAction dataclass + ActionType enum (Phase 3).

Per MASTER_PLAN.md §8 and codex (gpt-5.6-sol) review 2025-09-23.

Phase 3 actions (5 total — ERROR_GUIDED deferred because we don't yet have
leave-one-out residuals to distinguish "high uncertainty" from "high error"):
  - GLOBAL_EXPLORATION     Sobol sampling, ignores surrogate
  - LOCAL_EXPLOITATION     Perturb current best within bounds
  - SURROGATE_ACQUISITION  Fit GP (if needed), propose via LogEI, evaluate
  - RETRAIN_SURROGATE      Force a fresh GP fit, no evaluation
  - STOP                   Signal termination; does not mutate backend

`OptimizationAction.__post_init__` validates params at construction time so
invalid actions fail before reaching `execute_action`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ActionType(str, Enum):
    GLOBAL_EXPLORATION = "global_exploration"
    LOCAL_EXPLOITATION = "local_exploitation"
    UNCERTAINTY_SAMPLING = "uncertainty_sampling"
    ERROR_GUIDED_SAMPLING = "error_guided_sampling"
    BOUNDARY_SAMPLING = "boundary_sampling"
    SURROGATE_ACQUISITION = "surrogate_acquisition"
    RETRAIN_SURROGATE = "retrain_surrogate"
    STOP = "stop"


# Param schemas per action. Each entry lists (name, type, check_fn).
# check_fn returns the parsed value or raises ValueError.
def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name!r} must be int; got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name!r} must be > 0; got {value}")
    return int(value)


def _finite_float(name: str, value: Any, lo: float, hi: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name!r} must be a number; got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name!r} must be finite; got {value}")
    if not (lo <= value <= hi):
        raise ValueError(f"{name!r} must lie in [{lo}, {hi}]; got {value}")
    return value


def _strictly_positive_finite_float(name: str, value: Any, hi: float) -> float:
    """Float in (0, hi] — used for step sizes where 0 is meaningless."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name!r} must be a number; got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name!r} must be finite; got {value}")
    if not (0.0 < value <= hi):
        raise ValueError(f"{name!r} must lie in (0, {hi}]; got {value}")
    return value


def _non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name!r} must be int; got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name!r} must be >= 0; got {value}")
    return int(value)


_PARAM_SCHEMAS: dict[ActionType, dict[str, Any]] = {
    ActionType.GLOBAL_EXPLORATION: {
        "n_samples": _positive_int,
        "seed": lambda n, v: _non_negative_int(n, v) if v is not None else None,
    },
    ActionType.LOCAL_EXPLOITATION: {
        "step": lambda n, v: _strictly_positive_finite_float(n, v, hi=1.0),
        "seed": lambda n, v: _non_negative_int(n, v) if v is not None else None,
    },
    ActionType.UNCERTAINTY_SAMPLING: {
        "n_samples": _positive_int,
        "seed": lambda n, v: _non_negative_int(n, v) if v is not None else None,
    },
    ActionType.ERROR_GUIDED_SAMPLING: {
        "n_samples": _positive_int,
        "seed": lambda n, v: _non_negative_int(n, v) if v is not None else None,
    },
    ActionType.BOUNDARY_SAMPLING: {
        "n_samples": _positive_int,
        "tol": lambda n, v: _finite_float(n, v, 0.0, 0.5),
        "seed": lambda n, v: _non_negative_int(n, v) if v is not None else None,
    },
    ActionType.SURROGATE_ACQUISITION: {},
    ActionType.RETRAIN_SURROGATE: {},
    ActionType.STOP: {},
}

_ALLOWED_PARAMS: dict[ActionType, set[str]] = {
    ActionType.GLOBAL_EXPLORATION: {"n_samples", "seed"},
    ActionType.LOCAL_EXPLOITATION: {"step", "seed"},
    ActionType.UNCERTAINTY_SAMPLING: {"n_samples", "seed"},
    ActionType.ERROR_GUIDED_SAMPLING: {"n_samples", "seed"},
    ActionType.BOUNDARY_SAMPLING: {"n_samples", "tol", "seed"},
    ActionType.SURROGATE_ACQUISITION: set(),
    ActionType.RETRAIN_SURROGATE: set(),
    ActionType.STOP: set(),
}


@dataclass(frozen=True)
class OptimizationAction:
    """A structured optimization action.

    Parameters
    ----------
    type : ActionType
    params : Mapping[str, Any]
        Per-action-type params; validated in __post_init__.
        See _ALLOWED_PARAMS for the allowed keys per type.
    """

    type: ActionType
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.type, ActionType):
            raise TypeError(
                f"action.type must be ActionType; got {type(self.type).__name__}: "
                f"{self.type!r}"
            )
        params = dict(self.params) if self.params is not None else {}
        allowed = _ALLOWED_PARAMS[self.type]
        for k in params:
            if k not in allowed:
                raise ValueError(
                    f"Unsupported params {k!r} for action {self.type.value}. "
                    f"Allowed: {sorted(allowed) or '<none>'}."
                )
        schema = _PARAM_SCHEMAS[self.type]
        validated: dict[str, Any] = {}
        for k, fn in schema.items():
            if k in params:
                validated[k] = fn(k, params[k])
        # Copy validated back so default values like seed=None don't leak through.
        for k in allowed - set(schema):
            if k in params:
                validated[k] = params[k]
        # Bypass frozen to install validated params.
        object.__setattr__(self, "params", validated)


@dataclass(frozen=True)
class ActionResult:
    """What `execute_action` returns after performing an action.

    Attributes
    ----------
    action : OptimizationAction
        The action that was executed (echo for logging).
    n_new_samples : int
        How many new samples were added to the backend (0 for RETRAIN / STOP).
    n_model_fits : int
        How many GP fits were triggered. SURROGATE_ACQUISITION triggers 1
        implicit fit when the surrogate is not yet fitted; RETRAIN_SURROGATE
        always triggers 1 fit; other actions trigger 0.
    should_stop : bool
        True iff action.type == STOP. Caller should break its loop.
    """

    action: OptimizationAction
    n_new_samples: int
    n_model_fits: int = 0
    should_stop: bool = False

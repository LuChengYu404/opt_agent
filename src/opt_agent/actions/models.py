"""OptimizationAction / ActionResult.

h3 menu (docs/AF_SAMPLING_20261001.md): every point-selection action is an
LMABO-portfolio acquisition function; no action carries a `seed`.

  GLOBAL_EXPLORATION     TS (greedy posterior-sample argmax)
  LOCAL_EXPLOITATION     PI inside the step trust region
  UNCERTAINTY_SAMPLING   PosSTD
  SURROGATE_ACQUISITION  portfolio AF via `af` (default LogEI)
  RETRAIN_SURROGATE      force a GP fit, no evaluation
  STOP                   terminate
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from opt_agent.backend.acquisitions import ACQUISITION_SET


class ActionType(str, Enum):
    GLOBAL_EXPLORATION = "global_exploration"
    LOCAL_EXPLOITATION = "local_exploitation"
    UNCERTAINTY_SAMPLING = "uncertainty_sampling"
    SURROGATE_ACQUISITION = "surrogate_acquisition"
    RETRAIN_SURROGATE = "retrain_surrogate"
    STOP = "stop"


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name!r} must be a positive int; got {value!r}")
    return int(value)


def _unit_float(name: str, value: Any) -> float:
    """Float in (0, 1] — trust-region radius."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name!r} must be a number; got {value!r}")
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{name!r} must lie in (0, 1]; got {value!r}")
    return value


def _af_name(name: str, value: Any) -> str:
    if not isinstance(value, str) or value not in ACQUISITION_SET:
        raise ValueError(
            f"{name!r} must be one of the LMABO portfolio names; got {value!r}")
    return value


# One table: action -> {param: validator}. Params are validated at
# construction so invalid actions (e.g. LLM-emitted) fail early.
PARAM_VALIDATORS: dict[ActionType, dict[str, Any]] = {
    ActionType.GLOBAL_EXPLORATION: {"n_samples": _positive_int},
    ActionType.LOCAL_EXPLOITATION: {"step": _unit_float},
    ActionType.UNCERTAINTY_SAMPLING: {"n_samples": _positive_int},
    ActionType.SURROGATE_ACQUISITION: {"af": _af_name},
    ActionType.RETRAIN_SURROGATE: {},
    ActionType.STOP: {},
}


@dataclass(frozen=True)
class OptimizationAction:
    type: ActionType
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.type, ActionType):
            raise TypeError(f"action.type must be ActionType; got {self.type!r}")
        table = PARAM_VALIDATORS[self.type]
        params = dict(self.params) if self.params else {}
        for k in params:
            if k not in table:
                raise ValueError(
                    f"Unsupported params {k!r} for action {self.type.value}. "
                    f"Allowed: {sorted(table) or '<none>'}.")
        validated = {}
        for k, v in params.items():
            validated[k] = table[k](k, v)
        object.__setattr__(self, "params", validated)


@dataclass(frozen=True)
class ActionResult:
    """Echo of an executed action; n_model_fits counts fits actually
    triggered (AF samplers fit lazily per proposed point)."""

    action: OptimizationAction
    n_new_samples: int
    n_model_fits: int = 0
    should_stop: bool = False

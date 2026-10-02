"""Post-hoc temperature-scaling calibration for System-1 probability outputs.

Used by the router-sweep line (scripts/exp_rq3_router_sweep.py) to compare
trained calibration against a fitted temperature. The ECE/Brier/NLL and
top-k/KL metric modules were removed in the 2026-10-02 cleanup: no script
consumed them (experiments compute their metrics inline in the analyze_*
scripts).
"""

from opt_agent.metrics.temperature import (
    CalibrationResult,
    TemperatureCalibrator,
    fit_temperature,
)

__all__ = [
    "CalibrationResult",
    "TemperatureCalibrator",
    "fit_temperature",
]

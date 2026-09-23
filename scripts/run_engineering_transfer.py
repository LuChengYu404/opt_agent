"""Phase 14: Engineering transfer (placeholder).

Per MASTER_PLAN §5 / §25, the third benchmark level is a real
engineering optimization problem (vehicle suspension in the plan).
This script documents the transfer path and provides the integration
point for a future engineering simulator.

What is in scope today
----------------------
- The Phase 13 benchmark harness (`scripts/run_benchmark.py`) already
  exercises HybridPolicy on BOCoDe synthetic problems (Branin,
  Ackley, Hartmann). The closed-loop architecture (DecisionLoop +
  HybridPolicy + LLM System-2 + OOD-aware Router + Temperature
  Calibration) does NOT depend on the problem family — only on the
  candidate factory and the backend interface.

What is NOT in scope today
-------------------------
- A real vehicle-suspension simulator. MASTER_PLAN §5 lists
  "Vehicle Suspension Optimization" as the Level-3 benchmark, but the
  project has no such simulator in its dependency graph.
- BOCoDe engineering problems (Truss, Pressure Vessel, Gear Train,
  etc.) require extending `BocodeBotorchBackend`'s allowlist beyond
  the current {Branin, Hartmann, Ackley} (synthetic only). The
  allowlist extension requires verifying that the engineering
  problems are Phase-1-compatible (single-objective, unconstrained,
  continuous, well-behaved GP target).

To unblock Phase 14
------------------
Two paths forward:

1. Engineering simulator path: integrate a vehicle-suspension
   simulator behind a `UnifiedBackend` implementation (mirroring
   `BocodeBotorchBackend`'s interface). This requires a real simulator
   not currently in the project's dependency graph.

2. BOCoDe engineering-problem path: extend `BocodeBotorchBackend`'s
   allowlist to include e.g. Truss2D, PressureVessel, GearTrain after
   verifying their Phase-1 compatibility. The transfer harness
   becomes a thin wrapper around `run_benchmark.py` with the expanded
   problem set.

This script is intentionally a placeholder. Running it requires
either of the above paths; without them, only the synthetic
benchmark is meaningful.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# This script intentionally does NOT import BocodeBotorchBackend
# or HybridPolicy — it documents the transfer path without
# pretending engineering transfer is implemented.


_TRANSFER_PATHS: dict[str, str] = {
    "simulator": (
        "Integrate a vehicle-suspension simulator behind a "
        "UnifiedBackend implementation. Needs: real simulator in the "
        "project's dependency graph; this is OUT OF SCOPE today."
    ),
    "bocode_engineering": (
        "Extend BocodeBotorchBackend's allowlist beyond "
        "{Branin, Hartmann, Ackley} (currently synthetic-only). "
        "Add e.g. Truss2D, PressureVessel, GearTrain after "
        "Phase-1 compatibility verification."
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-paths", action="store_true",
                        help="List the two transfer paths and exit.")
    parser.add_argument("--output", type=Path,
                        default=Path("runs/phase14_engineering_stub.json"))
    args = parser.parse_args()

    if args.list_paths:
        for name, desc in _TRANSFER_PATHS.items():
            print(f"[{name}] {desc}")
        return 0

    payload = {
        "status": "placeholder",
        "scope": "Phase 14 engineering transfer not yet implemented",
        "paths": _TRANSFER_PATHS,
        "synthetic_baseline": (
            "scripts/run_benchmark.py demonstrates the HybridPolicy "
            "loop on BOCoDe synthetic problems. Use that as the "
            "template for engineering transfer once a real backend "
            "is available."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Phase 14 placeholder report: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
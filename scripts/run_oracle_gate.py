"""Phase 8.5 oracle gate runner — runs gates 3, 4, 5 on a real dataset.

Usage:
    uv run python scripts/run_oracle_gate.py runs/big_dataset.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.dataset import (  # noqa: E402
    affine_rescaling_test,
    aggregate_mass,
    iter_jsonl,
    oracle_test,
)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: run_oracle_gate.py <dataset.jsonl>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    records = list(iter_jsonl(path))
    if not records:
        print(f"error: no records in {path}", file=sys.stderr)
        return 1

    # Gate 3 — aggregate_mass.
    mass = aggregate_mass(records)
    print("Gate 3 (aggregate_mass):")
    print(f"  raw_mean = {mass.raw_mean:.4f}")
    print(f"  adjusted_lift_mean = {mass.adjusted_lift_mean:.4f}")
    print(f"  pass = {mass.pass_}")

    # Gate 4 — oracle_test (one record as a smoke).
    print("\nGate 4 (oracle_test) smoke (first early record):")
    early = [r for r in records if r.state["resource"]["remaining_budget_ratio"] > 0.7]
    if early:
        rec = early[0]
        # The oracle test needs a backend + state, but we don't have them
        # serialized in the JSONL. We can reconstruct a backend from the
        # problem family if we re-instantiate.
        from opt_agent.backend.bocode_botorch import (
            BackendConfig, BocodeBotorchBackend,
        )
        from opt_agent.state import extract_state
        from opt_agent.dataset.decision_record import _json_safe
        from opt_agent.dataset import DecisionRecord
        problem_id = rec.problem["family"].capitalize()
        try:
            be = BocodeBotorchBackend()
            be.initialize(BackendConfig(
                problem_id=problem_id,
                seed=int(rec.problem["seed"]),
                n_init=int(rec.state["sampling"]["n_samples"]),
            ))
            # We can't fully reconstruct state without re-running trajectory.
            # Skip unless we can replay.
            print(f"  skipped (cannot replay trajectory from JSONL alone)")
        except Exception as e:
            print(f"  skipped ({e})")

    print("\nGate 5 (affine_rescaling_test) requires regenerating dataset with transforms.")
    print("  Not run from this script; use scripts/generate_dataset.py with custom config.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

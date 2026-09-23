"""Phase 4 demo — run a full DecisionLoop on Brannin-2D with RulePolicy.

Demonstrates the plan §32 acceptance criterion end-to-end:

    state = loop.reset()
    action = policy(state)
    next_state, result = loop.step(action)

Usage:
    uv run python scripts/run_branin_demo.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend  # noqa: E402
from opt_agent.loop import DecisionLoop, RulePolicy  # noqa: E402


def main() -> int:
    print("=" * 78)
    print("opt_agent_v3 — Phase 4 demo on Brannin-2D")
    print("=" * 78)

    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(problem_id="Branin", seed=0, n_init=5))

    initial_best = backend.get_history().objectives.max().item()
    print(f"\nInitial best (5 Sobol points): {initial_best:.6f}")
    print(f"BOCoDe Brannin global maximum: -0.397887\n")

    loop = DecisionLoop(backend, total_budget=20)
    trajectory = loop.run(RulePolicy(seed=0), max_rounds=20)

    print(f"Ran {len(trajectory.steps)} step(s); termination: {trajectory.termination_reason.value}")
    print(
        f"Final: n_samples={trajectory.final_state.sampling.n_samples} "
        f"best={trajectory.final_state.optimization.best_value:.6f} "
        f"budget_left={trajectory.final_state.resource.remaining_budget}"
    )

    print("\nPer-step action log:")
    print(f"  {'round':>5}  {'action':<25}  {'n_new':>5}  {'best':>10}  {'stagnation':>9}")
    for s in trajectory.steps:
        print(
            f"  {s.round:>5}  {s.action.type.value:<25}  "
            f"{s.result.n_new_samples:>5}  "
            f"{s.next_state.optimization.best_value:>10.4f}  "
            f"{s.next_state.optimization.stagnation_steps:>9}"
        )

    # Save trajectory as JSONL for reproducibility (plan §19).
    out_path = ROOT / "runs" / "branin_demo_trajectory.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for s in trajectory.steps:
            row = {
                "round": s.round,
                "action_type": s.action.type.value,
                "action_params": dict(s.action.params),
                "n_new_samples": s.result.n_new_samples,
                "should_stop": s.result.should_stop,
                "best_value": s.next_state.optimization.best_value,
                "stagnation_steps": s.next_state.optimization.stagnation_steps,
                "n_samples": s.next_state.sampling.n_samples,
            }
            f.write(json.dumps(row) + "\n")
    print(f"\nTrajectory written to {out_path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

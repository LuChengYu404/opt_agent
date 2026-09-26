"""Run a zero-shot Laya System-1 baseline through the real BO decision loop.

Install the optional runtime first: ``uv sync --extra laya``. On first use,
the Laya SDK downloads the selected checkpoint from Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend  # noqa: E402
from opt_agent.loop import DecisionLoop  # noqa: E402
from opt_agent.system1 import LayaPolicy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="Branin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--model-id", default="convaiinnovations/laya")
    parser.add_argument("--subfolder", default=None)
    args = parser.parse_args()
    if args.budget <= args.n_init:
        parser.error("--budget must exceed --n-init")

    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(problem_id=args.problem, seed=args.seed, n_init=args.n_init))
    initial_best = float(backend.get_history().objectives.max().item())
    policy = LayaPolicy(model_id=args.model_id, subfolder=args.subfolder, seed=args.seed)
    trajectory = DecisionLoop(backend, total_budget=args.budget).run(
        policy, max_rounds=args.max_rounds
    )
    report = {
        "policy": "laya_zero_shot",
        "model_id": args.model_id,
        "subfolder": args.subfolder,
        "problem": args.problem,
        "seed": args.seed,
        "total_budget": args.budget,
        "initial_best": initial_best,
        "final_best": trajectory.final_state.optimization.best_value,
        "evaluations": trajectory.final_state.sampling.n_samples,
        "termination_reason": trajectory.termination_reason.value,
        "actions": [
            {
                "round": step.round,
                "type": step.action.type.value,
                "params": dict(step.action.params),
                "n_new_samples": step.result.n_new_samples,
                "best_value": step.next_state.optimization.best_value,
                "confidence": policy.decisions[step.round].confidence,
                "probabilities": policy.decisions[step.round].probabilities,
                "latency_s": policy.decisions[step.round].latency_s,
                "input_tokens": policy.decisions[step.round].input_tokens,
                "output_tokens": policy.decisions[step.round].output_tokens,
                "model": policy.decisions[step.round].model,
            }
            for step in trajectory.steps
        ],
        "aggregates": {
            "call_count": policy.call_count,
            "total_input_tokens": policy.total_input_tokens,
            "total_output_tokens": policy.total_output_tokens,
            "total_latency_s": policy.total_latency_s,
        },
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

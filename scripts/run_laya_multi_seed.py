"""Run local Laya zero-shot on the same paired BO protocol as Jev."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_jev_baseline import RuleFullBudget, run_one, summarize  # noqa: E402
from opt_agent.system1 import LayaPolicy  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="/home/lcy/.cache/opt_agent_v3/laya")
    parser.add_argument("--problems", nargs="+", default=["Branin", "Ackley", "Hartmann", "SixHumpCamel", "Rastrigin"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20, 25)))
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("runs/phase9_5_laya_zero_shot_5x5.json"))
    parser.add_argument("--state-format", choices=("structured", "narrative"),
                        default="structured",
                        help="Prompt state format; default 'structured' matches "
                             "the baseline that produced "
                             "runs/phase9_5_laya_zero_shot_5x5.json.")
    parser.add_argument("--criteria-format", choices=("legacy", "executable"),
                        default="legacy",
                        help="Prompt criteria format; default 'legacy' matches "
                             "the Laya baseline. 'executable' shifts the "
                             "prompt distribution and is not directly "
                             "comparable to that run.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    policy = LayaPolicy(
        model_id=args.model_id,
        state_format=args.state_format,
        criteria_format=args.criteria_format,
    )
    policy._get_agent()  # load before initializing paired backends
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for problem in args.problems:
        for seed in args.seeds:
            policy.seed = seed
            policy.reset()
            rule = run_one(problem, seed, args.budget, RuleFullBudget(seed=seed), "rule_full_budget", args.max_rounds)
            model = run_one(problem, seed, args.budget, policy, "laya", args.max_rounds)
            if rule["initial_best"] != model["initial_best"]:
                raise AssertionError("paired runs have different initial states")
            if len(model["actions"]) != len(policy.decisions):
                raise AssertionError("Laya decisions and trajectory steps differ")
            rows.extend((rule, model))
            args.output.write_text(json.dumps({
                "model_id": args.model_id,
                "state_format": args.state_format,
                "criteria_format": args.criteria_format,
                "torch_num_threads": 1,
                "rows": rows, "summary": summarize(rows),
            }, indent=2, sort_keys=True) + "\n")
            print(f"{problem} seed={seed}: rule={rule['terminal_best']:.4f}, laya={model['terminal_best']:.4f}, eval={model['new_evaluations']}, steps={len(model['actions'])}", flush=True)
    print(json.dumps(summarize(rows), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

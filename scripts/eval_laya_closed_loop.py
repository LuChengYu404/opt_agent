"""Parallel closed-loop audit for fine-tuned Laya System-1 checkpoints.

Branch `exp/rlcd-laya-finetune` (2026-09-27). Unlike the serial
`evaluate_trained_system1.py` (Jev-style checkpoints), this harness:

  - evaluates laya-loadable checkpoint dirs via `LayaPolicy(model_id=…)`;
  - runs trajectories in a ProcessPoolExecutor (CPU-parallel; each worker
    sets torch threads to 1 — the GP backend dominates, so CPU inference
    for the 0.4B model is fine and keeps the GPU free for training);
  - reports paired normalized gains vs `rule_full_budget` and
    `AlwaysSurrogate` with seed-clustered bootstrap CIs (house style).

Arms per (problem, seed): rule_full_budget, surrogate, plus one per
`--checkpoint NAME=PATH`. All arms share the backend seed, so starts are
paired.

Usage:
    .venv/bin/python scripts/eval_laya_closed_loop.py \
        --checkpoint proper=runs/rlcd_laya_proper_v1 \
        --checkpoint bt=runs/rlcd_laya_bt_v1 \
        --workers 8 --output runs/exp_rlcd_closed_loop.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _bootstrap_ci(values: list[float], repetitions: int = 2000,
                  seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(repetitions)
    )
    return means[int(0.025 * repetitions)], means[int(0.975 * repetitions)]


def _run_one(task: dict) -> dict:
    """Worker: one (problem, seed, arm) trajectory."""
    import torch

    torch.set_num_threads(1)
    from opt_agent.actions import candidate_actions_for_state
    from opt_agent.backend.bocode_botorch import (
        BackendConfig,
        BocodeBotorchBackend,
    )
    from opt_agent.loop import DecisionLoop
    from opt_agent.loop.policies import RulePolicy
    from opt_agent.state import OptimizationState

    class RuleFullBudget(RulePolicy):
        LOW_BUDGET_THRESHOLD = 0

    class AlwaysSurrogatePolicy:
        def __init__(self, seed: int = 0) -> None:
            self.seed = seed

        def select_action(self, state, backend=None):
            choices = candidate_actions_for_state(state, seed=self.seed)
            if "surrogate_acquisition" in choices:
                return choices["surrogate_acquisition"]
            return next(iter(choices.values()))

    problem, seed, budget, arm = (
        task["problem"], task["seed"], task["budget"], task["arm"]
    )
    if arm == "rule_full_budget":
        policy = RuleFullBudget(seed=seed)
    elif arm == "surrogate":
        policy = AlwaysSurrogatePolicy(seed=seed)
    else:
        from opt_agent.system1 import LayaPolicy

        policy = LayaPolicy(
            model_id=task["checkpoint_paths"][arm], seed=seed
        )

    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(problem_id=problem, seed=seed, n_init=5))
    initial = backend.get_history().objectives.flatten()
    scale = float(initial.std(correction=0)) or 1.0
    initial_best = float(initial.max())
    trajectory = DecisionLoop(backend, total_budget=budget).run(
        policy, max_rounds=40
    )
    history = backend.get_history()
    actions = [step.action.type.value for step in trajectory.steps]
    return {
        "problem": problem, "seed": seed, "budget": budget, "arm": arm,
        "initial_best": initial_best, "scale": scale,
        "terminal_best": float(history.objectives.max()),
        "new_evaluations": len(history) - 5,
        "termination": trajectory.termination_reason.value,
        "actions": actions,
        "action_counts": dict(Counter(actions)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="NAME=PATH; repeat for multiple checkpoints")
    parser.add_argument("--problems", nargs="+",
                        default=["Branin", "Ackley", "Hartmann",
                                 "SixHumpCamel", "Rastrigin"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(200, 210)),
                        help="Fresh test seeds (default 200-209)")
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/exp_rlcd_closed_loop.json"))
    args = parser.parse_args()

    checkpoint_paths = {}
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        if name in ("rule_full_budget", "surrogate") or not Path(path).is_dir():
            parser.error(f"invalid checkpoint spec: {item!r}")
        checkpoint_paths[name] = path

    arms = ["rule_full_budget", "surrogate", *checkpoint_paths]
    tasks = [
        {"problem": p, "seed": s, "budget": args.budget, "arm": arm,
         "checkpoint_paths": checkpoint_paths}
        for p in args.problems
        for s in args.seeds
        for arm in arms
    ]
    print(f"{len(tasks)} trajectories over {args.workers} workers "
          f"({len(args.problems)} problems × {len(args.seeds)} seeds × "
          f"{len(arms)} arms)", flush=True)

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            row = fut.result()
            rows.append(row)
            if i % 10 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] {t['problem']} seed={t['seed']} "
                      f"{t['arm']}: best={row['terminal_best']:.4f} "
                      f"eval={row['new_evaluations']}", flush=True)

    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by[(row["problem"], row["arm"])].append(row)

    summary = {}
    for (problem, arm), group in sorted(by.items()):
        for baseline_name in ("rule_full_budget", "surrogate"):
            if arm == baseline_name:
                continue
            base = {r["seed"]: r for r in by[(problem, baseline_name)]}
            diffs_by_seed = defaultdict(list)
            for r in group:
                b = base.get(r["seed"])
                if b is None:
                    continue
                diffs_by_seed[r["seed"]].append(
                    (r["terminal_best"] - b["terminal_best"]) / r["scale"]
                )
            seed_means = [statistics.fmean(v) for v in diffs_by_seed.values()]
            ci = _bootstrap_ci(seed_means) if len(seed_means) >= 5 else None
            summary[f"{problem}/{arm}/vs_{baseline_name}"] = {
                "n_seeds": len(seed_means),
                "mean_normalized_gain": statistics.fmean(seed_means),
                "ci95": ci,
                "ci_excludes_zero": (ci is not None and (
                    ci[0] > 0 or ci[1] < 0
                )),
            }
        summary[f"{problem}/{arm}"] = {
            "n": len(group),
            "mean_best": statistics.fmean(r["terminal_best"] for r in group),
            "mean_new_evaluations": statistics.fmean(
                r["new_evaluations"] for r in group
            ),
            "first_stop": sum(
                bool(r["actions"]) and r["actions"][0] == "stop" for r in group
            ),
            "action_counts": dict(sum(
                (Counter(r["actions"]) for r in group), Counter()
            )),
        }

    args.output.write_text(
        json.dumps({"checkpoints": args.checkpoint, "rows": rows,
                    "summary": summary}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

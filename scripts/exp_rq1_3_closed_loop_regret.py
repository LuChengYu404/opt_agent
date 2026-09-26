"""EXP-RQ1.3 closed-loop terminal simple regret — ultimate external-validity test.

Per codex review round 2:
> Closed-loop terminal simple regret is the primary external-validity endpoint.

For each (problem, seed), replay trajectory:
  1. At step 0, branch on each of the 8 actions as the first action.
  2. After the first action, continue with AlwaysSurrogate for remaining budget.
  3. Measure terminal best_y for each branch.
  4. The first-action regret = max_y - y(branch).

If regret correlates with ΔU (h=3 utility gap), the Decision Landscape
finding holds at the terminal level. If not, the h=3 finding is just a
short-horizon artifact.

Hardware: 32 cores available; trajectories are CPU-bound (BO step ~0.1s).
400 trajectories × ~10s each = 67 minutes serial → ~2 minutes parallel.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.actions import ActionType, OptimizationAction
from opt_agent.actions import candidate_actions_for_state
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend
from opt_agent.loop.decision_loop import DecisionLoop


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class AlwaysSurrogatePolicy:
    """Always picks SURROGATE_ACQUISITION. Used as continuation."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def select_action(self, state, backend=None):
        choices = candidate_actions_for_state(state, seed=self.seed)
        if "surrogate_acquisition" in choices:
            return choices["surrogate_acquisition"]
        return next(iter(choices.values()))


class FixedFirstActionPolicy:
    """Pick `first_action` at step 0; AlwaysSurrogate thereafter."""

    def __init__(self, first_action: OptimizationAction, seed: int = 0) -> None:
        self.first_action = first_action
        self.seed = seed
        self.first_done = False

    def select_action(self, state, backend=None):
        if not self.first_done:
            self.first_done = True
            return self.first_action
        choices = candidate_actions_for_state(state, seed=self.seed)
        if "surrogate_acquisition" in choices:
            return choices["surrogate_acquisition"]
        return next(iter(choices.values()))


# ---------------------------------------------------------------------------
# Trajectory runner
# ---------------------------------------------------------------------------


def run_one_trajectory(family: str, first_action: OptimizationAction,
                       budget: int, seed: int) -> float:
    """Run a fresh trajectory; return terminal best_y."""
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(
        problem_id=family,
        seed=seed,
        n_init=5,
    ))
    loop = DecisionLoop(backend, total_budget=budget)
    policy = FixedFirstActionPolicy(first_action, seed=0)
    loop.run(policy, max_rounds=budget + 5)
    history = backend.get_history()
    return float(history.objectives[:, 0].max().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+",
                        default=["Branin", "Hartmann", "Ackley",
                                 "SixHumpCamel", "Rastrigin"])
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of independent seeds (0..N-1).")
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs" / "exp_rq1_3_closed_loop_regret.json")
    args = parser.parse_args()

    # Build all 8 actions directly from the ALL_CANDIDATE_SPECS source.
    # STOP is added separately (not in the spec list but is a valid action).
    from opt_agent.actions.factory import ALL_CANDIDATE_SPECS, _make
    action_objs = []
    seen = set()
    for action_id, action_type, params in ALL_CANDIDATE_SPECS:
        action_objs.append(_make(action_type, params, seed=0))
        seen.add(action_id)
    # Add STOP explicitly.
    if "stop" not in seen:
        action_objs.append(OptimizationAction(ActionType.STOP, {}))

    # Build job list.
    jobs = []
    for family in args.families:
        for seed in range(args.seeds):
            for action in action_objs:
                jobs.append((family, action, args.budget, seed))
    print(f"Total jobs: {len(jobs)} = {len(args.families)} families × "
          f"{args.seeds} seeds × {len(action_objs)} actions")
    print(f"Workers: {args.workers}")

    # Run in parallel using multiprocessing.
    import multiprocessing as mp
    t0 = time.monotonic()
    with mp.Pool(args.workers) as pool:
        results = pool.starmap(run_one_trajectory, jobs)
    elapsed = time.monotonic() - t0
    print(f"Done in {elapsed:.1f}s ({len(jobs)/elapsed:.1f} trajectories/sec)")

    # Reshape into (family, seed, action) -> best_y.
    by_fam_seed_action = {}
    idx = 0
    for family in args.families:
        for seed in range(args.seeds):
            for action in action_objs:
                by_fam_seed_action[(family, seed, action.type.value)] = results[idx]
                idx += 1

    # Per-seed regret: max_y - y(a_branch) for each first action.
    summary = {}
    for family in args.families:
        summary[family] = {}
        for seed in range(args.seeds):
            y_by_action = {a.type.value: by_fam_seed_action[(family, seed, a.type.value)]
                            for a in action_objs}
            max_y = max(y_by_action.values())
            regrets = {a: max_y - y for a, y in y_by_action.items()}
            summary[family][seed] = {
                "max_y": max_y,
                "y_by_action": y_by_action,
                "regrets": regrets,
                "best_action": max(y_by_action, key=y_by_action.get),
                "surrogate_action": y_by_action["surrogate_acquisition"],
                "stop_action": y_by_action["stop"],
                "regret_surrogate": regrets["surrogate_acquisition"],
                "regret_stop": regrets["stop"],
            }

    # Aggregate per family.
    agg = {}
    for family in args.families:
        rows = [summary[family][s] for s in range(args.seeds)]
        for action_id in ("surrogate_acquisition", "stop", "local_exploitation",
                          "boundary_sampling", "global_exploration"):
            vals = [r["regrets"][action_id] for r in rows]
            agg.setdefault(family, {})[action_id] = {
                "mean": statistics.fmean(vals),
                "median": statistics.median(vals),
                "min": min(vals),
                "max": max(vals),
            }
        agg[family]["n_seeds"] = len(rows)
        # Best action frequency.
        from collections import Counter
        best_counter = Counter(r["best_action"] for r in rows)
        agg[family]["best_action_freq"] = dict(best_counter)

    print("\n=== Per-family closed-loop terminal regret (first action) ===")
    print(f"{'family':14s} {'action':22s} {'mean':>10s} {'med':>10s} {'max':>10s}")
    for family in args.families:
        for action_id in ("stop", "global_exploration", "surrogate_acquisition"):
            d = agg[family][action_id]
            print(f"{family:14s} {action_id:22s} {d['mean']:>10.4f} "
                  f"{d['median']:>10.4f} {d['max']:>10.4f}")
        print(f"  best_action_freq: {agg[family]['best_action_freq']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "config": {
            "families": args.families,
            "n_seeds": args.seeds,
            "budget": args.budget,
            "n_actions": len(action_objs),
            "elapsed_seconds": elapsed,
        },
        "per_seed_per_family": {f: {str(s): summary[f][s]
                                       for s in range(args.seeds)}
                                  for f in args.families},
        "per_family_aggregate": agg,
    }, indent=2) + "\n")
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

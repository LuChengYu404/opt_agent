"""EXP-RQ0.5 Dimension-Crossed — held-out validation of the frozen `dim>=5` router.

Per codex round-4 priority #1: freeze the router, evaluate on
dimension-crossed IOH functions (held-out from the BOCoDe benchmark
where the router was tuned).

IOH function set (different families from BOCoDe):
- fid=1  Sphere
- fid=2  Ellipsoid
- fid=3  Rastrigin
- fid=4  BuecheRastrigin
- fid=5  LinearSlope

Dimensions: 2D, 5D, 10D, 20D (4 levels, crossed with 5 functions = 20 problems)
Seeds: 5
Budget: 30 evaluations

Endpoint: normalized simple regret at terminal best.
  normalized_regret = (y - y_opt) / (y_init_best - y_opt)
  Range: 1.0 (no improvement from init) → 0.0 (reached optimum)

5 policies × 20 problems × 5 seeds = 500 trajectories.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import ioh
from opt_agent.actions import OptimizationAction, candidate_actions_for_state
from opt_agent.loop.decision_loop import DecisionLoop
from opt_agent.loop.policies import RulePolicy
from opt_agent.state import OptimizationState
from opt_agent.system1.jev_policy import JevPolicy
from opt_agent.system1.llm_system1_policy import LLMSystem1Policy
from ioh_backend import IOHBackend


# ---------------------------------------------------------------------------
# Policies (inlined from exp_rq0_5_full.py for portability)
# ---------------------------------------------------------------------------


class AlwaysSurrogatePolicy:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.call_count = 0

    def select_action(self, state: OptimizationState, backend=None) -> OptimizationAction:
        self.call_count += 1
        choices = candidate_actions_for_state(state, seed=self.seed)
        if "surrogate_acquisition" in choices:
            return choices["surrogate_acquisition"]
        return next(iter(choices.values()))


class FrozenRouter:
    """FROZEN `dim>=5` router — exact rule from EXP-RQ0.5 10-seed report."""

    def __init__(self, *, threshold: int = 5, seed: int = 0,
                 llm: LLMSystem1Policy | None = None,
                 surrogate: AlwaysSurrogatePolicy | None = None) -> None:
        self.threshold = threshold
        self.seed = seed
        self.llm = llm or LLMSystem1Policy(mode="llm", seed=seed,
                                           api_key=os.environ.get("DEEPSEEK_API_KEY"))
        self.surrogate = surrogate or AlwaysSurrogatePolicy(seed=seed)
        self.call_count = 0
        self.llm_call_count = 0
        self.surrogate_call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def select_action(self, state: OptimizationState, backend=None) -> OptimizationAction:
        self.call_count += 1
        if state.sampling.dimension >= self.threshold:
            self.llm_call_count += 1
            action = self.llm.select_action(state, backend)
        else:
            self.surrogate_call_count += 1
            action = self.surrogate.select_action(state, backend)
        self.total_input_tokens = self.llm.total_input_tokens
        self.total_output_tokens = self.llm.total_output_tokens
        return action


# ---------------------------------------------------------------------------
# IOH trajectory runner
# ---------------------------------------------------------------------------


def run_one_ioh(policy, fid: int, dim: int, instance: int,
                seed: int, budget: int = 30) -> dict:
    """Run a single IOH trajectory. Returns per-step y values for AUC."""
    problem = ioh.get_problem(fid=fid, dimension=dim, instance=instance,
                              problem_class=ioh.ProblemClass.REAL)
    backend = IOHBackend()
    backend.initialize(problem, n_init=5, seed=seed)
    loop = DecisionLoop(backend, total_budget=budget)
    try:
        traj = loop.run(policy, max_rounds=budget + 5)
    except Exception as exc:
        return {"fid": fid, "dim": dim, "instance": instance, "seed": seed,
                "error": repr(exc), "best_y": None, "y_trace": []}

    history = backend.get_history()
    y_trace = history.objectives[:, 0].cpu().numpy().tolist()
    best_y = float(min(y_trace))
    return {
        "fid": fid, "dim": dim, "instance": instance, "seed": seed,
        "best_y": best_y,
        "y_trace": y_trace,
        "y_init_best": float(min(y_trace[:5])),
        "y_opt": float(problem.optimum.y) if problem.optimum is not None else None,
        "problem_name": problem.meta_data.name,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _checkpoint_save(output_path, results, args, elapsed):
    """Save partial results without summary (final save does summary)."""
    for r in results:
        if r.get("error"):
            r["normalized_regret"] = None
            continue
        y_opt = r.get("y_opt")
        y_init = r.get("y_init_best")
        best = r.get("best_y")
        if y_opt is None or y_init == y_opt or y_init is None or best is None:
            r["normalized_regret"] = None
            continue
        nr = (best - y_opt) / (y_init - y_opt)
        r["normalized_regret"] = float(max(0.0, min(1.0, nr)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "config": {
            "dims": args.dims, "fids": args.fids,
            "instances": args.instances, "seeds": args.seeds,
            "budget": args.budget,
            "router_threshold": 5,
            "policies": args.policies,
        },
        "raw": results,
        "elapsed_seconds": elapsed,
    }, indent=2, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10, 20])
    parser.add_argument("--fids", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--instances", type=int, nargs="+", default=[1])
    parser.add_argument("--policies", nargs="+",
                        default=["Rule", "Surrogate", "LLM", "Router"])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs" / "exp_rq0_5_ioh_crossed.json")
    args = parser.parse_args()

    # Load API key.
    env_path = Path("/home/lcy/Documents/projects/opt_agent_v3/.env")
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        os.environ[k] = v

    print(f"dims={args.dims} fids={args.fids} instances={args.instances} "
          f"seeds={args.seeds} budget={args.budget}")

    def make_rule(seed): return RulePolicy(seed=seed)
    def make_surrogate(seed): return AlwaysSurrogatePolicy(seed=seed)
    def make_llm(seed):
        return LLMSystem1Policy(mode="llm", seed=seed,
                                api_key=os.environ.get("DEEPSEEK_API_KEY"))
    def make_router(seed):
        return FrozenRouter(threshold=5, seed=seed,
                            llm=make_llm(seed),
                            surrogate=make_surrogate(seed))

    # Instantiate factories now so each call passes an instance, not a function.
    def _resolve(name, seed):
        if name == "Rule": return make_rule(seed)
        if name == "Surrogate": return make_surrogate(seed)
        if name == "LLM": return make_llm(seed)
        if name == "Router": return make_router(seed)
        raise ValueError(name)

    factories = {
        "Rule": make_rule,
        "Surrogate": make_surrogate,
        "LLM": make_llm,
        "Router": make_router,
    }
    selected = {p: factories[p] for p in args.policies if p in factories}

    # Build jobs.
    jobs = []
    for policy_name in selected:
        for fid in args.fids:
            for dim in args.dims:
                for inst in args.instances:
                    for seed in range(args.seeds):
                        jobs.append((policy_name, fid, dim, inst, seed))
    print(f"Total jobs: {len(jobs)}")

    results = []
    t0 = time.monotonic()
    for policy_name, fid, dim, inst, seed in jobs:
        policy = _resolve(policy_name, seed)
        try:
            r = run_one_ioh(policy, fid, dim, inst, seed,
                            budget=args.budget)
        except Exception as exc:
            r = {"fid": fid, "dim": dim, "instance": inst, "seed": seed,
                 "policy": policy_name, "error": repr(exc),
                 "best_y": None, "y_trace": []}
        r["policy"] = policy_name
        results.append(r)
        # Periodic checkpoint save (avoid losing data on long runs).
        if (len(results) % 25 == 0):
            elapsed = time.monotonic() - t0
            print(f"  ... {len(results)}/{len(jobs)} done in {elapsed:.0f}s",
                  flush=True)
            _checkpoint_save(args.output, results, args, time.monotonic() - t0)
    elapsed = time.monotonic() - t0
    print(f"Done in {elapsed:.0f}s")

    # Compute normalized simple regret per (fid, dim, instance).
    # normalized_regret = (best_y - y_opt) / (y_init_best - y_opt).
    # Clamp to [0, 1] (1 means no improvement; 0 means reached optimum).
    for r in results:
        if r.get("error"):
            r["normalized_regret"] = None
            continue
        y_opt = r["y_opt"]
        y_init = r["y_init_best"]
        best = r["best_y"]
        if y_opt is None or y_init == y_opt:
            r["normalized_regret"] = None
            continue
        nr = (best - y_opt) / (y_init - y_opt)
        r["normalized_regret"] = float(max(0.0, min(1.0, nr)))

    # Aggregate per (policy, fid, dim).
    agg = {}
    for r in results:
        if r.get("normalized_regret") is None: continue
        key = (r["policy"], r["fid"], r["dim"])
        agg.setdefault(key, []).append(r["normalized_regret"])

    summary = {}
    for key, vals in agg.items():
        summary[str(key)] = {
            "n": len(vals),
            "mean": statistics.fmean(vals),
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
        }

    # Save.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "config": {
            "dims": args.dims, "fids": args.fids,
            "instances": args.instances, "seeds": args.seeds,
            "budget": args.budget,
            "router_threshold": 5,
            "policies": args.policies,
        },
        "raw": results,
        "summary": summary,
        "elapsed_seconds": elapsed,
    }, indent=2, default=str) + "\n")
    print(f"Saved: {args.output}")

    # Print per-dim aggregate.
    print()
    print("=== Mean normalized simple regret by (policy, dim) ===")
    print(f"{'policy':12s} " + " ".join(f"d={d:>3d}" for d in args.dims))
    for p in selected:
        row = [p]
        for d in args.dims:
            vals_per_d = []
            for fid in args.fids:
                vals_per_d.extend(agg.get((p, fid, d), []))
            row.append(f"{statistics.fmean(vals_per_d):>6.3f}" if vals_per_d else "  N/A")
        print(" ".join(row))

    print()
    print("=== Mean normalized simple regret by (policy, fid) ===")
    print(f"{'policy':12s} " + " ".join(f"fid={f:<3d}" for f in args.fids))
    for p in selected:
        row = [p]
        for fid in args.fids:
            vals_per_f = []
            for d in args.dims:
                vals_per_f.extend(agg.get((p, fid, d), []))
            row.append(f"{statistics.fmean(vals_per_f):>7.3f}" if vals_per_f else "  N/A")
        print(" ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

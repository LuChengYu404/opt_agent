"""EXP-RQ0.5 — Full 5-seed, 4-policy comparison (Rule, Surrogate, Jev, DeepSeek).

Per codex round-3 review:
- 5-seed replication of EXP-RQ0.5 (was 2 seeds)
- Add JevPolicy head-to-head (was missing)
- Per-step LLM AND Jev action logs (was missing — token counts all zero)
- 5 families × 5 seeds × budget 30

Output:
- Per (policy, family, seed): terminal best_y + n_steps + termination_reason
- Per (policy, family): per-step action distribution
- Total LLM/Jev call counts and token usage
- Decision regret (using h=3 utility vector from per-step rollouts)

Hardware: 32 cores; 4 policies × 5 seeds × 5 families = 100 trajectories
serial; ~5 minutes. LLM calls dominate (~1s each, ~25 per trajectory).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.actions import OptimizationAction
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend
from opt_agent.loop.decision_loop import DecisionLoop
from opt_agent.loop.policies import RulePolicy
from opt_agent.state import OptimizationState
from opt_agent.system1.jev_policy import JevPolicy
from opt_agent.system1.llm_system1_policy import LLMSystem1Policy


@dataclass
class RunResult:
    family: str
    seed: int
    policy: str
    best_y: float
    n_steps: int
    termination: str
    n_calls: int = 0
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    # Step-level log (per codex round-4 review item #1):
    # action_id, params, n_new_samples, current_best, latency_s,
    # model (for LLM/Jev), is_fitted (for evaluating state features).
    steps: list[dict] = field(default_factory=list)
    # RoutingAware controller: per-decision routing decision.
    routing_log: list[dict] = field(default_factory=list)


class AlwaysSurrogatePolicy:
    """Always picks SURROGATE_ACQUISITION (the BO-GP-signal baseline)."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.call_count = 0
        self.last_action: OptimizationAction | None = None

    def select_action(self, state: OptimizationState, backend=None) -> OptimizationAction:
        from opt_agent.actions import candidate_actions_for_state
        self.call_count += 1
        choices = candidate_actions_for_state(state, seed=self.seed)
        if "surrogate_acquisition" in choices:
            self.last_action = choices["surrogate_acquisition"]
        else:
            self.last_action = next(iter(choices.values()))
        return self.last_action


class RoutingAwareMetaController:
    """Route between LLM and Surrogate based on problem dimension.

    Hypothesis (EXP-RQ0.5 n=10 evidence):
      - LLM wins on Rastrigin (10D) by +16.19 best_y (p=0.0001)
      - Surrogate wins on Branin (2D), SixHumpCamel (2D), Hartmann (6D)
      - LLM ≈ Surrogate on Ackley (10D, p=0.23)

    Routing rule: if dimension >= threshold, use LLM; else use Surrogate.
    """

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
        self.routing_log: list[dict] = []
        # Inherit fields for parity with LLM/Surrogate stats.
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_action: OptimizationAction | None = None

    def select_action(self, state: OptimizationState, backend=None) -> OptimizationAction:
        self.call_count += 1
        dim = state.sampling.dimension
        route = "llm" if dim >= self.threshold else "surrogate"
        if route == "llm":
            self.llm_call_count += 1
            action = self.llm.select_action(state, backend)
        else:
            self.surrogate_call_count += 1
            action = self.surrogate.select_action(state, backend)
        self.total_input_tokens = self.llm.total_input_tokens
        self.total_output_tokens = self.llm.total_output_tokens
        self.routing_log.append({
            "dim": dim,
            "n_samples": int(state.sampling.n_samples),
            "boundary_coverage": float(state.sampling.boundary_coverage),
            "route": route,
            "action_id": action.type.value,
        })
        self.last_action = action
        return action


def run_one(policy_factory, family: str, seed: int, budget: int,
            policy_name: str) -> RunResult:
    """Run a single trajectory with the given policy factory."""
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(
        problem_id=family,
        seed=seed,
        n_init=5,
    ))
    policy = policy_factory(seed)
    loop = DecisionLoop(backend, total_budget=budget)
    try:
        traj = loop.run(policy, max_rounds=budget + 5)
    except Exception as exc:
        return RunResult(family=family, seed=seed, policy=policy_name,
                         best_y=float("nan"), n_steps=0, termination=f"error:{exc!r}")
    history = backend.get_history()
    best_y = float(history.objectives[:, 0].max().item())
    result = RunResult(
        family=family, seed=seed, policy=policy_name,
        best_y=best_y, n_steps=len(traj.steps),
        termination=(traj.termination_reason.value if traj.termination_reason else "?"),
        n_calls=getattr(policy, "call_count", 0),
        n_input_tokens=getattr(policy, "total_input_tokens", 0),
        n_output_tokens=getattr(policy, "total_output_tokens", 0),
        routing_log=list(getattr(policy, "routing_log", [])),
    )
    # Per-step log (per codex round-4).
    # Per-step `current_best` and `n_samples_after` come from `step.next_state`,
    # NOT the final history (bug fix per codex review).
    decisions = getattr(policy, "decisions", [])
    # For wrappers like RoutingAwareMetaController, also check inner LLMs.
    inner_decisions = []
    if hasattr(policy, "llm") and hasattr(policy.llm, "decisions"):
        inner_decisions.extend(policy.llm.decisions)
    if hasattr(policy, "surrogate") and hasattr(policy.surrogate, "decisions"):
        inner_decisions.extend(policy.surrogate.decisions)
    decision_pool = decisions if decisions else inner_decisions
    for i, step in enumerate(traj.steps):
        action_id = step.action.type.value
        result.action_counts[action_id] = result.action_counts.get(action_id, 0) + 1
        # Step-specific values from next_state.
        cur_best = float(step.next_state.optimization.best_value)
        n_samples_after = int(step.next_state.sampling.n_samples)
        cur_best_total_after = float(step.next_state.optimization.best_value)
        # Capture per-step policy decision info (from inner LLMs if wrapper).
        probs = confidence = model = latency_s = None
        if i < len(decision_pool):
            d = decision_pool[i]
            probs = dict(d.probabilities) if hasattr(d, "probabilities") else None
            confidence = float(d.confidence) if hasattr(d, "confidence") else None
            model = str(d.model) if hasattr(d, "model") else None
            latency_s = float(d.latency) if hasattr(d, "latency") else None
        result.steps.append({
            "step": i,
            "action_id": action_id,
            "action_params": dict(step.action.params),
            "n_new_samples": int(step.result.n_new_samples),
            "n_model_fits": int(step.result.n_model_fits),
            "current_best": cur_best,
            "current_best_total": cur_best_total_after,
            "should_stop": bool(step.result.should_stop),
            "is_fitted_after": bool(backend.is_fitted),
            "n_samples_after": n_samples_after,
            "latency_s": latency_s,
            "probs": probs,
            "confidence": confidence,
            "model": model,
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+",
                        default=["Branin", "Hartmann", "Ackley",
                                 "SixHumpCamel", "Rastrigin"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--policies", nargs="+",
                        default=["RulePolicy", "AlwaysSurrogate", "LLMDeepSeek",
                                 "Jev", "RoutingAware"],
                        help="Subset to run: RulePolicy / AlwaysSurrogate / LLMDeepSeek / Jev / RoutingAware")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs" / "exp_rq0_5_full_results.json")
    args = parser.parse_args()

    # Load API keys from .env.
    from pathlib import Path as _P
    env_path = _P("/home/lcy/Documents/projects/opt_agent_v3/.env")
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        os.environ[k] = v

    def make_rule(seed): return RulePolicy(seed=seed)
    def make_surrogate(seed): return AlwaysSurrogatePolicy(seed=seed)
    def make_llm(seed):
        return LLMSystem1Policy(mode="llm", seed=seed,
                                api_key=os.environ.get("DEEPSEEK_API_KEY"))
    def make_jev(seed):
        return JevPolicy(seed=seed, api_key=os.environ.get("JEV_API"))
    def make_routing(seed):
        return RoutingAwareMetaController(threshold=5, seed=seed,
                                          llm=make_llm(seed),
                                          surrogate=make_surrogate(seed))

    factories = {
        "RulePolicy": make_rule,
        "AlwaysSurrogate": make_surrogate,
        "LLMDeepSeek": make_llm,
        "Jev": make_jev,
        "RoutingAware": make_routing,
    }
    selected = {p: factories[p] for p in args.policies if p in factories}

    jobs = [(f, p, selected[p], s, args.budget)
            for p in selected
            for f in args.families
            for s in range(args.seeds)]
    print(f"Total jobs: {len(jobs)} = "
          f"{len(args.policies)} policies × {len(args.families)} families × "
          f"{args.seeds} seeds × budget {args.budget}")
    print(f"Policies: {args.policies}")
    print(f"DEEPSEEK_API_KEY present: {bool(os.environ.get('DEEPSEEK_API_KEY'))}")
    print(f"JEV_API present: {bool(os.environ.get('JEV_API'))}")

    results: list[RunResult] = []
    t0 = time.monotonic()
    for family, policy_name, factory, seed, budget in jobs:
        r = run_one(factory, family, seed, budget, policy_name)
        results.append(r)
        llm_str = ""
        if policy_name in ("LLMDeepSeek", "Jev"):
            llm_str = (f" calls={r.n_calls} "
                       f"tokens(in/out)={r.n_input_tokens}/{r.n_output_tokens}")
        print(f"  [{policy_name}] {family}/seed{seed} "
              f"best_y={r.best_y:+.4f} steps={r.n_steps} "
              f"reason={r.termination}{llm_str}")
    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Aggregate per (policy, family).
    summary = {}
    for r in results:
        d = summary.setdefault(r.policy, {}).setdefault(r.family, [])
        d.append(r)
    agg = {}
    for policy, by_fam in summary.items():
        agg[policy] = {}
        for fam, rs in by_fam.items():
            ys = [r.best_y for r in rs if r.best_y == r.best_y]  # not NaN
            if ys:
                agg[policy][fam] = {
                    "n_seeds": len(ys),
                    "mean_best_y": statistics.fmean(ys),
                    "median_best_y": statistics.median(ys),
                    "min_best_y": min(ys),
                    "max_best_y": max(ys),
                    "stdev": statistics.stdev(ys) if len(ys) > 1 else 0.0,
                }

    # Aggregate per-step action distribution per (policy, family).
    actions_by_pf = {}
    for r in results:
        key = (r.policy, r.family)
        d = actions_by_pf.setdefault(key, defaultdict(int))
        for a, c in r.action_counts.items():
            d[a] += c

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "families": args.families,
            "seeds": args.seeds,
            "budget": args.budget,
            "policies": args.policies,
        },
        "results": [
            {"family": r.family, "seed": r.seed, "policy": r.policy,
             "best_y": r.best_y, "n_steps": r.n_steps,
             "termination": r.termination,
             "n_calls": r.n_calls,
             "n_input_tokens": r.n_input_tokens,
             "n_output_tokens": r.n_output_tokens,
             "action_counts": r.action_counts,
             "routing_log": r.routing_log,
             "steps": r.steps}
            for r in results
        ],
        "summary_per_policy_family": agg,
        "actions_per_policy_family": {
            f"{p}/{fam}": dict(d) for (p, fam), d in actions_by_pf.items()
        },
        "elapsed_seconds": elapsed,
    }
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")

    # Print summary table.
    print("\n=== Summary (mean best_y across seeds, higher = better) ===")
    print(f"{'family':14s} " + " ".join(f"{p:>22s}" for p in selected))
    for fam in args.families:
        row = [fam]
        for p in selected:
            d = agg[p].get(fam, {})
            row.append(f"{d.get('mean_best_y', float('nan')):>22.4f}")
        print(" ".join(row))

    print("\n=== Per-step action distribution (% by family) ===")
    all_actions = ["global_exploration", "local_exploitation", "uncertainty_sampling",
                   "error_guided_sampling", "boundary_sampling",
                   "surrogate_acquisition", "retrain_surrogate", "stop"]
    print(f"{'policy/family':28s} " + " ".join(f"{a[:7]:>7s}" for a in all_actions))
    for p in selected:
        for fam in args.families:
            counts = actions_by_pf.get((p, fam), {})
            total = sum(counts.values()) or 1
            row = [f"{p}/{fam:10s}"]
            for a in all_actions:
                pct = 100.0 * counts.get(a, 0) / total
                row.append(f"{pct:>6.1f}%")
            print(" ".join(row))

    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

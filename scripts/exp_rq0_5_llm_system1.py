"""EXP-RQ0.5 quick verification: does LLM-as-System-1 ≈ AlwaysSurrogate?

Branch: exp/rq0.5-llm-system1 (off main).

Compares 3 policies head-to-head on the BOCoDe allowlist:
  - RulePolicy: hand-coded rules (Phase 4 baseline)
  - AlwaysSurrogatePolicy: always picks SURROGATE_ACQUISITION (the action
    that uses the BO GP signal — the "smart baseline")
  - LLMSystem1Policy: calls LLM at every decision point, parses response

If LLMSystem1 ≈ AlwaysSurrogate on most families, then trained Jev's value
over AlwaysSurrogate is just LLM's value over AlwaysSurrogate — i.e., the
training pipeline adds nothing beyond an LLM's in-context judgment.

If LLMSystem1 < AlwaysSurrogate on any family, the LLM is making bad
decisions; training Jev could conceivably compress those out.

If LLMSystem1 > AlwaysSurrogate on any family, the LLM has independent
decision intelligence worth studying.

Modes
-----
LLMSystem1Policy.mode="mock": deterministic heuristic (verifies wiring).
LLMSystem1Policy.mode="llm": real DeepSeek API call (requires DEEPSEEK_API_KEY).
By default this script runs mode="mock". Pass --llm to switch.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.actions import ActionType, OptimizationAction
from opt_agent.actions import candidate_actions_for_state
from opt_agent.actions.models import OptimizationAction as _OA
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend
from opt_agent.loop.decision_loop import DecisionLoop
from opt_agent.loop.policies import RulePolicy
from opt_agent.state import OptimizationState
from opt_agent.system1.llm_system1_policy import LLMSystem1Policy

logger = logging.getLogger("exp_rq0_5")
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class AlwaysSurrogatePolicy:
    """Always pick SURROGATE_ACQUISITION — the action that uses the BO GP signal.

    This is the "smart BO baseline": it lets the GP do its job and ignores
    any decision logic. If a learned/LLM policy can't beat this, the policy
    layer is adding no value.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.call_count = 0
        self.last_action: OptimizationAction | None = None

    def select_action(
        self, state: OptimizationState, backend=None
    ) -> OptimizationAction:
        self.call_count += 1
        choices = candidate_actions_for_state(state, seed=self.seed)
        # Prefer surrogate_acquisition if available; otherwise fall back to
        # the first sampling action (mirrors what a smart BO user would do).
        if "surrogate_acquisition" in choices:
            action = choices["surrogate_acquisition"]
        else:
            action = next(iter(choices.values()))
        self.last_action = action
        return action


# ---------------------------------------------------------------------------
# One trajectory runner
# ---------------------------------------------------------------------------


def run_one_trajectory(
    policy, family: str, budget: int, seed: int
) -> dict:
    """Run a single trajectory: init backend, run policy until termination."""
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(
        problem_id=family,
        seed=seed,
        n_init=5,
    ))
    loop = DecisionLoop(backend, total_budget=budget)
    try:
        traj = loop.run(policy, max_rounds=budget + 5)
    except Exception as exc:
        return {
            "family": family,
            "seed": seed,
            "error": repr(exc),
            "best_y": None,
            "n_steps": 0,
        }
    history = backend.get_history()
    best_y = float(history.objectives[:, 0].max().item())
    return {
        "family": family,
        "seed": seed,
        "best_y": best_y,
        "n_steps": len(traj.steps),
        "termination_reason": traj.termination_reason.value
            if traj.termination_reason else None,
        "n_policy_calls": getattr(policy, "call_count", 0),
        "n_llm_tokens": getattr(policy, "total_input_tokens", 0)
            + getattr(policy, "total_output_tokens", 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+",
                        default=["Branin", "Hartmann", "Ackley",
                                 "SixHumpCamel", "Rastrigin"])
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of independent problem seeds.")
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--llm", action="store_true",
                        help="Use real DeepSeek LLM (requires DEEPSEEK_API_KEY). "
                             "Default: mock mode (no API key needed).")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs" / "exp_rq0_5_results.json")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    use_llm = args.llm
    if use_llm and not api_key:
        print("WARNING: --llm requested but DEEPSEEK_API_KEY not set; "
              "falling back to mock mode.", file=sys.stderr)
        use_llm = False

    # Define the 3 policies.
    policies = {
        "RulePolicy": RulePolicy(seed=0),
        "AlwaysSurrogate": AlwaysSurrogatePolicy(seed=0),
        "LLMSystem1Policy": LLMSystem1Policy(
            mode="llm" if use_llm else "mock",
            api_key=api_key,
            seed=0,
        ),
    }

    print(f"Running EXP-RQ0.5: 3 policies × {len(args.families)} families × "
          f"{args.seeds} seeds × budget {args.budget}")
    print(f"  LLM mode: {'llm (DeepSeek)' if use_llm else 'mock (heuristic)'}")

    results: dict[str, list[dict]] = {name: [] for name in policies}
    t0 = time.monotonic()
    for policy_name, policy in policies.items():
        for family in args.families:
            for seed in range(args.seeds):
                r = run_one_trajectory(policy, family, args.budget, seed)
                results[policy_name].append(r)
                if r.get("best_y") is not None:
                    print(f"  [{policy_name}] {family}/seed{seed} "
                          f"best_y={r['best_y']:.4f} "
                          f"steps={r['n_steps']} "
                          f"reason={r.get('termination_reason','?')}")
                else:
                    print(f"  [{policy_name}] {family}/seed{seed} ERROR: "
                          f"{r.get('error','?')}")
    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Aggregate by (policy, family).
    summary = {}
    for policy_name, rows in results.items():
        summary[policy_name] = {}
        for family in args.families:
            fam_rows = [r for r in rows if r["family"] == family
                        and r.get("best_y") is not None]
            if fam_rows:
                best_ys = [r["best_y"] for r in fam_rows]
                summary[policy_name][family] = {
                    "n_seeds": len(best_ys),
                    "mean_best_y": statistics.fmean(best_ys),
                    "median_best_y": statistics.median(best_ys),
                    "min_best_y": min(best_ys),
                    "max_best_y": max(best_ys),
                }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "config": {
            "families": args.families,
            "seeds": args.seeds,
            "budget": args.budget,
            "llm_mode": "llm" if use_llm else "mock",
            "api_key_present": bool(api_key),
        },
        "raw": results,
        "summary": summary,
    }, indent=2, default=str) + "\n")

    # Print summary table.
    print("\n=== Summary (mean best_y across seeds) ===")
    print(f"{'family':15s} " + " ".join(f"{n:>22s}" for n in policies))
    for family in args.families:
        row = [family]
        for policy_name in policies:
            d = summary[policy_name].get(family, {})
            row.append(f"{d.get('mean_best_y', float('nan')):>22.4f}"
                       if d else "        n/a")
        print(" ".join(row))

    # Print the key comparison.
    print("\n=== Key question: LLM-System-1 vs AlwaysSurrogate ===")
    for family in args.families:
        llm_y = summary["LLMSystem1Policy"].get(family, {}).get("mean_best_y")
        surr_y = summary["AlwaysSurrogate"].get(family, {}).get("mean_best_y")
        rule_y = summary["RulePolicy"].get(family, {}).get("mean_best_y")
        if llm_y is None or surr_y is None:
            continue
        diff_llm_vs_surr = llm_y - surr_y
        diff_llm_vs_rule = llm_y - rule_y
        print(f"  {family:15s}: LLM={llm_y:+.4f}  Surrogate={surr_y:+.4f}  "
              f"Rule={rule_y:+.4f}  "
              f"Δ(L-S)={diff_llm_vs_surr:+.4f}  Δ(L-R)={diff_llm_vs_rule:+.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

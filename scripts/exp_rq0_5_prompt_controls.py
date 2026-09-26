"""EXP-RQ0.5 prompt/state controls (per codex round-4 priority #3).

Distinguishes "LLM picks global_exploration because of prompt prior"
from "LLM picks global_exploration because of state reasoning".

Controls:
  baseline:        real state → LLM
  no_state:        zero state → LLM (random-action baseline)
  shuffled_state:  state fields permuted → LLM (test sensitivity)
  state_swap:      state from a different trajectory → LLM (test responsiveness)

If LLM picks the same action across all controls → prompt prior.
If LLM's choice varies with state → state-sensitive reasoning.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from opt_agent.actions import OptimizationAction, candidate_actions_for_state
from opt_agent.state import OptimizationState
from opt_agent.system1.llm_system1_policy import LLMSystem1Policy
from opt_agent.system1._prompt import build_choice_request
from opt_agent.dataset import iter_jsonl


def get_state_at_step(dataset_path: str, family: str, seed: int, step: int):
    """Extract a specific (family, seed, step) state from the JSONL."""
    for r in iter_jsonl(dataset_path):
        if r.problem["family"] != family: continue
        if int(r.problem["seed"]) != seed: continue
        # Phase_of ratios aren't per-step; just pick by record id.
        if r.id.endswith(f":{step}"):
            return r
    return None


def make_state_minimal() -> OptimizationState:
    """Return a synthetic state dict — used for no-state baseline.

    Note: OptimizationState is frozen + has 5 sub-dataclasses. Build it
    with all fields filled but values set to neutral placeholders.
    """
    from opt_agent.state.models import (
        OptimizationState, SamplingState, SurrogateState,
        OptimizationProgress, ResourceState, HistoryState,
    )
    return OptimizationState(
        sampling=SamplingState(n_samples=5, dimension=2, min_pairwise_distance=0.0,
                               boundary_coverage=0.0),
        surrogate=SurrogateState(is_current=False, validation_rmse=0.0,
                                 mean_uncertainty=0.0, max_uncertainty=0.0),
        optimization=OptimizationProgress(best_value=0.0, last_value=0.0,
                                          recent_improvement=0.0,
                                          stagnation_steps=0),
        resource=ResourceState(used_budget=0, total_budget=30,
                                remaining_budget=30,
                                remaining_budget_ratio=1.0),
        history=HistoryState(last_action=None),
    )


def shuffle_state(state: OptimizationState, seed: int = 0) -> OptimizationState:
    """Permute numeric fields within each sub-dataclass."""
    from opt_agent.state.models import (
        OptimizationState, SamplingState, SurrogateState,
        OptimizationProgress, ResourceState, HistoryState,
    )
    rng = random.Random(seed)
    s = state.sampling
    su = state.surrogate
    o = state.optimization
    r = state.resource
    h = state.history
    return OptimizationState(
        sampling=SamplingState(
            n_samples=s.n_samples,
            dimension=s.dimension,
            min_pairwise_distance=su.mean_uncertainty,
            boundary_coverage=o.recent_improvement,
        ),
        surrogate=SurrogateState(
            is_current=su.is_current,
            validation_rmse=o.best_value,
            mean_uncertainty=s.min_pairwise_distance,
            max_uncertainty=r.remaining_budget_ratio,
        ),
        optimization=OptimizationProgress(
            best_value=su.validation_rmse or 0.0,
            last_value=su.max_uncertainty,
            recent_improvement=r.remaining_budget_ratio,
            stagnation_steps=h is not None and 1 or 0,
        ),
        resource=ResourceState(
            used_budget=r.used_budget,
            total_budget=r.total_budget,
            remaining_budget=s.n_samples,
            remaining_budget_ratio=o.last_value if o.last_value else 0.0,
        ),
        history=h,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "runs" / "phase8_6d_h3_10seed_5family.jsonl")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of (family, seed, step) tuples to test.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs" / "exp_rq0_5_prompt_controls.json")
    args = parser.parse_args()

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

    # Sample (family, seed, step) tuples from the dataset.
    families = ["Branin", "Hartmann", "Ackley", "SixHumpCamel", "Rastrigin"]
    samples = []
    for r in iter_jsonl(args.dataset):
        if r.problem["family"] not in families: continue
        samples.append(r)
        if len(samples) >= args.n_samples * 5:
            break
    rng = random.Random(42)
    rng.shuffle(samples)
    samples = samples[: args.n_samples]

    llm = LLMSystem1Policy(mode="llm", seed=0,
                            api_key=os.environ.get("DEEPSEEK_API_KEY"))

    results = []
    t0 = time.monotonic()
    for r in samples:
        # Reconstruct OptimizationState from the dataset record.
        state_dict = r.state
        from opt_agent.state.models import (
            OptimizationState, SamplingState, SurrogateState,
            OptimizationProgress, ResourceState, HistoryState,
        )
        # Convert NaN/inf to None in raw dict, then build state.
        def _f(x, default=0.0):
            try:
                v = float(x)
                if v != v or v in (float("inf"), float("-inf")):
                    return default
                return v
            except (TypeError, ValueError):
                return default
        sampling = state_dict.get("sampling", {})
        surrogate = state_dict.get("surrogate", {})
        optimization = state_dict.get("optimization", {})
        resource = state_dict.get("resource", {})
        history = state_dict.get("history", {})
        try:
            real_state = OptimizationState(
                sampling=SamplingState(
                    n_samples=int(sampling.get("n_samples", 0) or 0),
                    dimension=int(sampling.get("dimension", 0) or 0),
                    min_pairwise_distance=_f(sampling.get("min_pairwise_distance")),
                    boundary_coverage=_f(sampling.get("boundary_coverage")),
                ),
                surrogate=SurrogateState(
                    is_current=bool(surrogate.get("is_current", False)),
                    validation_rmse=_f(surrogate.get("validation_rmse")),
                    mean_uncertainty=_f(surrogate.get("mean_uncertainty")),
                    max_uncertainty=_f(surrogate.get("max_uncertainty")),
                ),
                optimization=OptimizationProgress(
                    best_value=_f(optimization.get("best_value")),
                    last_value=_f(optimization.get("last_value")),
                    recent_improvement=_f(optimization.get("recent_improvement")),
                    stagnation_steps=int(optimization.get("stagnation_steps", 0) or 0),
                ),
                resource=ResourceState(
                    used_budget=int(resource.get("used_budget", 0) or 0),
                    total_budget=int(resource.get("total_budget", 30) or 30),
                    remaining_budget=int(resource.get("remaining_budget", 0) or 0),
                    remaining_budget_ratio=_f(resource.get("remaining_budget_ratio")),
                ),
                history=HistoryState(last_action=history.get("last_action")),
            )
        except Exception as exc:
            results.append({"record_id": r.id, "error": f"state_build: {exc!r}"})
            continue

        no_state = make_state_minimal()
        shuf_state = shuffle_state(real_state, seed=hash(r.id) % 1000)

        # 4 conditions per record.
        for cond_name, st in [("baseline", real_state),
                              ("no_state", no_state),
                              ("shuffled", shuf_state)]:
            try:
                choices = candidate_actions_for_state(st, seed=0)
                state_arg, question = build_choice_request(
                    st, choices, state_format="structured",
                    criteria_format="executable",
                )
                llm.reset()
                # Bypass decide() to record only this single decision.
                from opt_agent.system1._prompt import validate_choice_response
                import httpx
                import json
                messages = [
                    {"role": "system", "content":
                     "You are an expert Bayesian-Optimization decision policy."
                     " Choose exactly one action_id from: "
                     + ", ".join(choices.keys())
                     + ". Respond with JSON only: "
                     + '{"action_id": "<one of those>", "rationale": "<short>"}'},
                    {"role": "user",
                     "content": f"State:\n{state_arg}\n\nQuestion:\n{question}"},
                ]
                body = {"model": llm.model, "messages": messages,
                        "temperature": llm.temperature, "max_tokens": llm.max_tokens}
                with httpx.Client(timeout=llm.timeout_s) as c:
                    r_resp = c.post(
                        f"{llm.base_url}/chat/completions",
                        json=body,
                        headers={"Authorization": f"Bearer {llm.api_key or ''}"},
                    )
                r_resp.raise_for_status()
                body_json = r_resp.json()
                parsed = LLMSystem1Policy._parse_openai_response(
                    body_json, candidates=choices
                )
                sel, probs, conf = validate_choice_response(parsed, choices)
                results.append({
                    "record_id": r.id,
                    "family": r.problem["family"],
                    "phase": r.phase,
                    "condition": cond_name,
                    "selected": sel,
                    "probs": probs,
                })
            except Exception as exc:
                results.append({"record_id": r.id, "condition": cond_name,
                                "error": repr(exc)})
    elapsed = time.monotonic() - t0

    # Aggregate: per family, distribution of selections per condition.
    agg = {}
    for r in results:
        if "error" in r or "selected" not in r: continue
        key = (r.get("family", "?"), r["condition"])
        agg.setdefault(key, []).append(r["selected"])

    summary = {}
    for key, sels in agg.items():
        c = Counter(sels)
        summary[str(key)] = {
            "n": len(sels),
            "selection_counts": dict(c),
            "selection_distribution": {k: v / len(sels) for k, v in c.items()},
            "top_action": c.most_common(1)[0][0],
            "top_action_share": c.most_common(1)[0][1] / len(sels),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "config": {"n_samples": args.n_samples},
        "results": results,
        "summary": summary,
        "elapsed_seconds": elapsed,
    }, indent=2, default=str) + "\n")
    print(f"Saved: {args.output}")

    print()
    print("=== Selection distribution per (family, condition) ===")
    print(f"{'(family, condition)':40s} top_action  share")
    for key in sorted(summary):
        s = summary[key]
        print(f"{key:40s} {s['top_action']:>12s}  {s['top_action_share']:.2%}")

    print()
    print("=== Cross-condition agreement ===")
    by_record = {}
    for r in results:
        if "error" in r or "selected" not in r: continue
        by_record.setdefault(r["record_id"], {})[r["condition"]] = r["selected"]
    agree = sum(1 for cs in by_record.values()
                if cs.get("baseline") == cs.get("shuffled"))
    n = sum(1 for cs in by_record.values() if "shuffled" in cs)
    if n:
        print(f"  baseline vs shuffled: {agree}/{n} = {agree/n:.1%} agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit actual LLM calls/tokens in the existing BOCoDe joint experiment.

This is a development-panel cost audit, not a confidence-router test. The
dimension router failed the separate held-out IOH performance check.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = ROOT / "runs/exp_rq0_5_joint_10seed.json"
    results = json.loads(source.read_text())["results"]
    by_policy = defaultdict(list)
    for row in results:
        by_policy[row["policy"]].append(row)
    full = by_policy["LLMDeepSeek"]
    router = by_policy["RoutingAware"]
    if len(full) != len(router) or not full:
        raise ValueError("missing paired LLM/router runs")
    keys = lambda rows: {(r["family"], r["seed"]) for r in rows}
    if keys(full) != keys(router):
        raise ValueError("LLM/router problem-seed panels differ")
    full_calls = sum(int(r["n_calls"]) for r in full)
    router_calls = sum(sum(step.get("route") == "llm" for step in r["routing_log"])
                       for r in router)
    if router_calls <= 0 or router_calls >= full_calls:
        raise ValueError("unexpected route counts")
    full_tokens = sum(int(r["n_input_tokens"]) + int(r["n_output_tokens"])
                      for r in full)
    router_tokens = sum(int(r["n_input_tokens"]) + int(r["n_output_tokens"])
                        for r in router)
    by_key = {(r["family"], r["seed"], r["policy"]): r for r in results}
    per_family = {}
    for family in sorted({r["family"] for r in full}):
        diffs = [by_key[family, seed, "RoutingAware"]["best_y"]
                 - by_key[family, seed, "LLMDeepSeek"]["best_y"]
                 for seed in sorted({r["seed"] for r in full if r["family"] == family})]
        per_family[family] = {
            "n_paired_seeds": len(diffs),
            "mean_router_minus_llm_best_y": statistics.fmean(diffs),
        }
    payload = {
        "source": str(source.relative_to(ROOT)),
        "panel": "BOCoDe development panel, 5 families x 10 seeds, budget 30",
        "full_llm_calls": full_calls,
        "dimension_router_llm_calls": router_calls,
        "llm_call_reduction": 1 - router_calls / full_calls,
        "full_llm_tokens": full_tokens,
        "dimension_router_tokens": router_tokens,
        "token_reduction": 1 - router_tokens / full_tokens,
        "per_family": per_family,
        "limitation": "The dimension router failed to show a gain over AlwaysSurrogate "
                      "on the held-out IOH panel; this cost saving is development-panel only.",
    }
    output = ROOT / "runs/exp_rq3_existing_cost_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

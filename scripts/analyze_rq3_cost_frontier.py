"""Merge RQ3 router-sweep files and compute paired cost-frontier statistics.

Combines the original 3-seed sweep (seeds 100–102) with later seed expansions
into one paired sample, verifies that every policy shares identical Sobol
starting evaluations per (family, seed), and reports per-family paired
bootstrap comparisons of every LLM-calling arm against the frozen System-1
(`rf`) baseline.

Example:
  uv run python scripts/analyze_rq3_cost_frontier.py \
      runs/exp_rq3_router_sweep.json runs/exp_rq3_router_sweep_seeds103_109.json \
      --output runs/exp_rq3_cost_frontier_10seed.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Config fields that must be identical across merged files: the frozen
# System-1 (dataset + training) and the System-2 backend must not change.
_MUST_MATCH = ("dataset_sha256", "training", "budget", "system2_mode", "system2_model")


def load_and_merge(paths: list[Path]) -> tuple[dict, list[dict]]:
    configs, runs = [], []
    for path in paths:
        data = json.loads(path.read_text())
        configs.append(data["config"])
        runs.extend(data["runs"])
    base = configs[0]
    for other in configs[1:]:
        for field in _MUST_MATCH:
            if other.get(field) != base.get(field):
                raise ValueError(f"config field {field!r} differs across input files")
    if base["system2_mode"] != "real":
        raise ValueError("refusing to analyze mock System-2 runs as evidence")
    seen: dict[tuple, dict] = {}
    for row in runs:
        key = (row["family"], row["seed"], row["policy"])
        if key in seen and seen[key] != row:
            raise ValueError(f"conflicting duplicate row for {key}")
        seen.setdefault(key, row)
    merged = sorted(seen.values(), key=lambda r: (r["family"], r["seed"], r["policy"]))
    return base, merged


def check_paired_starts(runs: list[dict]) -> list[str]:
    problems = []
    starts: dict[tuple, list] = defaultdict(list)
    for row in runs:
        starts[(row["family"], row["seed"])].append((row["policy"], row["initial_values"]))
    for (family, seed), group in sorted(starts.items()):
        values = {tuple(v) for _, v in group}
        if len(values) != 1:
            problems.append(f"{family} s{seed}: {len(values)} distinct initial_values "
                            f"across {len(group)} policies")
    return problems


def paired_bootstrap(diffs: list[float], n_boot: int = 10_000,
                     seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    n = len(diffs)
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(int(0.975 * n_boot), n_boot - 1)]
    return lo, hi


def analyze(base_config: dict, runs: list[dict]) -> dict:
    by_cell: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for row in runs:
        by_cell[(row["family"], row["policy"])][row["seed"]] = row
    families = base_config["families"]
    policies = base_config["policies"]
    sigma = {}
    for family in families:
        values = [r["terminal_best"] for r in runs if r["family"] == family]
        sigma[family] = statistics.stdev(values) if len(values) > 1 else float("nan")
    comparisons = {}
    for family in families:
        rf = by_cell.get((family, "rf"), {})
        for policy in policies:
            if policy == "rf":
                continue
            arm = by_cell.get((family, policy), {})
            common = sorted(set(rf) & set(arm))
            if not common:
                continue
            diffs = [arm[s]["terminal_best"] - rf[s]["terminal_best"] for s in common]
            lo, hi = paired_bootstrap(diffs)
            # W/T/L from the rf baseline's perspective (rf beats arm / tie / loses).
            rf_wins = sum(1 for d in diffs if d < 0)
            ties = sum(1 for d in diffs if d == 0)
            comparisons[f"{family}/{policy}"] = {
                "n_paired_seeds": len(common), "seeds": common,
                "mean_delta": statistics.fmean(diffs),
                "ci95": [lo, hi], "wtl_rf_vs_arm": [rf_wins, ties, len(diffs) - rf_wins - ties],
                "ci_excludes_zero_positive": lo > 0,
                "ci_excludes_zero_negative": hi < 0,
                "mean_calls_delta": statistics.fmean(
                    arm[s]["llm_calls"] - rf[s]["llm_calls"] for s in common),
                "mean_tokens_delta": statistics.fmean(
                    arm[s]["llm_tokens"] - rf[s]["llm_tokens"] for s in common),
                "regret_margin_ok": statistics.fmean(diffs) <= 0.10 * sigma[family],
            }
    summary = {}
    for (family, policy), cell in sorted(by_cell.items()):
        rows = list(cell.values())
        summary[f"{family}/{policy}"] = {
            "n_seeds": len(rows),
            "mean_terminal_best": statistics.fmean(r["terminal_best"] for r in rows),
            "mean_llm_calls": statistics.fmean(r["llm_calls"] for r in rows),
            "mean_llm_tokens": statistics.fmean(r["llm_tokens"] for r in rows),
            "total_fallbacks": sum(r["llm_fallbacks"] for r in rows),
        }
    return {"sigma_family": sigma, "summary": summary, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/exp_rq3_cost_frontier_merged.json")
    args = parser.parse_args()
    base, runs = load_and_merge(args.inputs)
    problems = check_paired_starts(runs)
    report = {
        "inputs": [str(p) for p in args.inputs],
        "config": {k: base[k] for k in _MUST_MATCH},
        "n_runs": len(runs),
        "sigma_definition": "sample stdev of terminal_best pooled over all "
                            "policies and seeds within a family",
        "paired_start_violations": problems,
        **analyze(base, runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    seeds = sorted({r["seed"] for r in runs})
    print(f"merged {len(runs)} runs over seeds {seeds} "
          f"({len(problems)} paired-start violations)")
    print(f"sigma_family: " + ", ".join(f"{k}={v:.3f}" for k, v in report["sigma_family"].items()))
    for key, comp in report["comparisons"].items():
        lo, hi = comp["ci95"]
        w, t, l = comp["wtl_rf_vs_arm"]
        flag = ("+" if comp["ci_excludes_zero_positive"]
                else "-" if comp["ci_excludes_zero_negative"] else " ")
        print(f"{flag} {key:28s} n={comp['n_paired_seeds']:2d} "
              f"meanΔ={comp['mean_delta']:+8.3f} CI[{lo:+8.3f},{hi:+8.3f}] "
              f"W/T/L={w}/{t}/{l} callsΔ={comp['mean_calls_delta']:+5.1f} "
              f"margin_ok={comp['regret_margin_ok']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    sys.exit(main())

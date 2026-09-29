"""EXP-META-CONFIG-SCREEN — meta-level pre-screen (guide §10.3 / codex E2).

Question (predeclared BEFORE running): before building any meta-level
supervision pipeline (configuration-level counterfactuals), is there
DETECTABLE configuration-conditionality on the current panel? Guide §4.4
warns the meta-level advantage can be nonpositive; this screen is the cheap
paired test the round-1 review asked for.

Design:
  - Config axis: initial design size n_init ∈ {5 ("small", the harness
    default), 12 ("large")} — a design-phase configuration per guide §1.2
    (初始设计/预算切分), the only meta-level knob the current backend
    exposes without new model code. Total budget FIXED at 30 evaluations
    for both arms (same spend; different split between design and BO).
  - Inner policy pinned to AlwaysSurrogate in both arms (isolates the
    config effect from policy effects).
  - Pairing: same family + same seed → same scrambled Sobol stream
    (nested: the first 5 points of draw(12) equal draw(5)).
  - Endpoint: terminal best objective (backend MAXIMIZES).

Readouts:
  1. Per-family paired advantage Δ_f = E_seed[y_large − y_small] with
     seed-level bootstrap CI.
  2. Cross-family meta-D on family-standardized utilities:
     ū_{a,f} = mean_seed[y_{a,f}] / s_f (s_f = pooled std of both arms),
     D_meta = E_f[max_a ū_{a,f}] − max_a E_f[ū_{a,f}]. With two configs
     this equals (E_f|d_f| − |E_f d_f|)/2 — it is > 0 iff the per-family
     advantage varies across families beyond its mean (interaction).
  3. Bootstrap over seeds within family for both.

Predeclared decision rule:
  - SCREEN POSITIVE (config-conditionality exists → meta pipeline justified
    on this axis) if ≥2 families have CI-excluding-zero Δ_f with OPPOSITE
    signs, or D_meta's bootstrap CI excludes zero.
  - SCREEN NEGATIVE otherwise: no detectable meta advantage at this
    axis/panel → per guide §4.4, do NOT build meta supervision on this
    axis; report and stop.

Hardware: CPU-bound trajectories ~10 s each; 5 families × 20 seeds × 2
configs = 200 trajectories, parallel pool.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.actions import candidate_actions_for_state
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend
from opt_agent.loop.decision_loop import DecisionLoop

CONFIGS = {"small": 5, "large": 12}


def _worker_init() -> None:
    """Cap torch threads per worker (oversubscription guard, repo precedent:
    the longhorizon driver hit load 315 with default threads)."""
    import torch

    torch.set_num_threads(1)


class AlwaysSurrogatePolicy:
    """Pinned inner policy: always SURROGATE_ACQUISITION."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def select_action(self, state, backend=None):
        choices = candidate_actions_for_state(state, seed=self.seed)
        if "surrogate_acquisition" in choices:
            return choices["surrogate_acquisition"]
        return next(iter(choices.values()))


def run_one_trajectory(family: str, config_name: str, budget: int,
                       seed: int) -> dict:
    """Run one (family, config, seed) episode; return terminal best + n_evals."""
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(
        problem_id=family,
        seed=seed,
        n_init=CONFIGS[config_name],
    ))
    loop = DecisionLoop(backend, total_budget=budget)
    loop.run(AlwaysSurrogatePolicy(seed=0), max_rounds=budget + 5)
    history = backend.get_history()
    return {
        "best": float(history.objectives[:, 0].max().item()),
        "n_evals": len(history),
        "termination": (
            loop.termination_reason.value if loop.termination_reason else None
        ),
    }


def paired_bootstrap_ci(diffs: list[float], n_boot: int = 5000,
                        seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return [lo, hi]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+",
                        default=["Branin", "Hartmann", "Ackley",
                                 "SixHumpCamel", "Rastrigin"])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs" / "exp_meta_config_screen.json")
    args = parser.parse_args()

    jobs = [(fam, cfg, args.budget, seed)
            for fam in args.families
            for seed in range(args.seeds)
            for cfg in CONFIGS]
    print(f"Total jobs: {len(jobs)} = {len(args.families)} families × "
          f"{args.seeds} seeds × {len(CONFIGS)} configs {sorted(CONFIGS)}")
    print(f"Budget {args.budget} (design+BO), pinned AlwaysSurrogate policy")

    import multiprocessing as mp
    t0 = time.monotonic()
    with mp.Pool(args.workers, initializer=_worker_init) as pool:
        results = pool.starmap(run_one_trajectory, jobs)
    elapsed = time.monotonic() - t0
    print(f"Done in {elapsed:.1f}s ({len(jobs)/elapsed:.1f} trajectories/s)")

    # Reshape: results follow the job order (family, seed, config).
    by_key: dict[tuple, dict] = {}
    idx = 0
    for fam in args.families:
        for seed in range(args.seeds):
            for cfg in CONFIGS:
                by_key[(fam, seed, cfg)] = results[idx]
                idx += 1

    # Sanity: both arms spent the same budget.
    n_evals = {r["n_evals"] for r in results}
    print(f"n_evals observed: {sorted(n_evals)}")

    # 1. Per-family paired advantage.
    per_family = {}
    for fam in args.families:
        diffs = [by_key[(fam, s, "large")]["best"] - by_key[(fam, s, "small")]["best"]
                 for s in range(args.seeds)]
        mean = statistics.fmean(diffs)
        ci = paired_bootstrap_ci(diffs, n_boot=args.n_boot)
        per_family[fam] = {
            "mean_diff_large_minus_small": mean,
            "ci95": ci,
            "ci_excludes_zero": (ci[0] > 0) or (ci[1] < 0),
            "per_seed_diffs": diffs,
            "best_config_by_mean": ("large" if mean > 0 else "small"),
        }

    # 2. Cross-family meta-D on standardized utilities.
    fam_stats = {}
    for fam in args.families:
        y_small = [by_key[(fam, s, "small")]["best"] for s in range(args.seeds)]
        y_large = [by_key[(fam, s, "large")]["best"] for s in range(args.seeds)]
        pooled = y_small + y_large
        s_f = statistics.pstdev(pooled)
        fam_stats[fam] = {
            "u_small": statistics.fmean(y_small) / s_f,
            "u_large": statistics.fmean(y_large) / s_f,
            "s_f": s_f,
        }
    u = {fam: {"small": st["u_small"], "large": st["u_large"]}
         for fam, st in fam_stats.items()}

    def d_meta(u_map: dict) -> float:
        fams = list(u_map)
        e_max = statistics.fmean(max(u_map[f][a] for a in CONFIGS) for f in fams)
        max_e = max(statistics.fmean(u_map[f][a] for f in fams) for a in CONFIGS)
        return e_max - max_e

    d_meta_point = d_meta(u)

    # Bootstrap: resample seeds within each family, recompute u and D_meta.
    rng = random.Random(0)
    boots = []
    for _ in range(args.n_boot):
        u_b = {}
        for fam in args.families:
            picks = [rng.randrange(args.seeds) for _ in range(args.seeds)]
            ys = [by_key[(fam, p, "small")]["best"] for p in picks]
            yl = [by_key[(fam, p, "large")]["best"] for p in picks]
            pooled = ys + yl
            s_b = statistics.pstdev(pooled) or 1.0
            u_b[fam] = {"small": statistics.fmean(ys) / s_b,
                        "large": statistics.fmean(yl) / s_b}
        boots.append(d_meta(u_b))
    boots.sort()
    d_meta_ci = [boots[int(0.025 * args.n_boot)],
                 boots[int(0.975 * args.n_boot) - 1]]

    # 3. Predeclared decision rule.
    sig = [f for f in args.families if per_family[f]["ci_excludes_zero"]]
    pos = [f for f in sig if per_family[f]["mean_diff_large_minus_small"] > 0]
    neg = [f for f in sig if per_family[f]["mean_diff_large_minus_small"] < 0]
    opposite = bool(pos) and bool(neg)
    d_meta_sig = (d_meta_ci[0] > 0) or (d_meta_ci[1] < 0)
    verdict = "POSITIVE" if (opposite or d_meta_sig) else "NEGATIVE"

    payload = {
        "experiment": "exp_meta_config_screen",
        "predeclared_rule": (
            "POSITIVE iff (>=2 families CI-significant in opposite directions) "
            "or (meta-D bootstrap CI excludes zero); else NEGATIVE"
        ),
        "families": args.families,
        "seeds": args.seeds,
        "budget": args.budget,
        "configs": CONFIGS,
        "policy": "AlwaysSurrogate (pinned)",
        "n_evals_observed": sorted(n_evals),
        "per_family": {
            fam: {k: v for k, v in row.items() if k != "per_seed_diffs"}
            for fam, row in per_family.items()
        },
        "meta_d": {
            "point": d_meta_point,
            "ci95": d_meta_ci,
            "ci_excludes_zero": d_meta_sig,
            "per_family_u": fam_stats,
            "note": "D_meta = E_f[max_a u_af] - max_a E_f[u_af] on "
                    "family-standardized terminal best; >0 iff per-family "
                    "config advantage varies beyond its cross-family mean",
        },
        "significant_families": sig,
        "opposite_direction_families": {"large_wins": pos, "small_wins": neg},
        "verdict": verdict,
        "wall_seconds": elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in
                      ("per_family", "meta_d", "significant_families",
                       "opposite_direction_families", "verdict")},
                     indent=2, default=str))
    print(f"\nVERDICT: {verdict}  (artifact: {args.output})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Proper paired-t statistics with Student-t distribution and CIs.

Per codex round-6 review, replaces normal-approximation p-values
(which overstate significance for n<30) with Student-t.
Adds 95% confidence intervals and Holm-adjustment for multiple
comparisons.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats as sp_stats


def paired_t(diffs):
    """Proper paired t-test with Student-t distribution + 95% CI."""
    n = len(diffs)
    if n < 2:
        return {"n": n, "mean": float("nan"), "se": float("nan"),
                "t": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p_two_sided": float("nan")}
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(n)
    t = mean / se
    ci_lo = mean - sp_stats.t.ppf(0.975, df=n - 1) * se
    ci_hi = mean + sp_stats.t.ppf(0.975, df=n - 1) * se
    p = 2.0 * (1.0 - sp_stats.t.cdf(abs(t), df=n - 1))
    return {"n": n, "mean": mean, "se": se, "t": t,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "p_two_sided": p}


def holm_adjust(p_values):
    """Holm-Bonferroni step-down correction for multiple comparisons.

    Returns adjusted p-values in original order. Adjusted p-values are
    monotonically non-decreasing; any adjusted p > 1 is clamped to 1.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    adjusted = [0.0] * n
    prev = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = max(prev, (n - rank) * p)
        adjusted[orig_idx] = min(adj, 1.0)
        prev = adj
    return adjusted


def main():
    # ----- IOH analytical router results -----
    print("=" * 72)
    print("EXP-RQ0.5-H: IOH analytical FrozenRouter (corrected Student-t)")
    print("=" * 72)

    ar = json.load(open("runs/exp_rq0_5_analytical_router.json"))
    router = ar["router"]
    fixed = json.load(open("runs/exp_rq0_5_ioh_crossed_fixed.json"))

    # Group Surrogate / LLM trajectories by (fid, dim, instance, seed).
    by_key = defaultdict(dict)
    for r in fixed["raw"]:
        k = (r["fid"], r["dim"], r["instance"], r["seed"])
        by_key[k][r["policy"]] = r["normalized_regret"]

    router_by_key = {(r["fid"], r["dim"], r["instance"], r["seed"]):
                     r["normalized_regret"] for r in router
                     if r.get("normalized_regret") is not None}

    diff_sur, diff_llm = [], []
    for k, ps in by_key.items():
        if "Surrogate" not in ps or "LLM" not in ps:
            continue
        router_v = router_by_key.get(k)
        if router_v is None:
            continue
        diff_sur.append(router_v - ps["Surrogate"])
        diff_llm.append(router_v - ps["LLM"])

    print(f"\n=== IOH analytical router (n={len(diff_sur)} paired) ===")
    for name, diffs in [("Router − Surrogate", diff_sur),
                        ("Router − LLM",       diff_llm)]:
        s = paired_t(diffs)
        print(f"  {name:25s}  mean={s['mean']:+.4f}  "
              f"95% CI=[{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}]  "
              f"t={s['t']:+.3f}  p={s['p_two_sided']:.4f}  n={s['n']}")

    # Holm adjustment for 2 router contrasts.
    p_sur = paired_t(diff_sur)["p_two_sided"]
    p_llm = paired_t(diff_llm)["p_two_sided"]
    holm = holm_adjust([p_sur, p_llm])
    print(f"\n  Holm-adjusted (2 contrasts): Router−Surrogate p_adj={holm[0]:.4f}; "
          f"Router−LLM p_adj={holm[1]:.4f}")

    # ----- BOCoDe joint 10-seed 5-policy results -----
    print()
    print("=" * 72)
    print("EXP-RQ0.5-BOCoDe: 10-seed 4-policy (corrected Student-t)")
    print("=" * 72)

    bmain = json.load(open("runs/exp_rq0_5_joint_10seed.json"))
    bres = bmain["results"]
    families = ["Branin", "Hartmann", "Ackley", "SixHumpCamel", "Rastrigin"]

    # Per-family paired t for each comparison.
    print(f"\n=== Per-family paired-t (LLM vs Jev; LLMSurrogate is the same since Jev≡Surrogate) ===")
    print(f"{'family':14s} {'mean':>10s} {'ci_lo':>10s} {'ci_hi':>10s} "
          f"{'t':>9s} {'p':>10s}")
    for fam in families:
        diffs = []
        for s in range(10):
            llm_y = next((r["best_y"] for r in bres
                          if r["family"] == fam and r["seed"] == s
                          and r["policy"] == "LLMDeepSeek"), None)
            jev_y = next((r["best_y"] for r in bres
                          if r["family"] == fam and r["seed"] == s
                          and r["policy"] == "Jev"), None)
            if llm_y is None or jev_y is None:
                continue
            diffs.append(llm_y - jev_y)
        s = paired_t(diffs)
        print(f"{fam:14s} {s['mean']:>+10.3f} {s['ci_lo']:>+10.3f} {s['ci_hi']:>+10.3f} "
              f"{s['t']:>+9.3f} {s['p_two_sided']:>10.4f}")

    # Aggregate per-seed TOTAL.
    print(f"\n=== BOCoDe TOTAL (sum across 5 families) paired-t ===")
    for pair in [("LLMDeepSeek", "Jev"),
                 ("LLMDeepSeek", "AlwaysSurrogate"),
                 ("RoutingAware", "AlwaysSurrogate"),
                 ("RoutingAware", "LLMDeepSeek"),
                 ("RoutingAware", "Jev")]:
        diffs = []
        for s in range(10):
            sum_p = sum_p_b = 0.0
            for fam in families:
                p_a = next((r["best_y"] for r in bres
                            if r["family"] == fam and r["seed"] == s
                            and r["policy"] == pair[0]), None)
                p_b = next((r["best_y"] for r in bres
                            if r["family"] == fam and r["seed"] == s
                            and r["policy"] == pair[1]), None)
                if p_a is None or p_b is None:
                    sum_p = sum_p_b = 0.0
                    break
                sum_p += p_a
                sum_p_b += p_b
            else:
                diffs.append(sum_p - sum_p_b)
                continue
            break
        s = paired_t(diffs)
        print(f"  {pair[0]:14s} − {pair[1]:18s}  mean={s['mean']:>+10.3f}  "
              f"CI=[{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}]  "
              f"t={s['t']:+.3f}  p={s['p_two_sided']:.4f}  n={s['n']}")

    # ----- Normalized BOCoDe aggregate (per-family, then mean) -----
    print()
    print("=" * 72)
    print("BOCoDe NORMALIZED aggregate (per-family z-score, then mean)")
    print("=" * 72)
    # z-score per family using y_init_best and (rough) y_opt from problem.
    # We don't have y_opt in BOCoDe; use family-level best_y from RulePolicy as
    # an empirical approximation (or skip; this is post-hoc).
    # We'll do simple within-family standardization (subtract mean, divide std)
    # over seed-level means per policy.
    print("\n(simple within-family std-normalized mean per policy, then mean across families):")
    policy_family_means = defaultdict(lambda: defaultdict(list))
    for fam in families:
        for p in ["RulePolicy", "AlwaysSurrogate", "LLMDeepSeek", "Jev", "RoutingAware"]:
            for s in range(10):
                ys = [r["best_y"] for r in bres
                      if r["family"] == fam and r["seed"] == s
                      and r["policy"] == p and r["best_y"] == r["best_y"]]
                if ys:
                    policy_family_means[p][fam].append(statistics.fmean(ys))
    # Standardize per family.
    norm_scores = defaultdict(dict)
    for p in policy_family_means:
        for fam in policy_family_means[p]:
            vals = policy_family_means[p][fam]
            if len(vals) < 2: continue
            m = statistics.fmean(vals)
            sd = statistics.stdev(vals)
            if sd == 0: continue
            for v in vals:
                norm_scores[p][fam] = (v - m) / sd
    # Mean across families per policy (each family contributes one std-normalized value).
    print(f"\n{'policy':18s} {'mean_norm':>12s}")
    for p in sorted(norm_scores):
        mean_z = statistics.fmean(list(norm_scores[p].values()))
        print(f"{p:18s} {mean_z:>12.4f}")


if __name__ == "__main__":
    main()

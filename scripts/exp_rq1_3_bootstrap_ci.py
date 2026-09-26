"""EXP-RQ1.3 step 2 — Seed-cluster bootstrap CIs.

Per codex round-2 review: 10 seeds per family are too few for paper-level
claims. Compute bootstrap CIs over the 10 seed-level means.

Output: per-family CI for ΔU and R_sur under family_median scale.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def bootstrap_ci_seed_clustered(seed_means: list[float], *,
                                repetitions: int = 2000,
                                seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap 95% CI for the mean of seed-level means.

    Resamples seed means with replacement and computes mean-of-means each
    iteration. Returns (mean, ci_lo, ci_hi).
    """
    import random
    if not seed_means:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(seed_means)
    means = sorted(
        statistics.fmean(seed_means[rng.randrange(n)] for _ in range(n))
        for _ in range(repetitions)
    )
    return (statistics.fmean(seed_means),
            means[int(0.025 * repetitions)],
            means[int(0.975 * repetitions)])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sensitivity_json", type=Path,
                        help="Output of exp_rq1_3_scale_sensitivity.py")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs" / "exp_rq1_3_bootstrap_ci.json")
    args = parser.parse_args()

    data = json.loads(args.sensitivity_json.read_text())
    schemes = data["results_by_scheme"]

    out = {}
    for scheme, scheme_data in schemes.items():
        seed_eq = scheme_data.get("seed_equal_mean", {})
        out[scheme] = {}
        for fam, metrics in seed_eq.items():
            out[scheme][fam] = {}
            for key in ("ΔU", "R_sur", "R_stop"):
                # metrics[key] is now {mean_of_seed_means, n_seeds, ...}
                # We need raw seed-level means. Reconstruct from
                # aggregate stats would lose info. So just use the
                # mean and report no CI for now — we need seed-level data.
                # Use the stored n_seeds to estimate SEM via delta-method
                # approximation:
                # SEM ≈ |mean| / sqrt(n_seeds) * fudge factor (rough).
                # Better: re-extract seed means from the JSON's per_record.
                # For now, return what we have:
                if key in metrics:
                    out[scheme][fam][key] = {
                        "n_seeds": metrics.get("n_seeds"),
                        "mean_of_seed_means": metrics.get("mean_of_seed_means"),
                    }
    # For bootstrap we need raw seed means; re-extract from per_record.
    per_record = data.get("per_record_canonical_scheme", [])
    by_seed: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in per_record:
        if r.get("phase") == "terminal":
            continue
        by_seed[(r["family"], r["seed"])].append(r)

    bootstrap = {}
    for fam in sorted({r["family"] for r in per_record}):
        bootstrap[fam] = {}
        seeds = sorted({s for (f, s) in by_seed if f == fam})
        for key in ("ΔU", "R_sur"):
            seed_means = []
            for s in seeds:
                vals = [r[key] for r in by_seed[(fam, s)]
                        if math.isfinite(r[key])]
                if vals:
                    seed_means.append(statistics.fmean(vals))
            if seed_means:
                mean, lo, hi = bootstrap_ci_seed_clustered(seed_means)
                bootstrap[fam][key] = {
                    "n_seeds": len(seed_means),
                    "seed_means": seed_means,
                    "mean": mean,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "ci_excludes_zero": lo > 0,
                }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "input": str(args.sensitivity_json),
        "bootstrap_family_median": bootstrap,
        "interpretation": (
            "For each (family, key), the 95% CI is over the bootstrap distribution "
            "of the mean across N=10 seeds. ci_excludes_zero=True means we can "
            "reject H0: μ=0 at α=0.05."
        ),
    }, indent=2) + "\n")

    print("=== Seed-cluster bootstrap 95% CIs (under family_median scale) ===")
    print(f"{'family':14s} {'key':6s} {'mean':>10s} {'ci_lo':>10s} {'ci_hi':>10s} "
          f"{'excl_0':>8s}")
    for fam in sorted(bootstrap):
        for key in ("ΔU", "R_sur"):
            d = bootstrap[fam][key]
            excl = "✓" if d["ci_excludes_zero"] else "✗"
            print(f"{fam:14s} {key:6s} {d['mean']:>10.4f} "
                  f"{d['ci95_lo']:>10.4f} {d['ci95_hi']:>10.4f} {excl:>8s}")
    print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""EXP-META-CONFIG-GRID — dose-response extension of the config screen.

Motivation. The two-level screen (exp_meta_config_screen.py; runs/
EXP_META_CONFIG_SCREEN.md) left the family x config interaction
unresolved: chi^2_4 = 8.71, p = 0.069, one confirmed family (Rastrigin,
d_z = 0.52), directional mechanistic agreement 5/5 — and identified the
two cheapest escalations: more seeds and more config levels. This script
executes both, predeclared below.

Estimands (declared before running).

  Model:  y_{f,s,L} = mu + alpha_f + beta_{c(f,L)} + error, where
          L in LEVELS = {5, 8, 12, 18} (raw n_init) and
          c(f,L) = L / (2 d_f + 1)  is the dimension-relative design size
          (the axis recommended by the screen report: raw n_init means
          different things at different d).
  Reference level: L0 = 5 (harness default).
  Per-family contrast:      Delta_{f,L} = E_s[y_{f,s,L} - y_{f,s,L0}].
  Per-family dose slope:    b_f = E_s[b_{f,s}], where b_{f,s} is the OLS
                            slope of y_{f,s,·} on c over the 4 levels
                            (seed-paired; levels share the Sobol prefix).
  H0_slope : the REFERENCE-STANDARDIZED slopes b_f / sd_ref(f) are equal
             across families (common dose response — no family x dose
             interaction). PRIMARY test. Standardization by each family's
             reference-level sd makes the test invariant to per-family
             objective rescaling (scale-stable; codex round-3 D1); the SEs
             come from joint seed resampling so the sd_ref estimation
             uncertainty is included (codex round-4 B1). Raw-unit slopes
             are reported secondarily.
  H0_pair  : Delta_{f,12} identical across families (direct comparability
             with the screen). SECONDARY.
  Mechanistic (directional, predeclared): multimodal families (Rastrigin,
  Ackley) have b_f > 0; smooth families (Branin, Hartmann, SixHumpCamel)
  have b_f < 0.

Design.

  - LEVELS = {5, 8, 12, 18}; budget 30 evaluations (design + BO) for all
    levels; pinned AlwaysSurrogate policy; same family + seed -> same
    scrambled Sobol stream with the prefix property across levels.
  - Seeds 0..59 (60): seeds 0-19 replicate the screen's sample exactly
    (nested), 20-59 are NEW — the replication/confirmation split is
    seeds 0-19 (discovery, prior sample) vs 20-59 (confirmation, fresh).
  - 5 families x 4 levels x 60 seeds = 1200 trajectories.

Inference.

  1. Per family x level (vs reference): paired mean, seed bootstrap CI,
     d_z, paired t, MDD_80 = 2.80 se, precision vs 0.25 sd floor.
  2. Dose slope per family: mean of seed-level OLS slopes, se, t, CI.
  3. PRIMARY: slope homogeneity Wald chi^2 (continuous Cochran's Q) with
     df = F-1; heterogeneity I^2.
  4. SECONDARY: Delta_{f,12} homogeneity chi^2 (the screen's statistic,
     recomputed at n=60 and on the fresh subset 20-59 alone).
  5. Mechanistic sign test on b_f (exact one-sided binomial).
  6. Descriptive: per-family best level; LOFO argmax prediction
     (majority of other families' best levels, tie -> reference).

Predeclared decision rules.

  - CONDITIONALITY_DETECTED iff slope-homogeneity p < 0.05.
  - FUTILE iff not detected AND, for every family, EVERY level-vs-reference
    contrast CI lies entirely within +/- 0.25 sd (equivalence bound).
  - Otherwise INCONCLUSIVE.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scipy import stats as scipy_stats

from opt_agent.actions import candidate_actions_for_state
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend
from opt_agent.loop.decision_loop import DecisionLoop

LEVELS = [5, 8, 12, 18]
REFERENCE = 5
MISE = 0.25          # minimal-interesting standardized effect
MDD_FACTOR = 2.80    # ~80% power, two-sided alpha=.05
MECH_SIGN = {"Rastrigin": +1, "Ackley": +1,
             "Branin": -1, "Hartmann": -1, "SixHumpCamel": -1}


def _worker_init() -> None:
    import torch

    torch.set_num_threads(1)


class AlwaysSurrogatePolicy:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def select_action(self, state, backend=None):
        choices = candidate_actions_for_state(state, seed=self.seed)
        if "surrogate_acquisition" in choices:
            return choices["surrogate_acquisition"]
        return next(iter(choices.values()))


def run_one_trajectory(family: str, n_init: int, budget: int, seed: int) -> dict:
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(problem_id=family, seed=seed, n_init=n_init))
    loop = DecisionLoop(backend, total_budget=budget)
    loop.run(AlwaysSurrogatePolicy(seed=0), max_rounds=budget + 5)
    history = backend.get_history()
    return {
        "family": family, "seed": seed, "n_init": n_init,
        "best": float(history.objectives[:, 0].max().item()),
        "n_evals": len(history),
        "dimension": int(history.X.shape[1]),
        "termination": (loop.termination_reason.value
                        if loop.termination_reason else None),
    }


def paired_bootstrap_ci(diffs: list[float], n_boot: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(statistics.fmean(
        [diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return [means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]]


def ols_slope(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else float("nan")


def analyze_contrast(diffs: list[float], n_boot: int,
                     disc_idx: list[int], conf_idx: list[int]) -> dict:
    n = len(diffs)
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(n)
    ci = paired_bootstrap_ci(diffs, n_boot)
    t_stat = mean / se if se > 0 else float("nan")
    p_t = (2.0 * (1.0 - scipy_stats.t.cdf(abs(t_stat), df=n - 1))
           if se > 0 else float("nan"))

    def half(idxs: list[int]) -> dict:
        d = [diffs[i] for i in idxs]
        m = statistics.fmean(d)
        s = statistics.stdev(d)
        c = paired_bootstrap_ci(d, max(1000, n_boot // 2), seed=1)
        return {"mean": m, "sd": s, "ci95": c,
                "ci_excludes_zero": (c[0] > 0) or (c[1] < 0)}

    disc, conf = half(disc_idx), half(conf_idx)
    return {
        "n_pairs": n, "mean": mean, "sd": sd, "se": se,
        "ci95": ci, "ci_excludes_zero": (ci[0] > 0) or (ci[1] < 0),
        "ci_half_width": (ci[1] - ci[0]) / 2.0,
        "paired_t": {"t": t_stat, "df": n - 1, "p_two_sided": p_t},
        "cohens_dz": mean / sd if sd > 0 else float("nan"),
        "mdd80": MDD_FACTOR * se,
        # Equivalence bound (codex round-3 E2): a precise null requires the
        # CI to lie ENTIRELY within +/- MISE*sd, not merely to be narrow
        # (a narrow CI centered on a large effect is not a null).
        "precise_enough_for_null": (ci[0] > -MISE * sd and ci[1] < MISE * sd)
        if sd > 0 else False,
        "discovery_half": disc, "confirmation_half": conf,
        "confirmed": (disc["mean"] > 0) == (conf["mean"] > 0)
                     and conf["ci_excludes_zero"],
    }


def homogeneity(means: dict[str, float], variances: dict[str, float]) -> dict:
    ms, vs = list(means.values()), list(variances.values())
    fixed = (sum(m / v for m, v in zip(ms, vs))
             / sum(1.0 / v for v in vs))
    chi2 = sum((m - fixed) ** 2 / v for m, v in zip(ms, vs))
    df = len(ms) - 1
    return {"statistic": chi2, "df": df,
            "p_value": 1.0 - scipy_stats.chi2.cdf(chi2, df=df),
            "pooled": fixed,
            "i2": max(0.0, (chi2 - df) / chi2) if chi2 > 0 else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+",
                        default=["Branin", "Hartmann", "Ackley",
                                 "SixHumpCamel", "Rastrigin"])
    parser.add_argument("--seeds", type=int, default=60)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs" / "exp_meta_config_grid.json")
    args = parser.parse_args()

    n_disc = args.seeds // 3           # discovery = prior screen's sample size
    disc_idx = list(range(0, n_disc))
    conf_idx = list(range(n_disc, args.seeds))

    jobs = [(fam, L, args.budget, seed)
            for fam in args.families for seed in range(args.seeds)
            for L in LEVELS]
    print(f"Jobs: {len(jobs)} = {len(args.families)} families x "
          f"{args.seeds} seeds x {len(LEVELS)} levels {LEVELS}; "
          f"budget {args.budget}; AlwaysSurrogate pinned; "
          f"discovery seeds 0-{n_disc - 1}, confirmation {n_disc}-{args.seeds - 1}")

    import multiprocessing as mp
    t0 = time.monotonic()
    with mp.Pool(args.workers, initializer=_worker_init) as pool:
        results = pool.starmap(run_one_trajectory, jobs)
    elapsed = time.monotonic() - t0
    print(f"Done in {elapsed:.1f}s ({len(jobs)/elapsed:.1f} trajectories/s)")

    by_key = {(r["family"], r["seed"], r["n_init"]): r for r in results}
    dims = {r["family"]: r["dimension"] for r in results}
    n_evals = {r["n_evals"] for r in results}
    print(f"n_evals observed: {sorted(n_evals)}; dims: {dims}")

    per_family: dict[str, dict] = {}
    for fam in args.families:
        cvals = [L / (2 * dims[fam] + 1) for L in LEVELS]
        # contrasts vs reference
        contrasts = {}
        for L in LEVELS:
            if L == REFERENCE:
                continue
            diffs = [by_key[(fam, s, L)]["best"] - by_key[(fam, s, REFERENCE)]["best"]
                     for s in range(args.seeds)]
            contrasts[L] = analyze_contrast(diffs, args.n_boot, disc_idx, conf_idx)
        # seed-paired dose slopes in c-space
        slopes = [ols_slope(cvals, [by_key[(fam, s, L)]["best"] for L in LEVELS])
                  for s in range(args.seeds)]
        ref_best = [by_key[(fam, s, REFERENCE)]["best"]
                    for s in range(args.seeds)]
        sd_ref = statistics.pstdev(ref_best) or 1.0
        # Reference-standardized slopes (PRIMARY estimand, scale-stable):
        # dividing by the family's reference-level sd makes the slope
        # invariant to affine rescaling of that family's objective, so the
        # cross-family homogeneity test does not depend on raw units
        # (codex round-3 D1). Raw-unit slopes are retained for reporting.
        std_slopes = [b / sd_ref for b in slopes]
        # SE of the standardized-slope mean by JOINT seed resampling: each
        # replicate re-estimates sd_ref from the resampled reference
        # outcomes, so the uncertainty of the standardizing sd enters the
        # SE (codex round-4 B1; treating sd_ref as fixed understates it).
        rng = random.Random(3)
        n_seeds = args.seeds
        boot_means = []
        for _ in range(args.n_boot):
            idx = [rng.randrange(n_seeds) for _ in range(n_seeds)]
            sd_star = statistics.pstdev([ref_best[i] for i in idx]) or 1.0
            boot_means.append(
                statistics.fmean([slopes[i] for i in idx]) / sd_star)
        boot_means.sort()
        slope = {
            "c_values": cvals,
            "mean": statistics.fmean(slopes),
            "sd": statistics.stdev(slopes),
            "se": statistics.stdev(slopes) / math.sqrt(n_seeds),
            "std_mean": statistics.fmean(std_slopes),
            "std_se": statistics.stdev(std_slopes) / math.sqrt(n_seeds),
            "std_se_joint_boot": statistics.pstdev(boot_means),
            "std_ci95_boot": [boot_means[int(0.025 * args.n_boot)],
                              boot_means[int(0.975 * args.n_boot) - 1]],
        }
        slope["ci95"] = paired_bootstrap_ci(slopes, args.n_boot, seed=2)
        slope["t"] = slope["mean"] / slope["se"] if slope["se"] > 0 else float("nan")
        slope["p_two_sided"] = (
            2.0 * (1.0 - scipy_stats.t.cdf(abs(slope["t"]), df=args.seeds - 1))
            if slope["se"] > 0 else float("nan"))
        slope["per_sd_ref"] = slope["mean"] / sd_ref
        best_level = max(
            LEVELS, key=lambda L: statistics.fmean(
                by_key[(fam, s, L)]["best"] for s in range(args.seeds)))
        per_family[fam] = {
            "dimension": dims[fam],
            "c_values": cvals,
            "contrasts_vs_reference": contrasts,
            "dose_slope": slope,
            "best_level": best_level,
            "best_vs_reference": contrasts.get(best_level)
            if best_level != REFERENCE else None,
            "level_means": {L: statistics.fmean(
                by_key[(fam, s, L)]["best"] for s in range(args.seeds))
                for L in LEVELS},
        }

    # PRIMARY: homogeneity of reference-standardized slopes (scale-stable),
    # with SEs from joint seed resampling (sd_ref uncertainty included).
    slope_homog = homogeneity(
        {fam: row["dose_slope"]["std_mean"] for fam, row in per_family.items()},
        {fam: row["dose_slope"]["std_se_joint_boot"] ** 2
         for fam, row in per_family.items()})
    slope_homog["se_method"] = (
        "joint seed bootstrap (sd_ref re-estimated per replicate; "
        "codex round-4 B1)")
    # SECONDARY: Delta_{.,12} homogeneity (the screen's statistic, n=60)
    pair12_homog = homogeneity(
        {fam: per_family[fam]["contrasts_vs_reference"][12]["mean"]
         for fam in args.families},
        {fam: per_family[fam]["contrasts_vs_reference"][12]["se"] ** 2
         for fam in args.families})
    # SECONDARY on fresh seeds only (20-59): Delta_{.,12}
    def subset_delta12(fam: str) -> tuple[float, float]:
        d = [by_key[(fam, s, 12)]["best"] - by_key[(fam, s, REFERENCE)]["best"]
             for s in conf_idx]
        return statistics.fmean(d), statistics.stdev(d) ** 2 / len(d)
    fresh12 = {fam: subset_delta12(fam) for fam in args.families}
    fresh12_homog = homogeneity({f: m for f, (m, _) in fresh12.items()},
                                {f: v for f, (_, v) in fresh12.items()})

    # Mechanistic sign test on slopes
    agreement = {fam: (1 if (per_family[fam]["dose_slope"]["mean"] > 0) else -1)
                 == MECH_SIGN.get(fam, 0) for fam in args.families if fam in MECH_SIGN}
    k_ok = sum(agreement.values())
    n_pred = len(agreement)
    sign_p = sum(math.comb(n_pred, i)
                 for i in range(k_ok, n_pred + 1)) / 2 ** n_pred

    # Descriptive LOFO on best level (undefined for single-family runs)
    lofo = []
    if len(args.families) > 1:
        for held in args.families:
            others = [f for f in args.families if f != held]
            counts: dict[int, int] = {}
            for f in others:
                counts[per_family[f]["best_level"]] = counts.get(
                    per_family[f]["best_level"], 0) + 1
            pred = max(counts, key=lambda L: (counts[L], -L))  # tie -> smaller level
            lofo.append({"held_out": held, "predicted": pred,
                         "actual": per_family[held]["best_level"],
                         "correct": pred == per_family[held]["best_level"]})
    lofo_acc = (sum(1 for r in lofo if r["correct"]) / len(lofo)) if lofo else None

    # Predeclared rules (cross-family; single-family arms have no such rule)
    detected = slope_homog["p_value"] < 0.05
    # Futility requires the equivalence bound on EVERY level contrast vs
    # the reference (not only the selected best; codex round-4 E2b) — a
    # reference-winning family is futile only if all its contrasts are
    # equivalence-bounded.
    precise_null = all(
        all(c["precise_enough_for_null"]
            for c in row["contrasts_vs_reference"].values())
        for row in per_family.values())
    if len(args.families) < 2:
        verdict = "SINGLE_FAMILY_ARM (cross-family rule undefined; evaluate " \
                  "the predeclared per-family question directly)"
    elif detected:
        verdict = "CONDITIONALITY_DETECTED"
    elif precise_null:
        verdict = "FUTILE"
    else:
        verdict = "INCONCLUSIVE"
    confirmed = [f for f in args.families
                 if per_family[f]["best_vs_reference"]
                 and per_family[f]["best_vs_reference"]["confirmed"]]

    payload = {
        "experiment": "exp_meta_config_grid",
        "relation_to_screen": "dose-response extension of "
                              "exp_meta_config_screen (levels 5/12 nested)",
        "estimands": {
            "contrast": f"Delta_fL = E[y(n_init=L) - y(n_init={REFERENCE})]",
            "dose_slope": "b_f = mean seed-paired OLS slope of y on "
                          "c = n_init/(2d+1) over levels",
            "primary_null": "b_f / sd_ref(f) identical across families "
                            "(no family x dose interaction)",
        },
        "predeclared_rules": {
            "conditionality": "slope-homogeneity p < 0.05 (chi^2_{F-1})",
            "futile": "not detected AND every level-vs-reference CI lies "
                      f"entirely within +/- {MISE} sd (equivalence bound)",
            "otherwise": "INCONCLUSIVE",
        },
        "families": args.families, "dimensions": dims, "levels": LEVELS,
        "reference_level": REFERENCE, "seeds": args.seeds,
        "discovery_seeds": disc_idx, "confirmation_seeds": conf_idx,
        "budget": args.budget, "policy": "AlwaysSurrogate (pinned)",
        "n_evals_observed": sorted(n_evals), "wall_seconds": elapsed,
        "per_family": per_family,
        "slope_homogeneity_primary": slope_homog,
        "delta12_homogeneity_n60": pair12_homog,
        "delta12_homogeneity_fresh_seeds": fresh12_homog,
        "mechanistic_sign_test": {
            "predicted_signs": MECH_SIGN, "agreement": agreement,
            "k_of_n": f"{k_ok}/{n_pred}", "exact_one_sided_p": sign_p},
        "lofo_best_level": {"folds": lofo, "accuracy": lofo_acc},
        "confirmed_families": confirmed,
        "verdict": verdict,
        "trajectories": [
            {k: r[k] for k in ("family", "seed", "n_init", "best",
                               "n_evals", "dimension", "termination")}
            for r in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({
        "level_means": {f: {L: round(v, 3) for L, v in row["level_means"].items()}
                        for f, row in per_family.items()},
        "slopes": {f: {"b": round(row["dose_slope"]["mean"], 4),
                       "p": round(row["dose_slope"]["p_two_sided"], 4),
                       "per_sd": round(row["dose_slope"]["per_sd_ref"], 4)}
                   for f, row in per_family.items()},
        "slope_homogeneity_primary": {k2: (round(v, 4) if isinstance(v, float) else v)
                                      for k2, v in slope_homog.items()},
        "delta12_homogeneity_n60": {k2: (round(v, 4) if isinstance(v, float) else v)
                                    for k2, v in pair12_homog.items()},
        "delta12_homogeneity_fresh": {k2: (round(v, 4) if isinstance(v, float) else v)
                                      for k2, v in fresh12_homog.items()},
        "sign_test": f"{k_ok}/{n_pred} p={sign_p:.4f}",
        "best_levels": {f: per_family[f]["best_level"] for f in args.families},
        "confirmed": confirmed, "lofo": lofo_acc, "verdict": verdict,
    }, indent=2))
    print(f"\nVERDICT: {verdict}  (artifact: {args.output})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

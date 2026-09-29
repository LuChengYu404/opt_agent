"""EXP-META-CONFIG-SCREEN — meta-level config-conditionality screen (guide §5.2/§10.3).

Research question. Before building any meta-level supervision pipeline, is
there detectable CONFIGURATION-CONDITIONALITY on the current panel — i.e.,
does the advantage of one design-phase configuration over another depend on
the problem family? Guide §4.4 warns the meta-level advantage may be
nonpositive everywhere; this screen is the cheap paired test the round-1
review (E2) asked for.

Estimands (declared before running).

  Model:  y_{f,s,a} = mu + alpha_f + beta_a + (alpha beta)_{f,a} + eps_{f,s,a}
          f = family (1..F), a in {small, large}, s = seed (paired).
  Delta_f = E_s[y_{f,s,large} - y_{f,s,small}]
          = (beta_large - beta_small) + (alpha beta)_{f,large} - (alpha beta)_{f,small}
          is the per-family config advantage (maximization units).
  H0_int : Delta_1 = ... = Delta_F  (no family x config interaction —
           the best config does not vary by family).
  H1_int : some Delta_f differ.

  Descriptor (NOT a test statistic): meta-D = E_f[max_a u_{a,f}] -
  max_a E_f[u_{a,f}] on family-standardized utilities u. In POPULATION terms
  meta-D > 0 iff the Delta_f are not all of one sign; it measures the value
  of an oracle that knows the family label. It says nothing about whether a
  LEARNABLE problem signature could recover that oracle — for that we report
  a leave-one-family-out (LOFO) sign-prediction probe as a descriptive
  lower-bound diagnostic.

Design.

  - Config axis: initial design size n_init in {5 ("small", harness
    default), 12 ("large")} — a design-phase configuration per guide §1.2.
    Both arms spend the SAME total budget (default 30 evaluations,
    design + BO; verified per trajectory).
  - Inner policy pinned to AlwaysSurrogate in both arms (isolates the
    config effect).
  - Pairing: same family + same seed -> same scrambled Sobol stream; the
    small arm's design is a PREFIX of the large arm's (verified in backend:
    draw(n) on a fresh seeded engine).
  - Endpoint: terminal best objective (backend maximizes).

Mechanistic hypotheses (declared before running; directional, tested
descriptively against sign(Delta_f)).

  H-mech-1 (multimodality -> space-filling): strongly multimodal families
  (Rastrigin, Ackley) benefit from a larger space-filling design because it
  covers more local basins before LogEI commits.
  H-mech-2 (smoothness -> model-guided budget): smooth families
  (Branin, Hartmann, SixHumpCamel) are hurt by spending budget on design
  points the GP could have placed itself (mild negative Delta_f).
  Predicted sign pattern: positive on {Rastrigin, Ackley}, negative on
  {Branin, Hartmann, SixHumpCamel}. We report n_init relative to the 2d+1
  heuristic per family to interpret the axis.

Inference (all primary tests predeclared).

  1. Per family: paired mean Delta_hat_f, seed-level percentile bootstrap
     CI (5000), paired t test, and Cohen's d_z = Delta_hat_f / sd(d_fs).
  2. Interaction (PRIMARY): homogeneity Wald test
     X^2 = sum_f (Delta_hat_f - Delta_hat_w)^2 / Var(Delta_hat_f),
     Var(Delta_hat_f) = sd_f^2 / n, referenced to chi^2_{F-1}. This is the
     continuous analogue of Cochran's Q and tolerates the family-scale
     heterogeneity that defeats naive pooling. Heterogeneity share
     I^2 = max(0, (X^2 - df) / X^2).
  3. Discovery / confirmation split: seeds 0..half-1 discovery,
     half..n-1 confirmation. A family is CONFIRMED iff both halves agree in
     sign AND the confirmation-half bootstrap CI excludes zero.
  4. Precision floor (futility semantics for a negative screen): per family
     report the 80%-power minimum detectable difference MDD_f ~ 2.80 *
     se_f and the CI half-width relative to 0.25 * sd_f (a declared
     minimal-interesting standardized effect).

Predeclared decision rules.

  - CONDITIONALITY DETECTED iff interaction p < 0.05 (chi^2_{F-1}).
  - Axis FUTILE (stop, per guide §5.5.4) iff NOT detected AND every
    family's CI half-width < 0.25 * sd_f (precise null).
  - Otherwise: INCONCLUSIVE (not detected but underpowered) — the axis is
    parked, not killed.
  - meta-D and LOFO are reported as descriptors and do NOT enter the rule.

Hardware: CPU-bound trajectories; F x n x 2 jobs, mp.Pool with a
torch single-thread initializer (oversubscription guard).
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

CONFIGS = {"small": 5, "large": 12}

# Declared minimal-interesting standardized config advantage.
MISE = 0.25  # in units of the family's within-pair sd
# Two-sided 80% power at alpha=.05 -> ~2.80 se multiplier.
MDD_FACTOR = 2.80


def _worker_init() -> None:
    """Cap torch threads per worker (oversubscription guard; the longhorizon
    driver hit load 315 with default threads)."""
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
    """One (family, config, seed) episode; return endpoint + bookkeeping."""
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
        "family": family,
        "seed": seed,
        "config": config_name,
        "best": float(history.objectives[:, 0].max().item()),
        "n_evals": len(history),
        "dimension": int(history.X.shape[1]),
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
    return [means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]]


def analyze_family(family: str, diffs: list[float], n_boot: int,
                   discovery: list[int], confirmation: list[int]) -> dict:
    n = len(diffs)
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(n)
    ci = paired_bootstrap_ci(diffs, n_boot=n_boot)
    t_stat = mean / se
    p_t = 2.0 * (1.0 - scipy_stats.t.cdf(abs(t_stat), df=n - 1))

    def half(indices: list[int]) -> dict:
        d = [diffs[i] for i in indices]
        m = statistics.fmean(d)
        s = statistics.stdev(d)
        c = paired_bootstrap_ci(d, n_boot=max(1000, n_boot // 2), seed=1)
        return {"mean": m, "sd": s, "ci95": c,
                "ci_excludes_zero": (c[0] > 0) or (c[1] < 0)}

    disc = half(discovery)
    conf = half(confirmation)
    sign_agree = (disc["mean"] > 0) == (conf["mean"] > 0)
    confirmed = sign_agree and conf["ci_excludes_zero"]
    return {
        "n_pairs": n,
        "mean_diff_large_minus_small": mean,
        "sd_pair_diff": sd,
        "se": se,
        "ci95_bootstrap": ci,
        "ci_excludes_zero": (ci[0] > 0) or (ci[1] < 0),
        "ci_half_width": (ci[1] - ci[0]) / 2.0,
        "paired_t": {"t": t_stat, "df": n - 1, "p_two_sided": p_t},
        "cohens_dz": mean / sd,
        "mdd80": MDD_FACTOR * se,
        "precise_enough_for_null": ((ci[1] - ci[0]) / 2.0) < MISE * sd,
        "discovery_half": disc,
        "confirmation_half": conf,
        "sign_agree_across_halves": sign_agree,
        "confirmed_family": confirmed,
        "best_config_by_mean": ("large" if mean > 0 else "small"),
    }


def interaction_test(per_family_mean: dict[str, float],
                     per_family_var: dict[str, float]) -> dict:
    """Homogeneity of per-family advantages (continuous Cochran's Q).

    H0: Delta_f identical across families; Wald X^2 ~ chi^2_{F-1}.
    """
    means = list(per_family_mean.values())
    variances = list(per_family_var.values())
    fixed = sum(m / v for m, v in zip(means, variances)) / \
        sum(1.0 / v for v in variances)  # inverse-variance weighted common mean
    chi2 = sum((m - fixed) ** 2 / v for m, v in zip(means, variances))
    df = len(means) - 1
    p = 1.0 - scipy_stats.chi2.cdf(chi2, df=df)
    i2 = max(0.0, (chi2 - df) / chi2) if chi2 > 0 else 0.0
    return {
        "statistic": chi2,
        "df": df,
        "p_value": p,
        "pooled_common_delta": fixed,
        "heterogeneity_i2": i2,
        "null": "Delta_1 == ... == Delta_F (no family x config interaction)",
    }


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
    print(f"Jobs: {len(jobs)} = {len(args.families)} families x "
          f"{args.seeds} seeds x {len(CONFIGS)} configs; budget {args.budget}; "
          f"AlwaysSurrogate pinned")

    import multiprocessing as mp
    t0 = time.monotonic()
    with mp.Pool(args.workers, initializer=_worker_init) as pool:
        results = pool.starmap(run_one_trajectory, jobs)
    elapsed = time.monotonic() - t0
    print(f"Done in {elapsed:.1f}s ({len(jobs)/elapsed:.1f} trajectories/s)")

    by_key = {(r["family"], r["seed"], r["config"]): r for r in results}
    n_evals = {r["n_evals"] for r in results}
    dims = {r["family"]: r["dimension"] for r in results}
    print(f"n_evals observed: {sorted(n_evals)}; dims: {dims}")

    half = args.seeds // 2
    discovery = list(range(0, half))
    confirmation = list(range(half, args.seeds))

    per_family: dict[str, dict] = {}
    for fam in args.families:
        diffs = [by_key[(fam, s, "large")]["best"] - by_key[(fam, s, "small")]["best"]
                 for s in range(args.seeds)]
        per_family[fam] = analyze_family(
            fam, diffs, args.n_boot, discovery, confirmation)

    inter = interaction_test(
        {fam: row["mean_diff_large_minus_small"] for fam, row in per_family.items()},
        {fam: row["se"] ** 2 for fam, row in per_family.items()},
    )

    # meta-D descriptor (oracle-among-known-families; NOT part of the rule).
    u: dict[str, dict] = {}
    for fam in args.families:
        ys = [by_key[(fam, s, "small")]["best"] for s in range(args.seeds)]
        yl = [by_key[(fam, s, "large")]["best"] for s in range(args.seeds)]
        s_f = statistics.pstdev(ys + yl)
        u[fam] = {"small": statistics.fmean(ys) / s_f,
                  "large": statistics.fmean(yl) / s_f}
    e_max = statistics.fmean(max(u[f][a] for a in CONFIGS) for f in args.families)
    max_e = max(statistics.fmean(u[f][a] for f in args.families) for a in CONFIGS)
    meta_d_point = e_max - max_e

    rng = random.Random(0)
    boots = []
    for _ in range(args.n_boot):
        u_b = {}
        for fam in args.families:
            picks = [rng.randrange(args.seeds) for _ in range(args.seeds)]
            ys = [by_key[(fam, p, "small")]["best"] for p in picks]
            yl = [by_key[(fam, p, "large")]["best"] for p in picks]
            s_b = statistics.pstdev(ys + yl) or 1.0
            u_b[fam] = {"small": statistics.fmean(ys) / s_b,
                        "large": statistics.fmean(yl) / s_b}
        eb = statistics.fmean(max(u_b[f][a] for a in CONFIGS) for f in args.families)
        mb = max(statistics.fmean(u_b[f][a] for f in args.families) for a in CONFIGS)
        boots.append(eb - mb)
    boots.sort()
    meta_d_ci = [boots[int(0.025 * args.n_boot)],
                 boots[int(0.975 * args.n_boot) - 1]]

    # LOFO sign-prediction probe (descriptive learnability lower bound).
    # Tie-break predicts the harness default ("small"); at F=5 with a near-even
    # split this probe has essentially no resolution — report, do not decide.
    fams = list(args.families)
    lofo = []
    for held in fams:
        others = [f for f in fams if f != held]
        votes_large = sum(
            1 for f in others if per_family[f]["mean_diff_large_minus_small"] > 0)
        votes_small = len(others) - votes_large
        pred = ("large" if votes_large > votes_small else "small")
        actual = per_family[held]["best_config_by_mean"]
        lofo.append({"held_out": held, "predicted": pred, "actual": actual,
                     "correct": pred == actual})
    lofo_acc = sum(1 for r in lofo if r["correct"]) / len(lofo)

    # Secondary mechanistic corroboration: predeclared directional signs.
    # H-mech-1 predicts Delta_f > 0 on multimodal {Rastrigin, Ackley};
    # H-mech-2 predicts Delta_f < 0 on smooth {Branin, Hartmann, SixHumpCamel}.
    predicted_sign = {"Rastrigin": +1, "Ackley": +1,
                      "Branin": -1, "Hartmann": -1, "SixHumpCamel": -1}
    agreement = {
        fam: (1 if (per_family[fam]["mean_diff_large_minus_small"] > 0) else -1)
             == predicted_sign.get(fam, 0)
        for fam in fams if fam in predicted_sign
    }
    k = sum(agreement.values())
    n_pred = len(agreement)
    sign_p = sum(
        math.comb(n_pred, i) for i in range(k, n_pred + 1)) / 2 ** n_pred

    # Predeclared rules.
    detected = inter["p_value"] < 0.05
    precise_null = all(row["precise_enough_for_null"] for row in per_family.values())
    if detected:
        verdict = "CONDITIONALITY_DETECTED"
    elif precise_null:
        verdict = "FUTILE"
    else:
        verdict = "INCONCLUSIVE"
    confirmed = [f for f in fams if per_family[f]["confirmed_family"]]

    payload = {
        "experiment": "exp_meta_config_screen",
        "estimands": {
            "Delta_f": "E_seed[y_large - y_small] per family (max units)",
            "interaction_null": inter["null"],
            "meta_d_descriptor": (
                "oracle-among-known-families value on standardized utilities; "
                ">0 in population iff Delta_f not all one sign; NOT a test "
                "statistic and NOT evidence of signature learnability"
            ),
        },
        "mechanistic_hypotheses": {
            "H_mech_1": "multimodal families (Rastrigin, Ackley) gain from a "
                        "larger space-filling design",
            "H_mech_2": "smooth families (Branin, Hartmann, SixHumpCamel) lose "
                        "model-guided budget to extra design points",
        },
        "predeclared_rules": {
            "conditionality": "interaction p < 0.05 (chi^2_{F-1})",
            "futile": "not detected AND all CI half-widths < "
                      f"{MISE} * sd_f (precise null)",
            "otherwise": "INCONCLUSIVE (parked, not killed)",
        },
        "families": args.families,
        "dimensions": dims,
        "n_init_over_2dp1": {
            fam: {"small": CONFIGS["small"] / (2 * dims[fam] + 1),
                  "large": CONFIGS["large"] / (2 * dims[fam] + 1)}
            for fam in args.families
        },
        "seeds": args.seeds,
        "discovery_seeds": discovery,
        "confirmation_seeds": confirmation,
        "budget": args.budget,
        "configs": CONFIGS,
        "policy": "AlwaysSurrogate (pinned)",
        "n_evals_observed": sorted(n_evals),
        "wall_seconds": elapsed,
        "per_family": per_family,
        "interaction": inter,
        "meta_d": {
            "point": meta_d_point,
            "ci95_bootstrap": meta_d_ci,
            "per_family_u": u,
            "role": "descriptor only (see estimands)",
        },
        "lofo_sign_prediction": {
            "folds": lofo, "accuracy": lofo_acc, "chance": 0.5,
            "tie_break": "predict harness default (small)",
            "caveat": "no resolution at F=5 with a near-even split; "
                      "descriptive only",
        },
        "mechanistic_sign_test": {
            "predicted_signs": predicted_sign,
            "agreement": agreement,
            "k_of_n": f"{k}/{n_pred}",
            "exact_one_sided_p": sign_p,
            "role": "secondary corroboration of the predeclared directional "
                    "hypotheses; does not enter the conditionality rule",
        },
        "confirmed_families": confirmed,
        "verdict": verdict,
        "trajectories": [
            {k: r[k] for k in ("family", "seed", "config", "best",
                               "n_evals", "dimension", "termination")}
            for r in results
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = {
        "per_family": {
            fam: {
                "mean": round(row["mean_diff_large_minus_small"], 4),
                "ci95": [round(x, 4) for x in row["ci95_bootstrap"]],
                "dz": round(row["cohens_dz"], 3),
                "p_t": round(row["paired_t"]["p_two_sided"], 4),
                "confirmed": row["confirmed_family"],
                "precise_null": row["precise_enough_for_null"],
            } for fam, row in per_family.items()
        },
        "interaction": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in inter.items()},
        "meta_d": {"point": round(meta_d_point, 4),
                   "ci95": [round(x, 4) for x in meta_d_ci]},
        "lofo_accuracy": lofo_acc,
        "confirmed_families": confirmed,
        "verdict": verdict,
    }
    print(json.dumps(summary, indent=2))
    print(f"\nVERDICT: {verdict}  (artifact: {args.output})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

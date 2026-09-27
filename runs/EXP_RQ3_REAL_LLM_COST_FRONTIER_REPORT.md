# EXP-RQ3 — Real-LLM cost frontier (RQ3 closed-loop, 2026-09-27)

> ## CORRECTION + 10-seed confirmation (2026-09-27, later the same day)
>
> **The original 3-seed prose below contains a sign inversion.** Its own
> recorded numbers show the `rf` arm (frozen System-1, 0 LLM calls) beat
> `llm` on Hartmann at seeds 100–102 by +0.910 (3/0/0; the W/T/L column
> was RF's wins, misread as the LLM's). The passages below that claim
> "Hartmann is the one family where the LLM is genuinely useful", the
> headline sentence "competitive on 4/5, indistinguishable on Ackley",
> the verdict row "FAIL on Ackley", and the per-family recommendation
> "Hartmann → llm or confidence_0.75" all inverted the sign of Δ and are
> **superseded**. The independent EXP-RQ0.5 joint run had already
> reported "Hartmann: LLM worse, −0.985, p = 0.0007" (workshop outline
> §5) — the corrected reading agrees with it; the 3-seed prose was the
> outlier.
>
> The 10-seed confirmation below (seeds 100–109, 450 paired trajectories)
> settles the direction: **no LLM-calling arm beats `rf` with a CI
> excluding zero on any family, while `llm` itself is CI-detectably worse
> than `rf` on Ackley, Hartmann, and SixHumpCamel.** See the next section.

## 10-seed confirmation (seeds 100–109)

**Setup.** Added seeds 103–109 to the original seeds 100–102 (315 new +
135 original = 450 paired trajectories; 5 families × 10 seeds × 9
policies). Same frozen RF System-1 (trained on rollout seeds 0–15,
temperature-calibrated on 16–19; identical `dataset_sha256` and training
block verified across both files), same `deepseek-chat` backend, budget
30, paired Sobol starts (0 paired-start violations across all 450), and
**zero LLM fallbacks** in the new run. Merged analysis:
`scripts/analyze_rq3_cost_frontier.py` →
`runs/exp_rq3_cost_frontier_10seed.json` (it reproduces the original
3-seed meanΔ/CI/W-T-L numbers exactly on the old file, which is how the
sign error was found).

**Headline (10 paired seeds; terminal best, higher is better).**

| Family | `rf` mean best | `llm` mean best | `rf − llm` | `llm` calls | `rf` calls |
|---|---:|---:|---:|---:|---:|
| Branin | -0.924 | -1.439 | +0.515 | 16.9 | **0** |
| Hartmann | +2.387 | +1.811 | **+0.576** | 14.6 | **0** |
| Ackley | -19.182 | -20.199 | **+1.017** | 15.6 | **0** |
| SixHumpCamel | +0.787 | -0.228 | **+1.015** | 12.4 | **0** |
| Rastrigin | -121.330 | -131.439 | +10.109 | 13.7 | **0** |

Bold = paired seed-bootstrap 95% CI of `llm − rf` entirely below zero
(RF detectably better): Hartmann [-1.032, -0.103], Ackley [-1.658,
-0.436], SixHumpCamel [-1.512, -0.587]. Branin [-1.224, +0.152] and
Rastrigin [-22.356, +0.404] are RF-better in mean with CI crossing zero.

**Cost-frontier table (10 seeds).** Paired mean Δ = `arm − rf` per seed
(negative = RF better); W/T/L is RF vs the arm; 10k-resample seed
bootstrap 95% CI.

| Family | Arm | mean Δ | CI 95 % | W/T/L | Calls Δ |
|---|---|---:|---|:-:|---:|
| Branin | llm | -0.515 | [-1.224, +0.152] | 5/0/5 | +16.9 |
| Branin | confidence_0.75 | **-0.725** | **[-1.454, -0.013]** | 6/0/4 | +17.0 |
| Branin | random_0.5 | **-0.959** | **[-1.908, -0.065]** | 7/0/3 | +11.2 |
| Hartmann | llm | **-0.576** | **[-1.032, -0.103]** | 7/0/3 | +14.6 |
| Hartmann | confidence_0.25 | +0.018 | [-0.492, +0.480] | 4/0/6 | +8.9 |
| Hartmann | random_0.25 | +0.149 | [-0.264, +0.690] | 5/2/3 | +6.0 |
| Ackley | llm | **-1.017** | **[-1.658, -0.436]** | 7/1/2 | +15.6 |
| Ackley | confidence_0.25 | **-0.588** | **[-1.213, -0.109]** | 5/3/2 | +9.1 |
| Ackley | random_0.25 | -0.278 | [-1.065, +0.739] | 6/2/2 | +5.8 |
| SixHumpCamel | llm | **-1.015** | **[-1.512, -0.587]** | 9/1/0 | +12.4 |
| SixHumpCamel | confidence_0.25 | **-1.015** | **[-1.512, -0.587]** | 9/1/0 | +12.9 |
| SixHumpCamel | random_0.25 | -0.236 | [-0.591, +0.084] | 6/1/3 | +7.2 |
| Rastrigin | llm | -10.109 | [-22.356, +0.404] | 6/1/3 | +13.7 |
| Rastrigin | confidence_0.25 | -5.186 | [-18.102, +6.706] | 5/1/4 | +12.9 |
| Rastrigin | random_0.25 | -5.447 | [-14.117, +1.388] | 5/4/1 | +5.5 |

Full grid (all 8 arms × 5 families) in `runs/exp_rq3_cost_frontier_10seed.json`.

**What changed vs the 3-seed run.**

1. **Ackley flips against the LLM.** The 3-seed read ("LLM may benefit
   Ackley, CI crosses zero") does not survive: at 10 seeds `llm` is
   CI-detectably *worse* than `rf` (-1.017, 7/1/2). The pre-specified
   regret-margin criterion no longer fails on Ackley.
2. **Hartmann corrects to an RF win** (sign error documented above):
   +0.576 with CI excluding zero. The "route LLM only to Hartmann"
   recommendation is withdrawn.
3. **Rastrigin's borderline 3-seed significance weakens**: +16.95
   (CI lo = 0) becomes +10.11 (CI [-22.4, +0.4]). RF remains better in
   mean; the effect is no longer near-significant.
4. **The random-rate-beats-confidence observation does not generalize.**
   At 10 seeds no random rate and no confidence threshold produces a
   CI-detectable gain over `rf` anywhere; the only positive point
   estimates on the whole grid are Hartmann/confidence_0.25 (+0.018,
   8.9 calls) and Hartmann/random_0.25 (+0.149, 6.0 calls), both
   CI-crossing. Meanwhile Branin random_0.5/0.75 are now CI-detectably
   *worse* than `rf`.

**Pre-specified acceptance verdict (10 seeds).**

| Criterion | Required | Observed | Verdict |
|---|---|---|---|
| Call reduction vs `llm` | ≥ 50 % | 100 % (`rf` = 0 calls) | **PASS** |
| Token reduction vs `llm` | ≥ 50 % | 100 % | **PASS** |
| Regret margin vs `llm` ≤ 0.10 σ_family | per family | `rf` ≥ `llm` in mean on 5/5 families (CI-detectably better on 3) | **PASS 5/5** |

σ_family = sample std of `terminal_best` pooled over arms and seeds per
family (Branin 1.408, Hartmann 0.599, Ackley 0.872, SixHumpCamel 0.692,
Rastrigin 12.418); definition fixed in the merged-analysis script. The
3-seed report's σ_Ackley ≈ 0.74 had no recorded definition.

**Decision for the paper (supersedes the 3-seed decision).**

1. **The frozen RF System-1 (0 LLM calls) is the recommended default on
   all five families.** The always-LLM policy is *dominated* — it never
   beats `rf` with CI excluding zero and loses detectably on three
   families at ~12–17 calls (~4.7–6.5k tokens) per trajectory.
2. **No selective-routing scheme earns its cost on this panel.** Neither
   confidence thresholds nor random rates produce any CI-detectable gain
   over `rf`; several are CI-detectably worse.
3. **The open problem changes.** It is no longer "per-family calibration
   of a router" (the LLM-favourable family does not exist here); it is
   *finding any regime where this LLM System-2 repays its cost* — e.g.
   longer horizons, stochastic objectives, or genuinely OOD states — or
   accepting the degenerate cost frontier: System-1 only.

---

## Original 3-seed run (seeds 100–102)

**Read the sections below with the sign correction above: the Hartmann
per-family paragraph, the headline sentence, the verdict table, and the
"Decision for the paper" contain the inverted reading. The tables and
recorded numbers are correct.**

**Date:** 2026-09-27
**Setup:** five BOCoDe families × seeds {100, 101, 102} × budget 30, paired starting Sobol evaluations.
**Policies:** `rf` (always System-1, 0 LLM calls); `llm` (always System-2);
`confidence_{0.25, 0.5, 0.75, 0.9}` (ConfidenceRouter with calibrated
softmax threshold); `random_{0.25, 0.5, 0.75}` (random rate to System-2);
total **135 trajectories**. All arms share the System-2 backend
(`deepseek-chat` via `DEEPSEEK_API_KEY`); mock runs are not evidence.

**Pre-specified acceptance criteria (declared before unblinding):**
- A practically acceptable regret margin of **0.10 × per-family objective std**.
  This is a *soft* criterion (a learned router is allowed a small
  performance tax for cost).
- A desired call reduction of **≥ 50 %** vs `llm` (always LLM). Token
  reduction in the same proportion.

## Headline

| Family | `rf` mean best | `llm` mean best | `rf − llm` mean | `llm` calls | `rf` calls |
|---|---:|---:|---:|---:|---:|
| Ackley | -18.788 | -19.802 | +1.014 | 16.3 | **0** |
| Branin | -1.135 | -1.944 | +0.809 | 14.3 | **0** |
| Hartmann | +2.441 | +1.531 | +0.910 | 12.3 | **0** |
| Rastrigin | -109.515 | -126.468 | +16.952 | 13.7 | **0** |
| SixHumpCamel | +0.394 | -0.032 | +0.427 | 11.7 | **0** |

The `rf − llm` column is *RF better than LLM*. Positive = RF wins.
**A frozen System-1 (RF only, 0 LLM calls) is competitive with or beats
the always-LLM policy on 4/5 families and is indistinguishable on the
fifth (Ackley).**

## Cost-frontier table

For each family, paired mean `terminal_best(baseline) − terminal_best(rf)`
(positive = baseline beats RF) and call/token deltas. `rf` baseline is the
frozen System-1 (no LLM cost). Paired seed bootstrap 95% intervals over
three seeds per cell.

| Family | Baseline | mean Δ | CI 95 % | W/T/L | Calls Δ | Tokens Δ |
|---|---|---:|---|:-:|---:|---:|
| Ackley | llm | -1.014 | [-2.428, +0.315] | 2/0/1 | +16.3 | +6299 |
| Ackley | confidence_0.75 | -1.057 | [-2.428, +0.189] | 2/0/1 | +17.3 | +6654 |
| Ackley | random_0.25 | -0.236 | [-1.120, +0.412] | 1/1/1 | +6.0 | +2327 |
| Branin | llm | -0.809 | [-2.201, +0.682] | 2/0/1 | +14.3 | +5536 |
| Branin | confidence_0.75 | -0.045 | [-0.909, +0.682] | 1/0/2 | +15.7 | +6062 |
| Branin | random_0.5 | +0.352 | [-0.503, +1.536] | 1/0/2 | +13.0 | +4988 |
| Hartmann | llm | **-0.910** | [-1.508, -0.530] | **3/0/0** | +12.3 | +4777 |
| Hartmann | confidence_0.75 | **-0.910** | [-1.508, -0.530] | **3/0/0** | +14.0 | +5392 |
| Hartmann | random_0.25 | +0.157 | [-0.432, +0.904] | 1/1/1 | +6.3 | +2431 |
| Rastrigin | llm | **+16.952** | [+0.000, +43.04] | **2/1/0** | +13.7 | +5255 |
| Rastrigin | confidence_0.5 | **+16.952** | [+0.000, +43.04] | **2/1/0** | +13.7 | +5243 |
| Rastrigin | random_0.25 | **+4.517** | [+0.000, +13.36] | **2/1/0** | +5.7 | +2190 |
| SixHumpCamel | llm | +0.427 | [+0.000, +0.892] | 2/1/0 | +11.7 | +4523 |
| SixHumpCamel | confidence_0.5 | +0.293 | [+0.000, +0.490] | 2/1/0 | +15.3 | +5909 |
| SixHumpCamel | random_0.5 | +0.302 | [-0.115, +0.629] | 1/0/2 | +10.7 | +4105 |

Bold cells are statistically detectable positive effects (CI lower bound > 0).

## Pre-specified acceptance verdict

| Criterion | Required | Observed | Verdict |
|---|---|---|---|
| Call reduction vs `llm` | ≥ 50 % | **100 %** (rf = 0 calls; llm ≈ 11–17) | **PASS by construction** |
| Token reduction vs `llm` | ≥ 50 % | **100 %** (rf = 0 tokens) | **PASS by construction** |
| Regret margin vs `llm` | ≤ 0.10 σ_family | RF wins on 4/5; Ackley is within CI but mean Δ +1.01 vs σ_Ackley ≈ 0.74 (1.4 σ) | **FAIL on Ackley**, **PASS on 4/5** |

## Per-family interpretations

**Ackley (10D noisy).** LLM and confidence_0.75 are the only arms that
beat RF mean terminal best, but the effect is modest (+1.0 best over
3 seeds; CI lower bound −2.4 crosses zero). Random_0.25 is essentially
tied with RF at one third of the LLM cost. The honest read is "Ackley
*may* benefit from selective LLM calls but 3 seeds do not establish this."

**Branin (2D).** No arm beats RF inside a 95 % CI. Confidence_0.75 has
mean Δ −0.045 with CI spanning zero. Random_0.5 wins by ~0.35 mean but
with wide CI. RF is competitive at zero cost.

**Hartmann (6D).** LLM and confidence_0.75 both beat RF with **CI lower
bound −1.508 < 0 ⇒ RF has higher mean regret**. This is the one family
where the LLM is genuinely useful. About 12–14 LLM calls per trajectory.
Random_0.25 is tied with RF on mean and ~6 LLM calls — a cost-aware
option if absolute regret is the only metric.

**Rastrigin (10D multimodal).** RF dominates every arm that makes LLM
calls. Even random_0.25 (6 calls) loses to RF mean by +4.5 best with
CI lower bound at zero. Adding LLM calls *hurts* on this family in our
sample. The mechanism is the same global-to-local schedule visible in
the closed-loop action distribution.

**SixHumpCamel (2D).** All confidence arms and `llm` lose to RF with
CI lower bound at or near zero. RF is clearly the cost-efficient choice.

## Confidence calibration diagnostic

Across 135 trajectories the RF softmax maximum is **rarely above 0.25**
on any family — most confidence thresholds route almost every step to
System-2. The random-rate baselines (rate = 0.25, 0.5) are *better than
or competitive with* the confidence-based router at every level we
tested. This is evidence that the existing `ConfidenceRouter.tau_conf`
mechanism is **not yet a useful routing heuristic**; random rate to
System-2 is a strong cost-aware baseline.

## Decision for the paper

1. **RF (System-1 only) is the recommended default** on these five
   BOCoDe families; it dominates at zero LLM cost on Rastrigin and
   SixHumpCamel, ties on Branin and Ackley, and loses only on Hartmann
   (~0.91 mean best terminal, ~12 LLM calls).
2. **Selective LLM routing did not pay off in this sample.** Neither
   confidence thresholds nor learned confidence beats a matched random
   rate. Hartmann is the only family where LLM calls produced a
   detectable gain.
3. **Per-family calibration is the open problem.** A router that knew
   *which* family it was on could route LLM only on Hartmann (≈ 12
   calls/trajectory) and System-1 elsewhere; that would meet the
   pre-specified cost reduction criterion while preserving Hartmann
   performance. The frozen confidence router does not learn this.

## Artifacts

- `runs/exp_rq3_router_sweep.json`: 135 paired trajectories, 5 families
  × 3 seeds × 9 policies, with `initial_values` for paired-start
  verification, action sequences, route history, call counts, tokens,
  fallbacks, elapsed time, and per-cell summary.
- `runs/exp_rq3_router_smoke.json`: pre-existing mock wiring check on
  Branin seed 300.

## Limitations

- Three seeds per cell. Per-family CIs are wide and few CI lower bounds
  exclude zero. Ackley in particular is underpowered; a 10-seed
  confirmation would be the minimum for any paper claim.
- A fixed `n_init = 5` Sobol start. Different starts may shift which
  family is the LLM-favourable one.
- The BOCoDe panel is not a stress test of long horizons or stochastic
  objectives; the experiment isolates the cost-versus-performance
  decision for the standard fixed-budget setting.
- Random-rate baselines are not a *learning* method but they beat the
  confidence-based router; that is a routing-calibration diagnostic, not
  a method endorsement.

Co-Authored-By: Claude Code <noreply@anthropic.com>
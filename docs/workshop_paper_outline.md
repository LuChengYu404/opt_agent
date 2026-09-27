# Workshop Paper Outline — Decision-Centric BO

> **Status**: draft outline, 2026-09-27 (updated same day: Tracks A/B/D complete;
> RQ3 real-LLM cost frontier added at 10 seeds; RQ2.2 dimension-transfer
> negative added; RQ3 router-sweep sign error corrected).
> **Target venue**: ICLR / NeurIPS workshop on Meta-Learning for Optimization
> or AutoML/BO workshop; NOT a main-track submission.
> **Length**: 8 pages + appendix.

---

## Title (draft)

**"Decision-Centric Bayesian Optimization: A Decision-Landscape Framework and the Surprising Value of an LLM as System-1"**

---

## Abstract (≤ 200 words)

Bayesian Optimization (BO) practice reduces many sequential decision points
(refit surrogate, switch acquisition, expand region, or stop) to a single fixed
acquisition function. We re-frame these as an explicit meta-decision: at every
state `s_t`, choose an action `a_t ∈ A` that updates the BO process. On a
5-family BOCoDe panel × 20 paired seeds we measure three things:

1. **Decision landscape** — we identify three regimes (Flat, Heavy-tailed,
   Frequent-opportunity) using a `adaptive-value` metric on counterfactual
   oracle labels.
2. **Affine invariance (partial)** — gold_probs **median rank correlation is
   Spearman ρ = 1.0000 across all 17 (problem, phase) buckets** under
   y → scale * y + shift. The predeclared joint Gate 5 (which also requires
   JS ≤ 0.02) **fails on 2/17 Hartmann buckets** due to GP-fit numerical
   fragility in the rollout's surrogate acquisition proposal. Per-record
   rank invariance is not established.
3. **LLM as System-1** — a deepseek-chat LLM used at every decision beats
   AlwaysSurrogate by **+17.54 best_y** on Rastrigin (paired-t **p=0.0058**,
   Student-t df=n-1, n=10) and exhibits a **global-to-local temporal action
   pattern** *associated with* its terminal advantage (no causal claim — a
   deterministic global-then-local baseline has not been run). A simple
   `dim≥5` router beats Surrogate on the BOCoDe panel (p=0.0120) but does
   NOT transfer to IOH/BBOB (p=0.1376, negative result).
4. **Cost frontier (10 seeds, closed loop)** — with a frozen RF System-1
   deployed through full BO trajectories, the always-LLM policy never beats
   System-1-only with a CI excluding zero and is detectably worse on Ackley,
   Hartmann, and SixHumpCamel; no confidence-threshold or random-rate routing
   arm earns its cost. A deterministic two-phase schedule dominates the LLM
   itself on 3/5 families — the frontier's degeneracy is a property of the
   System-2 decisions, formalized by a routing-dominance proposition.

Our contribution is a 7-component framework (backend/state/action/utility/
rollout/dataset/system-1+router) with **385 unit tests**, four formal
propositions verified against the data (`docs/THEORY.md`), and six
replicable experiments. We do not yet demonstrate cross-problem transfer of
the meta-policy, and within-function dimension transfer is a clean negative.

---

## §1 Introduction

- BO is usually presented as `x → f(x) → x*`, hiding the **sequential
  control decisions** the practitioner makes between sample acquisitions.
- We re-frame as `s_t → a_t → backend → s_{t+1}` — a meta-policy π(a|s).
- Three RQs: Learnability (RQ1), Generalization (RQ2), Selective Reasoning (RQ3).
- **Status**: RQ1 partially settled (Gate 3 adjusted_lift 1.294 CI [0.848, 1.954] at 20 seeds;
  Gate 4 stop-gain 5/5; **Gate 5 partial — median Spearman = 1.0 but predeclared joint
  fails on 2/17 buckets; Gate 2 still fails**); **RQ2 partially answered** (5-fold
  leave-one-family-out CV: cross-family RF wins vs the real state-conditional
  RulePolicy on **3/5 BOCoDe families** (Branin, Ackley, Rastrigin; seed-block
  95% CI upper < 0); aggregate Cross − Rule = -0.065; loses to a constant
  `local_exploitation` policy (AlwaysLocal) on aggregate); RQ3 has one strong
  pilot (LLM > Surrogate on Rastrigin, p=0.0058), one negative pilot
  (router does not transfer to IOH, p=0.1376), and a closed-loop 10-seed
  cost frontier (450 paired trajectories) in which a frozen RF System-1 with
  zero LLM calls beats or ties every LLM-calling arm on all five families.
  Within-function dimension transfer (RQ2.2) is a clean negative: the RQ2.1
  signal is a family signal, not a dimension signal.
- This paper presents the framework + RQ1 + RQ2 partial pilot + RQ3 pilots;
  full RQ2 transfer with engineering families and a crossed (family × dimension)
  design is deferred.

## §2 Background & Related Work

- BO and BoTorch (briefly).
- Meta-learning for algorithm selection (Biedenkapp et al. on algorithm
  configuration; our analogue is per-trajectory action selection).
- LLM-as-decision-maker literature (recent; e.g., AutoML-Agent, Voyager).
- Decision-theoretic foundations: counterfactual decision-making, instrument
  of decision theory (no need to be exhaustive).

## §3 The v3 Framework

Architecture diagram: `state → policy → action → backend → new_state`.

### §3.1 Backend — `BocodeBotorchBackend`

- BOCoDe problem + BoTorch SingleTaskGP + LogExpectedImprovement.
- Single-objective, unconstrained, continuous box-bounded (audited allowlist).
- Checkpoint / restore preserves GP state + RNG (`backend.checkpoint()`).
- Optional **affine objective transform** `y → scale * y + shift` (Gate 5).

### §3.2 State — `OptimizationState` (5 frozen dataclasses)

```
SamplingState    — n, coverage, boundary_fraction, gap_to_best
SurrogateState   — fitted?, model_output_at_current_best, posterior_var_at_current_best
OptimizationState — best_y, recent_best_y, n_no_improve
ResourceState    — budget_remaining, fraction_used
HistoryState     — last_action, last_n_actions, action_diversity
```

### §3.3 Action — 8 typed `OptimizationAction`

`global_exploration`, `local_exploitation`, `uncertainty_sampling`,
`error_guided_sampling`, `boundary_sampling`, `surrogate_acquisition`,
`retrain_surrogate`, `stop`. Each has fixed params, fixed eval cost,
and an `execute_action` runner.

### §3.4 Utility — 4-term weighted sum

```
U(a|s) = w_optimization · Δopt / σ_init
       − w_evaluation · n_new_samples / total_budget
       − w_fit · n_model_fits / total_budget
       + w_stop_opportunity · I[STOP]
P*(a|s) = softmax(U(a|s) / T)
```

Picked empirically per Phase 8.6A so that 1 sample ≈ 1/total_budget
utility units, comparable to Δopt fractions.

### §3.5 Counterfactual Rollout

`CounterfactualRollout(s, candidates, horizon)` — for each candidate, runs
`backend.restore(checkpoint) → execute(candidate) → run continuation policy
for `horizon-1` steps → return utility`. Tries to restore on every step
to keep the backend state clean.

### §3.6 Decision Dataset — JSONL

Each record: `state + candidate_actions + gold_probs + rollout_results +
metadata + replay_checkpoint`. 385 unit tests cover the schema, generation,
validation.

### §3.7 System-1 + Router

`LayaPolicy` (typed-decision adapter) and `LLMSystem1Policy` (DeepSeek-chat
API or mock). `HybridPolicy` wraps a System-1 + confidence-based router that
falls back to `System2LLMDeepSeek`.

### §3.8 Theoretical results (full statements + proofs: `docs/THEORY.md`)

Four short results formalize claims the experiments measure:

- **P1 (affine invariance, exact).** Under `y → αy + β` (α > 0) with the
  scale `σ_opt = std(y_init)` recomputed, every utility label is unchanged
  (β cancels in the Δopt difference; α cancels against the scale), so
  gold_probs and all rank statistics are invariant by construction. This
  makes Gate 5's Spearman = 1.0 *necessary*, and localizes the 2/17 JS
  breaches to GP-fit numerics rather than the supervision signal. Verified
  through the actual `compute_utility` code path on 5,304 rollout branches
  (max relative error 1.4e-12; `scripts/verify_theory_propositions.py`).
- **P2 (`adaptive-value` = tight value of state-conditionality).**
  `D = E[max_a U] − max_a E[U] ≥ 0` equals the one-step regret of the best
  constant action against per-state oracle selection and tightly bounds the
  gain of *any* state-conditional policy over the best constant action.
  The Flat families (SixHumpCamel D = 0.012 at 20 seeds) admit no
  state-conditional gain by bound, which is why the RF probe finds no
  signal there. Recomputed at 20 seeds: Hartmann 0.710 > Ackley 0.332 >
  Rastrigin 0.262 > Branin 0.040 > SixHumpCamel 0.012.
- **P3 (routing degeneracy).** If the System-2's per-state advantage
  `A(s) = Q(s, a₂(s)) − Q(s, a₁(s)) ≤ 0` a.s., *every* router — oracle or
  learned — has non-positive value over never routing. The 10-seed sweep
  (0/40 arms beat `rf` with CI excluding zero) is the empirical signature
  of `A ≤ 0`, so the routing failure is a property of the LLM's decisions,
  not of the routing heuristic.
- **L1 (exact imitation of axis-aligned rules).** A random forest whose
  features contain an axis-aligned threshold rule's inputs reproduces the
  rule exactly under noiseless labels with no feature aliasing — explaining
  RQ2.2's `rf − rule_full = +0.0000` and the inherited losses to
  Surrogate/Local.

## §4 RQ2 — Cross-Family Transfer (Leave-One-Family-Out CV)

> **NEW 2026-09-27**: this section fills the RQ2 gap that PROGRESS_BRANCH §1
> explicitly flagged as 0%. The pilot is **partial** — exploratory, one-step,
> synthetic-only.

**Setup.** 5-fold leave-one-family-out CV with multi-output
`RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5)`
System-1 (one regressor per action; argmax = chosen action). Trained on
`runs/phase8_6d_h3_20seed_5family.jsonl` (661 non-terminal records, 5 families
× 20 seeds × horizon=3). Features: 15-dim state vector (12 numeric + 3 phase
one-hot), via `scripts/_shared_rq_features.py` (shared with `exp_rq1_3_rf_probe.py`).

**Baselines (all run on every held-out fold).**
- **AlwaysSurrogate** (constant) — always picks `surrogate_acquisition`
- **In-family RF** (LOSO reference, **not** an upper bound — see below)
- **AlwaysLocal** (constant) — always picks `local_exploitation` (informational)
- **RulePolicy (real, state-conditional)** — the actual `RulePolicy.select_action()` from
  `src/opt_agent/loop/policies.py`: STOP if remaining_budget ≤ 4, GLOBAL_EXPLORATION
  if stagnation_steps ≥ 3 AND budget ≥ 2, else SURROGATE_ACQUISITION. Reads state
  from each record to produce a per-record action choice.

**Metric.** Held-out decision regret = U(s, a*) − U(s, a_hat), where a_hat is
the model's argmax over predicted utilities and a* is the oracle argmax.
Lower is better.

**Inference.** Per-family paired bootstrap 95% CIs in two flavors — (a)
record-level (underestimates variance due to within-seed correlation) and (b)
seed-block (resamples whole seeds with their records, preserving within-seed
correlation; preferred per codex gpt-6-sol 2026-09-27 REVISE).

**Per-family results** (lower regret is better; negative diff = cross is BETTER):

| Held-out | dim | cross-fam regret | in-fam (ref) | AlwaysSurrogate | RulePolicy (real) | AlwaysLocal | Cross − Sur CI (seed-block) | Cross − Rule CI (seed-block) | Cross − Local CI (seed-block) |
|---|---|---|---|---|---|---|---|---|---|
| **Branin** | 2D | +0.0550 | +0.0415 | +0.0582 | +0.0730 | +0.0488 | `[-0.0145, +0.0086]` | **`[-0.0301, -0.0059]`** | `[-0.0029, +0.0162]` |
| Hartmann | 6D | +0.9520 | +0.5776 | +0.7674 | +0.9249 | +0.7095 | `[+0.0414, +0.3466]` | `[-0.1155, +0.2101]` | `[+0.0925, +0.3910]` |
| **Ackley** | 10D | +0.4674 | +0.3117 | +0.7051 | +0.6575 | +0.3324 | **`[-0.4342, -0.0762]`** | **`[-0.3350, -0.0770]`** | `[+0.0185, +0.2487]` |
| SixHumpCamel | 2D | +0.0501 | +0.0128 | +0.0397 | +0.0471 | +0.0176 | `[+0.0075, +0.0134]` | `[-0.0013, +0.0075]` | `[+0.0259, +0.0391]` |
| **Rastrigin** | 10D | +0.2245 | +0.2858 | +0.3807 | +0.3793 | +0.2639 | **`[-0.2511, -0.0837]`** | **`[-0.2472, -0.0716]`** | `[-0.1012, +0.0163]` |

**Headline (per codex gpt-6-sol 2026-09-27 REVISE, exploratory)**:

- **vs AlwaysSurrogate** (seed-block 95% CI): cross WINS on **2/5** (Ackley,
  Rastrigin); LOSES on 2/5 (Hartmann, SixHumpCamel); inconclusive on Branin.
- **vs RulePolicy (real, state-conditional)**: cross WINS on **3/5** (Branin,
  Ackley, Rastrigin); inconclusive on Hartmann, SixHumpCamel. **0/5 losses.**
- **vs AlwaysLocal (constant local_exploitation)**: cross LOSES on 3/5
  (Hartmann, SixHumpCamel, Ackley); inconclusive on Branin + Rastrigin.
  This is the weakest baseline (it ignores budget awareness and keeps
  exploiting locally even at low remaining budget).

**Two findings that sharpen the claim**:

1. **Lower one-step regret than the real state-conditional RulePolicy on 3/5 BOCoDe families**, including the
   10D multimodal pair (Ackley + Rastrigin) AND the 2D smooth Branin. The
   real, state-conditional RulePolicy is a stronger baseline than the
   constant AlwaysSurrogate, and the cross-family RF still wins on 3/5.
   Cross − RulePolicy records-weighted mean = **−0.065** (cross is
   better on aggregate).
2. **In-family is not an upper bound**. On Rastrigin, cross-family regret
   (+0.225) is **lower** than in-family regret (+0.286) — the in-family fit
   has access to the held-out family but does not dominate. The in-family
   RF is best described as a **reference**, not an upper bound.

**Aggregate (records-weighted across 5 folds)**:
- Cross-family RF mean regret: +0.351
- In-family RF mean regret: +0.246 (reference)
- AlwaysSurrogate mean regret: +0.389
- RulePolicy (real) mean regret: +0.416 ← **always stops / global-explorates at low budget, picks surrogate the rest**
- AlwaysLocal mean regret: +0.275 ← **best in aggregate because it always exploits locally**
- Cross − Surrogate records-weighted mean: -0.038
- Cross − RulePolicy records-weighted mean: **-0.065** (cross is better)
- Cross − AlwaysLocal records-weighted mean: +0.076 (cross is worse)
- Cross − In-family records-weighted mean: +0.105

**Pattern**: cross-family transfer **wins on 2/5 vs AlwaysSurrogate**
(Ackley, Rastrigin — both 10D), and **wins on 3/5 vs RulePolicy** (adds
Branin 2D). The cross family signal correlates with dimensionality but
family × dim are confounded. We do not claim dimension-specificity from
this single 5-family panel.

**Recommended paper wording (per codex gpt-6-sol REVISE)**:

> "We test whether a learned System-1 (multi-output RandomForestRegressor)
> trained on 4 BOCoDe families transfers to a held-out 5th. **Lower
> one-step regret than the real state-conditional RulePolicy on three of
> five tested family–dimension combinations** (Branin, Ackley, Rastrigin;
> seed-block 95% CI upper bound < 0). The cross-family RF also has lower
> regret than AlwaysSurrogate on the two 10D multimodal families (Ackley,
> Rastrigin). It does **not** beat the constant `local_exploitation`
> policy (AlwaysLocal) on aggregate, but that constant policy is
> unrealistic — it does not respond to budget. We conclude that **a learned
> System-1 transfers usefully across related BOCoDe families**, but the
> transfer is heterogeneous (two inconclusive families, two losses vs
> AlwaysSurrogate). Whether this transfers to engineering problems or to
> closed-loop trajectory evaluation is left for future work."

**Limitations specific to §4 RQ2**:
- **One-step decision regret, not closed-loop trajectory** — the RF is
  evaluated on its one-step choice against the oracle; we do not measure
  the effect of deploying the RF through a full BO trajectory.
- **Single model class (RF)** — MLP / GBM / Transformer may behave differently.
- **Family × dimension confounded** — disentangling requires a crossed design
  (same family at multiple dimensions). EXP-RQ2.2 below addresses the
  dimension axis within a fixed function.
- **No multiplicity adjustment** on the 5 per-family CIs.
- **Aggregate records-weighted mean is arithmetic**, not inferential for
  an average effect across new families.

### §4.6 Within-function dimension transfer (EXP-RQ2.2) — clean negative

**Why.** Every BOCoDe family has a fixed dimension, so RQ2.1's cross-family
result confounds family with dimension. EXP-RQ2.2 fixes the function and
varies only the dimension: a multi-output RF (same hyperparameters as
RQ2.1) trained on the action labels of `RuleFullBudget` at IOH functions
f6 (AttractiveSector), f8 (Rosenbrock), f15 (RotatedRastrigin) × source
dims {2, 10} × instances {0, 1} × seeds {100–102}, then deployed on the
untouched target dim 5 × instance 2 (72 trajectories total).

| Cell (test d5/i2, n=10 seeds) | `rf − rule_full` | `rf − surrogate` | `rf − local` |
|---|---:|---:|---:|
| f6 | **+0.000** [+0.000, +0.000] | **+0.128** [+0.009, +0.282] | **+0.153** [+0.006, +0.303] |
| f8 | **+0.000** [+0.000, +0.000] | **+0.244** [+0.059, +0.453] | **+0.321** [+0.141, +0.526] |
| f15 | **+0.000** [+0.000, +0.000] | **+0.225** [+0.088, +0.378] | +0.105 [−0.033, +0.265] |

(10-seed confirmation of the original 3-seed run; W/T/L and the 3-seed
numbers in `runs/EXP_RQ2_2_WITHIN_FUNCTION_DIM_TRANSFER_REPORT.md`.)

The RF imitates the rule perfectly on the held-out dimension (THEORY.md
Lemma 1 makes this exact, not expected), and both lose to the fixed
Surrogate baseline on every function — with CI excluding zero on all three
at 10 seeds. **Within-function dimension transfer is a clean negative**;
the RQ2.1 cross-family signal is at least partly a *family* signal, not a
dimension signal.

## §5 RQ1 — Decision Landscape

### §4.1 Three regimes (Step 1)

`adaptive-value(family) = E[max_a U] − max_a E[U]` computed across 354
counterfactual oracle labels (5 families × 10 seeds × horizon=3).

| Family | adaptive-value | Regime |
|---|---|---|
| Hartmann | 0.684 | Frequent opportunity |
| Rastrigin | 0.368 | Heavy-tailed |
| Ackley | 0.309 | Heavy-tailed |
| Branin | 0.048 | Flat |
| SixHumpCamel | 0.015 | Flat (clean negative control: 0% above δ=0.1) |

### §4.2 Counterfactual oracle replay (Step 1 — Gate 3)

20-seed 5-family dataset (`phase8_6d_h3_20seed_5family.jsonl`, 663 records).
Gate 3 (`aggregate_mass`): **adjusted_lift = 1.294, seed-clustered 95% CI = [0.848, 1.954]**,
n_seeds=100 (up from 10-seed's 1.124, CI [0.695, 1.722]).
Gate 4 stop-gain CI lo > 0 on 5/5 families.
Gate 4 uniform-gain CI lo > 0 on 4/5 (Branin, Hartmann, Rastrigin, SixHumpCamel).
Gate 4 behavior-gain CI lo > 0 on 1/5 (Rastrigin only) — honest finding,
not overclaimed.

### §4.3 Affine invariance (Step 1 — Gate 5, PARTIAL PASS)

Gold_probs invariance under `y → scale * y + shift` (see `runs/EXP_AFFINE_REPORT.md`).
Result on the 5-family panel × 4 transforms:
- **Median Spearman ρ = 1.0000 across all 17 (problem, phase) buckets**
  (consistent with the analytic argument: shift cancels in Δopt; scale
  cancels via the `optimization_scale = std(initial_obj)` denominator)
- **Argmax agreement ≥ 0.83 on all 17 buckets** (full ≥ 0.8)
- **JS divergence ≤ 0.02 on 15/17 buckets; predeclared Gate 5 fails
  because 2/17 Hartmann (late, mid) buckets breach the threshold due to
  GP-fit numerical fragility** — BoTorch's `optimize_acqf(LogEI)` lands on
  a duplicate X under shift=5, producing `delta_opt = 0` and a sharp
  utility drop. **This is a known numerical fragility, NOT a violation of
  the analytic rank invariance. Per-record rank invariance is not
  established.**

Paper claim: "Counterfactual oracle labels have median rank correlation
1.0 under affine y transforms, with a known near-tie caveat in
decision-rich (Hartmann) states; the predeclared joint Gate 5 fails on
2/17 buckets due to GP-fit numerical fragility."

### §4.4 State conditionality (Step 2 — RF probe)

Hartmann: RF regret 0.47 vs Surrogate 0.68 (−29%) — paired bootstrap
CI [−0.038, 0.415], t p=0.141 → suggestive pilot. Rastrigin/Ackley:
RF ≈ best-fixed, no state-conditional learnability demonstrated.
Branin/SixHumpCamel: too flat for learnability signal.

### §4.5 Closed-loop regret (Step 3)

For each (problem, seed), replay with one first action then AlwaysSurrogate
continuation. **Best first action across opportunity-rich families is
`local_exploitation`** (Hartmann 0.33 vs Surrogate 0.48; Rastrigin 0.00
vs Surrogate 11.10; Ackley 0.005 vs Surrogate 0.03).

## §6 RQ3 — Selective Reasoning

### §5.1 Joint 10-seed 5-policy comparison (Student-t corrected)

| Family | LLM − Jev (=Surrogate) | p (paired-t, df=n-1, two-sided) |
|---|---|---|
| Branin | -2.267 (CI [-3.375, -1.159]) | **0.0012** (LLM worse) |
| Hartmann | -0.985 (CI [-1.423, -0.547]) | **0.0007** (LLM worse) |
| Ackley | +0.461 (CI [-0.372, +1.294]) | **0.2423** (LLM slightly better, NS) |
| SixHumpCamel | -0.247 (CI [-0.487, -0.007]) | **0.0446** (LLM worse) |
| **Rastrigin** | **+17.540** (CI [+6.500, +28.579]) | **0.0058** (LLM better) |

All p-values use Student-t distribution with df=n-1 (n=10 paired seeds).
Source: `scripts/exp_rq0_5_proper_stats.py`. The early-report
normal-approximation values were **stale** and inflated significance
(e.g., Branin and Hartmann were reported as p<0.0001; the corrected
Student-t values are 0.0012 and 0.0007 respectively).

### §5.2 Mechanism: global-to-local temporal action pattern (correlational)

From `runs/exp_rq0_5_action_distribution.md` — On Rastrigin (10D multimodal),
the LLM exhibits a two-phase temporal pattern: predominantly `global_exploration`
at steps 0–9 (90–100% across these steps; 1/10 trajectories spike to
`local_exploitation` at steps 3, 6, 8, 9); mixed at steps 10–12 (30–40% global,
60–80% local); 100% `local_exploitation` from step 13 onward. AlwaysSurrogate
picks `surrogate_acquisition` 100% of all 25 steps.

**Important caveats**:
- The earlier §5 of `EXP_RQ1_3_REPORT.md` hypothesized the LLM might be picking
  `local_exploitation` (the closed-loop best first action) — that hypothesis is
  **rejected** by the per-step distribution (LLM picks `global_exploration` 70%
  on Rastrigin, not `local_exploitation`).
- The LLM's **pooled** top action on Rastrigin is `global_exploration` at 70%,
  which is **similar to its share on smoother families** (Branin 76%, Hartmann
  67%, Ackley 66%, SixHumpCamel 68%) — *families where LLM does NOT beat
  Surrogate*. The pooled distribution alone does not explain the Rastrigin-
  specific advantage; only the *temporal pattern* is unique.
- **Deterministic schedule control now run (NEW)**: the predeclared
  `GlobalThenLocal` baseline (10 global decisions, then local) was run on
  the exact RQ3 panel (seeds 100–109, paired starts verified). It **beats
  the real LLM with CI excluding zero on Hartmann (+0.334), SixHumpCamel
  (+0.674), and Rastrigin (+9.628)** and never loses to it; it ties the
  frozen RF on three families (RF keeps Ackley and SixHumpCamel).
  On Rastrigin the ordering is schedule ≈ rf > llm, which **subsumes** the
  earlier "LLM > AlwaysSurrogate on Rastrigin" pilot: the schedule alone
  reproduces and exceeds the LLM's value. See §5.5.
- **Recommended paper wording (updated)**: *"On Rastrigin, the LLM
  exhibited a global-to-local temporal action pattern associated with its
  advantage over AlwaysSurrogate. A deterministic two-phase schedule
  realizing that pattern matches the LLM's value and exceeds it on three
  of five families, so the schedule — not LLM-specific per-step reasoning
  — accounts for the LLM's measurable decision value on this panel."*

### §5.3 FrozenRouter: BOCoDe advantage does NOT transfer

| Panel | FrozenRouter − Surrogate | p |
|---|---|---|
| BOCoDe (5-family) | +15.69 (sum) | 0.012 |
| IOH (f1–f5, dim 2/5/10/20) | +0.036 (mean normalized regret) | 0.138 |

**Negative result**: a simple `dim≥5` router trained on BOCoDe does not
generalize to IOH/BBOB at the same scale.

### §5.4 RoutingAware vs LLM-everywhere (negative finding)

On the BOCoDe panel sum, RoutingAware − LLM = +1.18 (paired-t p=**0.5134**,
not significant). The simple dimension-based routing rule is NOT measurably
better than LLM-everywhere on this benchmark. **The Pareto frontier
deferral**: more elaborate routing needs RQ2 transfer experiments to
become a defensible claim.

### §5.5 Real-LLM cost frontier, closed loop, 10 seeds (EXP-RQ3)

Frozen RF System-1 (trained on rollout seeds 0–15, temperature-calibrated on
16–19) deployed through full budget-30 BO trajectories against the same
`deepseek-chat` System-2, with confidence-threshold routers (τ ∈ {0.25,
0.5, 0.75, 0.9}) and random-rate baselines (rate ∈ {0.25, 0.5, 0.75}).
**450 paired trajectories** (5 families × seeds 100–109 × 9 policies),
identical Sobol starts per (family, seed), zero LLM fallbacks, identical
frozen-model hash across both sweep files.

| Family | `rf − llm` (mean) | `llm − rf` 95 % CI | verdict |
|---|---:|---|---|
| Branin | +0.515 | [-1.224, +0.152] | rf better in mean, CI crosses 0 |
| Hartmann | +0.576 | **[-1.032, -0.103]** | rf detectably better |
| Ackley | +1.017 | **[-1.658, -0.436]** | rf detectably better |
| SixHumpCamel | +1.015 | **[-1.512, -0.587]** | rf detectably better |
| Rastrigin | +10.109 | [-22.356, +0.404] | rf better in mean, CI crosses 0 |

**No arm on the whole grid (8 LLM-calling arms × 5 families) beats `rf`
with a CI excluding zero**; the only positive point estimates anywhere are
Hartmann/confidence_0.25 (+0.018) and Hartmann/random_0.25 (+0.149), both
CI-crossing. Always-LLM spends ~12–17 calls (~4.7–6.5k tokens) per
trajectory and is dominated. The pre-specified acceptance criteria (≥ 50 %
call/token reduction at ≤ 0.10 σ_family regret margin) pass trivially via
the System-1-only arm — the measured frontier is degenerate.

**Sign correction (documentation integrity).** The first 3-seed version of
this experiment inverted the sign of its own Hartmann numbers in prose
("the one family where the LLM is genuinely useful"); the recorded data had
`rf` winning 3/0/0. The corrected reading agrees with §5.1 above (LLM
−0.985 on Hartmann, p=0.0007). See the correction block at the top of
`runs/EXP_RQ3_REAL_LLM_COST_FRONTIER_REPORT.md`.

**Routing diagnostic.** The 3-seed "random rate beats the confidence
router" observation does not generalize: at 10 seeds neither scheme
produces a CI-detectable gain over `rf`, and Branin random_0.5/0.75 are
CI-detectably worse. The RF softmax maximum rarely exceeds 0.25, so
confidence thresholds route nearly every step to System-2;
`ConfidenceRouter.tau_conf` is not a useful routing signal on this panel.

**Schedule control (mechanism).** A predeclared deterministic
`GlobalThenLocal` control (10 global decisions, then local) run on this
exact panel (paired starts verified) **beats the LLM arm with CI excluding
zero on Hartmann (+0.334), SixHumpCamel (+0.674), and Rastrigin (+9.628)**
and never loses to it; it ties `rf` on Branin/Hartmann/Rastrigin while RF
keeps Ackley (−0.853) and SixHumpCamel (−0.341). The LLM's closed-loop
decision content is dominated by the fixed two-phase schedule its own
per-step distribution exhibits — and by THEORY.md P3, no router could have
rescued an arm with nonpositive per-state advantage. Artifacts:
`runs/exp_rq3_schedule_baseline.json`,
`scripts/exp_rq3_schedule_baseline.py`.

## §7 Limitations

(per codex round-1 review 2026-09-27; ordered roughly by importance)

1. **RQ2 partially answered** — `EXP-RQ2.1` (5-fold leave-one-family-out
   RF) shows cross-family transfer has **lower one-step regret than the
   real state-conditional RulePolicy on 3/5 BOCoDe families** (Branin,
   Ackley, Rastrigin; seed-block 95% CI upper < 0). It also has lower
   regret than AlwaysSurrogate on 2/5 (Ackley, Rastrigin — both 10D).
   It has higher regret than AlwaysSurrogate on 2/5 (Hartmann, SixHumpCamel);
   inconclusive on Branin. It loses to AlwaysLocal on aggregate. The pattern
   is family-specific; not a uniform "transfer works" claim. Family ×
   dimension confounded: a crossed design (same problem at multiple dimensions)
   is required to disentangle — deferred.
2. **Synthetic five-family panel only** — no real engineering objective.
   Family, dimension, smoothness, and multimodality are **confounded**;
   "Hartmann beats Branin" could be a dimensionality effect, a smoothness
   effect, or a multimodality effect.
3. **Gate 2 fails** — stable state-to-action learnability is **not demonstrated**
   (5/15 buckets pass at top_action_share ≥ 0.6 threshold).
4. **Gate 5 fails on 2/17 buckets** — predeclared joint test (`pass_: false`)
   because JS divergence exceeds 0.02 on Hartmann (late, mid) buckets. Median
   Spearman = 1.0 in all (problem, phase) buckets (consistent with the
   analytic rank-invariance argument; per-record invariance not established).
5. **20-seed is not an independent replication** — seeds 0–9 are reused
   from the 10-seed exploratory run. True replication would require an
   independent seed stream (e.g., 100–119).
6. **Pareto frontier measured but degenerate** — the closed-loop cost
   frontier (§5.5) is measured at budget 30 on the five-family panel with
   one provider (deepseek-chat): System-1-only dominates, so there is no
   interesting quality-cost tradeoff on this panel. Whether any regime
   (longer horizons, stochastic objectives, genuinely OOD states) gives
   the LLM System-2 positive marginal value is open.
7. **No external baseline** — lenz / PlugBO / TuRBO not yet compared. The
   internal baselines (Rule, AlwaysSurrogate, Jev) are sufficient for RQ3
   pilot framing but not for state-of-the-art comparison.
8. **Single LLM provider / prompt configuration** — only deepseek-chat via
   DeepSeek API. Other LLMs (GPT-4, Claude, Gemini) may exhibit different
   decision patterns; we have no evidence the Rastrigin finding transfers,
   and the §5.5 cost-frontier result is provider-specific.
9. **Mechanism: schedule-sufficiency shown, full causal decomposition not** —
   the deterministic global-then-local control now exists and dominates the
   LLM (3/5 families, CI excluding zero; never loses), which establishes
   that the schedule *suffices* for — and exceeds — the LLM's closed-loop
   value on this panel. This is stronger than the earlier correlational
   statement but still not a step-level causal decomposition of the LLM's
   individual decisions.
10. **Per-step trajectory attrition** — by step 15, only 1 LLM trajectory
    has not yet terminated; later-step means are computed on small subsamples.
    The +45.7 advantage at step 15 is illustrative, not inferential.
11. **Researcher-designed choices without sensitivity sweep** — utility
    weights, action set, horizon=3, regime taxonomy, and top_action_share
    threshold (0.6) are researcher-chosen; sensitivity to these choices is
    not analyzed.
12. **Multiple unadjusted comparisons** — per-family CIs are unadjusted for
    multiple comparisons; some borderline-pass CIs (Branin uniform CI lo
    +0.006; Rastrigin behavior CI lo +0.015) would not survive a
    Bonferroni-type correction.
13. **Learned System-1 closed-loop evidence is negative-to-heterogeneous** —
    EXP-RQ2.1 is one-step decision regret; the closed-loop audit
    (`runs/EXP_FINAL_GOAL_CLOSED_LOOP_REPORT.md`) and §5.5 deploy frozen RFs
    through full trajectories. The RF beats AlwaysSurrogate on Ackley but
    not the stronger AlwaysLocal control, and dominates every LLM arm at
    zero cost on this panel; a general learned-policy gain is not
    established. The "RoutingAware" rule remains hand-coded (dim ≥ 5).
14. **Summing raw best_y across families is not principled** — different
    problems have different objective scales. The TOTAL comparison is
    informative but not on a common scale; per-family + ratio-of-advantages
    is the cleaner framing.

## §8 Discussion

- "Is decision-intelligence learnable?" — partially yes (Hartmann
  suggestive; flat families no signal).
- "Does it transfer across problems?" — see §4 RQ2 below.
  Cross-family has lower regret vs real RulePolicy on 3/5 (Branin, Ackley,
  Rastrigin) and vs AlwaysSurrogate on 2/5 (Ackley, Rastrigin); higher regret
  vs AlwaysSurrogate on Hartmann + SixHumpCamel; inconclusive on Branin.
- "Is LLM-as-System-1 worth it?" — against AlwaysSurrogate, only on hard
  multimodal problems (Rastrigin +17.54, p=0.0058). But in closed loop
  against a frozen RF System-1 (§5.5), the always-LLM policy never wins
  with a CI excluding zero and loses detectably on 3/5 families at ~12–17
  calls per trajectory. The LLM's positive value on this panel is confined
  to the one-step comparison against weaker baselines.
- "Is routing a Pareto win?" — measured (§5.5): no. No confidence threshold
  or random rate earns its cost over System-1-only; the frontier is
  degenerate on this panel. Whether any regime gives routing positive
  value is the open question.
- "Does the trained System-1 transfer across problems?" — see §4 RQ2 below
  (wins vs real RulePolicy on 3/5; wins vs AlwaysSurrogate on 2/5; loses to a
  constant local policy on aggregate).

## §9 Conclusion

- Three contributions: (1) a 7-component decision-centric BO framework
  with **385 unit tests** (was 364 pre-Gate 5; +21 in `test_objective_transform.py`)
  and four formal propositions tied to the experiments (`docs/THEORY.md`:
  affine label invariance, adaptive-value bound, routing degeneracy,
  exact imitation of axis-aligned rules); (2) a counterfactual oracle
  dataset validation methodology that closes
  Gate 3 (PASS at 20 seeds, adjusted_lift CI [0.848, 1.954]) and Gate 4
  stop-gain (5/5), with **Gate 2 still failing** (5/15 buckets pass at
  top_action_share ≥ 0.6) and **Gate 5 partial** (median Spearman = 1.0
  necessarily by P1; predeclared joint fails on 2/17 buckets); (3) a
  global-to-local temporal action pattern whose *sufficiency* is now
  demonstrated: a deterministic two-phase schedule reproduces and exceeds
  the LLM's closed-loop value on this panel.
- **RQ2 (NEW)**: cross-family transfer **wins vs the real state-conditional
  RulePolicy on 3/5 BOCoDe families** (Branin, Ackley, Rastrigin; seed-block
  95% CI upper < 0). It also wins vs AlwaysSurrogate on 2/5 (Ackley,
  Rastrigin). Aggregate Cross − Rule = -0.065 (cross is better). The cross
  signal is heterogeneous (loses on Hartmann, SixHumpCamel vs Surrogate);
  family × dimension confounded.
- **RQ3 (cost frontier, NEW)**: in closed loop at 10 paired seeds the
  frozen RF System-1 (0 LLM calls) beats or ties every LLM-calling arm on
  all five families; always-LLM is CI-detectably worse on Ackley, Hartmann,
  and SixHumpCamel. No confidence-threshold or random-rate router earns
  its cost. The measured frontier is degenerate: System-1 only.
- Six negative results that sharpen the claim: (a) trained router does
  not transfer from BOCoDe to IOH; (b) RoutingAware does not beat
  LLM-everywhere on the same panel; (c) cross-family transfer does not
  help Hartmann or SixHumpCamel vs AlwaysSurrogate; (d) cross-family RF
  loses to a constant `local_exploitation` policy (AlwaysLocal) on
  aggregate because that policy ignores budget awareness; (e) within-
  function dimension transfer fails (RQ2.2 at 10 seeds — the RQ2.1 signal
  is a family signal, not a dimension signal); (f) in closed loop,
  always-LLM is dominated by both the frozen System-1 and a deterministic
  two-phase schedule (§5.5) — and by P3 no router could have fixed this.
- Future work: regimes where an LLM System-2 repays its cost (longer
  horizons, stochastic objectives, OOD states), external lenz/PlugBO
  baseline, fine-tuning a small decision model from LLM trajectories,
  dimension-crossed design to disentangle dimension vs family effects,
  engineering families (vehicle suspension).

---

## Appendix

- A1 — Test inventory (385 tests, file-by-file)
- A2 — Replay / checkpoint invariants
- A3 — Per-seed action distribution tables
- A4 — Full CI tables for Gate 3/4/5

---

## File references (one-line per claim)

- `docs/THEORY.md` — P1/P2/P3/L1 with proofs, scope conditions, and the
  proposition-to-evidence map
- `scripts/verify_theory_propositions.py` → `runs/verify_theory_propositions.json`
  — P1 checked through `compute_utility` on 5,304 branches (max rel err
  1.4e-12); P2 recomputed at 20 seeds
- `runs/exp_rq3_schedule_baseline.json` + `scripts/exp_rq3_schedule_baseline.py`
  — deterministic schedule control on the RQ3 panel (paired starts verified)
- `runs/exp_rq2_2_ioh_within_function_dim_transfer_10seed.json` — RQ2.2
  10-seed confirmation
- `runs/phase8_6d_h3_20seed_5family.jsonl` — Track A dataset
- `runs/exp_rq1_2_h3_20seed_report.json` — Track A gate results
- `runs/EXP_RQ0_5_REPORT.md` — Track C source (joint 10-seed)
- `runs/exp_rq0_5_action_distribution.md` — Track D mechanism analysis
- `runs/EXP_AFFINE_REPORT.md` — Track B Gate 5 result
- `runs/exp_rq3_router_sweep.json` + `runs/exp_rq3_router_sweep_seeds103_109.json`
  + `runs/exp_rq3_cost_frontier_10seed.json` — RQ3 cost frontier
  (450 paired trajectories; merged paired bootstrap analysis)
- `scripts/analyze_rq3_cost_frontier.py` — merged analysis; reproduces the
  3-seed numbers exactly on the original file (how the sign error was found)
- `runs/EXP_RQ2_2_WITHIN_FUNCTION_DIM_TRANSFER_REPORT.md` — RQ2.2 clean negative
- `runs/EXP_FINAL_GOAL_CLOSED_LOOP_REPORT.md` — closed-loop audit
  (`rf_all` / `rf_transfer` vs fixed controls, seeds 100–119 / 200–219)
- `src/opt_agent/backend/bocode_botorch.py` — affine transform (Gate 5)
- `src/opt_agent/dataset/generator.py` — DatasetConfig affine fields
- `tests/test_objective_transform.py` — 16 new tests for Gate 5
# Workshop Paper Outline — Decision-Centric BO

> **Status**: draft outline, 2026-09-27 (updated same day: Tracks A/B/D complete;
> RQ3 real-LLM cost frontier added at 10 seeds; RQ2.2 dimension-transfer
> negative added; RQ3 router-sweep sign error corrected).
> **2026-09-28 internal-consistency revision**: abstract no longer says the
> schedule control "has not been run" (it contradicted its own point 4 and
> §6.2/§6.5); RQ2.1 partial-transfer added to the abstract (was "not yet
> demonstrated"); §5.1 regime table moved to the 20-seed recomputation
> (10-seed Rastrigin/Ackley order swap noted); subsection numbering fixed
> (§4.1–4.5 under §5 and §5.1–5.5 under §6 renumbered to match their
> parents); test count 385→423; title aligned with the schedule-sufficiency
> finding (previous title kept below as an alternative).
> **2026-09-28 content update**: the branch RLCD line is integrated (§4.2
> fine-tuned-model cross-family transfer + GearTrain probe; §5.6 teacher
> noise + decision-level distillation + closed-loop collapse; limitations
> 15–16; discussion/conclusion/abstract updated). All quoted numbers were
> re-verified against `runs/*.json` in the same session. A fresh codex
> review pass is recommended before submission — the prior APPROVE predates
> this integration.
> **Target venue**: ICLR / NeurIPS workshop on Meta-Learning for Optimization
> or AutoML/BO workshop; NOT a main-track submission.
> **Length**: 8 pages + appendix.

---

## Title (draft)

**"Decision-Centric Bayesian Optimization: A Decision-Landscape Framework and an LLM System-1 Whose Value a Two-Phase Schedule Subsumes"**

> Alternative (previous title, dropped 2026-09-28): *"...and the Surprising
> Value of an LLM as System-1"* — it over-promises relative to the paper's
> own mechanism result: the deterministic two-phase schedule reproduces and
> exceeds the LLM's closed-loop value (§6.2/§6.5), and the LLM is detectably
> worse than Surrogate on 3/5 families in the one-step comparison (§6.1).
> ("Reduces to a Two-Phase Schedule" was also considered and dropped:
> reduction is a causal claim; the evidence establishes sufficiency —
> per codex design adjudication 2026-09-28.)

---

## Abstract (≤ 200 words)

Bayesian Optimization (BO) reduces sequential control decisions (refit,
switch acquisition, expand, stop) to one fixed acquisition function. We
re-frame them as a meta-decision: at each state choose one of eight typed
strategy actions. On a 5-family BOCoDe panel × 20 paired seeds
(independently seed-stream replicated) we measure: (1) a decision
landscape — three regimes via an adaptive-value bound on counterfactual
oracle labels; (2) label affine invariance — median Spearman ρ = 1.0 in
all 17 buckets; the predeclared joint gate fails on 2/17; (3) an
LLM System-1 — beats AlwaysSurrogate on Rastrigin (+17.54, p=0.0058) yet
is detectably worse on 3/5 families, and a deterministic two-phase
schedule subsumes its closed-loop value; a dim≥5 router does not transfer
to IOH; (4) a closed-loop cost frontier (450 paired trajectories) — a
frozen RF System-1 with zero LLM calls dominates every LLM-calling arm;
(5) distillation — RLCD fine-tuning lifts decision-level ranking 0.40 →
0.87 and transfers cross-family (BT arm +0.140/+0.188 on two seed streams), but collapses to a constant policy in closed loop.
Contributions: a 7-component framework (423 tests), four formal
results tied to the experiments, and honest negatives — on this panel no
learned policy beats a constant local baseline in closed loop.

---

## §1 Introduction

- BO is usually presented as `x → f(x) → x*`, hiding the **sequential
  control decisions** the practitioner makes between sample acquisitions.
- We re-frame as `s_t → a_t → backend → s_{t+1}` — a meta-policy π(a|s).
- Three RQs: Learnability (RQ1), Generalization (RQ2), Selective Reasoning (RQ3).
- **Status**: RQ1 partially settled (Gate 3 adjusted_lift 1.294 CI [0.848, 1.954] at 20 seeds;
  Gate 4 stop-gain 5/5; **Gate 5 partial — median Spearman = 1.0 but predeclared joint
  fails on 2/17 buckets; Gate 2 still fails**); the teacher itself is noise-dominated
  (single-roll pair flip 0.40 above-median) yet distillable at decision level
  (RLCD fine-tuning of Laya: 0.40 → 0.87 pairwise; closed loop collapses to a
  constant policy, §5.6); **RQ2 partially answered** (5-fold
  leave-one-family-out CV: cross-family RF wins vs the real state-conditional
  RulePolicy on **3/5 BOCoDe families** (Branin, Ackley, Rastrigin; seed-block
  95% CI upper < 0); aggregate Cross − Rule = -0.065; loses to a constant
  `local_exploitation` policy (AlwaysLocal) on aggregate; an RLCD fine-tuned
  model transfers at decision level, +0.140 / +0.188 on two independent
  seed streams, 5/5 families each — family-level locking still open, §4.2); RQ3 has one strong
  pilot (LLM > Surrogate on Rastrigin, p=0.0058), one negative pilot
  (router does not transfer to IOH, p=0.1376), and a closed-loop 10-seed
  cost frontier (450 paired trajectories) in which a frozen RF System-1 with
  zero LLM calls beats or ties every LLM-calling arm on all five families;
  a predeclared deterministic two-phase schedule reproduces and exceeds the
  LLM's closed-loop value (§6.2/§6.5).
  Within-function dimension transfer (RQ2.2) is a clean negative: the RQ2.1
  signal is a family signal, not a dimension signal.
- This paper presents the framework + RQ1 + RQ2 partial pilot + RQ3 pilots;
  full RQ2 transfer with engineering families and a crossed (family × dimension)
  design is deferred.

## §2 Background & Related Work

- BO and BoTorch (briefly).
- **Agentic BO (Sara/lenz, Brunzema et al. 2026)** — nearest neighbor: open
  command action space (probe/reconfigure/propose/evaluate/stop) over a
  BoTorch backend, full (trials, config, instructions, deliberation-history)
  state, no learned policy, token cost reported but not optimized. Our
  closed 8-action space over a compressed 15-scalar process state is the
  opposite trade: expressiveness for measurability — closure enables
  exhaustive per-decision counterfactual labels over our predefined menu
  (and with them the predeclared gates and the noninferiority cost
  frontier); it does not rule out counterfactual comparisons elsewhere.
  Full state/action design spectrum:
  `docs/ref_papers/state_action_design_survey.md`.
- Meta-learning for algorithm selection (Biedenkapp et al. on algorithm
  configuration; TempoRL's skip-actions are the anchor for our zero-cost
  actions and when-to-route; our analogue is per-trajectory action selection).
- LLM-as-decision-maker literature (LLAMBO ICLR 2024 — prompt-state with the
  full evaluation history, action = next x; GPTOpt 2025 — fine-tuned
  LLM-as-optimizer; both act in x-space, we act at the strategy layer).
- Decision-theoretic and classical-BO foundations (one paragraph in the
  paper, full map in `docs/THEORY.md` "Relation to classical theory"):
  the GP posterior is the classical sufficient statistic our 15-scalar
  state projects; EI/GP-UCB are closed-form one-step decision rules — our
  utility-plus-menu lifts that structure to the strategy level where no
  closed form exists; GP-Hedge (UAI 2011) and its portfolio successors
  allocate among rules by online bandits, and LMABO (ICLR 2026) delegates
  that allocation to a zero-shot LLM — our P2 is the decision-theoretic
  counterpart of those information-theoretic acquisitions (a
  perfect-information value measured in utility units, in the Howard-1966 /
  Lindley-1956 value-of-information lineage; ES/PES target the optimum's
  location and MES its value — a related but different functional, not a
  unit conversion) and P3 gives the
  degenerate case the portfolio line assumes away (per-state advantage
  ≤ 0 ⇒ no allocation helps), which our cost frontier measures.

## §3 The v3 Framework

Architecture diagram: `state → policy → action → backend → new_state`.

### §3.1 Backend — `BocodeBotorchBackend`

- BOCoDe problem + BoTorch SingleTaskGP + LogExpectedImprovement.
- Single-objective, unconstrained, continuous box-bounded (audited allowlist).
- Checkpoint / restore preserves GP state + RNG (`backend.checkpoint()`).
- Optional **affine objective transform** `y → scale * y + shift` (Gate 5).

### §3.2 State — `OptimizationState` (5 frozen dataclasses)

```
SamplingState        — n_samples, dimension, min_pairwise_distance, boundary_coverage
SurrogateState       — is_current, validation_rmse (in-sample), mean/max posterior std (64-pt Sobol grid; NaN when unfitted)
OptimizationProgress — best_value, last_value, recent_improvement, stagnation_steps
ResourceState        — used/total/remaining budget, remaining_budget_ratio
HistoryState         — last_action only (no action-sequence memory)
```

Frozen + JSON-safe (NaN→None); `to_payload()` exposes 15 keys for LLM
prompts; the RF System-1 uses the same 15 numbers + 3 phase one-hots.
Phase (early/mid/late) is derived downstream from remaining_budget_ratio,
not stored. 2026-09-28: field list corrected to match
`src/opt_agent/state/models.py` (the previous draft listed
gap_to_best / last_n_actions / action_diversity, which were never
implemented).

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
metadata + replay_checkpoint`. 423 unit tests cover the schema, generation,
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
  gain of *any* state-conditional policy over the best constant action —
  **scope: this fixed 8-action menu, the sampled states, and the rollout
  utility**; it does not bound closed-loop terminal value or gains from an
  expanded action menu.
  The Flat families (SixHumpCamel D = 0.012 at 20 seeds) admit no
  state-conditional gain by bound, which is why the RF probe finds no
  signal there. Recomputed at 20 seeds: Hartmann 0.710 > Ackley 0.332 >
  Rastrigin 0.262 > Branin 0.040 > SixHumpCamel 0.012.
- **P3 (routing degeneracy).** If the System-2's per-state advantage
  `A(s) = Q(s, a₂(s)) − Q(s, a₁(s)) ≤ 0` a.s., *every* router — oracle or
  learned — has non-positive value over never routing. The 10-seed sweep
  (0/40 arms beat `rf` with CI excluding zero) is CONSISTENT with this
  premise (we measure the routing outcomes, not A(s) itself — see §6.5's
  conditional wording and limitation 9).
- **L1 (representability of axis-aligned rules by trees).** Axis-aligned
  regression trees CAN represent an axis-aligned threshold rule exactly;
  under noiseless labels, no feature aliasing, and sufficient per-leaf
  samples, a fitted forest can recover it. The RQ2.2 observation
  (`rf − rule_full = +0.0000` on all functions) is the empirical
  realization of this representability, and explains the inherited
  losses to Surrogate/Local.

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
  evaluated on its one-step choice against the oracle. (Closed-loop RF
  deployment IS measured separately — §6.5's cost frontier and the
  final-goal closed-loop audit — but not under the §4 cross-family
  transfer comparison.)
- **Single model class (RF)** — MLP / GBM / Transformer may behave differently.
- **Family × dimension confounded** — disentangling requires a crossed design
  (same family at multiple dimensions). EXP-RQ2.2 below addresses the
  dimension axis within a fixed function.
- **No multiplicity adjustment** on the 5 per-family CIs.
- **Aggregate records-weighted mean is arithmetic**, not inferential for
  an average effect across new families.

### §4.1 Within-function dimension transfer (EXP-RQ2.2) — clean negative

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

### §4.2 Fine-tuned System-1 cross-family transfer (EXP-RLCD, analysis-rerun)

> NEW 2026-09-28 (branch `exp/rlcd-laya-finetune`; codex journal-round
> review APPROVE on the underlying reports). **Status caveat**: labeled
> *analysis-rerun* — the seed-reuse half was discharged same-day by the
> independent-stream replication below (+0.188, same 5/5 significance);
> the family-reuse half is NOT discharged (the same five families informed
> the design; a locked family-level eval requires new allowlisted
> problems — the backend allowlist currently holds exactly these 5 + the
> consumed GearTrain probe).

Setup: RLCD fine-tuned Laya checkpoints (BT and proper-score arms; trained
on seeds 0–15, early-stopping on 16–19 of the 20-seed 5-family dataset),
evaluated leave-one-family-out (5 folds, dual training seeds, 663 records;
`runs/exp_rlcd_loo_confirmatory.json`).

- Pooled cross-family pairwise accuracy (fine-tuned vs RF baseline, per-record
  paired bootstrap): **bt +0.140 [+0.126, +0.154]**; proper +0.019
  [+0.007, +0.032]. **Per-family directions differ by arm**: the BT arm
  improves over RF on 5/5 families (sixhump +0.305, ackley +0.209,
  rastrigin +0.151, branin +0.045, hartmann +0.016 — all CIs exclude 0);
  the proper arm is positive on 3/5 (ackley +0.089, sixhump +0.144,
  rastrigin +0.048) and NEGATIVE on branin (−0.095) and hartmann
  (−0.065) — its pooled positive delta is carried by the three winners.
- **Independent-seed-stream replication (2026-09-28, seeds 100–119, train
  AND eval both on the new stream;
  `exp_rlcd_loo_confirmatory_replication_s100_119.json`)**: bt − RF
  **+0.188 [+0.175, +0.202]** (5/5 families CI-positive again); proper −
  RF +0.052 [+0.041, +0.064] pooled, with the same per-arm pattern
  (positive ackley +0.161 / sixhump +0.072 / rastrigin +0.087; negative
  branin −0.034 / hartmann −0.053). The per-family strong/small grouping
  replicates (ackley/sixhump/rastrigin strong, branin/hartmann small;
  the top two exchange places across streams). The seed-reuse half of the
  analysis-rerun caveat is discharged; the family-reuse half is not
  (same 5 informed-design families — a locked family-level eval needs
  new allowlisted problems).
- Round-1 LOO (E5, `exp_rlcd_loo_transfer.json`): cross-family ranking
  gap vs in-family doubled (+0.056 → +0.146); bt > RF on 5/5 folds
  (one-sided sign test p=0.031).
- **Engineering probe (GearTrain)**: decision-level transfer holds —
  bt 0.893 ≈ RF 0.890 > proper 0.801 > zero-shot 0.423 (65 records,
  1,595 pairs; `exp_geartrain_transfer_probe.json`). Full
  engineering-simulator transfer remains unrun.
- Relation to §4 (RF): the RF System-1 transfers with lower one-step
  decision regret on 3/5 families; the fine-tuned model additionally
  recovers the teacher's *calibration* (proper arm, §5.6) and holds on an
  engineering family — but neither closes the loop (§5.6, limitation 15).

## §5 RQ1 — Decision Landscape

### §5.1 Three regimes (Step 1)

`adaptive-value(family) = E[max_a U] − max_a E[U]`. Primary numbers are the
**20-seed recomputation** on the confirmatory dataset
(`runs/verify_theory_propositions.json`, 663 records; same source as §3.8
P2). The original 10-seed landscape (354 labels) read Hartmann 0.684 >
Rastrigin 0.368 > Ackley 0.309 > Branin 0.048 > SixHumpCamel 0.015 — the
**Rastrigin/Ackley order swaps at 20 seeds**, but both remain in the same
regime, so the taxonomy is unchanged.

| Family | adaptive-value (20 seeds) | Regime |
|---|---|---|
| Hartmann | 0.710 | Frequent opportunity |
| Ackley | 0.332 | Heavy-tailed |
| Rastrigin | 0.262 | Heavy-tailed |
| Branin | 0.040 | Flat |
| SixHumpCamel | 0.012 | Flat (clean negative control: 0% above δ=0.1) |

### §5.2 Counterfactual oracle replay (Step 1 — Gate 3)

20-seed 5-family dataset (`phase8_6d_h3_20seed_5family.jsonl`, 663 records).
Gate 3 (`aggregate_mass`): **adjusted_lift = 1.294, seed-clustered 95% CI = [0.848, 1.954]**,
n_seeds=100 (up from 10-seed's 1.124, CI [0.695, 1.722]).
**Independent-stream replication (seeds 100–119, 654 records,
`exp_rq1_2_h3_20seed_replication_s100_119.json`): adjusted_lift 1.296,
CI [0.881, 1.867], PASS** — essentially identical.
Gate 4 stop-gain CI lo > 0 on 5/5 families (5/5 again on the replication
stream). Gate 4 uniform-gain CI lo > 0 on 4/5 (Branin, Hartmann, Rastrigin,
SixHumpCamel); the replication stream also passes 4/5 but with a DIFFERENT
set (Ackley, Branin, Hartmann, SixHumpCamel — Rastrigin swaps out), so the
count replicates while the passing family set does not. Gate 4
behavior-gain CI lo > 0 on 1/5 (Rastrigin only); the replication passes
3/5 with a disjoint set and Rastrigin reversed — report behavior-gain as
unstable across streams.

### §5.3 Affine invariance (Step 1 — Gate 5, PARTIAL PASS)

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

### §5.4 State conditionality (Step 2 — RF probe)

Hartmann: RF regret 0.47 vs Surrogate 0.68 (−29%) — paired bootstrap
CI [−0.038, 0.415], t p=0.141 → suggestive pilot. Rastrigin/Ackley:
RF ≈ best-fixed, no state-conditional learnability demonstrated.
Branin/SixHumpCamel: too flat for learnability signal.

### §5.5 Closed-loop regret (Step 3)

For each (problem, seed), replay with one first action then AlwaysSurrogate
continuation. **Best first action across opportunity-rich families is
`local_exploitation`** (Hartmann 0.33 vs Surrogate 0.48; Rastrigin 0.00
vs Surrogate 11.10; Ackley 0.005 vs Surrogate 0.03).

### §5.6 Teacher noise and decision-level distillation (EXP-RLCD)

> NEW 2026-09-28. Artifacts: `runs/EXP_JOURNAL_ROUND2_REPORT.md`,
> `exp_teacher_noise_diagnostic{,_r5}.json`, `runs/EXP_RLCD_REPORT.md`,
> `exp_rlcd_report.json` / `exp_rlcd_ci.json`, `exp_rlcd_sweep.json`,
> `exp_rlcd_longhorizon.json`, `exp_rlcd_closed_loop.json`,
> `exp_rlcd_router_panel.json`; theory `docs/RLCD_THEORY.md`.

**Label noise (P1a).** Single-roll counterfactual labels are
noise-dominated: per-re-roll pair flip probability **0.40** on
above-median-gap pairs (0.36 overall; conservative any-of-3 union 0.73
above-median / 0.65 overall); inter-roll gap spread 0.0915 exceeds the
median stored gap 0.05; argmax stable across re-rolls for only 29% of
records. R=5 replicate-averaging reduces the 2v2 split-half pair flip to
0.36 overall / 0.33 above-median — necessary hygiene, not stabilization.
A Gate-4-aligned teacher (rule continuation, pure terminal-Δbest utility,
R=5) still fails Gate 4 with wide upper bounds (+0.78 Ackley / +1.92
Hartmann): first-action labels cannot demonstrate terminal behavior-gain
on this panel.

**Distillation (4-arm, held-out 115 records / 2,889 pairs).** RLCD
fine-tuning of the 0.4B Laya typed-decision model on counterfactual pairs:

| Arm | Pairwise acc | Top-1 | Mean log score |
|---|---:|---:|---:|
| zero-shot | 0.396 (< chance) | 0.035 | −3.32 |
| RLCD proper-score | 0.800 | 0.252 | **−2.07** (only arm beating uniform −2.079) |
| BT pairs | **0.873** | 0.383 | −3.06 (below uniform — overconfidence) |
| RF (utilities) | 0.817 | 0.400 | — |

Pairwise CIs (record-clustered bootstrap, `exp_rlcd_ci.json`):
BT − RF +0.056 [+0.033, +0.080]; proper − BT −0.073 [−0.090, −0.055];
proper ≈ RF (CI crosses 0). Hyperparameter sweep (15 configs, re-run
2026-09-28 on the fixed margin plumbing): defaults near-optimal (best bt
0.878 at β1/margin0; best proper 0.795 at w_sph 0.5 / lr 5e-6) — the
objective choice is the lever, not the knobs. On the long-horizon partial
dataset the ordering is preserved at smaller margins (bt 0.660 > proper
0.576 > RF 0.545 > zero 0.485).

**Closed loop (negative).** Both fine-tuned arms collapse to the same
constant policy (1,250/1,250 decisions `local_exploitation`), so their
closed-loop results ARE the fixed-Local policy's: Rastrigin +0.624
[+0.288, +0.936] vs rule; Hartmann −9.041 [−17.89, −1.91]; Branin −0.016
[−0.025, −0.007] vs surrogate. A confidence router on top (quantile
thresholds) routes essentially every decision to surrogate on Hartmann
(routed ≡ always-surrogate, all diffs 0) and is negative-in-mean on
Rastrigin (−0.23, CI crosses 0): no free closed-loop win from routing the
fine-tuned model either. Decision-level recovery does not yield a deployed
optimizer — consistent with RLCD_THEORY T4/L1 (the h3 teacher's argmax is
dominated by its constant component) and with §6.5's schedule finding.

## §6 RQ3 — Selective Reasoning

### §6.1 Joint 10-seed 5-policy comparison (Student-t corrected)

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

### §6.2 Mechanism: global-to-local temporal action pattern (correlational)

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
  reproduces and exceeds the LLM's value. See §6.5.
- **Recommended paper wording (updated per codex design adjudication
  2026-09-28)**: *"On Rastrigin, the LLM exhibited a global-to-local
  temporal action pattern associated with its advantage over
  AlwaysSurrogate. A deterministic two-phase schedule realizing that
  pattern matches the LLM's value and exceeds it on three of five
  families — a two-phase schedule suffices to match or exceed the LLM's
  value on this panel, so LLM-specific per-step reasoning is not needed
  to explain it."*

### §6.3 FrozenRouter: BOCoDe advantage does NOT transfer

| Panel | FrozenRouter − Surrogate | p |
|---|---|---|
| BOCoDe (5-family) | +15.69 (sum) | 0.012 |
| IOH (f1–f5, dim 2/5/10/20) | +0.036 (mean normalized regret) | 0.138 |

**Negative result**: a simple `dim≥5` router trained on BOCoDe does not
generalize to IOH/BBOB at the same scale.

### §6.4 RoutingAware vs LLM-everywhere (negative finding)

On the BOCoDe panel sum, RoutingAware − LLM = +1.18 (paired-t p=**0.5134**,
not significant). The simple dimension-based routing rule is NOT measurably
better than LLM-everywhere on this benchmark. **The Pareto frontier
deferral**: more elaborate routing needs RQ2 transfer experiments to
become a defensible claim.

### §6.5 Real-LLM cost frontier, closed loop, 10 seeds (EXP-RQ3)

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
`rf` winning 3/0/0. The corrected reading agrees with §6.1 above (LLM
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
per-step distribution exhibits — none of the tested routers earned its
cost, consistent with THEORY.md P3 (IF an arm's per-state advantage is
≤ 0, no router can rescue it; our data are consistent with, but do not
independently establish, that premise). Artifacts:
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
3. **Gate 2 fails — at this state granularity** — stable state-to-action
   learnability is **not demonstrated** (5/15 buckets pass at
   top_action_share ≥ 0.6 threshold) *from the compressed 15-scalar process
   state*. Whether a richer state would help is **empirically probed,
   negative** (branch `exp/lenz-state-jev-presentation`, untouched
   s100–119 stream): (i) a trajectory-feature RF probe does not beat the
   15 scalars on 5/5 families; (ii) a rich-state LLM A/B (deepseek-chat,
   temperature 0, same frozen menu, 30 stratified records) changes 6/30
   decisions — 4 of 6 global_exploration → surrogate_acquisition, one
   local_exploitation → surrogate_acquisition, one global_exploration →
   local_exploitation — without
   moving gold-argmax agreement (0.133 both) at 2.1× token cost. A learned
   sequence encoder over trials remains untested, and the adaptive-value
   ceiling (P2, representation-independent) caps any such gain at D on
   four of five families. Design spectrum:
   `docs/ref_papers/state_action_design_survey.md`.
4. **Gate 5 fails on 2/17 buckets** — predeclared joint test (`pass_: false`)
   because JS divergence exceeds 0.02 on Hartmann (late, mid) buckets. Median
   Spearman = 1.0 in all (problem, phase) buckets (consistent with the
   analytic rank-invariance argument; per-record invariance not established).
5. **Independent replication ADDED (2026-09-28)** — the original 20-seed
   set reused seeds 0–9 from the exploratory run; a fully independent
   stream (seeds 100–119, 654 records, same frozen generator config) now
   exists (`phase8_6d_h3_20seed_5family_s100_119.jsonl`). Gate 3 replicates
   essentially exactly (1.296 [0.881, 1.867] vs 1.294 [0.848, 1.954],
   PASS); stop-gain replicates 5/5; uniform-gain replicates 4/5 in count
   but not in family set; behavior-gain does not replicate at family
   level (§5.2). Remaining limitation: the family PANEL is unchanged —
   the same five families informed the design on both streams.
6. **Pareto frontier measured but degenerate** — the closed-loop cost
   frontier (§6.5) is measured at budget 30 on the five-family panel with
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
   and the §6.5 cost-frontier result is provider-specific.
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
12. **Multiple unadjusted comparisons — sensitivity measured (2026-09-28)**:
    recomputing the per-family Gate-4 CIs at the Bonferroni-adjusted level
    (99%, α/5; `runs/exp_gate4_multiplicity_sensitivity.json`):
    **stop-gain survives 5/5**; uniform-gain degrades **4/5 → 2/5**
    (Rastrigin, SixHumpCamel hold; Branin lo −0.003 just misses; Hartmann's
    95% lower bound is itself bootstrap-fragile at ±0.05 around 0 — its
    family-level pass is not stable); behavior-gain degrades **1/5 → 0/5**.
    The borderline family-set instability also appears across seed streams
    (§5.2). We claim stop-gain as robust; uniform-gain as 2/5 robust +
    borderline; behavior-gain as not established.
13. **Learned System-1 closed-loop evidence is negative-to-heterogeneous** —
    EXP-RQ2.1 is one-step decision regret; the closed-loop audit
    (`runs/EXP_FINAL_GOAL_CLOSED_LOOP_REPORT.md`) and §6.5 deploy frozen RFs
    through full trajectories. The RF beats AlwaysSurrogate on Ackley but
    not the stronger AlwaysLocal control, and dominates every LLM arm at
    zero cost on this panel; a general learned-policy gain is not
    established. The "RoutingAware" rule remains hand-coded (dim ≥ 5).
14. **Summing raw best_y across families is not principled** — different
    problems have different objective scales. The TOTAL comparison is
    informative but not on a common scale; per-family + ratio-of-advantages
    is the cleaner framing.
15. **RLCD transfer is analysis-rerun, not locked** — the +0.140 5/5-fold
    cross-family result (§4.2) was produced after design iterations on the
    same panel; a locked evaluation on untouched families/tasks is required
    before calling it confirmatory. And the fine-tuned arms collapse to a
    constant policy in closed loop (§5.6) — no learned policy on this panel
    (RF or RLCD) beats the constant local baseline in closed loop.
16. **Teacher noise bounds the supervision** — counterfactual labels are
    noise-dominated at single roll (per-re-roll flip 0.40 above-median);
    all §5.6 distillation results are relative to that teacher, and Gate 4
    (terminal behavior-gain) was NOT established for first-action labels on
    this panel even after R=5 denoising and utility alignment (wide upper
    bounds are absence of evidence, not proof of impossibility).

## §8 Discussion

- "Is decision-intelligence learnable?" — partially yes (Hartmann
  suggestive; flat families no signal). At the distillation level: the
  noise-dominated counterfactual teacher is still learnable at decision
  level (0.40 → 0.87 pairwise; proper-score arm uniquely calibrated,
  §5.6) — but the learned policy collapses to the teacher's constant
  component in closed loop.
- "Does it transfer across problems?" — see §4 RQ2 below.
  Cross-family has lower regret vs real RulePolicy on 3/5 (Branin, Ackley,
  Rastrigin) and vs AlwaysSurrogate on 2/5 (Ackley, Rastrigin); higher regret
  vs AlwaysSurrogate on Hartmann + SixHumpCamel; inconclusive on Branin.
- "Is LLM-as-System-1 worth it?" — against AlwaysSurrogate, only on hard
  multimodal problems (Rastrigin +17.54, p=0.0058). But in closed loop
  against a frozen RF System-1 (§6.5), the always-LLM policy never wins
  with a CI excluding zero and loses detectably on 3/5 families at ~12–17
  calls per trajectory. The LLM's positive value on this panel is confined
  to the one-step comparison against weaker baselines.
- "Is routing a Pareto win?" — measured (§6.5): no. No confidence threshold
  or random rate earns its cost over System-1-only; the frontier is
  degenerate on this panel. Whether any regime gives routing positive
  value is the open question.
- "Does the trained System-1 transfer across problems?" — see §4 RQ2 below
  (wins vs real RulePolicy on 3/5; wins vs AlwaysSurrogate on 2/5; loses to a
  constant local policy on aggregate). The RLCD fine-tuned model transfers
  at decision level (+0.140, 5/5 folds, analysis-rerun; GearTrain probe
  holds, §4.2) — closed-loop transfer remains undemonstrated for both.

## §9 Conclusion

- Three contributions: (1) a 7-component decision-centric BO framework
  with **423 unit tests** (385 at first draft; 421 after the long-horizon
  and RLCD rounds; +2 in the 2026-09-28 audit round: partial-rollout
  validation, Gate-3 CI-branch coverage)
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
- **EXP-RLCD (NEW 2026-09-28)**: decision-level distillation of the
  counterfactual teacher into a 0.4B typed-decision model (pairwise
  0.40 → 0.87; the proper-score arm is the only calibrated one), with
  cross-family transfer (+0.140, 5/5 folds, analysis-rerun) and a
  decision-level engineering probe (GearTrain 0.89 ≈ RF). Closed loop
  collapses both fine-tuned arms to the teacher's constant component —
  decision-level recovery without a deployed optimizer, mirroring the
  schedule finding of §6.5.
- Seven negative results that sharpen the claim: (a) trained router does
  not transfer from BOCoDe to IOH; (b) RoutingAware does not beat
  LLM-everywhere on the same panel; (c) cross-family transfer does not
  help Hartmann or SixHumpCamel vs AlwaysSurrogate; (d) cross-family RF
  loses to a constant `local_exploitation` policy (AlwaysLocal) on
  aggregate because that policy ignores budget awareness; (e) within-
  function dimension transfer fails (RQ2.2 at 10 seeds — the RQ2.1 signal
  is a family signal, not a dimension signal); (f) in closed loop,
  always-LLM is dominated by both the frozen System-1 and a deterministic
  two-phase schedule (§6.5) — none of the tested routers earned its cost,
  and P3 supplies a sufficient-condition explanation (if per-state
  advantage ≤ 0, no router can help);
  (g) RLCD fine-tuned decision models collapse to a constant policy in
  closed loop (§5.6) — decision-level metrics do not imply
  trajectory-level value.
- Future work: regimes where an LLM System-2 repays its cost (longer
  horizons, stochastic objectives, OOD states), external lenz/PlugBO
  baseline, fine-tuning a small decision model from LLM trajectories,
  dimension-crossed design to disentangle dimension vs family effects,
  engineering families (vehicle suspension).

---

## Appendix

- A1 — Test inventory (423 tests, file-by-file)
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
- `runs/EXP_RLCD_REPORT.md` + `runs/exp_rlcd_report.json` / `exp_rlcd_ci.json` /
  `exp_rlcd_sweep.json` — RLCD 4-arm decision-level result, CIs, sweep
- `runs/EXP_JOURNAL_ROUND2_REPORT.md` + `runs/exp_teacher_noise_diagnostic.json` /
  `exp_teacher_noise_diagnostic_r5.json` / `runs/exp_rlcd_loo_confirmatory.json` /
  `exp_rlcd_loo_transfer.json` — teacher-noise chain + LOO transfer
- `runs/phase8_6d_h3_20seed_5family_s100_119.jsonl` +
  `runs/exp_rq1_2_h3_20seed_replication_s100_119.json` — independent-seed-stream
  gate replication (§5.2, limitation 5)
- `runs/exp_rlcd_loo_confirmatory_replication_s100_119.json` — LOO transfer
  replication on the independent stream (§4.2)
- `runs/exp_gate4_multiplicity_sensitivity.json` — Bonferroni sensitivity (limitation 12)
- branch `exp/lenz-state-jev-presentation`: `runs/exp_lenz_gate2_probe.json` +
  `runs/exp_lenz_vs_frozen_llm.json` + `runs/EXP_LENZ_JEV_REPORT.md` —
  rich-state A/B (both negative; limitation 3)
- `runs/exp_geartrain_transfer_probe.json` — engineering probe (§4.2)
- `runs/exp_rlcd_longhorizon.json` + `runs/exp_rlcd_closed_loop.json` +
  `runs/exp_rlcd_router_panel.json` — long-horizon, closed-loop collapse,
  router panel (§5.6)
- `docs/RLCD_THEORY.md` — R1–R4 / T1–T5 for the distillation line
- `runs/EXP_FINAL_GOAL_CLOSED_LOOP_REPORT.md` — closed-loop audit
  (`rf_all` / `rf_transfer` vs fixed controls, seeds 100–119 / 200–219)
- `src/opt_agent/backend/bocode_botorch.py` — affine transform (Gate 5)
- `src/opt_agent/dataset/generator.py` — DatasetConfig affine fields
- `tests/test_objective_transform.py` — 16 new tests for Gate 5
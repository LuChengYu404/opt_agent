# Theory companion — formal statements behind the empirical claims

> Status: 2026-09-27. This document formalizes four statements that the
> experiment reports state informally, proves them under explicitly listed
> scope conditions, and maps each to the experiment that tests it. Everything
> is stated at the level of rigor appropriate for a workshop paper appendix;
> no claim here goes beyond what the code (`src/opt_agent/`) implements.

## 0. Setup and notation

Let `s` denote an optimization state, `A` the 8 typed actions
(`|A| = m = 8`), and let the counterfactual utility of action `a` at state
`s` be the quantity implemented in `src/opt_agent/rollout/utility.py`:

```
U(a, s) = w_opt · Δopt(a, s) / σ_opt(s)
        + w_sur · Δsur(a, s) + w_cov · Δcov(a, s)
        − w_eval · n_new(a) / σ_eval
        − w_fit  · n_fit(a) / σ_fit
        − w_stop_opportunity · remaining_budget_ratio · 1[a = STOP]
```

with project defaults `w_sur = w_cov = 0` (Phase 8.5 decision) and
`w_stop_opportunity = 0.75` in the default `UtilityWeights` (the STOP
term is ACTIVE by default; `rollout/utility.py`),
`σ_eval = σ_fit = total_budget`, and
`σ_opt(s) = std` of the **initial** objective values of the trajectory
(computed by the dataset generator when `optimization_scale` is `None`).
`Δopt(a, s)` is the improvement of the best observed objective over the
short-horizon rollout branch that executes `a` and then a fixed
continuation — in particular it is a **difference** of two objective
values. `gold_probs(a|s) = softmax(U(·, s) / T)`, computed over the
rolled candidate set — which may be a quota-sampled subset of the 8
actions (`dataset/generator.py`).

States are drawn from a family-specific distribution ρ_family over the
state space induced by the rollout process. All expectations below are
over states of one family (or the pooled panel, as stated).

---

## P1. Affine invariance of the utility labels (Gate 5)

**Proposition 1.** Apply an affine objective transform `ỹ = αy + β`
(α > 0) to every objective evaluation of a trajectory, and recompute
`σ̃_opt = std(ỹ_init)`. If every objective-dependent term of `U` enters as
a ratio of an α-equivariant difference to an α-equivariant scale — in
particular under the project defaults, where the only such term is
`Δopt/σ_opt` — then

```
Ũ(a, s) = U(a, s)   for every (a, s),
```

exactly. Consequently `gold_probs`, the argmax action, and all rank
statistics of the labels are invariant.

**Proof.** `Δopt` is a difference of two objective values, so
`Δ̃opt = α · Δopt` (β cancels). The initial values transform the same way,
and the population standard deviation is positively homogeneous of degree
one: `σ̃_opt = α · σ_opt`. Hence
`w_opt · Δ̃opt / σ̃_opt = w_opt · Δopt / σ_opt`. The evaluation and fit
terms depend only on action counts and the fixed budget scales, which the
transform does not touch; the STOP term depends only on the remaining
budget ratio. Every term is unchanged. ∎

**Scope conditions.** (i) The scale must be *recomputed* on the
transformed initial values (the generator does this); a frozen σ would
leave a global α factor that softmax cancels only if T is rescaled too.
(ii) If `w_sur` or `w_cov` were nonzero, the TERMS `w_sur·Δsur` /
`w_cov·Δcov` would each have to be affine-invariant — an equivariant
difference alone is NOT sufficient (it leaves a global α factor on that
term); it must be divided by an α-equivariant scale, as `Δopt` is by
`σ_opt`. With unnormalized RMSE-like quantities the identity breaks.
The project runs with both weights zero, so the condition holds by
configuration.

**Consequence for Gate 5.** Spearman ρ = 1 across buckets is *necessary*,
not a lucky measurement; the observed 2/17 JS breaches on Hartmann cannot
come from the label definition and must come from numerics of the GP fit /
acquisition proposal (BoTorch `optimize_acqf` landing on duplicate X under
shift), which is what the affine report concludes empirically. The
predeclared joint Gate 5 therefore measures **GP-fit numerical stability
under shift**, not invariance of the supervision signal.

---

## P2. `adaptive-value` is the tight value of state-conditionality

**Proposition 2.** Define `D(family) = E_s[max_a U(a, s)] −
max_a E_s[U(a, s)]`. Then `D ≥ 0`, and for every (possibly randomized)
policy π,

```
E_s[U(π(s), s)] − max_a E_s[U(a, s)]  ≤  D,
```

with equality attained by the per-state oracle `π*(s) ∈ argmax_a U(a, s)`.
That is, `D` is exactly the one-step regret of the best **constant**
action against per-state oracle selection, and it tightly upper-bounds
what any state-conditional policy can gain over the best constant action.

**Proof.** For each π, `U(π(s), s) ≤ max_a U(a, s)` pointwise; take
expectations and subtract `max_a E[U(a, s)]` ≤ `E[max_a U(a, s)]`
(exchange of max and expectation). Equality for the oracle policy is
immediate. ∎

**Empirical tie.** The decision-landscape regime table (20-seed
recomputation, `runs/verify_theory_propositions.json`: Hartmann
D = 0.710 "frequent opportunity" vs SixHumpCamel D = 0.012 "flat"; the
original 10-seed landscape read 0.684 / 0.015 — same regime ordering) is a
direct readout of this bound: on flat families *no* state-conditional
policy — learned or LLM — can beat the best constant action by more than
≈ 0.012 in one-step utility, which is why the RF probe finds no signal
there. On Hartmann the bound is loose enough to permit state-conditional
gains, and the probe found a suggestive −29 % regret reduction.

**Caveat.** `D` bounds one-step utility gains. Closed-loop terminal value
is a different functional (the policy changes which states are visited);
Phase 8.6F showed first-action and continuation effects are not additive.
The bound is a screening result, not a closed-loop guarantee.

---

## P3. Routing cannot win under nonpositive System-2 advantage

**Proposition 3** (one-step routing model). Fix a state distribution ρ,
a System-1 action map `a₁(s)`, a System-2 action map `a₂(s)`, and a
per-call cost `c ≥ 0`. A router is a measurable set `S₂` of states on
which System-2 is consulted. With matched one-step continuation, define
the per-state net advantage `A(s) = Q(s, a₂(s)) − Q(s, a₁(s))`. Then the
router's value over never routing is

```
V(S₂) = E[ 1{s ∈ S₂} · A(s) ] − c · ρ(S₂).
```

If `A(s) ≤ 0` ρ-almost surely, then `V(S₂) ≤ 0` for **every** router —
oracle or learned, confidence-based or not — with strict inequality
unless `S₂` avoids all states where `A(s) < 0`.

**Proof.** Both terms are nonpositive term-by-term. ∎

**Corollary (frontier degeneracy).** On a panel where the System-2's
per-state advantage is nonpositive, no routing design can recover value;
the cost frontier is degenerate at "System-1 only". Routing research on
such a panel can only lose.

**Empirical reading.** The 10-seed EXP-RQ3 sweep measured eight
LLM-calling arms (always-LLM, four confidence thresholds, three random
rates); none beat `rf` with a CI excluding zero on any family, and
always-LLM lost detectably on three families. Under the one-step model
this pattern is CONSISTENT WITH `A(s) ≤ 0` across visited states — but
the sweep measures aggregate routing outcomes and cannot establish the
pointwise premise directly. `A(s) ≤ 0` is an unmeasured premise, not a
quantity the sweep estimated: the proposition says what FOLLOWS if the
premise holds; the sweep provides the conditional's empirical
illustration, not its antecedent's verification. Two supplementary observations fit: (a) the
calibrated RF softmax maximum rarely exceeds 0.25 with m = 8 (near the
uniform 0.125), so confidence thresholds τ ≥ 0.25 all induce the *same*
router (route-always); the tested router family collapses to its two
extremes, and the random-rate arms confirm the interior points also lose.
(b) The deterministic schedule control (below) shows the LLM's decisions
are dominated even by a fixed two-phase schedule — additional evidence
CONSISTENT WITH `A ≤ 0` (not a verification of the pointwise premise).

**Caveat.** `V` is additive only in the one-step model with matched
continuation. Sequential composition breaks additivity (Phase 8.6F), so
the closed-loop statement is verified empirically per-arm rather than
deduced. Read conditionally: IF the pointwise premise held on this
panel, P3 would imply the same outcome for any untested router — but
the premise is unmeasured, so this is a conditional statement about the
sweep's explanation, not a prediction about untested routers.

---

## L1. Random forests can represent axis-aligned threshold rules; exact imitation is then an empirical observation

**Lemma 1 (representability).** Let the target rule `r(s)` be a function of
a subset of axis-aligned features (e.g., `stagnation_steps`,
remaining-budget ratio). Suppose training pairs `(features(sᵢ), r(sᵢ))`
are noiseless and cover the rule's decision cells, and there is **no
aliasing**: equal feature vectors imply equal rule actions. Then there
EXIST CART trees (depth ≥ number of thresholds, one regressor per action
on indicator targets) whose ensemble reproduces `r` exactly:
`argmax_a f̂_a(features(s)) = r(s)` on covered states.

**Scope caveat (added per codex round 5).** This is a representability
statement, not a guarantee about the TRAINED forest: CART fitting with
random subsetting may grow trees that split elsewhere, and coverage plus
no-aliasing do not by themselves ensure the correct partitions dominate
the ensemble average. The claim that a fitted forest DOES recover the
rule exactly is an empirical observation (below), for which the lemma
supplies the explanation of possibility.

**Proof sketch.** Each CART tree partitions the feature space into
axis-aligned rectangles; the rule's cells are a partition of this kind, so
some tree realizations isolate them. With noiseless indicator targets and
sufficient leaf samples, per-action regressors CAN converge to the
indicator of the rule picking that action on every covered cell, making
the argmax unique for those realizations. ∎

**Empirical tie.** `RuleFullBudget` decides on
(`stagnation_steps ≥ 3` with budget, low-budget threshold 0) — both
features are in the 15-dim RF feature vector. EXP-RQ2.2 trained on
exactly these labels and measured `rf − rule_full = +0.0000`
(CI [+0.0000, +0.0000]) on the held-out dimension at both 3 seeds and —
after today's extension — 10 seeds, on all three functions. The lemma
also predicts the RF *inherits the rule's losses*: Surrogate and Local
beat `rf` on every function, now with CI excluding zero vs Surrogate on
all three (f6 +0.128, f8 +0.244, f15 +0.225) at 10 seeds. Perfect
imitation of a weak rule is a perfect reproduction of its weakness —
which is the mechanism behind the RQ2.2 negative result.

---

## Consequence: the LLM's decision content is dominated by a fixed schedule

The predeclared deterministic control `GlobalThenLocal` (10
`global_exploration` decisions, then `local_exploitation`), run today on
the exact EXP-RQ3 panel (seeds 100–109, paired Sobol starts verified
against the sweep files), gives:

- vs the real LLM (`llm` arm): schedule **wins with CI excluding zero on
  Hartmann (+0.334), SixHumpCamel (+0.674), Rastrigin (+9.628)**, and is
  CI-inconclusive-but-positive on Branin (+0.278) and Ackley (+0.165).
  It never loses to the LLM anywhere.
- vs the frozen RF System-1 (`rf` arm): `rf` wins on Ackley (−0.853) and
  SixHumpCamel (−0.341); they tie on Branin, Hartmann, Rastrigin.

Combined with P3: the LLM's per-state advantage is nonpositive on this
panel (a two-line schedule dominates it), so *no* router could have made
the LLM arm competitive — the frontier's degeneracy is a property of the
System-2 decisions, not of the routing scheme. The earlier per-step
"global-to-local" mechanism finding is thereby strengthened from
"associated pattern" to "the schedule alone suffices — and exceeds — the
LLM's closed-loop value on this panel."

---

## Relation to classical theory (added 2026-09-28)

The propositions are not built from scratch; each is a strategy-level
counterpart of a classical result. Making the lineage explicit (also the
related-work framing in the outline §2):

- **P2 (adaptive-value) is the decision-theoretic counterpart of the
  classical Value of Information.** Decision theory has long measured
  "what is knowing more worth": expected value of information (Howard,
  1966, *Information Value Theory*; Raiffa & Schlaifer, 1961) and expected
  information gain in Bayesian experimental design (Lindley, 1956).
  Information-theoretic BO acquisitions are a related but DIFFERENT
  functional, not a unit conversion: Entropy Search (Hennig & Schuler,
  2012) and Predictive Entropy Search (Hernández-Lobato et al., 2014)
  value a candidate point by posterior entropy reduction about the
  optimum's LOCATION, while Max-Value Entropy Search (Wang & Jegelka,
  2017) targets its VALUE. Our `D` is instead a perfect-information
  value measured in utility units about the STATE's oracle label:
  `E_s[max_a U(a,s)] − max_a E_s[U(a,s)]` is the most any policy could
  gain from perfectly knowing which state it is in (the
  Howard-style expected value of revealing the state identity). The
  analogy, not an identity: both ask what knowing more is worth, in
  different currencies about different unknowns. They compose
  empirically: `D ≈ 0` on the flat families means no state-revealing
  information can pay for itself in utility — consistent with richer
  state observation (branch `exp/lenz-state-jev-presentation`, E1/E2)
  buying nothing there.
- **The compressed state is a projection of the classical sufficient
  statistic — with the sufficiency model-conditional and the claim
  task-scoped.** Under the GP model the posterior `p(f | D_t)` is a
  sufficient summary of the trial log; process decisions additionally
  read budget, configuration, and execution state, which are not
  functions of the posterior. Our 15-scalar state is a
  fixed-dimensional projection of `(D_t, c_t)`. The E1 negative result
  (trajectory features add nothing over the projection on 5/5 families)
  shows the projection is near-lossless FOR PREDICTING THE ACTION LABEL
  under the tested feature families — it does not establish losslessness
  in general.
- **The utility-plus-action-set design lifts acquisition functions to the
  strategy level.** EI (Mockus, 1978) and GP-UCB (Srinivas et al., 2010)
  are closed-form one-step decision rules derived from the posterior; our
  closed 8-action set plus hand-specified utility is the same one-step-greedy
  structure at the level of *which optimizer action*, where no closed-form
  rule exists — which is precisely why supervision is needed.
- **P3 stands in an analogy relation to the portfolio/bandit line.**
  GP-Hedge (Hoffman, Brochu & Freitas, UAI 2011) allocates among
  acquisition functions by exponential weights, rewarding EVERY arm each
  round with the updated GP posterior mean at that arm's nominated point
  (a full-information reward, not the realized gain of the chosen arm);
  Shahriari et al.'s Entropy Search Portfolio and Vasconcelos et al.
  (2022) continue the line; LMABO (ICLR 2026) delegates the portfolio
  choice to a zero-shot LLM. That line assumes the portfolio contains an
  arm worth switching to. P3 is the analogous degenerate condition in our
  two-arm routing model — when the expensive arm's per-state advantage is
  ≤ 0, no allocation can extract positive value. The analogy is not an
  equivalence: our sweep measures routing outcomes, not the per-state
  premise itself (limitation 9 of the outline), and our action set (stop, refit,
  sampling strategies) has no per-arm nominated point, so GP-Hedge's
  reward is not directly defined over it.
- **L1 is a representability statement plus an empirical observation.**
  Axis-aligned regression trees CAN represent an axis-aligned threshold
  rule exactly; under noiseless labels, no feature aliasing, and
  sufficient per-leaf samples, a fitted forest can recover the partition
  — but coverage and no-aliasing alone do not GUARANTEE that the trained
  ensemble learns it or that correct trees dominate the average. The
  RQ2.2 observation (`rf − rule_full = +0.0000` on all functions) is the
  empirical realization; the proposition is the representability part.

Canon attribution note: the citations above are textbook-level; per repo
convention, double-check page-precise references at camera-ready
(`docs/ref_papers/model_interaction_survey.md` tracks verification state).

---

## Proposition-to-evidence map

| Statement | Type | Code anchor | Empirical test |
|---|---|---|---|
| P1 affine invariance | exact identity under scope conditions | `rollout/utility.py`, `backend` affine write-time | Gate 5 Spearman = 1.0 on 17/17 buckets; JS breaches attributed to GP numerics; `scripts/verify_theory_propositions.py` (code-path check) |
| P2 adaptive-value bound | tight upper bound (Jensen) | `exp_rq1_3_decision_landscape.py` metric | Regime table (Hartmann 0.684 vs SixHumpCamel 0.015); no learnability signal on flat families; `verify_theory_propositions.py` (20-seed recomputation) |
| P3 routing degeneracy | one-step dominance argument | router arms of `exp_rq3_router_sweep.py` | 0/40 arms beat `rf` with CI excluding zero; schedule control dominates LLM |
| L1 exact imitation | representation lemma (CART) | `exp_rq2_2_ioh_within_function_dim_transfer.py` | `rf − rule_full = +0.0000` on 3 functions × 10 seeds; inherited losses vs Surrogate/Local all CI-positive |

## What is deliberately *not* claimed

- P2's bound is one-step; closed-loop policy value is not bounded by it.
- P3's model is one-step with matched continuation; the closed-loop
  verification is empirical (all tested arms), and an untested *adaptive*
  router on a different panel could behave differently.
- L1 requires the no-aliasing condition; it held on the RQ2.2 data
  (imitation error exactly zero) but is not guaranteed a priori.
- Nothing here speaks to panels with stochastic objectives, longer
  horizons, or genuinely OOD states, where A(s) > 0 regions may exist.

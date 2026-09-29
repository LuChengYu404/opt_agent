# Workshop Paper §1 Introduction — Prose Draft (v2, research-grade rewrite)

> **Status**: 2026-09-28, second full rewrite. v1 was judged (correctly) to
> read like a positioning memo: it named the components but did not carry
> the method. v2 states the formal object of study, the estimator and its
> semantics, the four propositions at statement-level precision, and the
> empirical protocol with its estimands and inference discipline.
> Numbers cross-checked against `docs/workshop_paper_outline.md`,
> `docs/THEORY.md`, `docs/RLCD_THEORY.md`, and the `runs/*.json`
> references they cite. Claim wording follows the codex adjudications
> (schedule = sufficiency, not reduction; P2 scoped to the action set/states/
> utility; P3 conditional; G2 qualified by granularity; RLCD transfer
> labeled analysis-rerun).
> **Citations**: ✅-verified ref-library entries only. GPTOpt/FunBO cited
> by system name (author lists unverified). ⚠️ preprints excluded.
> **Length**: ~1,400 words — a full page; venues with tighter limits
> should cut Para 5's mechanism details first (see revision notes).

---

## 1 Introduction

**The decision layer that Bayesian optimization hides.** Bayesian
optimization (BO) is formally a sequential decision problem: at each step
the practitioner holds the trial log `D_t`, a fitted surrogate, and a
budget, and must decide what to do next. The textbook presentation
collapses that policy into a single acquisition function whose argmax is
the next evaluation. Classical theory has in fact solved several
*restricted instances* of the fuller problem in closed form: expected
improvement (Mockus, 1978) and GP-UCB (Srinivas et al., 2010) are
one-step-optimal rules under the GP posterior — the posterior being the
model-conditional sufficient statistic of the trial log; GP-Hedge
(Hoffman et al., 2011) allocates online among a portfolio of such rules
by exponential weights under a full-information reward; and
entropy-search acquisitions (Hennig & Schuler, 2012; Hernández-Lobato et
al., 2014; Wang & Jegelka, 2017) price a candidate point by the posterior
entropy it removes about the optimum. The practitioner's actual control
problem is strictly larger: when to explore globally versus exploit
locally, when to refit or distrust the surrogate, when to spend a scarce
evaluation or stop. For these *strategy-level* decisions no closed-form
rule exists; the decision must be learned — which requires per-decision
ground truth — or delegated to a general-purpose reasoner.

**The LLM frontier and its two blind spots.** Both routes have now been
taken. LLAMBO (Liu et al., 2024) prompts an LLM with the full trial
table to act as surrogate, warm-starter, and navigator: actions live in
x-space, nothing is learned, and cost is ungoverned. GPTOpt (2025)
fine-tunes an LLM to emit the next point directly from the raw
trajectory; OPRO (Yang et al., 2024) keeps the entire optimization
trajectory in natural language and prompts for the next candidate; and
FunSearch (Romera-Paredes et al., 2023, Nature) evolves programs with
an LLM proposer, with new mathematical results to its credit — all
three at the x-, prompt-, or program layer, where per-decision process
supervision is undefined. Agentic BO (Brunzema et al., 2026) — our
nearest neighbor —
formalizes the fuller problem as a metalevel MDP and installs a
zero-shot LLM as the policy over an open command space (probe the
backend, reconfigure the optimizer, propose, evaluate, stop); it
demonstrates capability, learns nothing, logs token cost in an appendix,
and defers cost-aware training to future work. LMABO (Ngo et al., 2026)
delegates acquisition selection to a zero-shot LLM at each iteration and
reports ~6k tokens per iteration without optimizing the quality–cost
trade-off. The wider model-interaction literature — routing and cascades
(Chen et al., 2023; Yue et al., 2024), System-1/System-2 agents (Lin et
al., 2023), learned help-seeking and abstention, active information
acquisition (Kapoor & Horvitz, 2009) — shares one structural assumption
with all of the above: the learned gate has positive value to harvest.
Two things are consequently missing from this literature. First,
*supervision*: no system exposes per-decision ground truth for
process-level actions — over an open command space, counterfactual
outcomes cannot be enumerated, so "did the agent decide well?" is not a
measurable quantity. Second, *cost governance*: LLM cost inside the
optimization loop is accounted, not optimized; the non-inferiority
question — does any routing of LLM calls repay its cost in closed loop?
— is not asked.

**Instrumenting the decision layer: expressiveness for measurability.**
We instrument the layer directly. States are a frozen 15-scalar schema in
five blocks — sampling geometry, surrogate validity and posterior
spread, optimization progress, budget, last action — a fixed-dimensional
projection of `(D_t, c_t)`. Actions are eight typed strategy moves
(global exploration, local exploitation, uncertainty, error-guided, and
boundary sampling, surrogate acquisition, surrogate retraining, stop)
with fixed parameters and recorded evaluation cost (`n_new ∈ {0,…,3}`
samples). The estimator is a paired counterfactual rollout: for each
recorded state and each candidate action, restore an exact backend
checkpoint (GP state and RNG), execute the action, run a matched
continuation for the remaining horizon, and score the branch with the
utility `U(a|s) = w_opt·Δopt/σ_opt − w_eval·n_new/B − w_fit·n_fit/B`
(stop-opportunity and surrogate-quality terms exist in the template and
are zero-weighted in this study; `σ_opt` is the trajectory's initial
objective spread, `B` the total budget) — a paired-intervention
estimator in the off-policy-evaluation lineage (Precup et al., 2000;
Dudík et al., 2011). Softmaxing the branch utilities yields
`gold_probs(a|s)`, an exhaustive per-decision oracle label over the
predefined action set: the object an open command space cannot produce, and
the substrate for everything that follows. Dataset validity itself
becomes a predeclared, testable hypothesis, via four gates on
action-preference stability (G2), early-exploration mass (G3, seed-
clustered adjusted-lift CI), terminal advantage of oracle actions (G4,
paired terminal runs with stop-/uniform-/behavior-gain readouts), and
affine invariance of the labels (G5). Three questions then organize the
paper: is the decision signal learnable (RQ1); does it transfer across
problem families (RQ2); and is selective LLM reasoning worth its cost
(RQ3)?

**Four results that bound what learning and routing can extract.** P1
(exact): under an affine objective map `y ↦ αy + β` with the scale
recomputed, every utility label is unchanged — the shift cancels in the
improvement difference and the improvement-to-scale ratio is homogeneous
of degree zero — so rank statistics are invariant by construction
(verified through the `compute_utility` code path on 5,304 rollout
branches, max relative error 1.4×10⁻¹²). This makes G5's median
Spearman ρ = 1.0 across all 17 evaluation buckets *necessary*, and
localizes its 2/17 joint-gate breaches to GP-fit numerics rather than to
the supervision signal. P2: `D = E_s[max_a U(a,s)] − max_a E_s[U(a,s)] ≥
0` equals the one-step regret of the best constant action against
per-state oracle selection, and tightly bounds the gain of *any*
state-conditional policy over that constant — a decision-theoretic
value of information, measured in utility units about the state's label
rather than in entropy about the optimum (ES/PES target the optimum's
location, MES its value; related functionals, not unit conversions).
Scope: this fixed action set, these sampled states, this rollout utility —
closed-loop terminal value is a different functional. P3: if a
System-2's per-state advantage `A(s) = Q(s, a₂(s)) − Q(s, a₁(s)) ≤ 0`
almost surely, then every router — oracle or learned — has non-positive
value over never routing; this is the degenerate case the portfolio
line (GP-Hedge and its successors, LMABO included) assumes away. L1:
axis-aligned threshold rules are exactly representable by CART
ensembles, so exact imitation of such a rule — and exact inheritance of
its weaknesses — is possible; our within-function dimension-transfer
experiment realizes both at once.

**Protocol and findings.** All measurements live on a five-family BOCoDe
panel (Branin, Hartmann, SixHumpCamel, Ackley-10D, Rastrigin-10D) × 20
paired seeds, with a fully independent seed-stream replication (seeds
100–119), paired-seed comparisons under seed-blocked bootstrap CIs, and
Bonferroni sensitivity on the multiplicity-exposed family sets. P2's
bound stratifies the panel into three regimes before any learning (D =
0.710 Hartmann, 0.332 Ackley, 0.262 Rastrigin, 0.040 Branin, 0.012
SixHumpCamel): the flat families admit no state-conditional gain by
bound, and the RF probe indeed finds no signal there. The signal is real
but noisy: G3 passes (adjusted lift 1.294 [0.848, 1.954]; replication
1.296 [0.881, 1.867]); G4 stop-gain passes on 5/5 families and survives
Bonferroni; G2 fails at our state granularity — and a predeclared
rich-state A/B closes the obvious objection within tested feature
families: trajectory-feature probes and Lenz-style rich-state
re-prompting change 6/30 LLM decisions, leave gold-argmax agreement
unchanged, and cost 2.1× the tokens. RQ3's answer is a measured,
degenerate frontier: across 450 closed-loop paired trajectories, a
frozen Random-Forest System-1 with zero LLM calls beats or ties all
eight LLM-calling arms (always-LLM, four confidence thresholds, three
random rates; 0/40 arm×family cells with a CI excluding zero in the
LLM's favor) on all five families, at a predeclared non-inferiority
margin of 0.10 σ_family; always-LLM is detectably worse on three
families while spending ~12–17 calls (~4.7–6.5k tokens) per trajectory.
One mechanism detail explains the router grid's shape: the calibrated
RF softmax maximum rarely exceeds 0.25 against the uniform floor of
1/8 = 0.125, so every threshold τ ≥ 0.25 induces the same route-always
policy — the tested router family collapses to its extremes, and the
random-rate arms confirm the interior loses too. The predeclared
deterministic control `GlobalThenLocal` (10 global decisions, then
local) beats the real LLM with CI excluding zero on three families and
never loses: a two-phase schedule *suffices for — and exceeds —* the
LLM's closed-loop value on this panel, and together with P3 this
attributes the frontier's degeneracy to the System-2's decisions rather
than to any routing scheme. Distillation separates the decision level
from deployment. RLCD fine-tuning of a 0.4B typed-decision model under
a strictly proper composite score (log + ½·spherical) — whose population
optimum is the teacher *distribution* itself, i.e. calibrated recovery,
in contrast to the ranking-only identification of the Bradley–Terry arm
— lifts held-out pairwise ranking from 0.40 (zero-shot, below chance)
to 0.87, is the only arm whose mean log score beats the uniform
baseline, and transfers cross-family (BT arm +0.140 [+0.126, +0.154]
over the RF baseline, replicated +0.188 on the independent stream; 5/5
families CI-positive on both streams; analysis-rerun — independent in
seeds, not yet in families; decision-level signal holds on a 65-record
engineering-family probe). Yet in closed loop both fine-tuned arms
collapse to the constant local-exploitation policy: on this panel, no
learned policy — RF or fine-tuned — beats a constant local baseline.

**Contributions.**

- **A measured decision substrate for BO's control layer**: frozen
  15-scalar state, closed typed action space (8 actions), paired counterfactual rollout
  labels with exact checkpoint replay, and a JSONL decision dataset
  with per-record replay checkpoints — 431 unit tests (423 at the
  reviewed baseline + 8 for the rich-state A/B instrumentation).
- **A dataset-validity methodology**: four predeclared gates,
  independent-seed-stream replication, and multiplicity sensitivity,
  with passes and failures reported symmetrically.
- **Four formal results tied to the experiments that measure them**
  (P1 affine label invariance; P2 adaptive-value bound; P3 routing
  degeneracy; L1 representability), each with an explicit
  proposition-to-evidence map and stated scope.
- **The first closed-loop quality–cost frontier for LLM participation
  inside a BO decision loop** — negative and explained: degenerate at
  System-1-only, with P3 the sufficient-condition explanation, the
  schedule control the mechanism, and the router-collapse detail the
  design lesson (a near-uniform confidence signal cannot gate
  anything). To a literature that reports LLM-in-the-loop gains, this
  is the control evidence it is missing.
- **Decision-level distillation with deployment honesty**: proper-score
  training recovers a noisy counterfactual teacher at decision level
  and transfers across families, yet yields no closed-loop winner —
  decision-level metrics do not imply trajectory-level value.

**Scope.** Every negative here is panel-scoped: a synthetic five-family
panel, one LLM provider, one-step bounds rather than closed-loop
guarantees (P2, P3 assume matched continuation; Phase 8.6F shows
first-action and continuation effects are not additive), and an
analysis-rerun distillation result. We report these limits because the
paper's claims are calibration claims: what the frontier literature
assumes, this panel measures — and on this panel the assumption fails.
Where the assumption might hold — longer horizons, stochastic
objectives, distribution-shifted states, engineering problems, regions
where A(s) > 0 — is precisely where the substrate's instruments should
be pointed next.

---

## Notes for the next revision

1. **Length**: ~1,400 words ≈ one full page. If the venue's intro budget
   is tighter, cut in this order: (a) the router-collapse softmax detail
   in Para 5 (it also lives in §6.5); (b) the GearTrain parenthetical;
   (c) compress the P1 numeric verification to a clause.
2. **Test-count propagation**: outline/abstract still say 423; main now
   has 431 after the lenz merge — update before submission.
3. **Utility wording**: the intro states the *instantiated* utility
   (w_stop = w_sur = w_cov = 0 per THEORY.md §0) while §3.4 of the
   outline presents the 4-term template. Keep the two consistent when
   the methods section is written.
4. **Citations**: author–year placeholders only; GPTOpt/FunBO by system
   name (author lists unverified). OPRO (ICLR 2024) and FunSearch
   (Nature 2023/24) were verified 2026-09-28 and added to Para 2;
   five-year optimization-theory anchors for §2 live in
   `docs/ref_papers/optimization_theory_survey.md` (JES, LogEI —
   must-cite, our backend runs LogEI —, SAWEI, probabilistic-regret
   stopping, PFN, GP-bandit regret line). ⚠️ preprints (HiL-Bench,
   Agentic Abstention, C3PO, System-1.x, adaptive-retrieval items) are
   intentionally absent — verify before adding.
5. **LMABO scope**: "delegates acquisition selection" is the adjudicated
   phrasing (do not imply it lacks strategy-level control elsewhere);
   "reports but does not optimize" mirrors `model_interaction_survey.md`.
6. Para 5's RQ2 coverage is deliberately thin (three clauses folded into
   the landscape/distillation sentences) because §4 owns it; if reviewers
   want RQ2 visible in the intro, promote "beats the deployed
   state-conditional RulePolicy on 3/5 families one-step; within-function
   dimension transfer is exactly-zero imitation (L1)" to its own
   sentence.

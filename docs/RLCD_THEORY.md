# RLCD theory companion — why contrastive distillation of the rollout teacher is sound

> Status: 2026-09-27, branch `exp/rlcd-laya-finetune`. This document is the
> theoretical support for fine-tuning Laya with RLCD (Reinforcement Learning
> from Contrastive Distillation; Yang et al., ICLR 2024, arXiv:2307.12950,
> code facebookresearch/RLCD) on the counterfactual rollout supervision. It
> follows the conventions of `docs/THEORY.md`: formal statements, proofs or
> proof sketches under explicitly listed scope conditions, and a
> proposition-to-evidence map. Code anchor: `src/opt_agent/system1/rlcd.py`.

## 0. Setup: RLCD in this project

RLCD's original pipeline (their §3):

1. generate **contrastive pairs** (o⁺, o⁻) by prompting a model under a
   positive vs a negative principle — no human labels;
2. train a **preference model** on the pairs with a Bradley-Terry (BT)
   objective: score each response independently, optimize
   −log σ(β(s(o⁺) − s(o⁻)));
3. **RL stage**: align the policy with PPO against the learned scores with a
   KL penalty to the base model.

The project mapping:

- **Teacher / pair source.** The counterfactual rollout utility
  `U(a, s)` of `docs/THEORY.md` §0 (executed by
  `src/opt_agent/rollout/utility.py`). A pair at state `s` is
  `(a⁺, a⁻) ⟺ U(a⁺, s) > U(a⁻, s)`; its label is the *sign of a measured
  difference* (`mine_pairs` in `rlcd.py`). This is strictly stronger
  information than RLCD's prompt-induced pairs: a quantitative teacher, not
  a binary principle.
- **Preference model.** Laya's `DecisionModel` per-option scorer: one
  forward pass at a decision point scores all offered options
  `f_θ(a | s) = z_θ(a, s)`; the BT loss is
  `ℓ = −log σ(β(z(a⁺|s) − z(a⁻|s)))` (`bradley_terry_loss`).
- **Policy.** `argmax_a f_θ(a|s)` — loadable through `LayaPolicy(model_id=
  <fine-tuned dir>)` with no new glue.
- **RL stage.** Designed in §5; against the CD model's own reward the argmax
  policy is already optimal, so the stage only pays off against closed-loop
  return and is deferred until per-step trajectory logs exist.

Notation follows `THEORY.md`: `A` is the 8-action vocabulary, `m = |A|`;
`R_k ⊆ A` is the rolled-out subset of a *quota-subsampled* record
(`rollout_action_subsample=k`, main-branch commit 8c70b79);
`gold_probs(a|s) = softmax(U(·,s)/T)`.

---

## T1. Pairwise labels inherit affine invariance and cancel per-state teacher shifts

**Proposition T1.** Consider the contrastive label map
`(s, a⁺, a⁻) ↦ 1[U(a⁺,s) > U(a⁻,s)]` with margin `Δ = U(a⁺,s) − U(a⁻,s)`.

(i) Under the scope conditions of P1 (affine objective transform with
recomputed scale), every contrastive label and every margin is unchanged.

(ii) For any perturbation of the teacher of the form
`Ũ(a, s) = U(a, s) + g(s)` — an arbitrary *state-dependent additive shift*
(e.g., a mis-set `w_stop · r(s)` term, a per-state utility offset, or any
utility component that does not vary across actions at the given state) —
every contrastive label is unchanged.

Consequently the population BT risk, its minimizers up to per-state
additive constants, the induced argmax policy, and every ranking metric are
invariant under both transformations.

**Proof.** (i) P1 states `Ũ(a,s) = U(a,s)` for every `(a,s)`; differences
are then equal term-by-term. (ii)
`Ũ(a⁺,s) − Ũ(a⁻,s) = U(a⁺,s) − U(a⁻,s)` because `g(s)` cancels. The BT
probability `σ(βΔ)` is a function of `Δ` alone; a per-state additive
constant on `f_θ` is likewise cancelled by the loss's difference structure,
so minimizers are identified exactly up to `s`-measurable offsets, which do
not affect `argmax_a f_θ(a|s)`. ∎

**Why it matters here.** DISCUSS.md Q-T1/Q-S5 worry that the 4-term utility
is mis-specified. T1 delimits *which* mis-specifications fine-tuning is
immune to: anything that shifts all actions equally at a state (STOP-term
calibration errors, per-state opportunity costs) is invisible to — and
cannot bias — the contrastive objective. What T1 does *not* cover:
action-dependent perturbations `h(a)` (e.g., a constant bias *for*
`boundary_sampling`), which rotate pair labels and must be fixed at the
utility level, as Phase 8.5 did.

---

## T2. Consistency under rollout label noise; margin filtering restores learnability

Rollout utilities are noisy: the continuation policy couples branches to
future decisions (Q-T2), and GP replay is not bit-reproducible (Q-T8). The
second scope condition of P1 aside, treat the observed utility as
`Ũ(a,s) = U*(a,s) + ε(a,s)` with true utility `U*` and noise `ε`.

**Assumptions.** For a pair with true gap `g = U*(a⁺,s) − U*(a⁻,s) > 0`,
the observed label flips iff `ε(a⁻,s) − ε(a⁺,s) > g`. Assume the noise
difference is symmetric about 0 with a continuous, strictly decreasing
one-sided tail; then the flip probability `η(g)` is strictly decreasing in
`g` with `η(0) = 1/2`, and `η(g) < 1/2 ⟺ g > 0`.

**Proposition T2.**
(i) *(Noisy-BT shrinkage, ranking preserved.)* The population BT risk with
flip-noise labels is minimized by scores whose differences satisfy

```
β( f*(a⁺|s) − f*(a⁻|s) ) = log( (1 − η(g)) / η(g) )
```

for each pair. Since `g ↦ log((1−η(g))/η(g))` is strictly increasing with
value 0 at `g = 0`, `f*` is a strictly increasing reparametrization of
`U*` within each state: the *ranking* — and hence `argmax_a f*` — equals
`argmax_a U*` whenever the true utilities are distinct. Noise shrinks the
score scale but does not reorder it.

(ii) *(Margin filtering.)* Dropping pairs with observed margin
`|Δ̃| < δ` bounds the flip rate of retained pairs by
`η(g_δ)` where `g_δ` solves `g_δ + (noise support condition)` — under a
noise scale `σ_ε` (e.g., sub-Gaussian), retained true gaps satisfy
`g ≥ δ − |ε⁺ − ε⁻|` w.h.p., so the retained flip rate is bounded strictly
below 1/2 for `δ` above the noise scale. Pairs with flip rate < 1/2 are
PAC-learnable by the classical classification-noise result (Angluin &
Laird 1988), and the excess BT risk scales with the flip rate rather than
saturating.

**Proof sketch.** (i) For a Bernoulli outcome with correct probability
`p = 1 − η(g)`, the minimizer of the expected log-loss
`−p log σ(x) − (1−p) log σ(−x)` is `x = log(p/(1−p))` (logistic MLE;
strict convexity in `x`). Substitute `p = 1 − η(g)` and use that
`η` strictly decreases through 1/2 at `g = 0`. (ii) is a tail bound plus
the noise-robust classification theorem; the homoscedastic special case
`ε ∼ N(0, σ²)` gives `η(g) = Φ(−g/(σ√2))` and retained flip rate
`≤ Φ(−(δ/(σ√2) − 1))`. ∎

**Empirical tie.** `RLCDConfig.margin` implements `δ`. The zero-shot vs
fine-tuned comparison of `scripts/train_rlcd_laya.py` measures the realized
ranking recovery; T2 predicts improvement whenever the teacher's within-state
gaps exceed the rollout noise scale — exactly the regime the long-horizon
quota fix buys (deeper horizon ⇒ larger separated gaps, per the Phase 8.6C
horizon diagnosis).

**Contrast with magnitude-matching objectives.** A regression/KL objective
that matches *utility values* or softmax *masses* is not sign-stable: its
optimum tracks the noisy magnitude `Ũ`, and under heteroscedastic noise the
argmax of the fitted surface can differ from `argmax U*` even in
expectation. The BT objective deliberately discards magnitude and keeps the
sign — the only part of the noisy teacher T1+T2 jointly certify.

---

## T3. Quota compatibility: pairwise supervision survives action subsampling; renormalized subset softmax does not

This is the theorem that binds the main-branch quota fix to the fine-tune
choice of objective. Let records roll out a random `k`-of-`m` subset
`R_k` per state (`rollout_action_subsample=k`), selected independently of
the utilities and their noise.

**Proposition T3.**
(i) *(Completeness of subsampled pairs.)* The pairs mined from `R_k` are
exactly the full pair set restricted to `R_k`. Any scorer whose induced
ranking is correct on every subset in the support of the subsampling design
has the correct ranking on `A`: for any ordered pair `(a, b)`, the
probability that both appear in one state's subset is
`p₂ = k(k−1)/(m(m−1)) > 0`, so over `n` states the pair's expected
observation count is `n·p₂` and the probability of never observing it is
`(1−p₂)ⁿ → 0`. Pairwise identification requires pair *co-observation*, not
full-vocabulary coverage per record.

(ii) *(Bias of renormalized subset softmax.)* Training a distribution-matching
objective (KL/cross-entropy) on `q_k = softmax(U|_{R_k})` — the subset
renormalization a naive loader would produce — is biased for the full-vocab
target `p = softmax(U)` and the bias does not vanish as records accumulate
at fixed `k`. Concrete instance: `A = {a, b, c}`, `U = (0, −1, −1)`, `T = 1`.
Full target `p_a = 1/(1 + 2e⁻¹) ≈ 0.576`. On subset `{a, b}`:
`q_a = 1/(1+e⁻¹) ≈ 0.731`; on subset `{b, c}`: the target assigns `a` mass
**0** although `a` is the global best. Every convex combination of subset
targets misplaces mass — the top action's expected target mass is neither
`p_a` nor monotone in the right direction — and the mass assigned when the
top action is *absent* is exactly the wrong supervision. The pairwise
objective on the same subsets uses only
`1[U(a|s) > U(b|s)]` and is untouched by all of this.

**Proof.** (i) counting + union bound as stated. (ii) direct computation in
the 3-action instance; the general statement follows because absent actions
receive renormalized-zero targets while present top actions receive
renormalized-inflated targets, and both distortions persist in expectation.
∎

**Consequence.** Under a subsampled rollout quota, RLCD's pairwise
objective is the *correct* estimator family for fine-tuning, and
softmax-target KL is structurally mismatched. This is why the quota fix
(main 8c70b79) records `partial_rollout` metadata warning against
renormalization, and why `mine_pairs` treats partial records as first-class
supervision (unit-tested in `tests/test_rlcd.py::
test_mine_pairs_partial_records_are_first_class`).

---

## T4. What fine-tuning can buy: the one-step regret ceiling and its decision-level readout

**Proposition T4.** Let `π̂(s) = argmax_a f̂(a|s)` be any policy trained on
`U`-supervision, and let the per-state utility range be
`range(s) = max_a U(a,s) − min_a U(a,s)`.

(i) *(Inherited ceiling.)* `E_s[U(π̂(s), s)] − max_a E_s[U(a, s)] ≤ D`
with `D = E_s[max_a U(a,s)] − max_a E_s[U(a,s)]` — P2 verbatim; no
fine-tuning method, RLCD included, escapes this bound on one-step value.

(ii) *(Decision-level readout.)*
`E_s[U(a*(s),s) − U(π̂(s),s)] ≤ E_s[ range(s) · 1{π̂(s) ≠ a*(s)} ]`
where `a*(s) ∈ argmax_a U(a,s)`. If the top-1 disagreement rate is
`τ = P(π̂(s) ≠ a*(s))` and `range(s) ≤ R̄` on the panel, the one-step regret
against the per-state oracle is at most `R̄ · τ`.

**Proof.** (i) is P2. (ii) on the event `{π̂(s) = a*(s)}` the loss is 0;
otherwise it is at most `range(s)`; take expectations. ∎

**Empirical tie.** This is why the branch's headline metrics are
*decision-level* (pairwise accuracy, top-1 agreement) and why they are
meaningful: by T4(ii) they bound the deployable one-step value of the
fine-tuned model, and by T4(i) that value is capped by `D` — Hartmann-class
families (large `D`) are where a better System-1 can matter at all;
SixHumpCamel-class families cannot reward any fine-tuning beyond the best
constant action (the RF probe already found no signal there, consistent
with `D ≈ 0.015`).

---

## 5. The RL stage: optimal by construction against its own reward, valuable only against closed-loop return

**Proposition T5.** Let `f̂` be the CD-trained scorer and consider the
learned-reward objective `J(π) = E_s[ f̂(π(s) | s) ]`. The greedy policy
`π̂(s) = argmax_a f̂(a|s)` maximizes `J` over all (possibly randomized)
policies. Hence an RL stage (PPO/REINFORCE) optimizing `J` cannot improve
it; its function in RLCD-for-LLMs is to trade `J` off against the KL
constraint and to average out reward-model errors under *generation-level*
actions. In this project's typed setting (m = 8 fixed actions, one network
producing the distribution and the scores), the only distinct quantity an
RL stage can optimize is the **closed-loop return**
`G(π) = terminal Δbest under full-budget execution`, which differs from
one-step `U` by the continuation coupling diagnosed in Phase 8.6C/F
(horizon flips ~50 % of top actions; continuation effects dominate first
effects on Ackley/Rastrigin). The faithful stage-2 design is REINFORCE
with a baseline and a KL anchor `κ·KL(π ‖ π_CD)`:

```
max_π  E[G(π)] − κ · KL(π ‖ π_CD)
```

which is a different functional from the CD risk and, per P2's caveat, is
not bounded by `D`. It is deferred until per-step trajectories are logged
(the existing `DecisionLoop` trajectory records suffice as the rollout
buffer); running it earlier would train on the same one-step signal the CD
stage already consumes provably optimally.

**Proof.** Pointwise `f̂(π(s)|s) ≤ max_a f̂(a|s)`; expectation preserves the
inequality; equality at `π̂`. ∎

**Secondary role (variance).** RLCD motivates contrastive pairs by
signal-to-noise: paired supervision differences out everything the two
responses share (their §2, citing contrastive RL losses, Tian et al.
2022). The same structure appears on both sides of our objective — teacher
differences (T1) and student score differences (the BT gradient per pair is
a function of `z⁺ − z⁻` only, so any state-level offset the encoder emits
across all options is invisible to training). The learned object is exactly
the ranking; nothing else is fitted.

---

## Relation to the existing theory (P1–P3, L1)

| New | Builds on | Statement in one line | Code anchor | Evidence |
|---|---|---|---|---|
| T1 | P1 | Pairwise labels are affine-invariant and cancel per-state teacher shifts | `mine_pairs` | affine Gate 5 Spearman = 1.0 (existing); shift-cancellation unit tests |
| T2 | — (new) | Rollout noise shrinks BT scores but preserves ranking; margin filtering restores learnability | `RLCDConfig.margin` | zero-shot vs fine-tuned pairwise accuracy (`scripts/train_rlcd_laya.py`) |
| T3 | quota fix (main 8c70b79) | Pairwise supervision is unbiased under k-of-m subsampling; renormalized subset softmax is biased and stays biased | `partial_rollout` metadata + `mine_pairs` | partial-record unit tests; long-horizon partial datasets |
| T4 | P2 | Top-1 disagreement × range bounds one-step regret; `D` still caps everything | `evaluate_ranking` | decision-level metrics are the readout of deployable value |
| T5 | P3 (spirit) | RL against the CD reward is a no-op for argmax; stage 2 only pays against closed-loop return | design only | deferred (needs trajectory logs) |
| — | L1 | The RF baseline imitates threshold rules exactly — its wins and losses are the teacher's | `_shared_rq_features` + RF | RQ2.2 `+0.0000` imitation (existing) |

## What is deliberately *not* claimed

- T2's noise model is symmetric one-sided-tail; adversarial or
  action-correlated noise (e.g., a buggy action executor that biases one
  action's branch) violates it and rotates labels — T1's exception clause.
- T3(i) is an identification statement (n → ∞); finite-n pair coverage is
  `n·k(k−1)/(m(m−1))`, which the experiment reports should quote.
- T4 bounds **one-step** value only; closed-loop value is a different
  functional (Phase 8.6F), which is precisely why §5 exists.
- The CD stage measures teacher-ranking recovery, not optimization
  performance. Any closed-loop claim needs the paired-trajectory protocol
  of EXP-RQ3 (and, per P3, a panel where `A(s) > 0` regions exist).
- RLCD's own Appendix N theory (unpublished in fetched HTML; paper-only)
  analyzes gradient behaviour of their BT variant; the propositions here
  are self-contained and do not depend on reading theirs.

## References

- Yang, K. et al. *RLCD: Reinforcement Learning from Contrastive
  Distillation.* ICLR 2024. arXiv:2307.12950.
- Bradley, R. A., Terry, M. E. *Rank Analysis of Incomplete Block
  Designs: The Method of Paired Comparisons.* Biometrika 39(3–4), 1952.
- Angluin, D., Laird, P. *Learning from Noisy Examples.* Machine
  Learning 2, 1988 (noise rate < 1/2 ⇒ learnability).
- Yue, Y., Joachims, T. *Interactively Optimizing Information Retrieval
  Systems as a Dueling Bandits Problem.* ICML 2009 (pairwise-only
  identification).
- Rafailov, L. et al. *Direct Preference Optimization (DPO).* NeurIPS
  2023 — the RL-free bound of the same BT objective; the natural
  stage-2 alternative if KL-anchored REINFORCE proves unnecessary.
- Tian, Y. et al. *Contrastive Representation Learning for RL.* 2022
  (contrastive-difference objectives as variance reduction), as cited in
  RLCD §2.

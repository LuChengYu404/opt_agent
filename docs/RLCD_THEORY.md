# RLCD theory companion — fine-tuning Laya on the counterfactual teacher

> Status: 2026-09-27, branch `exp/rlcd-laya-finetune`. **Objective
> reference corrected 2026-09-27 after verifying the term's origin** (see
> §0.1). This document is the theoretical support for fine-tuning Laya with
> RLCD — the training method of the Jev/Laya lineage of typed decision
> models — on the counterfactual rollout supervision. It follows the
> conventions of `docs/THEORY.md`: formal statements, proofs or proof
> sketches under explicitly listed scope conditions, and a
> proposition-to-evidence map. Code anchor: `src/opt_agent/system1/rlcd.py`.

## 0. Setup

### 0.1 Name disambiguation (verified 2026-09-27)

**RLCD, in the sense used by Laya and this project: "Reinforcement
Learning for Calibrated Decisions."** RL whose reward is a *strictly
proper scoring rule*, so that the expected-reward optimum is the true
outcome distribution — calibration by construction, unlike preference-based
RLHF/DPO-family training which identifies only a ranking (a tilted
distribution). Laya ships the objective in its SDK as
`laya.common.proper_reward` — log score + 0.5·spherical score, minus a
ranked-probability term on ordinal `score`-type questions — with
`td_lambda_targets` for multi-turn credit assignment and `act_costs`
escalation economics. **No method paper has been published** (noted by
arXiv:2609.24574 and third-party independent tests); the most complete
public description is the survey arXiv:2609.28940 ("JEV and Laya as System
One Decision Layers for LLM", Table II/IX), which also documents Laya
training against the Brier score — another member of the same strictly
proper family. Our implementation calls the SDK's `proper_reward` itself.

This is **not** the identically abbreviated "RLCD: Reinforcement Learning
from Contrastive Distillation" (Yang et al., ICLR 2024, arXiv:2307.12950),
which builds preference pairs by contrastive prompting. That paper's
pairwise construction is related to our *baseline arm* (§2, T1–T5), which
is why the confusion was easy to make and worth a paragraph here.

### 0.2 The two objectives implemented on this branch

Let `s` be an optimization state, `A(s)` the offered typed actions
(`m = |A| = 8` on full-vocabulary records), `U(a, s)` the counterfactual
rollout utility (`THEORY.md` §0), and
`gold_probs(·|s) = softmax(U(·, s)/T)` the teacher distribution. Laya's
forward pass produces per-option logits `z_θ(a|s)`; the reported
distribution is `q_θ(a|s) = softmax(z_θ)`.

- **RLCD (proper-score arm; primary).** Maximize the strictly proper score
  of the reported distribution against the teacher:

  ```
  max_θ  E_s [ S( q_θ(·|s), gold_probs(·|s) ) ],
  S(q, t) = log-score(q, t) + 0.5 · spherical(q, t)
          = Σ_a t_a log q_a  +  0.5 · (Σ_a t_a q_a) / ‖q‖₂
  ```

  (`proper_score_loss` = −S, using the SDK's `proper_reward` verbatim; the
  RPS term is masked to ordinal questions and inert for `choice`.)
  When the target distribution is known the RL objective reduces to this
  differentiable form — no sampling is needed; the genuinely RL form
  (score against *realized* outcomes) is §R4.

- **BT pairwise arm (baseline).** Bradley-Terry loss on utility-ordered
  pairs `a⁺ ≻ a⁻ ⟺ U(a⁺, s) > U(a⁻, s)`:
  `ℓ = −log σ(β(z_θ(a⁺|s) − z_θ(a⁻|s)))` (`bradley_terry_loss`). Kept
  because it is the estimator that remains unbiased under
  action-subsampled rollout quotas (T3).

---

## 1. RLCD (proper-score) theory

### R1. Strict properness ⇒ full-distribution identification and calibration by construction

**Proposition R1.** Let `S(q, t) = Σ_a t_a log q_a + 0.5·(Σ_a t_a q_a)/‖q‖₂`
(the SDK's composite on the simplex). Then:

(i) `S` is **strictly proper**: for every target `t` and every report
`q ≠ t` on the support of `t`, `S(t, t) > S(q, t)`.

(ii) Consequently the population RLCD objective
`E_s[S(q_θ(·|s), gold_probs(·|s))]` is uniquely maximized (over an
unconstrained conditional-distribution family) by
`q*(a|s) = gold_probs(a|s)` — the fine-tune target is the teacher
*distribution*, not merely its ranking, and the model's reported
confidence is calibrated against the teacher in the properness sense:
argmax, all pairwise orderings, and every moment of the reported
distribution match the teacher's.

(iii) Identification hierarchy: proper-score training identifies the
teacher up to equality; BT training (§2) identifies it only up to a
strictly increasing per-state transform (ranking alone). Properness buys
calibration; the price is exact sensitivity to target *magnitudes* (R2).

**Proof.** (i) The log score `Σ t_a log q_a` is strictly proper (Gibbs/
information inequality: `KL(t‖q) > 0` for `q ≠ t` on supp(t)). The
spherical score `(Σ t_a q_a)/‖q‖₂` is strictly proper (Shuford–Albert–
Massengill 1966). A positive-weighted sum of strictly proper scoring rules
is strictly proper: if each term is uniquely maximized at `q = t`, so is
the sum. (ii) Pointwise maximization; take expectation. (iii) is the
statement that BT's likelihood depends only on score *differences* (T1)
while `S` depends on `q` itself. ∎

**Why it matters here.** MASTER_PLAN §13 makes calibration the System-1
requirement ("does it know when it is unreliable?"), and the EXP-RQ3
router failure traced to the RF softmax never exceeding 0.25 — a
distribution-level pathology a ranking-only objective cannot even express.
RLCD is the objective whose optimum is a *calibrated* copy of the
counterfactual teacher, which is what the Router (Phase 12) needs as its
input signal. (Post-hoc temperature scaling — as Laya's checkpoint already
ships — is the cheap partial substitute; R1 makes it a no-op at the
optimum.)

### R2. Behavior under noisy rollout targets

Rollout utilities are noisy (GP refit nondeterminism Q-T8, continuation
coupling Q-T2), so the teacher distribution
`t̃ = softmax((U + ε)/T)` is a noisy instance of a clean `t*`.

**Proposition R2.** Both components are LINEAR in the target for fixed
`q`: the log-score objective `E[Σ_a t̃_a log q_a]` is maximized at
`q*(a|s) = E_t̃[t̃(a|s)]`, and the spherical objective
`E[(Σ_a t̃_a q_a)/‖q‖₂]` is likewise linear in `t̃` for fixed `q`, so its
population optimum also tracks `E[t̃]` (its first-order condition is
solved by the mean direction; interior uniqueness follows from strict
properness R1). The composite therefore anchors `q*` to the
noise-averaged teacher `E[softmax((U*+ε)/T)]` — a T-tempered,
noise-smoothed version of `softmax(U*/T)`. (Corrected 2026-09-28 per
codex review: the spherical statement is exact-linear, not "monotone
near interior points".)

**Contrast (deliberate).** BT under the same noise recovers only the
ranking of `U*` (T2) and is *sign-stable*; proper-score training tracks
the smoothed *distribution*, absorbing noise into a flatter target rather
than discarding magnitude. Neither is uniformly better: the proper-score
optimum is the right target when the noise is i.i.d.-across-actions
(averaging is the correct aggregator); the BT objective is the right one
when only ordering is trusted (asymmetric/action-correlated noise —
T1's exception clause). The branch's 4-arm comparison measures which
assumption the rollout noise actually satisfies.

**Proof.** `∇_q E[Σ_a t̃_a log q_a] = E[t̃]/q + λ` under the simplex
constraint ⇒ stationary at `q ∝ E[t̃]` (Gibbs). Linearity in `t̃` is the
whole argument. ∎

### R3. Subset conditioning changes the quota calculus (refining T3)

Deployment offers the model a *state-filtered* option subset
(`candidate_actions_for_state`), and Laya's architecture is
option-conditioned: the options are in the prompt and the softmax is over
the offered set only. Under quota subsampling (k-of-m rolled out per
record), training items carry the teacher renormalized **on the offered
subset**.

**Proposition R3.** For an option-conditioned model trained per subset
`R` with target `t^R = softmax(U|_{R})/Z_R`:

(i) On each subset, proper-score training identifies `t^R` exactly (R1
applies conditionally); no forced-zero supervision is emitted for absent
actions because absent actions receive no logit in that forward pass.

(ii) The composite full-vocabulary distribution the model would report if
offered all of `A` is *not* constrained to equal `softmax(U/T)` by
subset-only training: cross-subset consistency
(`t^R = softmax(U/T)|_R / Z_R` for all `R`) holds only if each option's
score is (approximately) INDEPENDENT of which other options are offered.
Laya emits one scalar `z_θ(a|s)` per option, which is consistent with a
shared-score representation, but the score is computed from a prompt
containing ALL offered options — so set-independence is an ARCHITECTURAL
HYPOTHESIS, not a guarantee; subset-perturbation tests of `z_θ` under
changed option sets are the required check (deferred; flagged per codex
review 2026-09-28).

(iii) The pairwise objective remains the weakest-assumption estimator: it
uses only co-observed ordering and requires no renormalization or
shared-score assumption at all.

**Empirical tie.** `metadata["partial_rollout"]` (main 8c70b79) exists
precisely because *non*-option-conditioned consumers (e.g. an RF over a
fixed 8-wide head) cannot consume subset targets without bias — T3's
original statement, which remains valid for fixed-head models. R3
delimits it: Laya-style option conditioning is the architecture under
which quota-subsampled records are consumable by proper-score training.

### R4. Where the "RL" lives: realized-outcome scoring and TD(λ)

The teacher-distribution form above is the differentiable reduction of
RLCD; the reinforcement-learning form matters when the target is not a
distribution but a **realized outcome** `o ∈ A(s)` (which action proved
right), because a strictly proper scoring rule `S(q, o)` scores a report
against a realized event using the same properness machinery
(`S(q, o) = Σ_a 1[a = o] s_a` for scoring functions `s`; the Brier and
log scores the Laya lineage documents are exactly this form —
arXiv:2609.28940 quotes Brier explicitly).

**Consequences for this project.**

- **Stage 2 (closed loop) gets a dense, per-decision reward for free.**
  Scoring `q_θ(·|s_t)` against the realized outcome at `s_t` (e.g. the
  action whose branch actually improved `best_value`, or a graded
  outcome) is a proper-score reward available at *every* decision, unlike
  sparse terminal-Δbest REINFORCE. Combined with the SDK's TD(λ)
  bootstrap (`td_lambda_targets`) over decision-loop trajectories, this
  is a PROPOSED design (unrun; marked per codex review 2026-09-28) for
  the closed-loop stage T5 sketches. That it optimizes the *reported
  distribution* is a property of the objective, but empirical
  calibration-under-RL and the absence of overconfidence are claims to
  be TESTED, not consequences (the Table II contrast in 2609.28940 is
  the lineage's own framing, not our result).
- **Escalation economics.** `act_costs`/`cost_wrong_act` in the shipped
  config make the reward cost-aware — the same term structure as our
  utility's evaluation/fit costs (`w_eval`, `w_fit`), so the teacher `U`
  and the RLCD reward can be composed without double-counting if stage 2
  adopts the SDK's convention.

---

## 2. The BT pairwise baseline arm (T1–T5, references corrected)

The propositions below analyze the pairwise objective kept as the
comparison arm. They are the theory of *paired-comparison distillation of
the rollout teacher* (Bradley-Terry 1952; the DPO literature analyzes the
same likelihood) — **not** of Laya's RLCD; the identically-named Yang
et al. 2024 paper is related work only.

**T1 (invariance/sufficiency).** Pairwise labels inherit P1's affine
invariance exactly and cancel any per-state additive teacher shift
`g(s)`: `Ũ(a,s) = U(a,s) + g(s)` leaves every pair label unchanged, and
the BT loss's difference structure makes minimizers identified only up to
`s`-measurable offsets, which do not affect argmax. *Delimits the
misspecification immunity of the pairwise arm:* STOP-term calibration
errors and per-state opportunity costs are invisible to it;
action-dependent perturbations `h(a)` are not. ∎ (differences cancel
`g(s)`; P1 term-by-term.)

**T2 (noise consistency).** ASSUMPTIONS (strengthened per codex review
2026-09-28): (a) symmetric zero-mean utility noise; (b) the pair
preference probabilities are jointly representable by a single per-state
score vector (BT transitivity) — noise that is correlated across pairs
within a state can violate this. Under these, the
flip rate `η(g)` of a pair with true gap `g > 0` is strictly decreasing
through `η(0) = 1/2`; the noisy-BT population optimum satisfies
`βΔf* = log((1−η(g))/η(g))` — a strictly increasing reparametrization of
`U*`, so the ranking (and argmax) is preserved while score magnitudes
shrink. Margin filtering (`margin=δ`) bounds the retained flip rate
strictly below 1/2, restoring learnability (Angluin–Laird). *Contrast
with R2:* BT discards magnitude by design and is sign-stable; proper-score
training keeps magnitude and averages noise instead.

**T3 (quota compatibility, for fixed-head consumers).** Pairs mined from
a k-of-m subsample are the full pair set restricted to the subset; any
scorer correct on all covered subsets ranks `A` correctly (pair
co-observation probability `k(k−1)/(m(m−1))`, misses vanish as records
accumulate). Renormalized subset softmax targets are biased for a *fixed
full-vocabulary* distribution and stay biased at fixed k (3-action
counterexample: `U = (0,−1,−1)`, subset targets give the global best mass
0 whenever absent, 0.73 vs 0.58 when present). See R3 for the
option-conditioned refinement under which proper-score training escapes
this objection.

**T4 (ceiling + decision-level readout).** Any U-supervised policy
inherits P2's `D` ceiling on one-step value; one-step regret vs the
per-state oracle is at most `E[range(s) · 1{top-1 wrong}]`, so the
branch's decision-level metrics bound deployable value. Applies to both
arms — and to the RF baseline.

**T5 (RL stage).** Against the BT model's own reward the argmax policy is
already optimal, so pairwise stage-2 RL is vacuous; the RL that pays is
against closed-loop return — now sharpened by R4: use proper-score
rewards on realized outcomes with TD(λ), not sparse terminal REINFORCE.

---

## Proposition-to-evidence map

| Statement | Type | Code anchor | Evidence |
|---|---|---|---|
| R1 strict properness ⇒ calibrated teacher recovery | identification (Gibbs + Shuford et al.) | `proper_score_loss` (SDK `proper_reward`) | proper-score arm vs zero-shot vs BT: pairwise acc **and** mean log score (`runs/EXP_RLCD_REPORT.md`) |
| R2 noise → mean-target tracking | population optimum (linearity) | idem | arm comparison under rollout noise; which arm wins separates R2 vs T2 noise models |
| R3 subset conditioning consumes quota records | architectural lemma | `build_item` target on offered options; `partial_rollout` metadata | partial-record tests; long-horizon partial datasets (main 8c70b79) |
| R4 realized-outcome RL + TD(λ) design | design proposition | deferred; uses SDK `td_lambda_targets` | stage-2 once trajectories logged |
| T1–T5 pairwise-arm theory | as above | `mine_pairs`, `bradley_terry_loss` | BT arm rows in the same report; unit tests |

## What is deliberately *not* claimed

- RLCD has **no published method paper**; R1–R4 analyze the objective as
  shipped in the SDK (and as described second-hand in arXiv:2609.28940 /
  2609.24574). If the lineage publishes the method and it differs (e.g.
  the Brier-score-only variant the survey reports), R1 still holds for
  the composite we run, and the Brier variant satisfies the same
  properness statements.
- R1's "calibration by construction" is calibration **against the
  counterfactual teacher**, not against ground-truth closed-loop
  outcomes; the teacher itself is horizon- and continuation-dependent
  (Phase 8.6C/F). Calibrating to a mis-specified teacher is precise
  error — the closed-loop stage (R4) is where that is addressed.
- T4's ceiling applies to both arms; decision-level wins do not
  establish closed-loop optimization gains.
- T2/R2's noise models are symmetric; action-correlated noise violates
  both.

## References

- Laya SDK (`pip install laya`, v0.3.7): `laya.common.proper_reward`,
  `td_lambda_targets`; checkpoint `convaiinnovations/laya`.
- *JEV and Laya as System One Decision Layers for LLM.* arXiv:2609.28940
  — closest public description of RLCD (Table II: calibration by
  construction; Table IX: Laya fine-tuning "RL with proper scoring
  rules"; Brier score quoted).
- *Evaluating Decision Models for Text Annotation.* arXiv:2609.24574 —
  notes RLCD has no published method paper.
- Gneiting, T., Raftery, A. E. *Strictly Proper Scoring Rules,
  Prediction, and Estimation.* JASA 102(477), 2007.
- Brier, G. W. *Verification of Forecasts Expressed in Terms of
  Probability.* Monthly Weather Review 78(1), 1950.
- Shuford, E. H., Albert, A., Massengill, H. E. *Admissible Probability
  Measurement Procedures.* Psychometrika 31(2), 1966 (spherical rule).
- Bernardo, J. M. *Expected Information as Expected Utility.* Ann.
  Statist. 7(3), 1979 (log score propriety).
- Bradley, R. A., Terry, M. E. *Rank Analysis of Incomplete Block
  Designs.* Biometrika 39(3–4), 1952 (BT arm).
- Rafailov, L. et al. *Direct Preference Optimization.* NeurIPS 2023
  (same likelihood family as the BT arm).
- Angluin, D., Laird, P. *Learning from Noisy Examples.* ML 2, 1988
  (T2). Yue, Y., Joachims, T. ICML 2009 (dueling-bandit identification,
  T3).
- Yang, K. et al. *RLCD: Reinforcement Learning from Contrastive
  Distillation.* ICLR 2024, arXiv:2307.12950 — **the other RLCD**;
  cited here only to anchor the disambiguation in §0.1.

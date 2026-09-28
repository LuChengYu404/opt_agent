# EXP-RLCD journal round — E1–E5 consolidated report (2026-09-27)

Branch `exp/rlcd-laya-finetune`. Five experiments closing the journal-level
gaps identified in `docs/ref_papers/journal_positioning.md` for the
fine-tuning line. All CPU tracks ran as parallel workers (the 450-trajectory
router panel: 6 GPU workers, ~5 min wall-clock).

## Summary table

| # | Question | Result | Verdict |
|---|---|---|---|
| E1 | Proper-score training vs BT + post-hoc temperature | proper is the only arm with good joint (log −2.07, ECE 0.11); BT+T* recovers log (−2.07) but destroys ECE (0.02→0.26); **no arm has confidence discrimination** (conf_gap ≤ 0.013, AURC ≈ 1−acc) | objective sets calibration *shape*; discrimination absent everywhere |
| E2 | Long-horizon quota teacher quality | Gate 3 adjusted_lift **12.9** (h3: 1.29 — 10×) ; Gate 2 FAIL, Gate 4 FAIL | much stronger early-exploration signal; terminal validity still open |
| E3 | Do long-horizon fine-tunes escape constant-collapse? | No — but to *different* modes: bt_lh → all `local_exploitation`, proper_lh → all `error_guided_sampling` (1250/1250 each) | teacher's argmax is mode-dominated at any horizon |
| E4 | Does calibrated confidence pay in routing? (predeclared) | **Criterion FAIL.** Hartmann: routed beats pure +5.87*** but by degenerating to exactly-surrogate trajectories (diffs ≡ 0); Rastrigin mirrors it negatively; no routed arm beats both parents anywhere | routing degenerates to family-level switching; consistent with E1's zero discrimination |
| E5 | Cross-family transfer (LOO, decision-level) | **bt 0.866 mean pairwise on UNSEEN families (in-family 0.873 — 99% retention); proper 0.746 with log −2.07 retained; RF 0.720. bt > RF on 5/5 folds (sign p = 0.031 one-sided); gap grows from +0.056 (in-family) to +0.146 (cross-family)** | **the journal positive: distilled decision models generalize across problem families where feature-imitation does not** |

Raw artifacts: `runs/exp_rlcd_calibration_ablation.json`,
`runs/exp_longhorizon_gates.json`,
`runs/exp_rlcd_closed_loop_longhorizon.json`,
`runs/exp_rlcd_router_panel.json` (+ log with taus),
`runs/exp_rlcd_loo_transfer.json`.

## E1 — calibration ablation (details)

Honest split (temperature fitted on seeds 16–17, metrics on 18–19; 60/55
records):

| arm | log score | ECE | conf_gap | AURC | max prob |
|---|---:|---:|---:|---:|---:|
| bt raw | −3.068 | 0.021 | 0.013 | 0.557 | 0.405 |
| bt + T* (16.5) | −2.073 | 0.257 | 0.0004 | 0.553 | 0.143 |
| proper | −2.070 | 0.114 | 0.005 | 0.657 | 0.158 |
| proper + T* (1.6) | −2.068 | 0.127 | 0.003 | 0.660 | 0.146 |

Reading: post-hoc temperature recovers the soft-distribution metric for BT
but breaks its argmax calibration; proper training is the only way to hold
both simultaneously (R1's joint identification). **However** the
routing-relevant property — confidence discriminating correct from
incorrect decisions — is absent in every arm (AURC ≈ 1 − accuracy). The
teacher's thin state-conditional signal (Gate 2 FAIL) leaves nothing for
confidence to carry.

## E4 — router panel (predeclared criterion and outcome)

Arms: surrogate / proper_only / bt_only / {proper,bt}_q{20,40,60} (quantile
thresholds frozen on the calib slice: proper τ ∈ [0.152, 0.157] — the
confidence distribution is nearly state-invariant; bt τ ∈ [0.376, 0.426]).
Panel: 5 families × seeds 300–309 × budget 30, paired (450 trajectories).

- Hartmann: every routed arm beats its pure arm (CI excludes 0) — five of
  the six routed arms by **+5.87 [+2.29, +10.59]**; `bt_q20` is the
  exception at +5.40 [+1.88, +9.74] — but at q40/q60 the routed
  trajectories are *identical*
  to always-Surrogate (all diffs 0): the router switched essentially every
  decision.
- Rastrigin: mirrored, negative (−0.23 vs pure, CI crosses 0; the same
  all-switch behavior lands on the wrong family).
- Branin/SixHumpCamel: small CI-positive gains over pure (+0.010***),
  losses vs surrogate for bt_q20.
- Predeclared success ("beat BOTH parents, never lose to the better one")
  is NOT met anywhere.

Mechanism: with zero confidence discrimination (E1), quantile routing
degenerates into near-deterministic whole-trajectory switching determined
by threshold placement relative to each family's confidence level — a
family-level, not state-level, decision. This is P3's degeneracy reproduced
for the Laya System-1 line, and it sharpens the RQ3 story: the missing
ingredient is a *discriminative* uncertainty signal, not a better-calibrated
one.

## E5 — cross-family transfer (the positive)

5-fold leave-one-family-out (train: other four families, early stop on
their held-out seeds; test: all records of the held-out family):

| held-out family | proper | bt | rf |
|---|---:|---:|---:|
| ackley | 0.758 | 0.869 | 0.666 |
| branin | 0.750 | 0.871 | 0.817 |
| hartmann | 0.757 | 0.829 | 0.812 |
| rastrigin | 0.734 | 0.844 | 0.697 |
| sixhumpcamel | 0.732 | 0.915 | 0.608 |
| **mean** | **0.746** | **0.866** | **0.720** |

- BT retains 99% of its in-family ranking accuracy (0.866 vs 0.873) on
  unseen families; proper retains its log score (−2.073 vs −2.071).
- bt > rf on 5/5 folds (one-sided sign test p = 0.031); the mean gap more
  than doubles under family shift (+0.056 → +0.146), and rf collapses
  hardest exactly where its features mislead (sixhumpcamel 0.608).
- Exploratory caveat: per-fold aggregates; the confirmatory rerun should
  log per-record predictions for record-clustered bootstrap CIs (the
  fold checkpoints are deleted after eval; retraining is ~2 min/fold).

Interpretation: the option-conditioned, semantically prompted Laya policy
distills a *representation* of the decision rule that transfers across
problem families; the axis-aligned-feature RF imitates the rule in-distribution
but does not carry it across shift (L1's corollary). This is the RQ2-style
positive the journal positioning asked for, and it is the strongest
candidate headline of the fine-tuning line.

## What this round establishes for the journal

1. **Positive (E5)**: counterfactual-distilled typed-decision models
   transfer across optimization families at ~full in-family accuracy,
   beating feature-imitation baselines with a gap that grows under shift.
2. **Method (E1 + theory R1)**: objective choice trades ranking vs
   calibration shape; proper-score training is uniquely joint-calibrated;
   post-hoc temperature cannot substitute.
3. **Mechanism (E1+E4+T4/L1)**: closed-loop value is capped by the
   teacher's thin state-conditional signal; confidence carries no
   state-level routing information, so routing degenerates to family-level
   switching — the RQ3 frontier stays degenerate for a deeper reason than
   "the RF is flat": *no current supervision provides discriminative
   uncertainty*.
4. **Open (E2/E3)**: the long-horizon teacher amplifies exploration mass
   10× but still fails terminal validity; fine-tunes at any horizon
   collapse to the teacher's dominant mode.

## Next steps toward the journal claim set

1. Confirmatory E5: per-record logging + record-clustered CIs; 2 seeds
   of training per fold for stability.
2. Discriminative-uncertainty direction: OOD/entropy router inputs
   (DISCUSS Q-T4), or learned abstention — motivated by E1's null.
3. Gate-4-valid teacher: utility revision on the long-horizon corpus
   (Phase 8.7+ line) — everything downstream (closed-loop, routing)
   inherits its validity.

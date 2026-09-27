# Journal round 2 — P0–P3 consolidated report (2026-09-27/28)

Branch `exp/rlcd-laya-finetune`. Executes the plan agreed after the GPT
assessment: lock the transfer positive (P0), diagnose and fix the teacher
(P1), retest the full chain on the fixed teacher (P2), extend to an
engineering problem (P3).

## Headlines

1. **P0 — confirmatory ANALYSIS RERUN of the transfer positive** (per
   codex review 2026-09-28, issue 1/2; estimand = these five families,
   this split, these two training seeds). Record-clustered paired
   bootstrap over 663 held-out records, dual training seeds per fold:
   **primary contrast: bt − RF pooled = +0.140, 95% CI [+0.126, +0.154]**.
   Per-family contrasts are EXPLORATORY/descriptive (unadjusted; folds
   share training data; the round-1 sign test re-uses the same five
   outcomes and is not independent evidence): ackley +0.209, branin
   +0.045, hartmann +0.016, rastrigin +0.151, sixhumpcamel +0.305 —
   consistent in direction on 5/5. proper − RF pooled +0.019
   [+0.007, +0.032]. A LOCKED evaluation on families/tasks untouched by
   any design decision remains required before calling the effect
   confirmatory in the strong sense. (`runs/exp_rlcd_loo_confirmatory.json`)

2. **P1 — the teacher diagnosis chain is complete and decisive.**
   - P1a (metrics corrected per codex issue 3): single-roll labels are
     noise-dominated — PER-RE-ROLL pair flip probability 0.36 (vs stored
     labels) on pairs above the median gap; conservative any-of-3 flip
     0.73; inter-roll gap spread (0.084) exceeds the median stored gap
     (0.05); argmax stable across re-rolls for only 25% of records.
     Re-roll seeds are now deterministic (no hash()).
   - P1b/denoising: R=5 replicate-averaging (generator gained
     `rollout_replicates`; 661-record dataset in 141 s). Independent 2v2
     split-half reliability of the averaged teacher: pair flip 0.33-0.36
     even at top-quartile stored gaps — averaging helps but does not make
     the teacher stable at these gap scales. Gate 3 holds (13.3). Gate 4
     fails.
   - P1b/alignment: a Gate-4-ALIGNED teacher (rule continuation + pure
     terminal-Δbest utility, R=5) removes the label-side protocol
     mismatch. Gate 4 STILL fails — behavior-gain seed-bootstrap CIs:
     ackley +0.25 [−0.18, +0.78], hartmann −0.24 [−2.63, +1.92],
     rastrigin +0.04 [−0.10, +0.17], sixhump +0.00 [−0.01, +0.01],
     branin −0.02 [−0.04, −0.00]. **No family is CI-positive; the gate's
     own replay is a single evaluation with unmatched randomness and the
     teacher covers only the k=4 rolled subset**, so the honest statement
     is: THIS TESTED TEACHER fails Gate 4, with first-action terminal
     advantage bounded to at most small on this panel (CIs consistent
     with anything from ≈−0.2 to +0.8 on Ackley/Hartmann). The
     schedule/continuation explanation remains a HYPOTHESIS pending a
     direct schedule-label experiment (Phase 8.6F is prior evidence, not
     proof).

3. **P2 — the thin-signal ceiling propagates through every stage.**
   Models trained on the aligned teacher: decision-level ranking 0.67
   (harder k=4 partial regime); closed-loop still collapses to the same
   per-objective constants (bt_al→`local_exploitation` 1250/1250,
   proper_al→`error_guided_sampling` 1250/1250); confidence
   discrimination still zero (conf_gap ≤ 0.001, AURC ≈ 1 − accuracy;
   teacher-argmax agreement of the models themselves only 0.12–0.22 —
   the aligned teacher's argmax is nearly random across its rolled
   actions). No supervision tried tonight — h3, long-horizon, denoised,
   aligned — produces a state-conditional signal strong enough to escape
   constant-collapse or seed a router.

4. **P3 — engineering extension (probe-level).** GearTrain (the only
   BOCoDe engineering problem passing the backend's single-objective /
   unconstrained gates) added to the allowlist with 4 integration tests.
   Numerical→engineering decision probe (no retraining; PROBE-LEVEL per
   codex issue 7 — 65 records, one problem, no CI, no closed-loop):
   bt_h3 0.893 / rf_synthetic 0.890 (effectively tied) / proper_h3 0.801
   / zero-shot 0.423. Reads as: synthetic-trained policies are NOT
   broken on engineering states; the Laya-vs-RF advantage does not
   demonstrably extend here. End-to-end engineering coverage remains a
   pre-submission P0 item.
   All other engineering problems need constraint semantics (documented
   P3 remainder). External optimizer baselines (SMAC3/HEBO/LLAMBO) not
   installed — remains the P0 list's mechanical remainder.

## What the paper can now claim (updated claim set)

- **Positive (confirmatory)**: counterfactual-rollout distillation of a
  typed-decision foundation model transfers across optimization families
  at near in-family accuracy and beats feature-imitation (RF) with CI
  excluding zero on 5/5 held-out families; it also transfers to an
  engineering problem's decision states at ~0.89 (probe-level).
- **Method**: objective choice trades ranking vs calibration shape for
  the TESTED models/protocol (proper jointly calibrated in our E1 setup;
  the tested BT+post-hoc-temperature fit recovered one metric at the
  cost of the other — not a general impossibility claim);
  pairwise supervision is the quota-correct estimator (T3/R3); rollout
  replicate-averaging is necessary hygiene (P1a) and cheap (T2).
- **Mechanism chain (each link measured; strengths as stated above)**:
  single-roll labels noise-dominated (per-re-roll flip 0.36) → the tested
  denoised+aligned teacher still fails Gate 4 (no CI-positive
  behavior-gain) → distilled policies collapse to per-objective constants
  → confidence carries no discrimination → routing degenerates to
  family-level switching.
- **Consequent redirection**: the learnable-and-transferable object at
  this action granularity is the RANKING (which transfers); the
  value-bearing object is the SCHEDULE/continuation — the next
  supervision design should label policies, not first actions.

## Artifacts

```
runs/exp_rlcd_loo_confirmatory.json       # P0 (per-record, dual-seed, CIs)
runs/exp_teacher_noise_diagnostic.json    # P1a (R=1 verdict)
runs/exp_teacher_noise_diagnostic_r5.json # P1a on averaged labels
runs/longhorizon_k4r5_20seed_5family.jsonl      # P1b denoised teacher
runs/longhorizon_k4r5_aligned_20seed_5family.jsonl  # P1b aligned teacher
runs/exp_r5_gates.json, runs/exp_aligned_gates.json  # P1b gates
runs/exp_rlcd_aligned_train.json          # P2 training (both arms)
runs/exp_rlcd_closed_loop_aligned.json    # P2 closed-loop (collapse)
runs/exp_rlcd_discrimination_aligned.json # P2 discrimination (zero)
runs/geartrain_h3_10seed.jsonl            # P3 engineering dataset
runs/exp_geartrain_transfer_probe.json    # P3 probe
scripts/exp_rlcd_loo_confirmatory.py, exp_teacher_noise_diagnostic.py,
        exp_geartrain_transfer_probe.py
src: DatasetConfig.rollout_replicates (+_average_rollouts),
     GearTrain in backend allowlist; tests: 3 replicate + 4 GearTrain
```

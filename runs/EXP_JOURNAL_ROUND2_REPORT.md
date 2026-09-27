# Journal round 2 — P0–P3 consolidated report (2026-09-27/28)

Branch `exp/rlcd-laya-finetune`. Executes the plan agreed after the GPT
assessment: lock the transfer positive (P0), diagnose and fix the teacher
(P1), retest the full chain on the fixed teacher (P2), extend to an
engineering problem (P3).

## Headlines

1. **P0 — the transfer positive is now confirmatory.** Record-clustered
   paired bootstrap over 663 held-out records, dual training seeds per
   fold: **bt − RF pooled = +0.140, 95% CI [+0.126, +0.154]; CI excludes
   zero on 5/5 folds** (ackley +0.209, branin +0.045, hartmann +0.016,
   rastrigin +0.151, sixhumpcamel +0.305); per-fold training seeds stable
   (e.g. rastrigin bt [0.840, 0.850]). proper − RF pooled +0.019
   [+0.007, +0.032] — CI-positive but small; per-fold mixed (loses
   Branin/Hartmann CI-detectably).
   (`runs/exp_rlcd_loo_confirmatory.json`)

2. **P1 — the teacher diagnosis chain is complete and decisive.**
   - P1a: single-roll labels are **noise-dominated** — re-roll flip rate
     0.74 on pairs above the median gap; inter-roll gap spread (0.084)
     exceeds the median stored gap (0.05); argmax stable across re-rolls
     for only 25% of records.
   - P1b/denoising: R=5 replicate-averaging (generator gained
     `rollout_replicates`; 661-record dataset in 141 s) reduces split-half
     flip to 0.37 (residual label-vs-true ≈ 19% at median gap). Gate 3
     holds (13.3). **Gate 4 still fails.**
   - P1b/alignment: a Gate-4-ALIGNED teacher (rule continuation + pure
     terminal-Δbest utility, R=5) removes the label/evaluation mismatch
     entirely — **Gate 4 still fails**: behavior-gain ackley +0.25,
     rastrigin +0.04, branin −0.02, hartmann −0.24, sixhump +0.00. With
     noise reduced AND alignment exact, the residual is the truth: **the
     first action's true terminal advantage over the behavior policy is
     ≈ 0 at these budgets/action granularity.** Gate 4 as a criterion for
     single-action labels is unattainable in this regime; decision value
     lives in schedules/continuations (Phase 8.6F's conclusion, now
     reproduced on clean, aligned labels).

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
   Numerical→engineering decision probe (no retraining): bt_h3 0.893 /
   rf_synthetic 0.890 / proper_h3 0.801 / zero-shot 0.423 on 65 GearTrain
   decision records — synthetic-trained policies transfer to an
   engineering problem's states at ~0.89; the Laya-vs-RF advantage does
   not clearly extend on this single small probe (exploratory, no CI).
   All other engineering problems need constraint semantics (documented
   P3 remainder). External optimizer baselines (SMAC3/HEBO/LLAMBO) not
   installed — remains the P0 list's mechanical remainder.

## What the paper can now claim (updated claim set)

- **Positive (confirmatory)**: counterfactual-rollout distillation of a
  typed-decision foundation model transfers across optimization families
  at near in-family accuracy and beats feature-imitation (RF) with CI
  excluding zero on 5/5 held-out families; it also transfers to an
  engineering problem's decision states at ~0.89 (probe-level).
- **Method**: objective choice trades ranking vs calibration (proper
  uniquely joint-calibrated; post-hoc temperature cannot substitute);
  pairwise supervision is the quota-correct estimator (T3/R3); rollout
  replicate-averaging is necessary hygiene (P1a) and cheap (T2).
- **Mechanism (complete chain)**: single-roll labels are noise-dominated
  (0.74 flip) → even noise-free, Gate-4-aligned single-action labels have
  ≈0 true terminal advantage → distilled policies collapse to per-objective
  constants → confidence carries no discrimination → routing degenerates
  to family-level switching. Every link now has a direct measurement.
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

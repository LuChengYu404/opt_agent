# EXP-RLCD closed-loop audit — fine-tuned Laya checkpoints (2026-09-27)

Branch `exp/rlcd-laya-finetune`. Parallel harness: `scripts/
eval_laya_closed_loop.py` (ProcessPoolExecutor, 8 workers, paired backend
seeds). Panel: 5 BOCoDe families × fresh seeds 200–209 × budget 30 × 4 arms
(rule_full_budget, AlwaysSurrogate, `runs/rlcd_laya_proper_v1`,
`runs/rlcd_laya_bt_v1`) = 200 trajectories. Raw rows + seed-clustered
bootstrap CIs: `runs/exp_rlcd_closed_loop.json`.

## Headline finding: both fine-tuned arms collapse to the same constant policy

The two fine-tuned checkpoints produce **identical action sequences on
every trajectory** — 1250/1250 decisions are `local_exploitation` (one
`retrain_surrogate` on the proper arm). Their closed-loop results are
therefore the results of the fixed Local policy:

| Family | proper/bt vs rule_full | proper/bt vs surrogate |
|---|---:|---:|
| Rastrigin | **+0.624, CI [+0.288, +0.936]*** | +0.259, CI crosses 0 |
| Ackley | +2.339, CI crosses 0 | +2.171, CI crosses 0 |
| Hartmann | −0.820, CI crosses 0 | **−9.041, CI [−17.89, −1.91]*** |
| Branin | +0.000, CI crosses 0 | **−0.016, CI [−0.025, −0.007]*** |
| SixHumpCamel | +0.005, CI crosses 0 | −0.011, CI [−0.024, −0.000]*** |

(\* = CI excludes 0.) This is the known fixed-Local profile: strong where
Local is strong (Rastrigin), dominated by Surrogate on Hartmann/Branin —
the same pattern as the EXP-RQ1.3 closed-loop table and the
GlobalThenLocal schedule control.

## Interpretation (ties directly to the theory)

1. **T4/L1 logic confirmed.** The h3 teacher's argmax is dominated by its
   constant component (`local_exploitation`, per P2's best-constant readout
   and the schedule-control finding). Distilling the teacher — under
   either objective — reproduces a mostly-constant policy, and its
   closed-loop value is exactly that constant action's value. The
   decision-level ranking advantages over RF (BT +0.056 CI-excl-0) did
   **not** translate into closed-loop gains beyond fixed Local: the
   state-conditional signal the teacher carries is too thin on this panel
   (Gate 2 FAIL; C3 heterogeneity).
2. **The calibration advantage is invisible under forced argmax.** The
   proper-score arm's distribution-level win (log score −2.07 vs −3.06)
   cannot matter in a forced-System-1 loop that takes argmax at every
   step. It is designed to matter where the *reported distribution* is
   consumed — router thresholds, escalation (Phase 12), or the RLCD
   proper-score reward itself in the closed-loop RL stage (R4). Testing
   that requires the router panel, not this audit.
3. **Consistency check passed.** Both objectives, trained independently,
   converge to the same argmax policy — the teacher's dominant mode —
   which is exactly what two consistent estimators of the same teacher
   should do when the teacher is dominated by its mode.

## What this does and does not establish

- Fine-tuning works (both arms), is hyperparameter-robust (sweep:
  defaults near-optimal; best proper 0.791 @ w_sph 0.5 / lr 5e-6, best bt
  0.879 @ β 1 / margin 0), and the two arms trade ranking vs calibration
  as R1(iii) predicts.
- No closed-loop gain over the best fixed action on this panel — matching
  the RQ1 open status, not contradicting it. The panel's adaptive-value
  ceiling D and the teacher's Gate-2 instability bound what any
  teacher-distilled policy can do here.
- Next: (a) router-panel evaluation of the proper arm (where calibration
  pays); (b) long-horizon quota teachers (the Phase 8.6C horizon
  diagnosis says horizon is what makes labels state-conditional — the
  k=4 long-horizon dataset `runs/longhorizon_k4_20seed_5family.jsonl`
  exists for this); (c) the R4 closed-loop RL stage on realized outcomes.

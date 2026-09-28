# Opt Agent v3 — Branch Snapshot: RLCD fine-tuning of Laya

**Date**: 2026-09-27 (objective reference corrected same day, see §1)
**Branch**: `exp/rlcd-laya-finetune` (off `main` @ 8c70b79)
**Purpose**: Separately evaluate fine-tuning Laya as the System-1 decision
model (vs the RandomForest "semantic routing" baseline), with the
theoretical support for **RLCD — the Jev/Laya-lineage method** — as the
first deliverable.

> The previous content of this file (the merged `exp/rq0.5-llm-system1`
> snapshot, 2026-09-26) lives on in `main`'s history; this branch reuses
> the filename per repo convention.

---

## 1. What RLCD is (name verified 2026-09-27)

**RLCD = "Reinforcement Learning for Calibrated Decisions"** — the
training method of the Jev/Laya line of typed, non-autoregressive
System-1 decision models (the "TypeSafe-style" family the user pointed
at): RL whose reward is a **strictly proper scoring rule**, so the
expected-reward optimum is the true outcome distribution — calibration by
construction. Laya ships it in the SDK (`laya.common.proper_reward`:
log score + 0.5·spherical − RPS-on-ordinal; `td_lambda_targets` for
multi-turn credit assignment; `act_costs` escalation economics). **No
method paper exists** (arXiv:2609.24574; independent tests); the closest
public description is arXiv:2609.28940 (which also documents Brier-score
training — same proper family). It is NOT the identically-named
contrastive-distillation paper of Yang et al. ICLR 2024. An early commit
on this branch (642617d) made exactly that confusion; corrected after
verification, following the repo's sign-correction precedent.

## 2. Theory (the requested focus): `docs/RLCD_THEORY.md`

**R-propositions (RLCD proper):**
- **R1 strict properness** — the SDK's composite (log + 0.5·spherical) is
  strictly proper (Gibbs inequality + Shuford et al.; positive sums of
  strictly proper rules stay proper) ⇒ the population optimum is the
  teacher *distribution*; identification strictly stronger than
  ranking-only BT (R1iii hierarchy).
- **R2 noisy targets** — log-score optimum = **mean of noisy targets**
  (linearity): the arm tracks the noise-averaged teacher; contrast with
  BT's sign-stability (T2). Which arm wins empirically separates the two
  noise models.
- **R3 subset conditioning** — Laya's option-conditioned architecture is
  what makes quota-subsampled records consumable by proper-score training
  (no forced-zero supervision for absent options; cross-subset
  consistency via the shared scalar score per (s, a)); refines T3, which
  remains valid for fixed-head consumers (RF-style).
- **R4 where the RL lives** — proper scores are realizable against
  *outcomes* (Brier/log vs realized one-hot), giving the closed-loop
  stage a dense per-decision reward + TD(λ) bootstrap (the SDK's own
  machinery), with calibration pressure persisting during RL.

**T-propositions (BT baseline arm):** T1 invariance (inherits P1, cancels
per-state shifts), T2 noise consistency (noisy-BT shrinkage preserves
ranking; margin filtering restores learnability), T3 quota bias of
renormalized subset softmax (fixed-head consumers), T4 ceiling (inherits
P2's D; decision-level metrics bound deployable value), T5 RL-stage
design.

## 3. Implementation

- `src/opt_agent/system1/rlcd.py` — `proper_score_loss` (calls the SDK's
  `proper_reward` verbatim; the RLCD objective) and `bradley_terry_loss`
  (baseline arm); `objective` switch in `RLCDConfig`; `mine_pairs`
  (partial records first-class); `build_item` now carries both supervision
  forms (pairs + teacher distribution on offered options); training with
  early stop on val ranking; laya-loadable checkpoint saving;
  `evaluate_ranking` now also reports mean log score (floored like
  `proper_reward` for comparability).
- `scripts/train_rlcd_laya.py` — 4-arm driver: zero-shot / RLCD
  proper-score / BT pairs / RF.
- `scripts/analyze_rlcd_ci.py` — record-clustered paired bootstrap CIs +
  per-arm log scores.
- `tests/test_rlcd.py` — 15 tests (pair mining, both losses incl. strict-
  properness preference and gradient flow, config validation,
  real-tokenizer item alignment, stub-model backward).

## 4. Headline result (`runs/EXP_RLCD_REPORT.md`; 115 held-out records, 2889 pairs)

| Arm | Pairwise acc | Top-1 | Mean log score |
|---|---:|---:|---:|
| Laya zero-shot | 0.396 (< chance) | 0.035 | −3.32 |
| **RLCD proper-score** | 0.800 | 0.252 | **−2.07** |
| BT pairs (baseline) | **0.873** | 0.383 | −3.06 |
| RandomForest | 0.817 | 0.400 | — |

Reference points: uniform-8 log score −2.079; teacher self-score −1.977
(entropy 1.98 nats).

Estimand note (2026-09-28 audit): Top-1 in this table is the
training-report estimand (`evaluate_ranking`, artifact
`exp_rlcd_report.json`). The CI artifact `exp_rlcd_ci.json` uses the
`per_record_laya` estimand (raw-utility argmax, valid-only) and reads
top-1 bt 0.365 / proper 0.252 — same ordering, not directly comparable;
it also wobbles ±1 record across GPU runs. The bootstrap-denominator
audit fix moved CIs by ≤0.003 (no conclusion changes).

**Reading (matches the corrected theory):**
- Both fine-tunes dominate zero-shot (CIs exclude 0 by wide margins).
- **Ranking:** BT > RF (+0.056, CI [+0.033, +0.080]); RLCD-proper ≈ RF
  (CI crosses 0); BT > RLCD-proper (+0.073, CI excludes 0).
- **Calibration/distribution:** RLCD-proper is the only arm whose log
  score beats uniform and approaches the teacher; BT's log score
  *degrades below uniform* while its ranking climbs (−2.77→−3.06) — the
  overconfidence pathology the proper-scoring design exists to prevent.
- Practical: the Router (Phase 12) consumes confidence; the EXP-RQ3
  router failure traced to a never-above-0.25 RF softmax. The RLCD-proper
  arm is the one whose reported distribution is meaningful for routing;
  the BT arm is the ranking specialist. Combining them is a follow-up.

Not claimed: closed-loop optimization gains (T4 caps one-step value at D;
R4 designs the closed-loop stage), cross-family transfer (RQ2).

## 5. Parallel follow-up round (2026-09-27, same day; hardware: 32 cores + RTX 3090)

Four tracks run with explicit parallelization (per user direction):

- **A — long-horizon quota dataset** (`runs/longhorizon_k4_20seed_5family.jsonl`,
  661 records: 5 families × 20 seeds, horizon=remaining mean 15.7/max 25,
  k=4, 3,966 pairs) in **47 s wall-clock** with 24 single-thread spawn
  workers (vs ≈90 min single-worker full-8 equivalent). Driver gained
  `torch.set_num_threads(1)` + spawn context (the first attempt
  oversubscribed to load 315 with default threads).
- **B — hyperparameter sweep** (`scripts/sweep_rlcd.py`, `runs/exp_rlcd_sweep.json`,
  15 configs; RE-RUN 2026-09-28 after the margin-plumbing audit fix — the
  original run's margin axis was inert): defaults are near-optimal — best
  proper 0.795 (w_sph 0.5, lr 5e-6), best bt 0.878 (β 1, margin 0); all bt
  configs within 0.852–0.878. margin 0 beats margin 0.05 at every β
  (+1.5–2.6 pp; the margin arms also evaluate on fewer pairs, so treat the
  margin comparison as directional). The objective choice is the lever,
  not the knobs.
- **C — training on the long-horizon partial dataset**
  (`runs/EXP_RLCD_LONGHORIZON_REPORT.md`): bt 0.660 > proper 0.576 >
  RF 0.545 > zero-shot 0.485 ranking; proper −2.071 log score again the
  only calibration win. Same ordering as h3 at smaller margins (fewer
  pairs: C(4,2)=6/record).
- **D — parallel closed-loop audit** (`scripts/eval_laya_closed_loop.py`,
  8 workers, 200 trajectories, fresh seeds 200–209;
  `runs/EXP_RLCD_CLOSED_LOOP_REPORT.md`): both fine-tuned arms produce
  **identical action sequences — 1250/1250 `local_exploitation`** — i.e.
  the distilled policy equals fixed Local: +0.62 vs rule_full on Rastrigin
  (CI excludes 0), −9.04 vs Surrogate on Hartmann (CI excludes 0).
  Decision-level ranking wins did NOT translate into closed-loop gains
  beyond the best constant action — consistent with T4's D-ceiling, L1's
  imitation logic, and the RQ1 open status (Gate 2 FAIL, thin
  state-conditional signal in the h3 teacher). The proper arm's
  calibration advantage is structurally invisible under forced argmax;
  it is designed for router/escalation panels and the R4 RL stage.

## 6. Next steps on this branch

1. Router-panel evaluation of the proper arm (calibration should pay
   exactly where EXP-RQ3's router failed).
2. Closed-loop teachers from the long-horizon dataset (Phase 8.6C: horizon
   is what makes labels state-conditional) — train + Gate-4-style replay.
3. R4 closed-loop RL stage: proper-score reward on realized outcomes +
   `td_lambda_targets` over decision-loop trajectories.
4. Two-stage schedule (BT-rank → proper-calibrate) if both properties are
   needed in one deployed model.

## 7. File index

```
docs/RLCD_THEORY.md                     # disambiguation + R1-R4 + T1-T5 (theory deliverable)
src/opt_agent/system1/rlcd.py           # proper_score_loss (RLCD) + bradley_terry_loss (baseline)
scripts/train_rlcd_laya.py              # 4-arm train + comparison driver
scripts/sweep_rlcd.py                   # parallel GPU sweep (spawn, save=False)
scripts/eval_laya_closed_loop.py        # parallel closed-loop audit (ProcessPool)
scripts/analyze_rlcd_ci.py              # record-clustered paired bootstrap CIs
tests/test_rlcd.py                      # 15 unit tests
runs/rlcd_laya_proper_v1/, runs/rlcd_laya_bt_v1/     # h3 checkpoints (gitignored weights)
runs/rlcd_laya_proper_lh/, runs/rlcd_laya_bt_lh/     # long-horizon checkpoints
runs/longhorizon_k4_20seed_5family.jsonl             # quota dataset (Track A)
runs/EXP_RLCD_REPORT.md                 # 4-arm decision-level report + CIs
runs/EXP_RLCD_LONGHORIZON_REPORT.md     # Track C
runs/EXP_RLCD_CLOSED_LOOP_REPORT.md     # Track D (collapse finding + interpretation)
runs/exp_rlcd_report.json, exp_rlcd_ci.json, exp_rlcd_sweep.json,
runs/exp_rlcd_longhorizon.json, runs/exp_rlcd_closed_loop.json
```

## 8. Journal round E1–E5 (2026-09-27 evening; parallel CPU+GPU)

Five experiments against the journal gaps (`docs/ref_papers/journal_positioning.md`).
Consolidated report: `runs/EXP_RLCD_JOURNAL_ROUND_REPORT.md`.

- **E1 calibration ablation**: proper-score is the only jointly calibrated
  arm (log −2.07 + ECE 0.11); BT+temperature recovers log but breaks ECE
  (0.02→0.26); **confidence discrimination ≈ 0 in every arm** (conf_gap
  ≤ 0.013, AURC ≈ 1−acc).
- **E2 long-horizon gates**: Gate 3 lift 12.9 (10× the h3 teacher's 1.29);
  Gate 2/4 still FAIL. Partial-record Gate 2 machinery fixed
  (`validation.gate2_stability` now averages per-action mass over records
  that rolled the action out).
- **E3**: long-horizon fine-tunes still collapse to constants — different
  modes per objective (bt_lh→local, proper_lh→error_guided).
- **E4 router panel (predeclared)**: FAIL — quantile routing degenerates to
  family-level all-or-nothing switching (Hartmann +5.87*** vs pure but
  trajectories ≡ surrogate). Consistent with E1's null discrimination;
  sharpens RQ3: the missing ingredient is discriminative uncertainty.
- **E5 LOO transfer (the positive)**: bt retains 0.866/0.873 of in-family
  ranking on unseen families; RF drops to 0.720; bt > rf on 5/5 folds
  (sign p=0.031); the gap doubles under shift (+0.056 → +0.146).
  Exploratory (per-fold aggregates); confirmatory per-record rerun queued.

New scripts: `exp_rlcd_calibration_ablation.py`, `exp_rlcd_router_panel.py`,
`exp_rlcd_loo_transfer.py`. Gate-2 partial-record fix in
`src/opt_agent/dataset/validation.py`.

## 9. Journal round 2 — P0–P3 (2026-09-27/28)

Consolidated report: `runs/EXP_JOURNAL_ROUND2_REPORT.md`.

- **P0 confirmatory**: bt−RF pooled **+0.140 [+0.126, +0.154]**, CI excludes
  0 on 5/5 folds, dual training seeds, 663 records.
- **P1 teacher chain**: single-roll labels noise-dominated (flip 0.74);
  R=5 replicate averaging (new `rollout_replicates`) cuts split-half flip
  to 0.37; Gate-4-ALIGNED teacher (rule continuation + terminal-Δbest
  utility) removes label/evaluation mismatch — **Gate 4 still fails:
  first-action true terminal advantage ≈ 0**. Value lives in
  schedules/continuations (8.6F reproduced on clean labels).
- **P2**: aligned-teacher models still collapse to per-objective constants;
  discrimination still zero — the thin-signal ceiling propagates.
- **P3**: GearTrain added (only gate-passing engineering problem; 4 tests);
  numerical→engineering probe: bt 0.893 / rf 0.890 / zero-shot 0.423.
  Constraint-semantics problems + external baselines documented as remainder.

## 10. Codex review loop (2026-09-28): REVISE → REVISE → REVISE → APPROVE

4 rounds via `codex exec` (proxy per user instruction). Round 1: 7 issues
(3 BLOCKER / 4 MAJOR) — all substantive; logs `runs/codex_review_round2.log`,
`codex_rereview_round2.log`, `codex_rereview3.log`, `codex_rereview4.log`.
Revisions (commits 34f55cf, e0d0502, 159a050): P0 relabeled analysis-rerun
(fixed-family/fixed-training estimand; locked new-family eval required);
pooled CI = single primary contrast, 5/5 = descriptive; noise metrics
redefined and recomputed (per-re-roll 0.40 above-median; 2v2 split-half
persisted, 0.36/0.33/0.33); Gate-4 claim scoped to the tested teacher with
wide-CI acknowledgment (upper bounds +0.78/+1.92 on Ackley/Hartmann);
claim narrowing (RF singularity, temperature, teacher-relative calibration);
theory fixes (R2 exact-linear spherical, R3 set-independence as hypothesis,
T2 direct tail assumption, R4/T5 marked proposals). Round 4: both remaining
nits ADDRESSED — **VERDICT: APPROVE**.

Pre-submission remainder (unchanged, from review): locked transfer
evaluation on untouched families/tasks; benchmark breadth (BBOB/IOH, HPO,
end-to-end engineering); external optimizer anchors (SMAC3/HEBO/LLAMBO).

## 11. Full-repo consistency audit + Codex adjudication (2026-09-28)

Independent 4-agent audit (src / scripts / docs-vs-artifacts / tests) +
lead spot-verification, then one `codex exec` adjudication round
(logs: `runs/AUDIT_INCONSISTENCY_FINDINGS.md`,
`runs/codex_audit_review.log`). Verdict on 20 findings: 15 CONFIRMED /
5 PARTIAL / 0 REFUTED, VERDICT: SELECTIVE. All findings fixed this
session; full suite green (see below).

Code fixes (conclusion-affecting):
- `exp_rlcd_loo_transfer.py` NameError on `torch` (regression from the
  e523119 cleanup) — LOO transfer re-runnable.
- `RLCDConfig.margin` was dead config (never reached `mine_pairs`): the
  sweep's margin 0.0-vs-0.05 arms were computationally identical —
  differences in the original `exp_rlcd_sweep.json` were GPU
  nondeterminism, NOT a margin effect. margin is now threaded through
  `build_item`; the sweep was RE-RUN on the fixed plumbing (same session):
  margin 0 wins at every β (+1.5–2.6 pp), best configs unchanged
  (bt β1/margin0 0.878; proper w_sph0.5/lr5e-6 0.795) — "defaults
  near-optimal" stands on real margin arms now.
- `analyze_rlcd_ci.py`: paired bootstrap normalized both arms by arm A's
  pair total (each arm now uses its own resampled total) and the NaN
  filter missed the new `None` log-score sentinel. `exp_rlcd_ci.json`
  regenerated: CIs move ≤0.003, no conclusion changes. NOTE:
  `per_record_laya` top-1 wobbles ±1 record (~0.009) across GPU runs on
  a borderline-tie record (0.365↔0.374 observed for bt).
- `executor.py` ERROR_GUIDED_SAMPLING with n_samples>1 evaluated the SAME
  perturbed point n times (`repeat`); now draws n independent
  perturbations (n=1 bit-identical). The factory can emit n_samples up to
  3 for this action, so historical datasets may contain duplicate
  evaluations from it — semantics were "averaging the same point" (wasted
  budget, no directional bias), but any such records are flagged by this
  note.
- `validate_record` rejected partial-rollout (subsample k<8) records its
  own generator produces; invariant relaxed to gold_probs ⊆ candidates
  (+ regression test).
- `_rlcd_common.per_record_laya` training-mode restore was inverted.

Consistency fixes (docstrings/defaults/telemetry): `w_stop_opportunity`
docstring now states default 0.75; RandomPolicy/UniformPolicy docstrings
no longer claim a nonexistent 5-vs-8 distinction (implementations are
identical — `uniform` continuation arms equal `RandomPolicy`; historical
ablation text relying on the distinction should be read accordingly);
`HeuristicLLMClient` docstring matches its uniform mock; `oracle_test`
default max_actions 50→200 (matches `oracle_test_from_record`);
executor step fallback 0.05→0.1 (factory-aligned) + fallback-convention
comments; `JevPolicy.reset()` clears counters / stop branch increments
`call_count` (symmetric with Laya/LLMSystem1 — earlier cost comparisons
across adapters were not apples-to-apples after reset); `_shared_rq_features`
header no longer lists the archived rf_probe as a live consumer.

Test hardening: `test_rlcd` teacher fixture was anti-correlated with its
utilities (silent wrong-teacher risk); tautological assertion in
`test_phase8_5_gates` replaced with finiteness checks; the fake
"affine-invariant" test now actually tests scale/shift invariance through
`to_gold_probs`; Gate 3's ≥5-seed seed-clustered CI branch got its first
coverage; partial-rollout validation regression test added.

Doc-vs-artifact fixes (reviewer/paper-facing): review brief no longer
claims Gate 5 "all buckets pass" (predeclared gate FAILED, 2/17 JS
breaches) or the withdrawn p=0.0003 (Student-t 0.0058) or stale RQ2/
frontier status; `PROGRESS.md` self-contradiction on the cost frontier
resolved (frontier ran, 450 trajectories; engineering transfer is
probe-level only); teacher-noise satellites 0.084/25% traced to the
superseded log and updated to 0.0915/29% (headline 0.40/0.36 re-verified
reproducible); SixHumpCamel `***` scoped to the bt arm; "+5.87 every
routed arm" corrected (+5.40 for bt_q20); "+Hartmann now passes" →
+Branin; affine §3.2 heading 15/17→17/17; THEORY.md D values moved to
the 20-seed recomputation; P3/L1 no longer claimed as script-verified.

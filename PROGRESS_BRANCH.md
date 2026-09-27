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
  15 configs): defaults are near-optimal — best proper 0.791 (w_sph 0.5,
  lr 5e-6), best bt 0.879 (β 1, margin 0); all bt configs within
  0.86–0.879. The objective choice is the lever, not the knobs.
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

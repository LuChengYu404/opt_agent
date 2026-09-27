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

## 5. Next steps on this branch

1. Closed-loop evaluation of `runs/rlcd_laya_proper_v1` (and the BT arm)
   via `LayaPolicy(model_id=…)` on the EXP-RQ3 paired panel — the
   calibration advantage should matter exactly where the router failed.
2. Train on a long-horizon partial dataset (main's
   `generate_dataset_longhorizon.py`, k=4) — R3/T3 quota analysis says
   both arms consume partial records correctly under option conditioning.
3. Margin/β/w_sph sweeps; two-stage (BT-rank → proper-calibrate) schedule.
4. Closed-loop RL stage per R4: proper-score reward on realized outcomes
   + `td_lambda_targets` over decision-loop trajectories.

## 6. File index

```
docs/RLCD_THEORY.md                     # disambiguation + R1-R4 + T1-T5 (theory deliverable)
src/opt_agent/system1/rlcd.py           # proper_score_loss (RLCD) + bradley_terry_loss (baseline)
scripts/train_rlcd_laya.py              # 4-arm train + comparison driver
scripts/analyze_rlcd_ci.py              # record-clustered paired bootstrap CIs
tests/test_rlcd.py                      # 15 unit tests
runs/rlcd_laya_proper_v1/, runs/rlcd_laya_bt_v1/   # checkpoints (weights gitignored)
runs/EXP_RLCD_REPORT.md                 # experiment report + CIs + interpretation
runs/exp_rlcd_report.json, runs/exp_rlcd_ci.json
```

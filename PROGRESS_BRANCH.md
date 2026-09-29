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

## 12. Paper outline: consistency revision + RLCD integration (2026-09-28)

Two commits on `docs/workshop_paper_outline.md`:

1. **Internal-consistency revision** (7 contradictions fixed): abstract
   point 3 said the schedule control "has not been run" while its own
   point 4 and §6.2/§6.5 reported its results; the abstract still said
   cross-problem transfer "not yet demonstrated" (RQ2.1 predates the
   sentence); §5.1 regime table carried 10-seed D values whose
   Rastrigin/Ackeli ordering flips in §3.8's 20-seed table; subsection
   numbering was off (§4.1–4.5 under §5, §5.1–5.5 under §6 → renumbered);
   test count 385→423 (x4); title aligned with the schedule-sufficiency
   finding (old title kept as noted alternative).
2. **RLCD line integrated** (content update, numbers re-verified against
   `runs/*.json` in the same session): new §4.2 (fine-tuned-model
   cross-family transfer +0.140 5/5 folds analysis-rerun; GearTrain
   probe), new §5.6 (teacher-noise chain; 4-arm distillation table; sweep
   re-run numbers; long-horizon; closed-loop collapse + router panel),
   limitations 15–16, discussion/conclusion/abstract point 5, negative
   result (g). Fresh codex review recommended before submission — the
   prior APPROVE predates this integration.

Verification this pass (all personally re-pulled): §5.1 D values
(verify_theory_propositions.json), §4.2 LOO confirmatory + per-family +
E5 recompute (gap +0.146 = 0.8657−0.7200 ✓, 5/5 ✓, one-sided p=0.031 ✓),
GearTrain, split-half 2v2, aligned Gate-4 upper bounds (+0.78/+1.92),
4-arm + CIs, sweep re-run, long-horizon, closed-loop regrets recomputed
from exp_rq1_3_closed_loop_regret.json (local 0.332/0.0/0.005 vs
surrogate 0.477/11.099/0.031 ✓), Hartmann RF probe (0.47 vs 0.68,
CI [−0.038, 0.415], p=0.141 ✓), Gate 5 Spearman per-bucket = 1.0 (x17).

## 13. Supplementary experiment round (2026-09-28, seeds 100–119)

Three additions closing the two cheapest outline gaps + one bug found:

1. **Independent-seed-stream gate replication** (limitation 5).
   `generate_phase8_6d_20seed_replication.py` → 654 records (frozen config,
   seeds 100–119 only) → `validate_phase8_6b.py` →
   `exp_rq1_2_h3_20seed_replication_s100_119.json`:
   - Gate 3: **1.296 [0.881, 1.867] PASS** vs original 1.294 [0.848, 1.954]
     — essentially exact.
   - Gate 2: FAIL on both streams (consistent).
   - Gate 4 stop-gain: **5/5 on both streams** (identical family sets).
   - Gate 4 uniform-gain: 4/5 on both streams but DIFFERENT family sets
     (rastrigan↔ackley swap) — count replicates, family identity does not.
   - Gate 4 behavior-gain: 1/5 (rastrigin) → 3/5 (disjoint set; rastrigin
     reversed) — family-level claim NOT replicated; report as unstable.

2. **Gate-4 multiplicity sensitivity** (limitation 12).
   `exp_gate4_multiplicity_sensitivity.py` recomputes per-family
   seed-clustered bootstrap CIs at Bonferroni 99% (α/5) from the stored
   per-seed gains: **stop survives 5/5; uniform 4/5 → 2/5** (rastrigin,
   sixhump hold; branin lo −0.003 just misses; **hartmann's 95% lower
   bound is itself bootstrap-fragile at ±0.05 around 0** — official +0.022
   vs recompute −0.044); **behavior 1/5 → 0/5**.

3. **RLCD LOO transfer replication on the independent stream** (§4.2
   seed-reuse half of the analysis-rerun caveat).
   `exp_rlcd_loo_confirmatory.py --dataset …_s100_119.jsonl` (train AND
   eval both on the new stream):
   - bt − RF pooled **+0.188 [+0.175, +0.202]** vs original +0.140
     [+0.126, +0.154]; proper − RF +0.053 [+0.041, +0.064] vs +0.019.
   - **5/5 family CIs exclude 0 on both streams**, same per-family strength
     ordering (ackley/sixhump/rastrigin strong, branin/hartmann small).
   - Family-level locking remains open: the backend allowlist holds exactly
     the 5 design-touched families + GearTrain (consumed by the probe) — a
     locked family eval requires allowlisting new BOCoDe problems (frozen
     decision, needs its own review).

**Bug found & fixed during (3)**: `exp_rlcd_loo_confirmatory.py` split its
early-stopping pool by the literal `seed < 16`; on the 100–119 stream that
made the TRAIN SLICE EMPTY (first replication attempt silently trained
nothing — both arms evaluated as zero-shot, pooled diff −0.249 identical
for both arms). Fixed to split by seed RANK (16 smallest train / rest val),
which is identity on the original 0–19 data and correct for any seed
offset. The void artifact was overwritten by the fixed re-run.

## 14. State/action design survey + codex adjudication (2026-09-28)

Literature spectrum survey (Sara/lenz full-text fetched; LLAMBO / GPTOpt /
Volpp / TempoRL / FunBO search-verified) in
`docs/ref_papers/state_action_design_survey.md` — the project sits at
"most-compressed state + mid-level strategy actions + unique per-decision
counterfactual supervision"; the closed action space is the expressiveness-for-
measurability trade that makes gates/cost-frontier possible.

Codex adjudication (`runs/codex_design_review.log`):
**VERDICT: KEEP_FROZEN_QUALIFY.** (a)+(e) now: freeze the design, qualify
the wording; (b) sidecar / (d) reconfigure-actions are journal-version
options; (c) trajectory-level state not recommended. No lossless
state/action change exists. Six claim-wording fixes applied to the outline
(P2 scope, P3 conditionality, schedule sufficiency — title now "…Whose
Value a Two-Phase Schedule Subsumes", action-space closure scoped, Gate-2
granularity, Gate-4 "not established" not "unattainable").

## 15. Design A/B executed: rich state shows no decision value (2026-09-28)

The KEEP_FROZEN_QUALIFY adjudication (§14) was theory-only at the time.
Branch `exp/lenz-state-jev-presentation` (commits 06deec4, bf7eccf)
implemented the lenz-style rich state (full trial log D_t + config c_t +
decision history H_t, with the frozen 15-scalar summary carried as a
projection) with Jev-style narrative + executable-criteria presentation,
then ran BOTH follow-up experiments on the untouched s100–119 stream
(`runs/EXP_LENZ_JEV_REPORT.md`):

- **E1 learnability probe** (`runs/exp_lenz_gate2_probe.json`): per-family
  LOSO RF, frozen15 vs frozen+trajectory-features vs traj-only — trajectory
  features do NOT beat frozen15 on 5/5 families (traj-only clearly worse
  where signal exists: branin 0.50→0.24, sixhump 0.66→0.57).
- **E2 LLM A/B** (`runs/exp_lenz_vs_frozen_llm.json`): deepseek-chat at
  temperature 0, same frozen 8-action space, 30 stratified records — rich
  state changes 6/30 decisions — 4 of 6 global_exploration →
  surrogate_acquisition, one local_exploitation → surrogate_acquisition,
  one global_exploration → local_exploitation; concentrated in mid/late
  phases — but gold-argmax agreement is
  IDENTICAL (0.133 both) at 2.1× token cost (766 → 1,599 per call).

Reading: the compression is not the bottleneck on this panel — the thin
signal lives in the problems (P2 ceiling), not in the 15-scalar state.
Empirical closure for §14; cited from outline limitation 3. Caveats
(handcrafted features only, n=30 single-model A/B, H_t truncated to
last_action in dataset records) in the report. Scripts are
dataset-agnostic — rerun first on any journal-version panel.

## 16. Theory grounded in classical lineage + investigation rounds indexed (2026-09-28)

- `docs/THEORY.md` new section "Relation to classical theory": P2 = the
  utility-unit Value of Information (Howard 1966 / Lindley 1956 lineage;
  ES/PES/MES are the nats version); 15-scalar state = projection of the GP
  posterior sufficient statistic (E1 negative consistent); utility+closed
  action set =
  acquisition-function structure lifted to the strategy level; P3 = the
  degenerate case the GP-Hedge→LMABO portfolio line assumes away; L1 =
  standard tree representational behavior.
- outline §2: classical-chain paragraph added (posterior → EI/UCB →
  GP-Hedge → LMABO; P2/P3 positioning).
- `docs/ref_papers/exploration_rounds.md`: six investigation rounds
  indexed with prioritized future directions (P0 GP-Hedge baseline,
  LMABO citation/scoping, codex re-review; P1 locked-eval/external
  baselines; P2 journal-version panel swaps with branch A/B scripts as
  first diagnostic).

## 17. Paper-package codex review loop (2026-09-28): REVISE → REVISE → APPROVE

Three rounds on the post-integration package (outline + THEORY classical
section + exploration rounds):
- Round 5 (`runs/codex_paper_review_round5.log`): REVISE — 10 MAJOR +
  5 MINOR. Substantive catches: proper-arm per-family direction obscured
  by "5/5" (branin/hartmann NEGATIVE both streams); A/B transition counts
  4/6 not 5/6; +0.052 rounding; P2≠entropy unit conversion (ES/PES
  location vs MES value); GP-Hedge reward is posterior-mean
  full-information, not realized gains; L1 restated as representability;
  abstract 496→≤200 words; 4.x renumbering; LMABO scoping ("reports but
  does not optimize a quality-cost frontier"); GP-Hedge baseline
  re-scoped (spec-first Exp3-style alternative, not a direct port).
- Round 6 (`runs/codex_paper_rereview6.log`): 12 ADDRESSED / 3 PARTIAL
  (abstract 210 words; two residual unit-conversion/GP-Hedge lines).
- Round 7 (`runs/codex_paper_rereview7.log`): all three ADDRESSED —
  **VERDICT: APPROVE**.

Commits: 555981a (round-5 revisions), 69764be (round-6 residuals), plus
the lenz-branch report fix (A/B counts).

## 17. Codex paper-package review loop rounds 5-7 (2026-09-28): REVISE → REVISE → APPROVE

Round 5 (`runs/codex_paper_review_round5.log`): 10 MAJOR + 5 MINOR across
internal consistency / new-number spot checks / classical-theory accuracy /
positioning fairness / priorities sanity. Substantive catches: P2-vs-entropy
is a decision-theoretic counterpart NOT a unit conversion (ES/PES target the
optimum's location, MES its value); GP-Hedge's reward is the updated
posterior mean at each nominated arm, not realized terminal gain; L1 is
representability, exact imitation stays empirical; LMABO reports but does
not optimize a quality-cost frontier; proper-score arm loses to RF on
branin/hartmann in both streams (directions now stated); A/B transitions
4/6 not 5/6; +0.052 not +0.053; §4 renumbered 4.1/4.2; test history
385→421→423 complete. Fixes: commits 555981a, 69764be, 9d47f61 (also
brought the lenz A/B artifacts onto the mainline so citations resolve).
Round 6 (`codex_paper_rereview6.log`): 12/15 ADDRESSED, 3 PARTIALLY
(abstract word count; a leftover unit-conversion phrase; deferred bandit
spec). Round 7 (`codex_paper_rereview7.log`): all three ADDRESSED —
**VERDICT: APPROVE**. Current outline/THEORY/survey package is the
reviewed submission baseline.

## 18. Research boundary guide: drafted + codex round-1 + initial fixes (2026-09-29)

New normative document `docs/RESEARCH_BOUNDARY_GUIDE.md` (research
boundary & modification spec: abstract architecture, A/B/C component
taxonomy, two-level decision structure, interface contracts, modification
procedures, anchors, glossary). Codex round-1 review
(`runs/codex_boundary_guide_review.log`, gpt-6-sol, 105k tokens):
**VERDICT: REVISE** — 5 BLOCKER / 13 MAJOR / 1 MINOR.

Error-class fixes applied same session (guide + THEORY.md):

- Fitting contract contradiction (BLOCKER ×2): "per-round MLE non-optional"
  removed; described the actual lazy-fit lifecycle (evaluation invalidates,
  GP-dependent actions fit on demand, RETRAIN explicit). Bull 2011
  re-characterized correctly (sequential-MLE non-convergence counterexamples
  + alternative estimators; NOT a per-round-refit mandate).
- C-class reframed from "no statable theorem" to a project protocol
  (protecting attribution/comparability/theorem premises); PFN note: a
  calibrated posterior does not automatically inherit GP bounds.
- §3.4 gate-2 algebra fixed: an equivariant difference alone leaves a
  global α on the term — invariance requires dividing by an α-equivariant
  scale; THEORY.md P1 scope (ii) corrected the same way.
- THEORY.md §0 corrected: U has SIX terms, `w_stop_opportunity=0.75`
  default (STOP term ACTIVE — the old "w_stop = 0" claim was wrong vs
  `rollout/utility.py`); gold_probs noted as computed over the possibly
  quota-sampled candidate set. P1's proof itself is unaffected (STOP term
  depends only on budget ratio).
- Restart ownership: removed from the inner action submenu (meta-level
  only); A_dec/A_str renamed A_pt (point-selection) / A_flow
  (process-control) to stop colliding with layer names.
- §5.4 success criteria re-stated per arm/seed-stream (BT > RF both
  streams; proper ≈ RF original / +0.053 replication; closed-loop
  collapse), Gate 5 as necessary-condition (median ρ=1, joint gate FAILED
  2/17), Gate 4 stop-gain estimand clarified; rich-state A/B updated to
  the completed negative result.
- Adaptedness decoupled from determinism/temperature-0; regret glossary
  sign fixed (repo internally maximizes — convention now explicit);
  warm-start "theorem applies as usual" caveated (adaptive selection,
  budget accounting, uniformity over initial histories); meta level
  qualified as a non-stationary budgeted bandit; P2/P3 positioned as
  screening tests; multi-objective "two plugins" scoped to the abstract
  architecture with the real implementation surface listed.
- NEW §6.6: supervision-defining components (S-class: utility
  weights/scales, continuation policy, horizon, gold temperature, quota,
  replicate count) — minimal interim procedure (utility_version bump +
  same-data A/B + P1 recheck + flip-rate reporting); NEW §10: scheme-class
  review leftovers (D3 failure criteria, E1 supervision-construction
  status, E2 meta-level pre-screen, E3 open-problem ordering, D2
  enforcement machinery) flagged for the user's manual review.
- §7 rebuilt as a claim→source map with per-item local-PDF status
  (verified against `docs/ref_papers/pdfs/`); non-load-bearing anchors
  dropped; Bull 2011 + GP-UCB rows added; LMABO 7.42 min scoped to the
  Gemini arm's 50-iteration average.

Not done (deliberately, pending manual scheme review): §10 items.

## 19. Round-2: scheme-class items landed on exp/boundary-scheme-round2 (2026-09-29)

Branch `exp/boundary-scheme-round2` (off the round-1 commit). The five
§10 scheme-class leftovers from the boundary-guide review are now
executed, not just planned:

1. **Kill criteria (D3)** — guide §5.5: four predeclared termination
   conditions (teacher reliability after label-unit redesign: flip must
   drop below the current 0.37; closed-loop increment vs best
   constant/two-phase schedule CI-positive on ≥3 families or the route
   stops at decision-level; per-label rollout cost >10× closed-loop eval
   → redesign the label unit; meta-axis screen NEGATIVE → no pipeline on
   that axis).
2. **Supervision-construction status (E1)** — guide §0.3 + §5.2: the
   myopic-label limits (flip 0.37 @ R=5; aligned Gate-4 first-action
   terminal advantage ≈0 with wide Ackley/Hartmann CIs; 8.6F
   non-additivity; closed-loop collapse) now carry equal weight with the
   contribution claim; schedule/continuation-level labeling + a matched
   closed-loop test is the declared next decision point BEFORE scaling
   first-action labels.
3. **Meta-level pre-screen (E2) — EXECUTED, INCONCLUSIVE (rigor-corrected
   same session)** — `scripts/exp_meta_config_screen.py` +
   `runs/EXP_META_CONFIG_SCREEN.md` (+ .json with all 200 per-trajectory
   records). Config axis n_init ∈ {5, 12}, budget 30 both arms, pinned
   AlwaysSurrogate, paired nested Sobol streams, 5 families × 20 seeds ×
   2 configs. Estimands + mechanistic hypotheses + decision rules all
   predeclared in the script. PRIMARY test = homogeneity Wald χ² on
   per-family advantages: χ²₄ = 8.71, p = 0.069 (not detected; I² = 0.54).
   Rastrigin individually confirmed (Δ = +8.36, d_z = 0.52, paired t
   p = 0.032, stable across discovery/confirmation seed halves);
   predeclared directional mechanistic hypotheses get 5/5 sign agreement
   (exact one-sided p = 0.031); the null is imprecise everywhere (no
   family reaches the 0.25·sd precision floor) → axis PARKED (no
   pipeline, not killed). [SUPERSEDED same day: the Rastrigin
   "confirmed family" reading in this entry was retracted by the grid's
   cross-block replication failure (see §20); the verdict was also
   corrected POSITIVE -> INCONCLUSIVE (method error, see the correction
   note in the screen report §7).] CORRECTION: the first-pass read (POSITIVE via
   meta-D bootstrap CI excluding zero) tested a nonnegative statistic on
   the boundary and was wrong (codex round-2 D1); meta-D is demoted to a
   descriptor and the report rewritten paper-style with estimands,
   dimension context (n_init/(2d+1): both arms starved at d=10),
   MDD₈₀ power readouts, LOFO learnability probe (3/5, no resolution at
   F=5), and limitations. Worker-side torch.set_num_threads(1) guard
   (85 s/traj → ~1 s/traj oversubscription fix, longhorizon precedent).
4. **Open-problem reordering (E3)** — guide §8: locked transfer
   evaluation (the only twice-replicated positive) first,
   schedule/continuation labeling second, meta axes third (screen already
   run), state enrichment later.
5. **Enforcement machinery (D2, minimal)** — new
   `src/opt_agent/actions/boundary_registry.py` (ActionType →
   boundary-class/submenu/note) + `tests/test_boundary_registry.py`
   enforcing registry ↔ menu ↔ guide §2.4 table three-way sync (guide
   table parsed, so a doc-only edit cannot desync either); new
   `src/opt_agent/dataset/versioning.py` (HARNESS_VERSION "h1" /
   UTILITY_VERSION "phase8.6b", stamp_metadata, check_record_versions
   with ok/legacy/mismatch semantics) wired into the generator (replaces
   the hardcoded utility_version) + `tests/test_versioning.py` (12 new
   tests total, all green).

Suite status at commit time: full `uv run pytest tests/` green —
**443 passed** in 21:32 (includes the RLCD GPU tests; 431 pre-existing +
12 new). Remaining open from round-1: §6.6 full
procedure formalization + consumer-side hard refusal policy for
legacy/mismatch stamps.

## 20. Config-axis dose-response grid: conditionality DETECTED; screen's Rastrigin claim retracted (2026-09-29)

Executed the screen's predeclared escalation paths (more levels + more
seeds): `scripts/exp_meta_config_grid.py`, artifact
`runs/exp_meta_config_grid.json` (5 families x n_init {5,8,12,18} x 60
seeds = 1200 trajectories, 107 s) + `runs/exp_meta_config_grid_
rastrigin240.json` (Rastrigin-only deep arm, 4 x 240, 33 s). Estimands
predeclared in-script: per-level contrasts vs reference L0=5; per-family
OLS dose slope in c = n_init/(2d+1) space over seed-paired levels;
PRIMARY test = slope homogeneity (Wald chi^2, Cochran's-Q form).

- PRIMARY (as reported at the time, raw-unit slopes; superseded by
  the standardized estimand, see section 22/23): chi^2_4 = 30.87,
  p < 1e-4, I^2 = 87% -> CONDITIONALITY_
  DETECTED. The family x dose interaction is decisively present.
- Structure: smooth families individually significant NEGATIVE slopes
  (Hartmann -0.84 sd/unit c, p<1e-4; Branin -1.56, p=0.002; SixHumpCamel
  -0.30, p=0.0007; monotone level means); Ackley (multimodal, d=10)
  significantly POSITIVE (+0.45, p=0.029, best level 18); mechanistic
  sign agreement 5/5 (exact p=0.031).
- RETRACTION: the screen's Rastrigin 12-vs-5 finding (Delta=+8.36,
  p=0.032, "confirmed" by within-0-19 halves) does not replicate: seeds
  0-19 reproduce bit-exact, seeds 20-59 mean -3.30; at n=60 Delta=0.58
  (p=0.81); deep arm n=240: L=18-vs-5 +2.20 [-0.47,4.82] d_z=0.11,
  slope one-sided p=0.049. Lesson recorded: within-block half-split
  confirmation is insufficient for very-high-variance families; require
  replication across disjoint seed blocks.
- Method note: on the SAME 60 seeds, the two-level contrast homogeneity
  stays marginal (Delta_12: p=0.059; fresh-only p=0.114) while the
  4-level slope homogeneity is decisive — the screen's INCONCLUSIVE is
  attributed to single-contrast power, vindicating the predeclared
  escalation design.
- License level (guide 5.2/8.3): signature features + held-out
  prediction is the next justified investment; the configuration-level
  supervision pipeline remains NOT started (oracle value != signature
  learnability; featureless LOFO 3/5 has no resolution at F=5).
- Script fixes en route: LOFO guard for single-family runs; single-family
  verdict labeled SINGLE_FAMILY_ARM instead of misapplying the
  cross-family rule.

Report: `runs/EXP_META_CONFIG_GRID.md` (paper-style, formal register).
Guide synced (5.2 two-stage reading, 5.5.4 not triggered, 8.3 next step
= signature prediction, 10.3 updated).

## 21. Round-3 closure: codex round-2 leftovers landed (2026-09-29)

All remaining review-class items from codex round-2 (14 MAJOR / 4 MINOR)
addressed on exp/boundary-scheme-round2:

- C3 (version enforcement at the loading boundary): versioning.py gains
  VersionPolicy (allow/warn/strict) + enforce_record_versions (legacy
  summarizes once per source with a do-not-pool reminder; unknown stamps
  raise under strict); decision_record.read_jsonl(path,
  version_policy=...) materializes + enforces; exported from the package;
  train_rlcd_laya.py reads via read_jsonl with --version-policy (default
  warn). Tests: warn/strict/legacy/mismatch + read_jsonl roundtrip.
- C1 (parser hardening): guide 2.4 table gained a submenu column (and
  proper per-row notes instead of ditto marks); the sync-test parser is
  a pure function with strict schema checks (header ActionType/归类/
  子菜单/备注; malformed short rows, duplicate rows, unknown action
  rows all raise) and now compares class AND submenu columns; unit
  tests feed synthetic broken tables and assert fail-closed behavior.
- C2 (registry scope): boundary_registry docstring states its scope
  honestly (CI-consistency artifact + query API; runtime menu source is
  the factory); added boundary_class_of/submenu_of accessors; new
  test_factory_candidates_match_registry enforces two-way coverage
  between ALL_CANDIDATE_SPECS and the registry (the specs contain all 8
  actions incl. stop, so the invariant is exact equality).
- Doc/judgment fixes: 5.5.1 flip estimator pinned (2v2 split-half, R=5,
  current 0.359 per exp_teacher_noise_diagnostic_r5.json); 5.5.2
  rewritten as paired success + futility rules (fixed 5-family set,
  declared seed counts, MDD_80 power, 0.25 sd effect floor); 5.5.3 cost
  denominator defined (per-label rollout vs one budget-30 trajectory,
  same hardware); 6.6 full procedure formalized (items 4-6:
  reliability / invariance / outcome-alignment checks) with automation
  status noted; 0.2 + 2.2-B1 + B3 mark periodic refit as proposed-not-
  implemented; 0.4 + 6.1 state the exact verification scope of the sync
  tests; 10.5 updated with round-3 dispositions.
- THEORY.md P3 empirical reading corrected: aggregate nonsignificance is
  CONSISTENT WITH but does not establish the pointwise A(s) <= 0 premise;
  the premise is explicitly unmeasured (codex round-1 C6 / round-2 A2).

New/changed tests: 23 in test_boundary_registry + test_versioning
(was 12). Suite at commit: 439 (excl. test_rlcd) + 15 (test_rlcd) = 454 passed; was 443 before round-3 (12 new tests: parser hardening, enforcement, factory link).

## 22. Round-3 review verdict + full disposition (2026-09-29)

Codex round-3 (`runs/codex_boundary_round3_review.log`, 103k tokens):
VERDICT: REVISE — 0 BLOCKER / 9 MAJOR / 4 MINOR (trajectory: 7 BLOCKER
-> 0/14 -> 0/9). Round-2 items: 12 ADDRESSED / 7 PARTIAL / 0 UNADDRESSED.
All 13 new findings fixed in the same session:

- D1 (real statistical catch): the grid PRIMARY test ran on raw-unit
  slopes — not scale-stable. Primary re-declared as reference-
  standardized slopes (b_f/sd_ref, sd_ref treated as fixed): chi^2_4
  = 38.69, p < 1e-4, I^2 = 90% [superseded by the joint-bootstrap SE
  version, see section 23]
  — matches codex's own recomputation; verdict unchanged
  (CONDITIONALITY_DETECTED). Raw-unit version demoted to secondary.
- E2/C1: "precise null" now an equivalence bound (CI entirely within
  +/- 0.25 sd) in the grid script and guide 5.5.4/5.5.2; success rule
  requires CI lower bound > 0 on >= 3 families.
- B1/B2/E1: iter_jsonl documented as raw unenforced access (replay
  only); pooling consumers must use read_jsonl with an explicit
  version_policy; train_system1_compare.py migrated; mixed legacy+
  current pooling warns (warn) / raises (strict) unless pool_legacy=True
  records the explicit opt-in; guide 4.5 aligned with enforcement.
- A1/C2: guide 6.5 and THEORY P3 no longer call the aggregate sweep a
  "signature" of pointwise A<=0; schedule control is "consistent with";
  the untested-router claim is explicitly conditional.
- D2/D3: screen-report Rastrigin statements marked historical +
  retracted; grid 8.2 relabeled candidate baseline pending held-out;
  PROGRESS 19 carries the supersession note.
- C3: 5.5.1 records estimator details (replicas 0-3; replica 5 unused;
  pair-sampling rule re-declared after redesign).
- B3: parser requires exactly 4 cells, exactly one table, and the
  guide's presence (skipifs removed).

Tests: 55 across affected modules (4 new). Grid + deep arm re-run under
the corrected estimand (deterministic; artifacts refreshed).
## 23. Round-4 review verdict + full disposition (2026-09-29)

Codex round-4 (`runs/codex_boundary_round4_review.log`): VERDICT REVISE —
0 BLOCKER / 5 MAJOR / 4 MINOR (14 -> 9 -> 5 MAJOR). All fixed same
session:

- B1 (statistical): the standardized-slope SE treated the estimated
  sd_ref as fixed. Fixed with JOINT SEED RESAMPLING: every bootstrap
  replicate re-estimates sd_ref from the resampled reference outcomes;
  the primary homogeneity now uses these SEs — chi^2_4 = 20.06,
  p = 5e-4, I^2 = 80% (matches codex's jackknife Q~20.8 / p~3.5e-4;
  verdict unchanged). Both grid arms re-run.
- B2: train_system1_compare.py now passes version_policy="warn"
  explicitly; phase9_baselines.py migrated from iter_jsonl to
  read_jsonl(version_policy="warn").
- B3: pooling check compares RAW stamp values (two different known-past
  stamps, and absent-vs-present stamps, both count as a mix);
  pool_legacy=True now REQUIRES a pooling_note and records the decision
  in an audit trail (pooling_decisions()); the one-source boundary
  limitation (cross-file combination) documented in guide 4.5.
- B4/E2: futility now requires the equivalence bound on EVERY
  level-vs-reference contrast (not only the selected best); guide 5.5.4
  and the script's declared + serialized rules synchronized to the
  equivalence wording.
- B5: report/guide/PROGRESS all carry the corrected primary numbers
  (20.06 / 5e-4 / 80%) with the raw-unit version labeled secondary and
  superseded notes in PROGRESS 20.
- Minors: guide 5.4 closed-loop success = CI lower bound > 0; screen
  6.3 Rastrigin statement marked retracted; grid 8.2 relabeled
  candidate baseline pending held-out; parser header requires exactly
  four cells and detects a second table in the section.

Tests: 24 in registry/versioning (pooling-note path added).
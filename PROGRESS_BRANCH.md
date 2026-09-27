# Opt Agent v3 — Branch Snapshot: RLCD fine-tuning of Laya

**Date**: 2026-09-27
**Branch**: `exp/rlcd-laya-finetune` (off `main` @ 8c70b79)
**Purpose**: Separately evaluate the effect of fine-tuning Laya as the
System-1 decision model (replacing/comparing the RandomForest "semantic
routing" baseline), with the theoretical support for RLCD as the first
deliverable.

> The previous content of this file (the merged `exp/rq0.5-llm-system1`
> snapshot, 2026-09-26) lives on in `main`'s history; this branch reuses
> the filename per repo convention.

---

## 1. Why RLCD

RLCD (Yang et al., ICLR 2024, arXiv:2307.12950) trains a preference model
on **contrastive pairs** with a Bradley-Terry objective and then aligns
the policy with RL — replacing human preference labels. In this project
the contrastive pairs come from the counterfactual rollout teacher:
`a⁺ ≻ a⁻ ⟺ U(a⁺|s) > U(a⁻|s)` — a strictly stronger label source than
RLCD's prompt-induced pairs (a quantitative utility's sign, exactly the
statistic a BT likelihood models).

## 2. Theory (the requested focus): `docs/RLCD_THEORY.md`

Five propositions in the house style of `docs/THEORY.md`:

- **T1 (invariance/sufficiency)** — pairwise labels inherit P1's affine
  invariance *exactly* and cancel any per-state additive teacher shift
  `g(s)` (delimits which utility misspecifications fine-tuning is immune
  to; covers the `w_stop·r(s)` worry class).
- **T2 (noise consistency)** — rollout noise (GP refit, continuation
  coupling) shrinks the BT-optimal scores but preserves the ranking
  (`f* = h∘U*`, h strictly increasing); margin filtering (`margin=δ`)
  bounds the flip rate strictly below 1/2 ⇒ learnable (Angluin–Laird).
  Contrast: magnitude-matching (KL/regression) objectives track noisy
  magnitudes and can flip argmax in expectation.
- **T3 (quota compatibility)** — the theorem binding this branch to the
  main-branch quota fix (8c70b79): pairwise supervision is unbiased under
  k-of-m action subsampling (pair identification needs co-observation
  probability `k(k−1)/(m(m−1))` > 0, coverage → 1 as n grows), while
  renormalized subset softmax targets are biased and stay biased at fixed
  k (3-action counterexample inside). RLCD-CD is the correct estimator
  family for the long-horizon quota regime.
- **T4 (ceiling + readout)** — any U-supervised policy inherits P2's
  `D` ceiling; one-step regret ≤ range × top-1 disagreement, making the
  branch's decision-level metrics a bound on deployable value.
- **T5 (RL stage)** — against the CD model's own reward the argmax policy
  is already optimal, so stage-2 RL only pays against **closed-loop**
  return (the Phase 8.6C/F continuation coupling); designed as
  REINFORCE + KL anchor, deferred until trajectory logs exist. DPO noted
  as the RL-free alternative.

## 3. Implementation

- `src/opt_agent/system1/rlcd.py` — pair mining (`mine_pairs`; partial
  records first-class), Bradley-Terry loss, Laya sequence plumbing via the
  SDK's own `build_sequence`/`collate_items` with the **deployment prompt
  shape** (`build_choice_request`, structured/legacy = the phase9_5
  baseline shape), `train_rlcd` (early stop on val pairwise acc,
  warmup+cosine, bf16 autocast), `save_laya_checkpoint` (output loads via
  `laya.load()` / `LayaPolicy(model_id=…)`), `evaluate_ranking`.
- `scripts/train_rlcd_laya.py` — zero-shot vs RLCD vs RF comparison on
  the canonical 20-seed dataset (train seeds 0–15, val 16–19).
- `scripts/analyze_rlcd_ci.py` — record-clustered paired bootstrap CIs.
- `tests/test_rlcd.py` — 11 tests (pair mining incl. partial/tie/invalid,
  BT loss incl. gradient direction, real-tokenizer item alignment, stub
  DecisionModel backward flow).
- `_prompt.py`: `build_choice_request` now accepts a payload Mapping
  (training from records); runtime paths unchanged.

## 4. Headline result (decision-level, `runs/EXP_RLCD_REPORT.md`)

On 115 held-out records (seeds 16–19, 2889 pairs):

| Model | Pairwise acc | Top-1 agreement |
|---|---:|---:|
| Laya zero-shot | 0.396 (below chance) | 0.035 |
| **Laya + RLCD (CD)** | **0.879** | 0.383–0.400 |
| RandomForest baseline | 0.817 | 0.400 |

Paired bootstrap (record-clustered): fine-tuned − RF = **+0.062,
95% CI [+0.039, +0.087]** (excludes 0); fine-tuned − zero-shot = +0.483
[+0.447, +0.518]. Training cost: 56 s on the RTX 3090 (13,618 pairs).

**Reading.** RLCD fine-tuning turns Laya from below-chance to
significantly better than the incumbent RF on teacher-ranking recovery —
the first positive fine-tuning result for the System-1 line (Phase 9.3's
Jev-style KL model was never closed-loop-validated, and Phase 9.5's
zero-shot Laya collapsed). Not claimed: closed-loop optimization gains
(T4 caps one-step value at D; T5 says the RL stage is where closed-loop
value would come from), cross-family transfer (RQ2).

## 5. Next steps on this branch

1. Closed-loop evaluation of `runs/rlcd_laya_v1` via
   `LayaPolicy(model_id="runs/rlcd_laya_v1")` on the EXP-RQ3 paired panel
   (5 families × fresh seeds × budget 30) vs `rf`/schedule/LLM arms.
2. Train on a long-horizon partial dataset (main's
   `generate_dataset_longhorizon.py`, k=4) — T3 predicts this is where the
   BT objective's quota advantage bites against any softmax-target model.
3. Margin-filter sweep (T2: δ ∈ {0, 0.05, 0.1, 0.2}) and β sweep.
4. Stage-2 REINFORCE once per-step trajectories are logged (T5).

## 6. File index

```
docs/RLCD_THEORY.md                     # T1–T5 + evidence map (the theory deliverable)
src/opt_agent/system1/rlcd.py           # RLCD CD pipeline
scripts/train_rlcd_laya.py              # train + 3-way comparison driver
scripts/analyze_rlcd_ci.py              # paired bootstrap CIs
tests/test_rlcd.py                      # 11 unit tests
runs/rlcd_laya_v1/                      # fine-tuned checkpoint (laya-loadable)
runs/EXP_RLCD_REPORT.md                 # experiment report (+ CI section)
runs/exp_rlcd_report.json, runs/exp_rlcd_ci.json
```

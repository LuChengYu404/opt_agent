# EXP-RLCD — RLCD fine-tuning of Laya on counterfactual pairs

Date: 2026-09-27 · branch `exp/rlcd-laya-finetune`
Dataset: `runs/phase8_6d_h3_20seed_5family.jsonl` (663 records; 548 train / 115 val by problem seed)

Method: contrastive pairs (a⁺ ≻ a⁻ ⟺ U(a⁺|s) > U(a⁻|s)) from the
counterfactual rollout teacher; Bradley-Terry loss −log σ(β(z⁺−z⁻))
on Laya's per-option scorer. Theory: `docs/RLCD_THEORY.md`.

| Model | Pairwise acc | Top-1 agreement | Pairs | Records |
|---|---:|---:|---:|---:|
| Laya zero-shot | 0.396 | 0.035 | 2889 | 115 |
| Laya + RLCD (CD) | 0.879 | 0.383 | 2889 | 115 |
| RandomForest (utilities) | 0.817 | 0.400 | 2889 | 115 |

Training: 13618 pairs over 548 usable records; 56s on cuda.
Saved checkpoint: `runs/rlcd_laya_v1` (loadable via `LayaPolicy(model_id=...)`).

Caveats: decision-level metrics only (teacher-ranking recovery),
not closed-loop optimization performance; same-family split by seed
(cross-family transfer is RQ2 and not claimed here).

## Paired bootstrap CIs (clustered by record; `scripts/analyze_rlcd_ci.py`)

| Contrast | Pairwise-acc diff | 95% CI | P(diff > 0) |
|---|---:|---|---:|
| RLCD fine-tuned − zero-shot | +0.483 | [+0.447, +0.518] | 1.00 |
| RLCD fine-tuned − RandomForest | **+0.062** | **[+0.039, +0.087]** | 1.00 |
| RandomForest − zero-shot | +0.420 | [+0.385, +0.457] | 1.00 |

The fine-tuned Laya beats the current RF System-1 on teacher-ranking
recovery with a record-clustered CI excluding zero. Top-1 agreement is tied
(RF 0.400 vs fine-tuned 0.383–0.400; the spread is bf16 batching-order
numerics flipping ~2/115 near-tie records, not a stable difference).
Zero-shot Laya is *below chance* (0.396 < 0.5): its option ordering is
systematically anti-correlated with the teacher's, consistent with the
Phase 9.5 action-collapse diagnosis.

Additional caveats: decision-level metrics only (teacher-ranking
recovery), not closed-loop optimization performance; same-family split by
seed (cross-family transfer is RQ2 and not claimed here); ranking
recovery's deployable value is capped by the adaptive-value ceiling D
(RLCD_THEORY T4).

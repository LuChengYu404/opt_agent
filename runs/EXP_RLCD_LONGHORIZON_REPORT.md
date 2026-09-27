# EXP-RLCD — RLCD fine-tuning of Laya on the counterfactual teacher

Date: 2026-09-27 (objective reference corrected same day) · branch `exp/rlcd-laya-finetune`
Dataset: `runs/longhorizon_k4_20seed_5family.jsonl` (661 records; 546 train / 115 val by problem seed)

**RLCD = the Jev/Laya-lineage method** ("Reinforcement Learning for
Calibrated Decisions"): reward = strictly proper scoring rule (SDK's
`proper_reward`: log score + 0.5·spherical). No method paper exists;
the implementation is the one shipped with the checkpoint. NOT the
identically-named Yang et al. ICLR 2024 paper. Theory:
`docs/RLCD_THEORY.md`.

| Model | Pairwise acc | Top-1 agreement | Mean log score | Pairs | Records |
|---|---:|---:|---:|---:|---:|
| Laya zero-shot | 0.485 | 0.174 | -3.3691 | 618 | 115 |
| **Laya + RLCD (proper score)** | 0.576 | 0.348 | -2.0715 | 618 | 115 |
| Laya + BT pairs (baseline arm) | 0.660 | 0.487 | -2.2199 | 618 | 115 |
| RandomForest (utilities) | 0.545 | 0.383 | — | 618 | 115 |

RLCD arm: 2843 pairs / 546 records seen; 54.5s on cuda.
Saved checkpoints: `runs/rlcd_laya_proper_lh` (proper-score arm), `runs/rlcd_laya_bt_lh` (BT arm) (loadable via `LayaPolicy(model_id=...)`).

Caveats: decision-level metrics only (teacher recovery), not
closed-loop optimization performance; same-family split by seed
(cross-family transfer is RQ2 and not claimed here); value capped
by the adaptive-value ceiling D (RLCD_THEORY T4).

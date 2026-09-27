# EXP-RLCD — RLCD fine-tuning of Laya on the counterfactual teacher

Date: 2026-09-27 (objective reference corrected same day) · branch `exp/rlcd-laya-finetune`
Dataset: `runs/phase8_6d_h3_20seed_5family.jsonl` (663 records; 548 train / 115 val by problem seed)

**RLCD = the Jev/Laya-lineage method** ("Reinforcement Learning for
Calibrated Decisions"): reward = strictly proper scoring rule (SDK's
`proper_reward`: log score + 0.5·spherical). No method paper exists;
the implementation is the one shipped with the checkpoint. NOT the
identically-named Yang et al. ICLR 2024 paper. Theory:
`docs/RLCD_THEORY.md`.

| Model | Pairwise acc | Top-1 agreement | Mean log score | Pairs | Records |
|---|---:|---:|---:|---:|---:|
| Laya zero-shot | 0.396 | 0.035 | -3.3145 | 2889 | 115 |
| **Laya + RLCD (proper score)** | 0.800 | 0.252 | -2.0712 | 2889 | 115 |
| Laya + BT pairs (baseline arm) | 0.873 | 0.383 | -3.0567 | 2889 | 115 |
| RandomForest (utilities) | 0.817 | 0.400 | — | 2889 | 115 |

RLCD arm: 13618 pairs / 548 records seen; 53.9s on cuda.
Saved checkpoints: `runs/rlcd_laya_proper_v1` (proper-score arm), `runs/rlcd_laya_bt_v1` (BT arm) (loadable via `LayaPolicy(model_id=...)`).

Caveats: decision-level metrics only (teacher recovery), not
closed-loop optimization performance; same-family split by seed
(cross-family transfer is RQ2 and not claimed here); value capped
by the adaptive-value ceiling D (RLCD_THEORY T4).

## Paired bootstrap CIs — pairwise accuracy (clustered by record; `scripts/analyze_rlcd_ci.py`)

| Contrast | Diff | 95% CI | P(diff > 0) |
|---|---:|---|---:|
| RLCD proper − zero-shot | +0.403 | [+0.372, +0.435] | 1.00 |
| RLCD proper − RandomForest | −0.017 | [−0.038, +0.004] | 0.06 |
| RLCD proper − BT pairs | −0.073 | [−0.090, −0.055] | 0.00 |
| BT pairs − zero-shot | +0.475 | [+0.439, +0.512] | 1.00 |
| BT pairs − RandomForest | +0.056 | [+0.033, +0.080] | 1.00 |
| RandomForest − zero-shot | +0.420 | [+0.385, +0.457] | 1.00 |

## Interpretation — the two objectives trade ranking against calibration

Reference points for the mean log score: uniform-8 = −2.079; the teacher
itself (Σ t·log t, same floor as `proper_reward`) = **−1.977** (teacher
entropy 1.98 nats — a deliberately flat softmax-of-utilities target).

- **RLCD (proper score)** is the only arm that improves the *distribution*
  metric (−2.071, i.e. between uniform and the teacher, versus −3.32
  zero-shot). Its ranking recovery (0.800) is statistically
  indistinguishable from the RF baseline (CI crosses 0).
- **BT pairs** wins ranking outright (0.873, beats RF with CI excluding
  zero) but its log score *degrades below uniform* during training
  (−2.77 → −3.06 across epochs) while pairwise accuracy rises —
  over-sharpening past the teacher's temperature, the classic
  preference-training overconfidence pathology that the RLCD lineage's
  proper-scoring design exists to avoid (arXiv:2609.28940 Table II).
- Both fine-tunes crush the zero-shot baseline, which is *below chance*
  on ranking (0.396 < 0.5) and confidently miscalibrated (−3.32).

This is the empirical form of the corrected theory's identification
hierarchy (RLCD_THEORY R1(iii)): proper-score training identifies the
teacher distribution (calibration, Router-ready confidence), BT training
identifies only its ranking — and pays for the ranking with calibration.

Practical reading for this project: the Router (Phase 12) consumes
confidence as signal, and the EXP-RQ3 router failure traced to a
never-above-0.25 RF softmax; an RLCD-proper System-1 is the arm whose
reported distribution is meaningful, while the BT arm is the ranking
specialist. The two are combinable (rank with BT scores, calibrate with
proper scores — or a two-stage schedule), which is a branch follow-up.

Note: an earlier version of this report (commit 642617d) mis-labeled the
BT arm as "RLCD" under the Yang et al. (ICLR 2024) reading of the
acronym; corrected after verifying the term's origin in the Laya lineage.

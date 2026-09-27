# EXP-RQ2.1 — Cross-Family Transfer (Leave-One-Family-Out CV)

**Date**: 2026-09-27
**Dataset**: `runs/phase8_6d_h3_20seed_5family.jsonl` (661 records, 5 families × 20 seeds × horizon=3)
**Model**: multi-output RandomForestRegressor (one regressor per action)
**CV**: 5-fold leave-one-family-out
**Baselines**: AlwaysSurrogate, In-family RF (reference, NOT upper bound), **RulePolicy (real, state-conditional)**, AlwaysLocal (constant local_exploitation, informational)
**Metric**: held-out decision regret = U(s, a*) − U(s, a_hat) (lower is better)

**Statistical inference**: per-family paired bootstrap CIs in two flavors — (a) **record-level** (understates variance due to within-seed correlation), (b) **seed-block** (preferred per codex gpt-6-sol 2026-09-27 REVISE). Both reported below; only seed-block CIs used for headline claims.

## 1. Per-family results

| Held-out | n_test | cross-fam regret | in-fam (reference) | AlwaysSurrogate | **RulePolicy (real)** | AlwaysLocal | Cross − Sur CI (seed-block) | Cross − **Rule** CI (seed-block) | Cross − Local CI (seed-block) |
|---|---|---|---|---|---|---|---|---|---|
| branin | 145 | +0.0550 | +0.0415 | +0.0582 | **+0.0730** | +0.0488 | `[-0.0145, +0.0086]` | `[-0.0301, -0.0059]` | `[-0.0029, +0.0162]` |
| hartmann | 135 | +0.9520 | +0.5776 | +0.7674 | **+0.9249** | +0.7095 | `[+0.0414, +0.3466]` | `[-0.1155, +0.2101]` | `[+0.0925, +0.3910]` |
| ackley | 130 | +0.4674 | +0.3117 | +0.7051 | **+0.6575** | +0.3324 | `[-0.4342, -0.0762]` | `[-0.3350, -0.0770]` | `[+0.0185, +0.2487]` |
| sixhumpcamel | 123 | +0.0501 | +0.0128 | +0.0397 | **+0.0471** | +0.0176 | `[+0.0075, +0.0134]` | `[-0.0013, +0.0075]` | `[+0.0259, +0.0391]` |
| rastrigin | 128 | +0.2245 | +0.2858 | +0.3807 | **+0.3793** | +0.2639 | `[-0.2511, -0.0837]` | `[-0.2472, -0.0716]` | `[-0.1012, +0.0163]` |

**Reading the table (negative diff = cross is BETTER)**:
- Cross − Surrogate CI < 0 → cross beats AlwaysSurrogate
- Cross − **RulePolicy** CI < 0 → cross beats the **real, state-conditional** rule
- Cross − AlwaysLocal CI < 0 → cross beats the constant `local_exploitation` policy

**Important note on baselines (per codex gpt-6-sol REVISE)**:
- **RulePolicy** is the actual state-conditional `RulePolicy.select_action()` from `src/opt_agent/loop/policies.py`, **not** a constant `local_exploitation` policy. It picks STOP if remaining_budget ≤ 4, GLOBAL_EXPLORATION if stagnation_steps ≥ 3 AND budget ≥ 2, else SURROGATE_ACQUISITION.
- **AlwaysLocal** is the constant `local_exploitation` policy (informational only — not what was previously called "RulePolicy" in round-1).

## 2. Aggregate (records-weighted across folds)

- **Cross-family RF mean regret**: +0.3512
- **In-family RF mean regret (reference)**: +0.2461
- **AlwaysSurrogate mean regret**: +0.3893
- **RulePolicy (real, state-conditional) mean regret**: +0.4164
- **AlwaysLocal mean regret**: +0.2753
- **Cross − Surrogate (records-weighted mean)**: -0.0381
- **Cross − RulePolicy (records-weighted mean)**: -0.0652
- **Cross − AlwaysLocal (records-weighted mean)**: +0.0759
- **Cross − In-family (records-weighted mean)**: +0.1051

## 3. Interpretation

**Cross-family transfer test (RQ2)**: if the cross-family regret is lower than a baseline's on a held-out family, the System-1 learned on 4 families transfers usefully to it.

**In-family reference (NOT upper bound)**: the in-family RF (trained on N-1 seeds of the held-out family) is a reference, not an upper bound. On Rastrigin, the cross-family RF has lower regret than the in-family RF (Cross − In-family diff < 0). The two fits also have different amounts of training data (4 families vs N-1 seeds).

**Headline findings (per codex gpt-6-sol 2026-09-27 REVISE, exploratory)**:

- **vs AlwaysSurrogate** (seed-block 95% CI):
  - **2/5 families**: cross-family has lower regret
  - **2/5 families**: cross-family has higher regret
  - **1/5 families**: inconclusive

- **vs RulePolicy (real, state-conditional)**:
  - **3/5 families**: cross-family has lower regret
  - **0/5 families**: cross-family has higher regret
  - **2/5 families**: inconclusive

- **vs AlwaysLocal (constant local_exploitation)** — informational:
  - **0/5 families**: cross-family has lower regret
  - **3/5 families**: cross-family has higher regret
  - **2/5 families**: inconclusive

**Pattern**: cross-family transfer is heterogeneous across the five tested family–dimension combinations, but neither *family-specific* nor *dimension-specific* in a clean sense — family and dimension are confounded, and dimension is itself an RF feature.

**Hypothesis (descriptive, untested)**: cross-family training helps when the held-out family shares structural features with the training set (e.g., 10D multimodal structure common to Ackley + Rastrigin in the training set). Hartmann's decision-rich 6D landscape may have family-specific structure that the cross model does not capture; SixHumpCamel is too flat for the cross model to add value over AlwaysSurrogate.

## 4. Limitations

- **Single model class (RF)** — we don't claim this generalizes to MLPs, Transformers, or other System-1 architectures.
- **Single dataset (20-seed 5-family BOCoDe)** — engineering families not included.
- **Family × dimension confounded**: Branin/SixHumpCamel are 2D, Hartmann is 6D, Ackley/Rastrigin are 10D. The cross-family signal may be partially a dimension-transfer signal. A crossed design (same family at multiple dimensions) is required to disentangle.
- **Trained RF vs baselines is heterogeneous** — wins vs RulePolicy on 3/5, wins vs AlwaysSurrogate on 2/5, loses vs AlwaysLocal on aggregate (AlwaysLocal is an unrealistic constant `local_exploitation` policy that ignores budget). The headline number depends on which baseline you pick.
- **One-step decision regret, not closed-loop trajectory** — the RF is evaluated on its one-step decision regret against the oracle. We do not measure the effect of deploying the RF through a full BO trajectory.
- **5 per-family 95% CIs without multiplicity adjustment** — some borderline-pass CIs may not survive Bonferroni or Holm correction. Per codex gpt-6-sol recommendation, the 95% intervals are exploratory.
- **Aggregate records-weighted mean is arithmetic, not inferential** — it hides per-family heterogeneity.

## 5. Files

- `runs/exp_rq2_1_report.json` — raw per-fold results (record-level + seed-block CIs)
- `runs/EXP_RQ2_1_REPORT.md` — this report
- Source data: `runs/phase8_6d_h3_20seed_5family.jsonl` (20-seed 5-family)
- Shared helpers: `scripts/_shared_rq_features.py` (also used by `exp_rq1_3_rf_probe.py`)

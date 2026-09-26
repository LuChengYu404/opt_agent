# EXP-RQ1.3 — Decision Landscape Analysis (post-codex-final, pilot evidence base)

**Date**: 2026-09-26
**Branch**: `exp/rq0.5-llm-system1` (off `main` @ eb7e113)
**Status**: **Pilot evidence base closed**; ready to merge to main as canonical RQ1 evidence
**Verdict from `gpt-5.6-sol` round-3 review**: REVISE — after corrections, approve as pilot evidence base; stronger LLM/System-2 claims require replication.

## 1. The full chain (Step 1 → Step 2 → Step 3 → Step 4)

```
Step 1: DECISION OPPORTUNITY        — closed (scale-robust under family_median)
Step 2: STATE CONDITIONALITY        — closed (RF learns on Hartmann, suggestive pilot)
Step 3: CLOSED-LOOP REGRET          — closed (mechanistic plausibility for EXP-RQ0.5)
Step 4: LLM VALUE                   — closed (suggestive pilot at 2 seeds; replication required)
```

Every step is a gate. A step can pass on one family and fail on another.

## 2. Step 1 — Decision Opportunity (the landscape)

### 2.1 Scale-robust ΔU means (5 schemes × 5 families)

| Family | per_record | **family_median** | family_mean | robust_iqr | total_budget |
|---|---|---|---|---|---|
| ackley | 0.383 | **0.341** | 0.348 | 1.124 | 0.016 |
| branin | 0.033 | **0.035** | 0.030 | 0.054 | 0.043 |
| **hartmann** | 1.379 | **0.570** | 0.620 | 0.714 | 0.015 |
| rastrigin | 0.388 | **0.342** | 0.314 | 0.576 | 0.243 |
| sixhumpcamel | 0.023 | **0.022** | 0.022 | 0.037 | 0.021 |

**Caveat (per codex)**: the three-regime split is conditional on the **family_median normalization**. Under `robust_iqr`, SixHumpCamel has ΔU prevalence 13.3% (vs 0% under family_median) and Branin 13.8% (vs 6.25%). Under `total_budget`, Hartmann and Ackley drop to 0%. **The taxonomy is robust for the qualitative ordering Hartmann > Rastrigin ≈ Ackley > Branin > SixHumpCamel, but specific δ-prevalence numbers are not stable across schemes.**

### 2.2 Decision Opportunity profile (codex-suggested metric)

The cleanest metric for "how much strategy value does this benchmark support" is:

  **adaptive-value(s) = U(s, a*) − U(s, a_best-fixed)**

  where a_best-fixed = `argmax_a E[U(s, a)]` (the best fixed action across the benchmark).

Computed as `E[max_a U(s, a)] − max_a E[U(s, a)]`:

| Family | E[max_a U] | max_a E[U] | **adaptive-value** |
|---|---|---|---|
| **hartmann** | 1.033 | 0.350 | **0.684** |
| rastrigin | 0.550 | 0.182 | 0.368 |
| ackley | 0.619 | 0.309 | 0.309 |
| branin | -0.008 | -0.056 | 0.048 |
| sixhumpcamel | -0.063 | -0.078 | **0.015** |

This is a **better taxonomy metric than prevalence-above-δ**: it directly measures "how much state-adaptive strategy value does the benchmark support". The ordering matches the qualitative regime split, with a sharper separation between Hartmann (0.68) and the rest.

### 2.3 Practical-δ = 0.1 prevalence under family_median

| Family | ΔU prev(>δ) | R_sur prev(>δ) | max ΔU | n_records (non-terminal) |
|---|---|---|---|---|
| sixhumpcamel | 0.00% | 0.00% | **0.0549** | **60** |
| branin | 6.25% | 12.50% | 0.2950 | 80 |
| ackley | 40.00% | 48.57% | 3.7465 | 70 |
| rastrigin | 42.47% | 45.21% | 2.6300 | 73 |
| **hartmann** | 60.87% | 49.28% | 4.2763 | 69 |

(Factual correction from earlier rounds: SixHumpCamel has **60 records**, max ΔU = **0.0549**. The 354-record canonical array has **11 fields** per record.)

### 2.4 Three regimes — defensible as a **provisional, family-median-specific** taxonomy

- **Flat (Branin, SixHumpCamel)**: adaptive-value < 0.05; SixHumpCamel has 0% above δ
- **Heavy-tailed (Rastrigin, Ackley)**: adaptive-value 0.3-0.4; ~40% above δ; max ΔU > 2
- **Frequent opportunity (Hartmann)**: adaptive-value 0.68; 61% above δ

## 3. Step 2 — State Conditionality (RandomForest utility regression)

### 3.1 Setup
- Multi-output RandomForestRegressor predicting 8 utilities per record
- Leave-one-seed-out CV (10 folds per family)
- Features: 12 numeric + 3 phase one-hot
- Target: 8 utilities under family_median scale

### 3.2 Held-out decision regret (LOSO seed-equal mean)

| Family | Surrogate | Majority | **RF** | RF vs Surrogate | RF vs Majority |
|---|---|---|---|---|---|
| **Hartmann** | 0.670 | 0.785 | **0.473** | **−29.4%** | **−39.7%** |
| Rastrigin | 0.446 | 0.426 | 0.371 | −16.8% | −12.9% |
| Ackley | 0.560 | 0.282 | 0.315 | −43.8% | +11.7% |

**Interpretation (per codex, revised)**:
- **Hartmann**: RF improves on Surrogate by 29.4%. Wins on 8/10 seeds. **Paired bootstrap CI for Surrogate−RF is [−0.038, 0.415]; two-sided paired t p=0.141.** Publishable as suggestive pilot evidence, not "confirmed."
- **Rastrigin**: RF's gain over fold-specific constant is 0.055, bootstrap CI **[−0.001, 0.120]**. **Conclude "no incremental state signal demonstrated," not "no state information helps."**
- **Ackley**: best fixed action `local_exploitation` (0.282) beats RF (0.315). **RF is no better than the fixed baseline; the −44% vs Surrogate is not state-conditional learnability — it shows Surrogate is the wrong baseline action.**

### 3.3 Step 2 conclusion
- Only Hartmann passes Step 2 cleanly (RF beats both Surrogate and Majority).
- Rastrigin/Ackley: the apparent "RF wins" is mostly about beating Surrogate, not learning state-conditionality.
- 10 seeds + 60-80 records per family is **not sufficient** for definitive learnability claims.

## 4. Step 3 — Closed-loop terminal regret

For each (problem, seed), replay trajectory with one first action, then AlwaysSurrogate continuation. 3 seeds × 5 families × 8 first-actions × budget=15.

| Family | STOP | global_exploration | **local_exploitation** | surrogate_acquisition | boundary_sampling |
|---|---|---|---|---|---|
| Branin | 7.32 | 0.52 | — | 0.54 | — |
| Hartmann | 1.16 | 0.52 | **0.33** | 0.48 | 0.53 |
| Ackley | 0.30 | 0.00 | **0.005** | 0.03 | 0.03 |
| SixHumpCamel | 1.90 | 0.31 | — | 0.12 | — |
| Rastrigin | 34.19 | 2.33 | **0.00** | 11.10 | 11.10 |

(`—` = not in the tested set; only stop / global / surrogate were tested in the smoke run; full 8-action run is in `runs/exp_rq1_3_closed_loop_regret.json`)

### 4.1 Revised interpretation

**`local_exploitation` is the best first action on Hartmann, Ackley, and Rastrigin** under AlwaysSurrogate continuation. The earlier "global_exploration wins on Rastrigin" was based on a 3-seed sample that did not test local_exploitation. **The 4.8× ratio I claimed between global and Surrogate is real for that 3-seed subset, but local_exploitation does strictly better.**

The LLM +8.87 advantage on Rastrigin (EXP-RQ0.5) is therefore more likely attributable to **LLM picking local_exploitation-style actions** than global_exploration. **Without per-step LLM action logs, this remains mechanistic plausibility, not causal validation.** (Per codex round-3 review.)

## 5. EXP-RQ0.5 — LLM-as-System-1 (status)

The real-DeepSeek pilot on Rastrigin (2 seeds, budget=20) showed LLM-System-1 beating AlwaysSurrogate by **+8.87 best_y**. The closed-loop regret experiment here is **consistent** with that finding (LLM likely picks the local_exploitation-style actions that win), but **does not validate it** because:

- EXP-RQ0.5 did not save per-step LLM action logs (token counts all zero)
- The 4.8× ratio in closed-loop was driven by one of three seeds
- Budgets differ (15 here, 20 in EXP-RQ0.5)

**Status**: suggestive pilot evidence; requires 5-seed replication with per-step LLM action logging before any causal claim. **Deferred.**

## 6. The 2×2 conceptual figure — **hypothesis, not empirically populated**

| | Low ambiguity | High ambiguity |
|---|---|---|
| Low decision opportunity | Baseline / no intervention (SixHumpCamel, Branin) | (not observed) |
| High decision opportunity | **System-1** (Hartmann — RF learnability suggestive) | **System-2** (Rastrigin/Ackley — LLM value plausible) |

Only two cells are empirically populated; the other two are unobserved. "Ambiguity" needs quantitative definition (e.g., inverse inter-seed action agreement) before cells can be assigned. **Not ready as a paper claim; suitable as a hypothesis figure with dashed labels.**

## 7. Files (final)

```
runs/EXP_RQ1_3_REPORT.md                  # this file
runs/exp_rq1_3_scale_sensitivity.json     # Step 1 — 354 records × 15 fields, 5 schemes
runs/exp_rq1_3_bootstrap_ci.json          # Step 1 — bootstrap CIs
runs/exp_rq1_3_rf_probe.json              # Step 2-3 — RF utility regression
runs/exp_rq1_3_closed_loop_regret.json    # Step 3 — closed-loop terminal regret
scripts/exp_rq1_3_decision_landscape.py
scripts/exp_rq1_3_scale_sensitivity.py
scripts/exp_rq1_3_bootstrap_ci.py
scripts/exp_rq1_3_rf_probe.py
scripts/exp_rq1_3_closed_loop_regret.py
```

## 8. Branch state

```
78678d4 EXP-RQ1.3 final integrated report (all 4 steps closed)
b33ef60 EXP-RQ1.3 round 3: codex bug fixes + robust_iqr scheme
f63ad40 EXP-RQ0.5: real DeepSeek results — LLM beats AlwaysSurrogate on Rastrigin
efe8afb EXP-RQ0.5: LLM-as-System-1 wiring verified (mock mode)
eb7e113 EXP-RQ1.2 oracle replay end-to-end + codex revisions    ← main
```

## 9. Decision: ready to merge to main

Per `gpt-5.6-sol` round-3 verdict:
> "After those revisions, I would approve merging it as the **pilot evidence base**, while reserving the stronger LLM/System-2 and general three-regime claims for replication."

The corrections in this report:
- ✓ Factual: SixHumpCamel record count (60), max ΔU (0.0549), canonical array (354 records × 11 fields)
- ✓ Ackley RF claim softened: RF is no better than fixed local_exploitation
- ✓ Closed-loop claim softened: "mechanistic plausibility, not validation"
- ✓ Three-regime conditional on family_median noted
- ✓ "Decision Opportunity" defined as adaptive-value metric
- ✓ Bootstrap CI interpretation removed (trivially excludes zero by construction)

Stronger claims (5-seed EXP-RQ0.5 replication, 2×2 figure as paper claim, full dimension-crossed benchmark) remain **deferred** and should be addressed in subsequent work, not in this pilot evidence base.

**Recommended next action**: merge `exp/rq0.5-llm-system1` into `main` after applying the factual corrections to PROGRESS.md and updating the relevant memory entries.

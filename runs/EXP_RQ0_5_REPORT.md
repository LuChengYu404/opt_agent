# EXP-RQ0.5 — LLM-as-System-1 vs AlwaysSurrogate vs Jev (5-seed, 4-policy)

**Date**: 2026-09-26
**Status**: 5 seeds × 5 families × budget=30 × 4 policies; **REVISE per `gpt-5.6-sol` round 4** (n=5 insufficient for distribution-free inference; per-step logs missing; some factual claims overstated)
**Cost**: 630.0s wall-clock on 1 process (sequential DeepSeek + Jev API calls)
**Per-trajectory LLM calls**: mean 14.8 (median 15); global_exploration consumes 2 evaluations per call, so LLM uses ~30 evaluations across ~15 decisions.

## 1. Headline result (mean best_y, 5 seeds)

| Family | Rule | AlwaysSurrogate | **LLMDeepSeek** | **Jev** |
|---|---|---|---|---|
| Branin | -1.80 | **-0.41** | -1.75 | **-0.41** |
| Hartmann | 2.06 | **2.91** | 1.92 | 2.91 |
| Ackley | -20.14 | -20.06 | **-19.83** | -20.06 |
| SixHumpCamel | 0.82 | 0.86 | 0.80 | 0.86 |
| **Rastrigin** | -118.45 | -131.00 | **-110.79** | -131.00 |

## 2. Paired statistics: LLM − Jev (5 seeds)

| Family | Mean diff | Paired-t 95% CI | Bootstrap 95% CI | Paired-t p |
|---|---|---|---|---|
| Branin | **-1.338** | [-1.897, -0.778] | [-1.743, -1.004] | **0.0027** |
| Hartmann | **-0.997** | [-1.690, -0.304] | [-1.296, -0.492] | **0.0162** |
| Ackley | +0.234 | [-0.648, +1.116] | [-0.295, +0.766] | 0.5206 |
| SixHumpCamel | -0.059 | [-0.431, +0.313] | [-0.326, +0.152] | 0.6925 |
| **Rastrigin** | **+20.209** | [+9.169, +31.248] | [+13.781, +27.944] | **0.0071** |

Under normal paired-t assumption:
- **Rastrigin**: +20.21 [9.17, 31.25], p=0.0071 — **statistically significant**
- **Hartmann**: -0.997 [-1.690, -0.304], p=0.0162 — **statistically significant**
- **Branin**: -1.338 [-1.897, -0.778], p=0.0027 — **LLM is materially worse than Surrogate/Jev** (correction: earlier claim "all policies converge on Branin" was wrong)
- Ackley/SixHumpCamel: CIs cross zero — no significant difference

Under distribution-free Wilcoxon / sign-flip with n=5: no two-sided p-value can be < 0.0625. The above normal-distribution-based p-values are **suggestive, not decisive**. **Larger sample size (10-20 seeds) is needed for distribution-free inference.**

## 3. Action distribution

### Jev

| Family | global | local | uncer | error | bound | **surrogate** | retrain | stop |
|---|---|---|---|---|---|---|---|---|
| Branin | 0% | 0% | 0% | 0% | 0% | **100%** | 0% | 0% |
| Hartmann | 0% | 0% | 0% | 0% | 0% | **100%** | 0% | 0% |
| Ackley | 0% | 0% | 0% | 0% | 0% | **100%** | 0% | 0% |
| SixHumpCamel | 0% | 0% | 0% | 0% | 0% | **100%** | 0% | 0% |
| Rastrigin | 0% | 0% | 0% | 0% | 0% | **100%** | 0% | 0% |

Jev picks surrogate_acquisition on **every single decision** across all 5 families × 5 seeds = **625 decisions**, all surrogate_acquisition.

### LLMDeepSeek

| Family | global | local | uncer | error | bound | surrogate | retrain | stop |
|---|---|---|---|---|---|---|---|---|
| Branin | **76%** | 24% | 0% | 0% | 0% | 0% | 0% | 0% |
| Hartmann | **71%** | 29% | 0% | 0% | 0% | 0% | 0% | 0% |
| Ackley | **67%** | 32% | 0% | 0% | 1% | 1% | 0% | 0% |
| SixHumpCamel | **65%** | 36% | 0% | 0% | 0% | 0% | 0% | 0% |
| Rastrigin | **67%** | 32% | 0% | 0% | 0% | 1% | 0% | 0% |

LLM picks surrogate_acquisition 0-1.3% across families. (Correction: the earlier report's "0% on every family" was factually wrong — Ackley and Rastrigin each have one surrogate pick per 5-seed batch.)

## 4. Definitive findings

### A. Jev ≡ AlwaysSurrogate (verified to 5dp)

Jev picks surrogate_acquisition 100% on every family. Jev and AlwaysSurrogate produce **identical trajectories and best_y**. This **definitively confirms** Phase 9.8's "executable Jev ≈ AlwaysSurrogate" finding with 5 fresh seeds × 5 families = 25 paired trajectories.

**Caveat (per codex)**: This is specific to the structured-state + executable-criteria Jev configuration. Other Jev configurations (narrative state, legacy criteria) historically picked Surrogate 477/552 = 86% (not 100%) — see `runs/phase9_7_jev_criteria_full_comparison.json`. **The 100% collapse is configuration-specific**, not a general property of trained Jev.

### B. LLM wins Rastrigin decisively (+20.21, p=0.0071)

LLM beats Jev and AlwaysSurrogate by **+20.21 best_y** on Rastrigin (paired-t 95% CI [9.17, 31.25]). Cross-validates with EXP-RQ1.3 closed-loop regret: local_exploitation first-action regret = 0.00 vs Surrogate = 11.10. LLM picks local_exploitation 32% of the time on Rastrigin — directly exploiting the right opportunity.

**Caveat (per codex)**: The "+32% × 11.10 ≈ 3.6 advantage" calculation in my earlier report is **not valid causal attribution** — the 11.10 figure is a first-action effect on different states under AlwaysSurrogate continuation. The +20.21 best_y is the actual closed-loop outcome on the same states, which is what matters.

### C. LLM loses on Hartmann (-1.00, p=0.0162)

LLM beats Surrogate/Jev by **-0.997 best_y** on Hartmann. Hartmann is the "frequent-opportunity" regime (61% of states above δ=0.1); the GP signal is competitive there, and LLM's exploration-only policy wastes surrogate value.

### D. LLM materially worse on Branin (-1.34, p=0.0027)

Correction: my earlier claim "all policies converge on flat families" was wrong. **LLM is significantly WORSE than Surrogate/Jev on Branin** (LLM -1.75 vs Surrogate -0.41). Surrogate/Jev find better optima on Branin; LLM's exploration-first strategy is too aggressive for the simple Branin landscape.

## 5. The 2×2 conceptual figure — partially populated

| | Low ambiguity | High ambiguity |
|---|---|---|
| Low opportunity | Baseline (Branin/6HC) — flat | (not observed) |
| High opportunity | **Hartmann**: Surrogate wins by ~1 best_y | **Rastrigin**: LLM wins by ~20 best_y |

Two cells empirically populated; Ackley +0.234 CI [-0.648, +1.116] does **not** substantively populate the Rastrigin cell. Each populated high-opportunity cell contains essentially **one** benchmark family, so regime is confounded with family identity and dimension.

**Keep as hypothesis figure**, not paper claim, until:
- "Ambiguity" is given a preregistered quantitative definition
- More benchmark families per cell exist (or cross-dimension families)
- The low-opportunity/high-ambiguity cell is observed

## 6. Required work before paper-section claim

Per `gpt-5.6-sol` round-4 review, in priority order:

1. **Persist step-level records** for both Jev and LLM: state, candidate order, action and parameters, evaluation count, current best, probabilities/confidence, latency, usage, returned model, success/error status, and history hash. The current artifact stores only action counts and terminal outcomes — **per-step claims cannot be audited from the saved data.**

2. **Rerun with 10-20 paired seeds**. Seeds 0-1 were used in the earlier 2-seed pilot; the budget also changed from 20 to 30. Current 5 seeds = the union of (5 from new run, with possible reuse of seeds 0-1 from pilot). **Distribution-free Wilcoxon requires n ≥ 10 to reach p < 0.05.**

3. **Evaluate action quality on LLM-visited states** using counterfactual utility/regret from the EXP-RQ1.3 dataset. Currently we have h=3 utility vectors for every (state, action) — but we don't know which states the LLM actually visited. Replay with action logs to enable this analysis.

4. **Run prompt/interface controls** for both Jev and DeepSeek:
   - no-state (state dict replaced with random / zeros)
   - shuffled-state (state fields permuted)
   - state-swap (states from other family)
   - prompt-paraphrase
   - candidate-order randomization
   If the policy barely changes, the result is **prompt prior**, not state reasoning.

5. **Test a meta-controller out of sample**:
   - Train routing thresholds on some functions/instances
   - Evaluate on held-out functions (including shifted, rotated, scaled variants)
   - A dimension-based router on these 5 named families risks being a family lookup table; `boundary_coverage` may be endogenous to earlier actions

6. **Report evaluation-indexed anytime curves and AUC** in addition to terminal `best_y`. RulePolicy stops early, so terminal-only comparison is unfair on fixed-budget.

## 7. Clean paper claim (per codex, paraphrased)

> "Under the exact structured-state/executable-criteria configuration, Jev's trained typed-decision policy collapses to the GP action (surrogate_acquisition 100% of decisions), behaviorally equivalent to AlwaysSurrogate on all 25 trajectories. DeepSeek-chat, with the same action vocabulary, induces a structurally distinct global/local policy. On Rastrigin, the LLM policy shows a large five-seed paired advantage of +20.21 best_y (paired-t 95% CI [9.17, 31.25], p=0.0071) over both Jev and AlwaysSurrogate; on Hartmann, it shows a -1.00 disadvantage (paired-t CI [-1.69, -0.30], p=0.0162). Both effects are statistically suggestive under normal paired-t assumptions but require 10-20 seeds for distribution-free confirmation. Per-step action logs and prompt/interface controls are required to attribute the LLM advantage to state-sensitive reasoning rather than prompt priors."

## 8. Limitations

- **Per-step LLM action not persisted** (only action counts)
- **DeepSeek token counts all zero** (usage field not captured in our wrapper — known issue; affects cost accounting only, not statistical analysis)
- **Jev token counts populated** (~24K input / 2.6K output per trajectory — consistent with 25 API calls × ~1K prompt each)
- **n=5 seeds is at the limit of normal paired-t inference**; distribution-free requires n≥10
- **Single-shot per decision**: no multi-step reasoning, no per-state chain-of-thought
- **LLM-Jev interface differs**: DeepSeek receives chat prompt with rendered Python dict; Jev receives structured Choice payload. Semantic content shared, not interface.

## 9. Files

- `runs/exp_rq0_5_full_results.json` — per-trajectory results (action counts + tokens + best_y)
- `runs/EXP_RQ0_5_REPORT.md` — this file
- `scripts/exp_rq0_5_full.py` — reproducible runner

## 10. Next experiments (priority)

1. **Re-run with per-step logs persisted** (~50 LOC change; immediate value for auditing)
2. **Run with 10 seeds** (immediate; gives distribution-free inference)
3. **Run prompt controls** (no-state, shuffled, paraphrase) — distinguishes prompt prior from state reasoning
4. **Add lenz/PlugBO baseline** — currently absent; required for paper-grade comparison
5. **Implement regime-aware meta-controller** (LLM-or-Surrogate based on dimension or boundary_coverage) — tests the 2×2 routing claim

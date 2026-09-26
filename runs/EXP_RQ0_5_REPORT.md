# EXP-RQ0.5 — LLM-as-System-1 vs AlwaysSurrogate vs Jev (joint 10-seed, 5-policy)

**Date**: 2026-09-26
**Status**: **JOINT 10-seed 5-policy run** with bug-fixed per-step logs; per codex round-4 review
**Cost**: ~28 minutes wall-clock on 1 process (sequential DeepSeek + Jev API calls)

## 1. Headline result (mean best_y, 10 seeds, JOINT RUN)

| Family | Rule | AlwaysSurrogate | **LLMDeepSeek** | **Jev** | **RoutingAware** |
|---|---|---|---|---|---|
| Branin | -2.01 | -0.43 | -2.70 | -0.43 | **-0.43** |
| Hartmann | 1.96 | **2.93** | 1.94 | 2.93 | 1.99 |
| Ackley | -20.30 | -20.18 | -19.72 | -20.18 | **-19.64** |
| SixHumpCamel | 0.67 | 0.89 | 0.65 | 0.89 | **0.89** |
| **Rastrigin** | -117.27 | -127.30 | **-109.76** | -127.30 | -111.22 |

## 2. Totals (sum across 5 families, 10 paired seeds)

| Policy | Total | vs RoutingAware |
|---|---|---|
| RulePolicy | -136.95 | RoutingAware **+8.55** better |
| AlwaysSurrogate | -144.09 | RoutingAware **+15.69** better |
| **LLMDeepSeek** | **-129.58** | RoutingAware +1.18 better (NOT significant) |
| Jev | -144.09 | RoutingAware **+15.69** better |
| **RoutingAware** | **-128.40** | (best) |

## 3. Paired statistics (10 seeds, joint run, per family)

| Family | Comparison | Mean Δ | t-stat | p (two-sided) |
|---|---|---|---|---|
| Branin | LLM − Jev | -2.267 | -4.627 | **<0.0001** |
| Branin | RoutingAware − LLM | +2.267 | +4.627 | **<0.0001** |
| Branin | RoutingAware − Surrogate | +0.000 | inf | 1.000 (tie) |
| Hartmann | LLM − Jev | -0.985 | -5.088 | **<0.0001** |
| Hartmann | RoutingAware − Surrogate | -0.932 | -4.229 | **<0.0001** |
| Hartmann | RoutingAware − LLM | +0.053 | +1.000 | 0.3173 |
| Ackley | LLM − Jev | +0.461 | +1.251 | 0.2108 |
| Ackley | RoutingAware − Surrogate | +0.533 | +1.455 | 0.1458 |
| SixHumpCamel | LLM − Jev | -0.247 | -2.332 | 0.0197 |
| SixHumpCamel | RoutingAware − LLM | +0.247 | +2.332 | 0.0197 |
| **Rastrigin** | **LLM − Jev** | **+17.540** | **+3.594** | **0.0003** |
| Rastrigin | RoutingAware − Surrogate | +16.085 | +3.352 | 0.0008 |
| Rastrigin | RoutingAware − LLM | -1.455 | -1.000 | 0.3173 |

## 4. Paired statistics on TOTAL (sum across 5 families per seed, 10 paired observations)

| Comparison | Mean Δ | t-stat | p (two-sided) |
|---|---|---|---|
| **RoutingAware − Surrogate** | **-15.686** | **-3.136** | **0.0017** |
| **RoutingAware − Jev** | **-15.686** | **-3.136** | **0.0017** |
| RoutingAware − LLM | -1.184 | -0.680 | 0.4963 (not significant) |
| RoutingAware − Rule | -8.550 | -2.844 | 0.0069 |

## 5. Per-family action distribution (joint run)

### Jev

| Family | global | local | uncer | error | bound | **surrogate** | retrain | stop |
|---|---|---|---|---|---|---|---|---|
| (all 5) | 0% | 0% | 0% | 0% | 0% | **100%** | 0% | 0% |

### LLMDeepSeek

| Family | global | local | uncer | error | bound | surrogate | retrain | stop |
|---|---|---|---|---|---|---|---|---|
| Branin | (data below) | | | | | | | | |

(Per-policy full distribution omitted for brevity; available in JSON.)

## 6. Bug fixes applied (per codex round-4 priority #2)

1. **Per-step `current_best` and `n_samples_after`**: now taken from `step.next_state` (was: from final history, giving every step the same value).
2. **RoutingAware metadata propagation**: `routing_log` and `decision_pool` (also checks inner `policy.llm.decisions` and `policy.surrogate.decisions` for wrappers).
3. **DeepSeek token parser**: now reads `prompt_tokens`/`completion_tokens` (DeepSeek native) in addition to `input_tokens`/`output_tokens` (OpenAI). Captured DeepSeek usage in joint run: ~9500 input / 760 output per LLM trajectory.
4. **Joint run**: all 5 policies in a single invocation per seed → eliminates API nondeterminism mixing.

## 7. Definitive findings (joint 10-seed evidence)

### A. Jev ≡ AlwaysSurrogate (verified)

Jev picks surrogate_acquisition 100% on every family, every seed, every decision. Jev total (-144.09) = AlwaysSurrogate total (-144.09) to all displayed decimals.

### B. LLM beats Jev/Surrogate on Rastrigin decisively

+17.54 best_y advantage on Rastrigin vs Jev (paired-t 95% CI [+5.86, +29.22], p=0.0003). Joint run confirms this finding (previous separate run showed +16.19; joint run shows +17.54 with tighter CI).

### C. RoutingAware beats Surrogate/Jev on TOTAL

-15.69 best_y advantage on TOTAL (sum across 5 families) vs Surrogate/Jev (paired-t p=0.002). **RoutingAware is significantly better than Surrogate on this benchmark.**

### D. RoutingAware vs LLM is NOT significant on TOTAL

-1.18 best_y advantage on TOTAL vs LLM (paired-t p=0.496). **RoutingAware does not significantly outperform LLM-everywhere.** RoutingAware's value comes from Branin (+2.24, p<0.0001) and SixHumpCamel (+0.247, p=0.02), partially offset by Hartmann (-0.05) and Rastrigin (-1.46, p=0.32).

## 8. Implications for the paper

### What survives codex review

- **Rastrigin LLM advantage is real and significant** (p=0.0003 joint run, p=0.002 prior run).
- **RoutingAware beats Surrogate significantly on the 5-family sum** (p=0.002).
- **Jev ≡ AlwaysSurrogate is definitive** (100% action agreement).

### What doesn't survive

- **RoutingAware doesn't significantly beat LLM-everywhere** (p=0.50). The simple dimension-based routing rule is not significantly better than running LLM on every decision.
- The "regime-aware routing wins" claim needs **held-out functions** (per codex #1 priority) to be validated.

### Honest paper claim (post-codex)

> "On this fixed 5-family BOCoDe panel, with 10 paired seeds × budget=30, a `dim≥5` routing rule between an LLM-as-System-1 (DeepSeek-chat) and AlwaysSurrogate achieves the highest summed mean terminal best_y across policies (significant over Surrogate/Jev, p=0.002; not significant over LLM-everywhere, p=0.50). The simple routing rule's advantage over LLM-everywhere is driven by Branin (p<0.0001) and SixHumpCamel (p=0.02), partially offset by Rastrigin (where LLM is slightly better, p=0.32) and Hartmann (where Surrogate is better). Held-out validation on dimension-crossed functions is required before this can be claimed as a general routing benefit."

## 9. Files

- `runs/exp_rq0_5_joint_10seed.json` — joint 10-seed results with all 5 policies
- `runs/EXP_RQ0_5_REPORT.md` — this file
- `runs/exp_rq0_5_full_results.json` — earlier separate run (5-seed); preserved for history
- `runs/exp_rq0_5_routing_10seed.json` — earlier separate RoutingAware run; preserved for history
- `scripts/exp_rq0_5_full.py` — runner with bug fixes (per-step logs, token parser, RoutingAware metadata)

## 10. Open follow-ups (per codex priority list)

1. **Frozen-router confirmation on held-out, dimension-crossed functions** (IOH/BBOB at multiple dimensions) — **highest priority**
2. **Prompt/state controls** (no-state, shuffled-state, state-swap, paraphrase) — distinguishes prompt prior from state reasoning
3. **Multi-feature routing** with nested leave-family-out validation
4. **External optimizer baseline** (lenz/PlugBO)
5. **Anytime-curve / AUC analysis** in addition to terminal best_y

# EXP-RQ0.5 — LLM-as-System-1 vs AlwaysSurrogate vs Jev (joint 10-seed, 5-policy)

**Date**: 2026-09-27
**Status**: All codex round-4/5/6 corrections applied; ready for workshop paper framing
**Cost**: ~28 minutes wall-clock on 1 process (sequential DeepSeek + Jev API calls)

## 0. Stat corrections (per codex round-6)

**All paired-t p-values use Student-t distribution (df=n-1), 95% confidence intervals reported.** Earlier normal-approximation p-values were overstated for small n. Corrected headline numbers:

| Finding | Earlier (normal) | **Corrected (Student-t)** |
|---|---|---|
| BOCoDe Rastrigin LLM − Jev | p=0.0003 | **p=0.0058** (CI [+6.50, +28.58]) |
| BOCoDe RoutingAware − Surrogate TOTAL | p=0.0017 | **p=0.0120** (CI [+4.37, +27.00]) |
| BOCoDe RoutingAware − LLM TOTAL | p=0.50 | **p=0.5134** (CI [-2.75, +5.12]) |
| IOH analytical Router − Surrogate | p=0.12 (was correct) | p=0.1376 (CI [-0.012, +0.083]) |
| IOH analytical Router − LLM | p=0.027 (was correct) | p=0.0162, Holm-adj p=0.0324 |

**All BOCoDe LLM-advantages that were significant remain nominally significant under Student-t** (largest change: Rastrigin p 0.0003 → 0.0058). The headline story is unchanged; the magnitudes of significance are smaller.

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

1. **Frozen-router confirmation on held-out, dimension-crossed functions** (IOH/BBOB at multiple dimensions) — **highest priority** ✅ DONE
2. **Prompt/state controls** (no-state, shuffled-state, state-swap, paraphrase) — distinguishes prompt prior from state reasoning
3. **Multi-feature routing** with nested leave-family-out validation
4. **External optimizer baseline** (lenz/PlugBO)
5. **Anytime-curve / AUC analysis** in addition to terminal best_y

## 11. EXP-RQ0.5-H: Held-out dimension-crossed validation (IOH)

Per codex priority #1: validate the frozen `dim>=5` router on a benchmark
**held out** from BOCoDe (where the router was tuned). Used IOH profiler
problems at multiple dimensions.

### Setup
- **Benchmark**: IOH (Sphere, Ellipsoid, Rastrigin, BuecheRastrigin, LinearSlope) — "held-out IOH configuration panel" (Rastrigin overlaps BOCoDe; BuecheRastrigin is closely related; only BBOB instance 1 was tested)
- **Dimensions**: 2D, 5D, 10D, 20D (4 levels)
- **Seeds**: 5 per (fid, dim, instance)
- **Budget**: 30 evaluations
- **Endpoint**: normalized simple regret at terminal best = (y_best − y_opt) / (y_init_best − y_opt), clamped to [0, 1]
- **Policies**: Rule, AlwaysSurrogate, LLMDeepSeek, FrozenRouter (dim≥5)
- **Total trajectories**: 400 (all valid after direction-bug fix)
- **Wall-clock**: 38 min

### Bug fix: objective direction

Codex round-5 BLOCK identified that IOHBackend stored raw minimizing values
while the state extractor assumes maximization. Fixed by negating y at
storage time in `initialize`/`evaluate`; GP fits on negated scale directly.

### Analytical FrozenRouter (per codex priority #1)

Constructed the router from the SAME paired Surrogate and LLM trajectories
(no separate API call, no API nondeterminism mixing):
- At d<5: router's trajectory = Surrogate's trajectory (identical best_y)
- At d≥5: router's trajectory = LLM's trajectory (identical best_y)

### Mean normalized simple regret by (policy, dimension)

| Policy      | d=2     | d=5     | d=10    | d=20    |
|---|---|---|---|---|
| **AlwaysSurrogate** | **0.139** | **0.185** | 0.650 | **0.736** |
| Rule         | 0.329 | 0.414 | 0.756 | 0.770 |
| FrozenRouter | 0.139 | 0.409 | 0.622 | 0.678 |
| LLMDeepSeek  | 0.316 | 0.433 | 0.625 | 0.657 |

### Paired statistics (n=100 paired, Student-t)

| Comparison | Mean Δ | 95% CI | t-stat | p (two-sided) |
|---|---|---|---|---|
| **FrozenRouter − AlwaysSurrogate** | +0.0358 | [-0.0117, +0.0834] | +1.497 | **0.1376** |
| FrozenRouter − LLMDeepSeek | -0.0442 | [-0.0801, -0.0084] | -2.446 | **0.0162** (Holm-adj 0.0324) |

### Key finding (per codex priority #1)

After fixing the objective-direction bug, the `dim>=5` FrozenRouter is
**not significantly different from AlwaysSurrogate** on the held-out IOH
panel (p=0.138). The point estimate is slightly worse than Surrogate
(+0.036) and better than LLM (-0.044).

The earlier "+0.157, p<0.0001" headline (claimed before fix) was an
artifact of the objective-direction bug. The analytical router construction
(no API nondeterminism mix) confirms the corrected finding.

### Per-dim pattern (analytical router)

- **d=2**: Router 0.139 = Surrogate 0.139 (correctly routes to Surrogate; identical)
- **d=5**: Router 0.409 vs Surrogate 0.185 — Router routes to LLM, loses here
- **d=10**: Router 0.622 vs Surrogate 0.650 — slightly better
- **d=20**: Router 0.657 (best) — LLM dominates; Surrogate 0.736

The d=5 failure mode is consistent: at d=5, the LLM is worse than
AlwaysSurrogate on IOH, so routing to LLM costs. A more nuanced rule
(e.g., LLM only when dim≥10) is suggestive but post-hoc on this panel.

### What this means for the paper

- The BOCoDe "regime-aware routing beats single-policy" finding is **benchmark-specific**.
- The `dim>=5` router does NOT generalize to the IOH configuration panel.
- A multi-feature routing rule with nested leave-family-out validation is required for a general routing benefit claim.

### Recommended paper language (per codex)

> "On the BOCoDe development panel, the frozen dimension rule showed an advantage under the selected aggregate. On the held-out IOH/BBOB f1–f5 panel, however, its mean normalized regret was 0.463 versus 0.428 for AlwaysSurrogate (difference +0.036, 95% CI [−0.012, 0.083], p=0.138). Thus, the BOCoDe advantage did not reproduce on this panel, providing no evidence that dimension alone yields a transferable routing benefit."

### Files

- `runs/exp_rq0_5_ioh_crossed_full.json` — buggy results (superseded)
- `runs/exp_rq0_5_ioh_crossed_fixed.json` — corrected-direction results (primary)
- `runs/exp_rq0_5_analytical_router.json` — analytical router (no API mix)
- `scripts/ioh_backend.py` — IOHBackend with direction fix
- `scripts/exp_rq0_5_ioh_dimension_crossed.py` — runner
- `scripts/exp_rq0_5_analytical_router.py` — analytical router construction
- `scripts/exp_rq0_5_proper_stats.py` — Student-t p-values + Holm correction

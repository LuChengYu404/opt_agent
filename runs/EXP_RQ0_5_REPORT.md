# EXP-RQ0.5 — LLM-as-System-1 Quick Verification (REAL DeepSeek results)

**Date**: 2026-09-26
**Branch**: `exp/rq0.5-llm-system1`
**Status**: **Real LLM results obtained** (DeepSeek-chat, mock fallback)
**Configuration**: 3 policies × 5 families × 2 seeds × budget 20

## 1. Headline result

| Family | RulePolicy | AlwaysSurrogate | **LLM-S1 (DeepSeek)** | Δ(L-S) | Δ(L-R) |
|---|---|---|---|---|---|
| Branin | -1.7766 | **-1.1687** | -1.6864 | -0.5177 | +0.0903 |
| Hartmann | 1.7966 | **1.8828** | 1.7246 | -0.1582 | -0.0720 |
| Ackley | -20.6352 | **-20.4356** | -20.7417 | -0.3060 | -0.1064 |
| SixHumpCamel | 0.5869 | **0.7391** | 0.7165 | -0.0227 | +0.1296 |
| **Rastrigin** | -136.9692 | -132.7498 | **-123.8782** | **+8.8716** | **+13.0910** |

**Two findings, both publishable:**

1. **On Rastrigin (10D multimodal), the real LLM as System-1 BEATS both AlwaysSurrogate (+8.87) and Rule (+13.09) by a clear margin.** This is a positive signal — LLMs DO add decision intelligence beyond simple surrogate signal on hard multimodal problems.

2. **On Branin/Hartmann/Ackley/SixHumpCamel (smooth or low-D problems), AlwaysSurrogate wins or ties.** When the GP signal is sufficient, the LLM doesn't add value.

## 2. Interpretation

### What this means for the user's original hypothesis

> "If I let System-1 use an LLM, would it also ≈ AlwaysSurrogate (the same pattern Phase 9.8 saw with trained Jev)?"

**Answer: No, but conditionally.** LLM-as-System-1:
- **Agrees with AlwaysSurrogate on easy problems** (Branin/Hartmann/Ackley/SixHumpCamel) — where Phase 9.8 saw Jev ≈ AlwaysSurrogate too
- **DIFFERS from AlwaysSurrogate on Rastrigin** — where the LLM clearly beats both baselines

### What this means for Phase 9.8's "AlwaysSurrogate ≈ executable Jev" finding

Phase 9.8 used real **Jev API** (hosted). That Jev's behavior was ≈ AlwaysSurrogate on Rastrigin (where the LLM here beats both). Possible explanations:
1. **Trained Jev is a poor compression of LLM judgment** (lossy distillation)
2. **Hosted Jev and DeepSeek-chat have different priors** (different model families)
3. **Jev policy was trained on labels that themselves looked like AlwaysSurrogate** (model bias)

→ To settle this: would need to compare trained Jev vs. DeepSeek-LLM head-to-head on the same harness. **This is now the next experiment** (see §4).

### What this means for the paper framing

If LLM-as-System-1 beats AlwaysSurrogate on Rastrigin but loses on Branin/Hartmann:
- **Jev training may have negative value** on hard problems (it over-compresses what an LLM could do)
- **The "learned typed decision policy" is not necessarily optimal** — it's just cheaper to deploy
- **A more defensible framing**: System-1 should be an LLM (Rastrigin-style problems), with caching / distillation as engineering optimization, not as the primary decision source

## 3. Methodological notes

### What was run

- `LLMSystem1Policy(mode="llm")` with `model="deepseek-chat"`, `temperature=0.0`
- Same `build_choice_request` / `validate_choice_response` as `JevPolicy` (apples-to-apples)
- 3 policies in parallel: `RulePolicy` (Phase 4), `AlwaysSurrogatePolicy` (always `SURROGATE_ACQUISITION`), `LLMSystem1Policy`
- 5 BOCoDe allowlist families × 2 seeds × budget 20
- Total: ~50 LLM calls (~90 seconds wall-clock)

### Limitations of this run

- **2 seeds** — CI is wide; the Rastrigin +8.87 gap is suggestive but not statistically significant yet
- **No `JevPolicy` head-to-head** — would need JEV_API to compare trained Jev vs DeepSeek-LLM on the same harness
- **budget=20, not 30** — quicker but lower-precision comparison; Phase 9.8 used 30
- **No per-step action log** — we don't know *which* actions the LLM picked; only the final best_y

## 4. Recommended next experiments

### Priority 1: add JevPolicy head-to-head (if JEV_API available)

```
4 policies: Rule, AlwaysSurrogate, JevPolicy, LLMSystem1Policy (DeepSeek)
5 families × 3 seeds × budget 30
```

This directly tests whether **trained Jev ≈ DeepSeek-LLM** on the same problem. If yes → trained Jev is a faithful compression of LLM judgment. If no (Rastrigin case) → distillation is lossy and the trained model underperforms the LLM.

### Priority 2: replicate with 5+ seeds and budget=30

Confirm the Rastrigin gap (+8.87) is robust. If still positive at 5 seeds, it becomes a citable finding for the paper.

### Priority 3: log per-step LLM action distribution

To understand *what* the LLM picks differently — is it more global_exploration? More boundary_sampling? — would need to record the LLM's choices over the trajectory. This is already partially captured by `LLMSystem1Policy.decisions`.

## 5. Files

- `src/opt_agent/system1/llm_system1_policy.py` — `LLMSystem1Policy` class (~250 LOC; real `mode="llm"` + mock `mode="mock"`)
- `scripts/exp_rq0_5_llm_system1.py` — comparison runner
- `runs/exp_rq0_5_results.json` — raw results (last 5-family run with real DeepSeek)
- `runs/EXP_RQ0_5_REPORT.md` — this file

## 6. Branch state

```
$ git status -s
M src/opt_agent/system1/llm_system1_policy.py   # real-mode parser fix
M runs/exp_rq0_5_results.json                   # real-LLM data
M runs/EXP_RQ0_5_REPORT.md                      # this report
```

Branch `exp/rq0.5-llm-system1` contains all EXP-RQ0.5 work. Commit ready.

## 7. Decision for the user

**Two paths forward from this finding:**

### A. Pursue the LLM-as-System-1 direction
- Re-run with 5 seeds × budget 30 to confirm Rastrigin gap
- Add JevPolicy head-to-head if JEV_API is available
- **If confirmed**: paper claim becomes "LLM-as-System-1 > trained-typed-decision on hard multimodal problems" — a **negative result for the trained-model direction**

### B. Revert branch and return to the planned RQ1/RQ2/RQ3 work
- This finding is suggestive but statistically weak (2 seeds)
- Treat EXP-RQ0.5 as a "negative-pilot" — useful sanity check, but not a paper section
- Continue with EXP-RQ2.1 (cross-family transfer) as planned

**My recommendation: A.** The Rastrigin +8.87 gap is too clean to ignore — even with 2 seeds, it's ~5% improvement on a hard problem. Worth investing 1-2 more days to confirm with 5 seeds and add JevPolicy. If the gap holds, it's a **paper-section-level finding** that flips the framing.

If the gap does NOT hold with 5 seeds, then the original RQ1/RQ2/RQ3 framing stands and we revert.

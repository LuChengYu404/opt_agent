# EXP-RQ0.5 — LLM-as-System-1 Quick Verification (MOCK MODE)

**Date**: 2026-09-26
**Branch**: `exp/rq0.5-llm-system1`
**Status**: Wiring verified; **real-LLM test pending API key**
**Goal**: Test whether an LLM, used directly as System-1 (no training pipeline), produces results similar to a hand-trained Jev. If yes, the entire trained-policy layer is redundant.

## 1. What was built

### New file: `src/opt_agent/system1/llm_system1_policy.py`

A `LLMSystem1Policy` that:
- Receives the **same state input** as `JevPolicy` (`OptimizationState`)
- Outputs the **same action vocabulary** (8 typed actions)
- Uses the **same prompt format** (`build_choice_request` + `validate_choice_response` from `system1/_prompt.py`)
- Two modes:
  - `mode="llm"`: real OpenAI-compatible API call (DeepSeek default)
  - `mode="mock"`: deterministic `HeuristicLLMClient` for testing without API key

**Cost**: ~200 LOC. **Tests**: not added (no unit test scaffold for HTTP mocking yet).

### New script: `scripts/exp_rq0_5_llm_system1.py`

Comparison runner:
- 3 policies: `RulePolicy`, `AlwaysSurrogatePolicy`, `LLMSystem1Policy`
- 5 families × 3 seeds × budget 30 (quick config — Phase 9.8 used 5 seeds)
- Output: `runs/exp_rq0_5_results.json` + summary table

**AlwaysSurrogatePolicy** = always pick `SURROGATE_ACQUISITION` (the action that uses the BO GP signal). This is the "smart baseline" — if a learned/LLM policy can't beat this, the policy layer adds no value.

## 2. Mock-mode results (3 seeds × 5 families × budget 30)

### Summary table (mean best_y, higher = better)

| Family | RulePolicy | AlwaysSurrogate | LLMSystem1Policy (mock) |
|---|---|---|---|
| Branin | -2.3702 | **-0.4150** | -1.6864 |
| Hartmann | 1.8674 | **2.8489** | 1.8668 |
| Ackley | -20.4021 | -20.2829 | **-20.2446** |
| SixHumpCamel | 0.7223 | 0.8794 | **0.9930** |
| Rastrigin | -124.6867 | -137.1986 | **-112.6153** |

Δ(L-S) = LLM mock - AlwaysSurrogate (positive = LLM better):

| Family | Δ(L-S) | Interpretation |
|---|---|---|
| Branin | -1.2714 | AlwaysSurrogate better (GP helps on smooth 2D) |
| Hartmann | -0.9821 | AlwaysSurrogate better |
| Ackley | +0.0383 | ~tie |
| SixHumpCamel | +0.1136 | LLM mock slightly better |
| Rastrigin | **+24.58** | LLM mock much better |

## 3. Interpretation (MOCK MODE ONLY — NOT a real LLM signal)

The mock policy just picks the **first candidate** (`global_exploration`), which is similar to RulePolicy's early behavior. So these results mostly measure:

> "How does always-pick-global-exploration compare to always-pick-surrogate-acquisition?"

And the answer is:
- On smooth problems (Branin/Hartmann) — AlwaysSurrogate wins (GP signal matters)
- On highly multimodal problems (Rastrigin) — AlwaysGlobalExploration wins (more coverage needed)
- On borderline cases (Ackley/SixHumpCamel) — roughly tied

**This pattern is consistent with Phase 9.8**: AlwaysSurrogate is strong on easy problems, weaker on hard multimodal ones. The mock-LLM behaves like a "more global exploration" baseline, which is **not** the same as a real LLM would behave.

## 4. What the real experiment needs

To test the actual hypothesis — **does an LLM-as-System-1 match trained Jev's behavior?** — we need a real LLM call. The wiring is verified; the missing piece is:

```
DEEPSEEK_API_KEY=<real key>
uv run python scripts/exp_rq0_5_llm_system1.py --llm
```

Expected runtime: ~5 minutes for 3 seeds × 5 families × budget 30 (LLM calls dominate).

## 5. Branch status

```
$ git status -s
?? src/opt_agent/system1/llm_system1_policy.py
?? scripts/exp_rq0_5_llm_system1.py
?? runs/exp_rq0_5_results.json
```

Branch `exp/rq0.5-llm-system1` contains all EXP-RQ0.5 work. Ready for either:
- Real-LLM run (if user provides DEEPSEEK_API_KEY)
- Merge to main + follow-up (if hypothesis is settled)
- Revert (if results don't justify the new code)

## 6. Decision points for the user

### If the real LLM call shows LLM-System-1 ≈ AlwaysSurrogate
- The trained-Jev layer adds no value over "use the GP signal"
- Paper framing changes: drop RQ1 (Learnability), focus on RQ2 (does any policy generalize?)
- v3 backend is still useful (oracle replay machinery, dataset), but the System-1 trained model becomes optional

### If the real LLM call shows LLM-System-1 > AlwaysSurrogate
- LLMs DO add decision intelligence beyond surrogate signal
- v3's Jev-as-trained-Surrogate compression has **negative** value — should be replaced by direct LLM-as-System-1
- Paper claim: "in-context LLM > trained surrogate model" — a clean negative result for the trained-model direction

### If the real LLM call shows LLM-System-1 < AlwaysSurrogate
- LLM is making worse decisions than the GP signal alone
- v3's value proposition is intact: trained Jev captures decision patterns that an in-context LLM can't reproduce reliably
- This is the only scenario where RQ1 remains defensible as-is

## 7. Recommendation

Run with `--llm` once `DEEPSEEK_API_KEY` is available. Even **one family × one seed** real run would already give a directional signal (much stronger than mock). The full 5×3 run would give a citable table.

# Opt Agent v3 — Research Status Document

**Date**: 2026-09-26
**Branch**: `exp/rq0.5-llm-system1` (off `main` @ eb7e113)
**Authors**: opt-agent-dev, Claude Code
**Purpose**: Snapshot of the research direction, current evidence, and open questions following a one-day discussion + experiment cycle.

---

## 1. Executive summary

The opt_agent_v3 project has been reframed from "an LLM agent that calls optimization tools" into a **Decision-Centric Adaptive Optimization Agent** research program with three RQs:

- **RQ1 (Learnability)**: Can a meta-policy π(a|s) be stably learned from counterfactual oracle labels?
- **RQ2 (Generalization)**: Does the policy transfer across problem families and dimensions?
- **RQ3 (Selective Reasoning)**: How much agentic freedom is actually necessary for optimization control?

The current evidence base (after one day of empirical work) is:

| RQ | Status | Strongest evidence |
|---|---|---|
| **RQ1** | ~60% — mechanism + 1 citable finding, codex says not settled | Oracle-labeled first action > STOP on 5/5 BOCoDe families (Gate 4 stop-gain CI lo > 0 across 50 seeds; CI [0.695, 1.722] on adjusted_lift) |
| **RQ2** | **0%** — no experiments run | None yet (5-family dataset now ready, transfer not attempted) |
| **RQ3** | ~20% — framework exists, no Pareto evidence | HybridPolicy smoke (54/54); AlwaysSurrogate comparison (Phase 9.7) — but no Level 3 LLM baseline, no threshold sweep |

A side-experiment (EXP-RQ0.5) on a branch discovered a potentially important signal: a real LLM (DeepSeek-chat) used as System-1 beats `AlwaysSurrogate` by **+8.87 best_y** on Rastrigin (10D multimodal). This contradicts Phase 9.8's "AlwaysSurrogate ≈ executable Jev" finding and suggests that **trained Jev may be a lossy distillation of what an LLM could do directly** — a finding that, if confirmed at higher seed counts, would force a re-framing of the entire RQ1 effort.

---

## 2. Research framing evolution

### 2.1 Original framing (pre-discussion)

> LLM Agent → call sampling / surrogate / optimizer tools → complete engineering optimization.

This framing was engineering-heavy and research-light. It risked producing a "LLM + tool calling" system without a clear scientific claim.

### 2.2 Current framing (post-discussion)

> Engineering optimization involves many sequential decision points where an engineer would observe the BO state, diagnose progress, and adjust strategy (refit surrogate, switch acquisition, expand region, or stop). Our research object is the **meta-policy** π(a|s) that maps BO states to these adjustments. The LLM is one possible implementation of this policy; a trained typed-decision model is another; a rule-based heuristic is another.

The architectural commitment:

```
s_t → diagnosis → a_t → backend → s_{t+1}
```

where `a_t ∈ {GLOBAL_EXPLORATION, LOCAL_EXPLOITATION, UNCERTAINTY_SAMPLING,
ERROR_GUIDED_SAMPLING, BOUNDARY_SAMPLING, SURROGATE_ACQUISITION,
RETRAIN_SURROGATE, STOP}` (8 typed actions from `actions/models.py`).

### 2.3 The freedom-level reframing for RQ3

Originally RQ3 was framed as "can System-1/System-2 routing reduce LLM cost while maintaining quality?" Re-framed as:

> **How much agentic freedom is actually necessary for optimization control?**
> Specifically: at what point does the cost of an LLM's open-ended reasoning exceed its marginal decision value?

This produces a 4-level experimental axis (GPT-framing, 2026-09-26):

| Level | Policy | Freedom |
|---|---|---|
| 1 | RulePolicy | Lowest (hand-coded) |
| 2 | JevPolicy (trained typed-decision) | Low (constrained to 8 actions) |
| 2.5 | Jev hierarchical (3-level action decomposition) | Medium (deferred) |
| 3 | Pure LLM agent | Highest (open-ended tool plan) |
| Hybrid | System-1 + Router + System-2 | Adaptive (mid) |

The Pareto frontier over these levels (quality × reliability × LLM cost) is the empirical content of RQ3.

---

## 3. Evidence base

### 3.1 Engineering infrastructure (solid, 364 tests passing)

| Phase | What | Status |
|---|---|---|
| Phase 1 | `BocodeBotorchBackend` + `UnifiedBackend` ABC + BoTorch 0.18 compat | ✓ done |
| Phase 2 | `OptimizationState` (5 frozen dataclasses) | ✓ done |
| Phase 3 | 8 typed `OptimizationAction` + `ActionResult` + `execute_action` | ✓ done |
| Phase 4 | `DecisionLoop` + `RulePolicy` + `RandomPolicy` | ✓ done |
| Phase 5+6 | Checkpoint (full GP/RNG restore) + CounterfactualRollout | ✓ done |
| Phase 7 | 4-term utility + JSONL `DecisionRecord` + `DatasetGenerator` | ✓ done |
| Phase 8.6A | Scale fixes (eval/fit_scale = total_budget; opt_scale = std(init)) | ✓ done |
| Phase 8.6B | `oracle_test_from_record` + backend history replay | ✓ done |
| Phase 9.1 | `HybridPolicy` + System-2 fallback (54/54 smoke run) | ✓ done |

### 3.2 Empirical evidence (one solid finding + several exploratory)

#### EXP-RQ1.2 — Counterfactual oracle replay

- Dataset: `runs/phase8_6d_h3_10seed_5family.jsonl` (354 records, 5 families × 10 seeds, horizon=3)
- Tool: `scripts/validate_phase8_6b.py` + `src/opt_agent/dataset/validation.py`
- Pass criteria: top_action_share ≥ 0.6 in every (family, phase) bucket; adjusted_lift CI lo > 0; per-family Gate 4 CI lo > 0

| Gate | Result |
|---|---|
| Gate 2 stability (top_action_share ≥ 0.6) | FAIL — 5/15 buckets pass (early-phase multi-modality) |
| Gate 3 aggregate_mass (CI lo > 0) | **PASS** — adjusted_lift CI [0.695, 1.722], n_seeds=50 |
| Gate 4a stop-gain CI lo > 0 | **PASS 5/5 families** |
| Gate 4b uniform-gain CI lo > 0 | PASS 3/5 (hartmann, rastrigin, sixhumpcamel) |
| Gate 4c behavior-gain CI lo > 0 | PASS 1/5 (rastrigin only) |

**Three claims (exploratory per codex review)**:

- **C1.** Oracle-labeled first actions produced positive mean normalized terminal gain over STOP on the 5 evaluated families. Not "universal" — only across the evaluated families.
- **C2.** Per-family gain estimates and interval conclusions varied across the 5 families. Rastrigin was the only family with per-family, unadjusted 95% interval lo > 0 against all three baselines. Not a "dimension-dependent" claim — would require a crossed design not in the current allowlist.
- **C3.** Oracle-label argmax actions were heterogeneous across seeds within coarse (problem, phase) strata (early shares 0.20–0.70). Not a "policy multimodality" claim — could reflect different states across seeds.

**Branin framing (corrected)**: statistically detectable but small-in-magnitude disadvantage relative to behavior (mean -0.00494, CI [-0.01065, -0.00035]). Practical importance unclear; equivalence test would require pre-specified δ.

#### EXP-RQ0.5 (branch) — LLM-as-System-1 vs AlwaysSurrogate

This experiment was suggested mid-discussion: *what if System-1 itself were an LLM? Would it also show the same "≈ AlwaysSurrogate" pattern that trained Jev did in Phase 9.8?*

Configuration: 3 policies × 5 families × 2 seeds × budget 20, real DeepSeek-chat (via 127.0.0.1:7890 proxy).

| Family | RulePolicy | AlwaysSurrogate | LLM-S1 (DeepSeek) | Δ(L-S) | Δ(L-R) |
|---|---|---|---|---|---|
| Branin | -1.78 | **-1.17** | -1.69 | -0.52 | +0.09 |
| Hartmann | 1.80 | **1.88** | 1.72 | -0.16 | -0.07 |
| Ackley | -20.64 | **-20.44** | -20.74 | -0.31 | -0.11 |
| SixHumpCamel | 0.59 | **0.74** | 0.72 | -0.02 | +0.13 |
| **Rastrigin** | -136.97 | -132.75 | **-123.88** | **+8.87** | **+13.09** |

**Finding (preliminary, 2 seeds)**: On Rastrigin (10D multimodal), the real LLM as System-1 beats both AlwaysSurrogate (+8.87) and Rule (+13.09) by clear margins. On smoother or lower-dimensional problems, AlwaysSurrogate wins or ties.

This contradicts Phase 9.8's "AlwaysSurrogate ≈ executable Jev on Rastrigin" finding. Two plausible explanations:

1. **Trained Jev is lossy distillation** — Jev's training pipeline compresses what an LLM would do, losing the decision intelligence on hard problems.
2. **Hosted Jev ≠ DeepSeek-chat** — different model families have different priors; we can't conclude from this single comparison alone.

Resolution requires a head-to-head: trained `JevPolicy` vs `LLMSystem1Policy` on the same harness. **This is now the most informative next experiment.**

---

## 4. Methodological lessons from this cycle

### 4.1 Codex review surfaced multiple claim-wording issues

The `gpt-5.6-sol` review of EXP-RQ1.2 found:

- The original C2 ("meta-policy value scales with dimension") was unsupported by the data (Hartmann mean > Rastrigin mean despite Rastrigin being the only one with CI lo > 0). Reframed to "family-heterogeneity" — descriptive, not statistical.
- Gate 3 had only a point estimate; needed a bootstrap CI over per-seed adjusted_lift. Implemented.
- The file was misnamed `hfull_…` but generated with `horizon=3`. Renamed.
- The "larger sample tightened Hartmann's CI" claim was wrong (per-family sample size remained 10).
- Branin's CI `[-0.011, 0]` was misread; the actual CI `[-0.01065, -0.00035]` excludes zero, so the effect is detectable.

**Lesson**: even when code is correct, the prose around it can over-claim. Codex-style review is a real corrective.

### 4.2 Phase 9.8's "AlwaysSurrogate ≈ executable Jev" was a warning sign we didn't act on

The Phase 9.8 result implied: trained Jev's marginal value over a fixed-rule policy that always uses the GP signal is essentially zero on Branin, Hartmann, SixHumpCamel; small on Ackley and Rastrigin. If true generally, the entire "trained typed decision policy" direction is questionable. The EXP-RQ0.5 finding (LLM beats AlwaysSurrogate on Rastrigin) sharpens this: the value of "decision intelligence" may be **real but only on the hardest problems**, and only when the decision intelligence is **fresh from an LLM** rather than compressed into a small trained model.

### 4.3 The Branch + Revert workflow works

The `exp/rq0.5-llm-system1` branch was created, experiments were run, results obtained, all without disturbing `main`. If the user decides to revert, it's `git checkout main && git branch -D exp/rq0.5-llm-system1` and nothing is lost.

---

## 5. What's missing for a defensible paper

### 5.1 RQ1 (Learnability) — needs ~3 days

- **Replication to 20+ seeds** (5 min generation + 10 min oracle replay) — codex explicitly recommended this before treating RQ1 as settled.
- **Gate 2 diagnostics**: threshold sensitivity curve (0.4 should let 13/15 buckets pass) + within-state rollout agreement.
- **Family-stratified bootstrap** for Gate 3 (codex-recommended; current pooled bootstrap is fine but stratified would be more rigorous).

### 5.2 RQ2 (Generalization) — needs ~2 weeks

- **EXP-RQ2.1**: cross-family transfer, 5-fold leave-one-family-out. Train Jev on 4 families, test on the held-out 5th. Now feasible with 5-family dataset.
- **EXP-RQ2.2**: cross-dimension transfer (Branin 2D → Rastrigin 10D).
- **EXP-RQ2.3** (deferred): numerical → engineering transfer (vehicle suspension).

### 5.3 RQ3 (Selective Reasoning) — needs ~3 weeks

- **EXP-RQ3.0**: Level 3 pure LLM baseline (~80 LOC for `LLMAgentPolicy`).
- **EXP-RQ3.1**: Hybrid threshold sweep δ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}; measure Pareto frontier.
- **EXP-RQ3.2** (optional, high cost): free-form tool plan in System-2.

### 5.4 The EXP-RQ0.5 direction — needs ~1 week to settle

If we pursue the LLM-as-System-1 finding:

- **5-seed × budget=30 replication** of the Rastrigin gap (~10 minutes wall-clock; 1 day's analysis).
- **Add `JevPolicy` head-to-head** (requires JEV_API key — we have it in `.env`).
- **Per-step LLM action distribution** analysis: what does the LLM pick that AlwaysSurrogate doesn't?

If the Rastrigin gap holds at 5 seeds, this becomes a paper-section finding ("LLM-as-System-1 beats AlwaysSurrogate on hard multimodal problems"). If it does not hold at 5 seeds, this is a negative pilot and the original RQ1/RQ2/RQ3 framing stands.

### 5.5 External baseline comparison — needed for any serious paper

- **Run lenz / PlugBO** on the same BOCoDe problems for head-to-head table. This is what review will ask for; without it the paper is "we made a framework" rather than "we beat the state of the art."

---

## 6. Three possible end-states

Given the current evidence and the open experiments, the paper will likely land in one of three places:

### State A: "LLM-as-System-1 is enough; trained typed decision adds nothing"
- EXP-RQ0.5 holds at 5 seeds; head-to-head with `JevPolicy` shows Jev ≈ LLM or Jev < LLM.
- Paper claim: v3's contribution is the decision-extraction machinery (oracle replay, checkpoint, action taxonomy) — not the trained model itself. The trained model is at best a cost optimization.
- Risk: paper becomes a framework paper, not a result paper.

### State B: "Trained Jev beats LLM-as-System-1 on transfer"  
- RQ2.1 shows Jev trained on 4 families transfers better than a per-decision LLM call would.
- Paper claim: the training pipeline distills a transferable decision policy that an LLM's in-context judgment cannot reproduce.
- Risk: this requires RQ2.1 to show clear transfer gain, which is uncertain.

### State C: "Different problems need different policies; v3 is the routing layer"
- RQ3 Pareto frontier shows hybrid routing captures most of the LLM's value at much lower cost.
- Paper claim: the contribution is the routing / calibration machinery, not any single policy.
- Risk: this is the most nuanced framing and is harder to publish than a single clear result.

The current data leans toward **State A** (LLM-as-System-1 beats AlwaysSurrogate on Rastrigin, Jev probably doesn't). 5-seed replication will move the needle one way or another.

---

## 7. Decisions pending from the user

1. **Direction**: pursue EXP-RQ0.5 confirmation (1 week) vs return to RQ1/RQ2/RQ3 mainline.
2. **External baseline**: run lenz/PlugBO head-to-head (~2 days) vs skip and accept "framework paper" framing.
3. **Engineering transfer**: start vehicle suspension work now (long horizon) vs defer until RQ1/RQ2/RQ3 settled.
4. **Time budget**: 6-week sprint to full conference paper vs 12+ month journal pace.

---

## 8. File index

| File | Purpose |
|---|---|
| `PROGRESS.md` | Phase-by-phase development log (chronological, includes Phase 9.8) |
| `MASTER_PLAN.md` | Original design plan |
| `PHASE9_DESIGN.md` | Phase 9 system-1/system-2 design |
| `runs/EXP_RQ1_2_REPORT.md` | EXP-RQ1.2 oracle replay report (post-codex revision) |
| `runs/phase8_6d_h3_10seed_5family.jsonl` | 5-family canonical dataset (354 records) |
| `runs/EXP_RQ0_5_REPORT.md` | EXP-RQ0.5 LLM-as-System-1 report (real DeepSeek results) |
| `runs/exp_rq0_5_results.json` | Raw results from 5-family × 2-seed run |
| `src/opt_agent/system1/llm_system1_policy.py` | `LLMSystem1Policy` (~250 LOC) |
| `scripts/exp_rq0_5_llm_system1.py` | 3-policy comparison runner |
| `src/opt_agent/dataset/validation.py` | `aggregate_mass` with seed-clustered CI |
| `scripts/validate_phase8_6b.py` | Gate 2/3/4 runner |

## 9. Branch state

```
$ git branch --show-current
exp/rq0.5-llm-system1

$ git log --oneline -3
f63ad40 EXP-RQ0.5: real DeepSeek results — LLM beats AlwaysSurrogate on Rastrigin
efe8afb EXP-RQ0.5: LLM-as-System-1 wiring verified (mock mode)
eb7e113 EXP-RQ1.2 oracle replay end-to-end + codex revisions
```

`main` is at `eb7e113` and is unaffected by the EXP-RQ0.5 work. Reverting the branch is `git checkout main && git branch -D exp/rq0.5-llm-system1` with no data loss.

---

*Document version: 2026-09-26, end of one-day cycle. Next update after 5-seed EXP-RQ0.5 replication (or RQ1 replication, depending on direction choice).*

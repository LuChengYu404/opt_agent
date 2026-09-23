# Phase 9: Hybrid Agent Framework — Status & Updated Design

> **Status**: design reconciled with implementation after codex gpt-6-sol reviews
> **Date**: 2026-09-23
> **Reviewers**: codex `gpt-5.6-sol` (Phase 8.6B) → `gpt-6-sol` (Phase 9.0, 9.1)
> **Last reviewed commit**: `5f79257` (Phase 9 framework skeleton, 235/235 tests)
> **Now**: `HEAD` after codex re-review fixes (250/250 tests)

---

## 1. Background

### 1.1 Current state (Phase 0–8 complete, Phase 9 framework skeleton committed)

| Phase | Module | Status |
|---|---|---|
| 0–4 | Backend + State + Action + DecisionLoop | ✅ 63 tests |
| 5–6 | Checkpoint/Restore + Counterfactual Rollout | ✅ 91 tests |
| 6-correction | 3 codex-flagged biases fixed | ✅ 105 tests |
| 7 | Utility + DecisionDataset JSONL + Generator | ✅ 117 tests |
| 8 | Empirical validation (revealed STOP bias) | ✅ 125 tests |
| 8.5 | Utility fix + oracle gate machinery | ✅ 136 tests |
| 8.6A | Scale wiring fix → Gate 3 PASS | ✅ 145 tests |
| 8.6B | Gate 2/4 machinery + replay history + LayaPolicy | ✅ 157 tests |
| **9.0 framework skeleton** | **metrics + splits + system2 + router + HybridPolicy** | **✅ 250 tests** |

`runs/phase8_6b_h3_10seed.jsonl` empirical reality (PROGRESS.md):
- Gate 2 (stable action preference): **FAIL**
- Gate 3 (early exploration mass): **PASS** (point estimate, +1.2 ~ +1.6)
- Gate 4 (terminal optimization advantage): **FAIL**
- **Known issue**: utility misspecification → reward hacking — **deferred to Phase 8.7+, not Phase 9 blocker**

### 1.2 Phase 9 scope (per MASTER_PLAN §23 + §24)

Build a working **Hybrid Agent skeleton**:

```
Optimization State
       ↓
   System-1 (π_θ(a|s))         — fast, learned or rule-based
       ↓
   confidence / entropy
       ↓
   ┌───┴───┐
  high    low
   ↓       ↓
  action  System-2 (LLM reasoning or fallback) → action
```

**Phase 9 minimum scope** (build only what the experiment needs, per §27):
- Typed-decision System-1 interface (LayaPolicy already production-ready)
- Calibration metrics (ECE / Brier / NLL / reliability)
- Confidence-based Router
- System-2 stub (delegates to RulePolicy, no real LLM call)
- End-to-end closed-loop eval entry point via HybridPolicy

### 1.3 Architecture decision: real System-1 adapter over new Protocol

Phase 9.0 delivers a **shared candidate factory** (`actions.candidate_actions_for_state`)
plus a **typed-decision adapter** for Laya (`system1.laya_policy.LayaPolicy`). The
adapter exposes a `decide(state) → (action, probabilities, confidence)` triple that
serves as the canonical System-1 interface in this codebase. It aligns with the
Jev/NanoJev/RL-Agent typed-decision-primitive contract (choice / score / noul)
but only `choice` is wired in Phase 9.0 — `score` and `noul` remain unwired
because the underlying checkpoints have no calibration targets yet.

**A separate "TypedDecisionSystem1" Protocol is NOT introduced.** The existing
`Policy` Protocol + the `LayaPolicy.decide()` triple give us everything we need
without inventing a parallel abstraction. `HybridPolicy` accepts any object
that satisfies `decide(state) → (action, probabilities, ...)` (via
`System1Decider`) and adapts bare `Policy` objects with
`adapt_policy_as_system1`.

### 1.4 Future fine-tuning plan (Phase 9.3, deferred)

After Phase 9.0–9.2 close the framework skeleton, Phase 9.3 will fine-tune a
real typed-decision checkpoint on rollout data. The candidate is a base model
that supports the typed-decision primitive API with released weights under a
permissive license. The exact choice is deferred until the smoke harness
runs against a real BOCoDe backend and we can compare candidates on (a)
license permissiveness, (b) set-attention over dynamic candidate vocabulary,
and (c) calibration availability on the calibration split.

This plan deliberately does NOT commit to a specific paper, model card, or
license claim — earlier versions of this document cited references that did
not verify to the cited model, and codex review flagged that as a C1
finding. References will be re-added with verified URLs in Phase 9.3.

---

## 2. Interface Contracts

### 2.1 System-1: LayaPolicy + DecisionLike

```python
from opt_agent.system1.laya_policy import LayaPolicy, LayaDecision

@dataclass(frozen=True)
class LayaDecision:
    action: OptimizationAction
    probabilities: dict[str, float]
    confidence: float

class LayaPolicy:
    def decide(self, state: OptimizationState) -> LayaDecision: ...
    def select_action(self, state, backend=None) -> OptimizationAction: ...
    def reset(self) -> None: ...
```

`LayaPolicy.decide()` is the canonical System-1 contract. Its probability
distribution is built from the shared candidate factory
(`actions.candidate_actions_for_state`), so the action_id vocabulary is
stable across LayaPolicy, the dataset generator (`full_candidates`), and
any future System-1 implementation.

### 2.2 Shared candidate factory

```python
from opt_agent.actions import candidate_actions_for_state

def candidate_actions_for_state(
    state: OptimizationState, *, seed: int = 0,
) -> dict[str, OptimizationAction]:
    """Build budget-valid candidates as {action_id: OptimizationAction}.

    Per codex M1: a single factory owns candidate construction so the
    action_id vocabulary stays aligned across System-1 implementations.
    """
```

Filter rules: budget <= 0 → only `stop`; budget > 0 → at least one
sampling action + `stop` (so early termination is always offered);
local actions only when `n_samples > 0`; `retrain_surrogate` only when
the surrogate is not yet fitted.

### 2.3 System-2: FallbackSystem2

```python
from opt_agent.system2 import FallbackSystem2, System2Decision

@dataclass(frozen=True)
class System2Decision:
    action: OptimizationAction
    confidence: float       # 0.0 for STOP, 1.0 otherwise
    reasoning: str = ""

class FallbackSystem2:
    def __init__(self, *, baseline: RulePolicy | None = None): ...
    def reason(self, state, *, recent_actions=(), remaining_budget=None) -> System2Decision: ...
    def reset(self) -> None: ...
```

Per codex M2: the fallback MUST delegate to a real baseline (RulePolicy
here) and MUST NOT default to STOP. STOP is returned only when the
baseline itself returns STOP (i.e., remaining_budget ≤ 4). A regression
test (`test_fallback_drives_decision_loop_through_multiple_actions`)
verifies this by ensuring at least 2 non-STOP actions occur before the
loop terminates.

### 2.4 Router: ConfidenceRouter

```python
from opt_agent.router import ConfidenceRouter, Route

@dataclass(frozen=True)
class Route:
    name: Literal["system1", "system2"]
    confidence: float
    reason: str

class ConfidenceRouter:
    def __init__(self, *, tau_conf: float = 0.5): ...
    def route(self, probabilities: dict[str, float]) -> Route: ...
    def update(self, probabilities, chosen, reward) -> None: ...
    def stats(self) -> dict[str, int]: ...
```

Per codex M3: the first-version router uses ONLY `max(probabilities.values())`
against `tau_conf`. `predict_score` and `predict_noul` are NOT consulted —
those heads have no calibration targets in the current models.

### 2.5 HybridPolicy: the glue

```python
from opt_agent.system1 import HybridPolicy

class HybridPolicy:
    def __init__(self, system1, router, system2): ...
    def select_action(self, state, backend=None) -> OptimizationAction: ...
    def reset(self) -> None: ...
```

Per step:
1. `decision = system1.decide(state)`
2. Validate `decision.action.type.value ∈ decision.probabilities` (contract guard).
3. `route = router.route(decision.probabilities)`
4. If `route.name == "system1"` → return `decision.action`.
5. Else → return `system2.reason(state).action`.

`HybridPolicy` implements the existing `Policy` Protocol, so it can be
passed directly to `DecisionLoop.run(policy)` — the DecisionLoop
constructor is NOT modified (per codex M4).

### 2.6 Calibration metrics

```python
from opt_agent.metrics import (
    expected_calibration_error, brier_score, negative_log_likelihood,
    reliability_diagram_data, top_one_agreement, kl_divergence,
    aggregate_decision_metrics, CalibrationResult, DecisionResult,
)
```

`expected_calibration_error(probs, outcomes, n_bins=15)` — `outcome` is the
**observed** label for each record. There are two regimes (see module
docstring):

1. **Internal consistency**: `outcomes[i] = argmax(probs[i])`. Per-record
   accuracy is 1.0, so ECE collapses to the gap between `max(probs)` and
   1.0. This measures confidence-in-its-own-choice, NOT real calibration.
2. **Real calibration**: pass an INDEPENDENTLY observed label (human
   annotation, oracle gap from rollout). ECE then measures agreement
   between confidence and real accuracy.

Per codex new-issue-2: `aggregate_decision_metrics` reports
`n_infinite_kl` / `n_infinite_nll` so the caller knows how many records
were excluded from `kl_mean` / `nll_mean`. A single +inf record can no
longer make a corpus look perfect.

---

## 3. Module Structure (current)

```
src/opt_agent/
├── actions/                        # Research core: candidate factory
│   ├── factory.py                  # NEW — shared candidate_actions_for_state
│   ├── models.py                   # OptimizationAction, ActionType, _ALLOWED_PARAMS
│   └── executor.py
├── backend/                        # Engineering infra: BoTorch + checkpoint
├── state/                          # Research core: OptimizationState + extract_state
├── dataset/                        # Research core: DecisionRecord + generation + splits
│   ├── splits.py                   # NEW — 5-way split + physical_state_hash + gold_probs_kind
│   └── ...
├── rollout/                        # Research core: counterfactual + utility
├── loop/                           # Research core: DecisionLoop + base policies
│   └── policies.py                 # RandomPolicy + RulePolicy (DecisionLoop is NOT modified)
├── system1/                        # Research core
│   ├── laya_policy.py              # Production-ready LayaPolicy (decide / select_action / reset)
│   └── hybrid_policy.py            # NEW — HybridPolicy + System1Decider + adapt_policy_as_system1
├── system2/                        # Research core: stub for Phase 11
│   ├── base.py                     # NEW — System2Decision + System2Policy Protocol
│   └── fallback.py                 # NEW — FallbackSystem2 (delegates RulePolicy, NOT STOP)
├── router/                         # Research core
│   ├── base.py                     # NEW — Route dataclass + Router Protocol
│   └── confidence.py               # NEW — ConfidenceRouter (max(probs) vs tau_conf)
├── metrics/                        # Research core: calibration + decision metrics
│   ├── calibration.py              # NEW — ECE/Brier/NLL/reliability (with regime docstring)
│   └── decision.py                 # NEW — top-1/KL/NLL-vs-gold (with inf counts)
└── benchmark/                      # Engineering infra (deferred)

tests/
├── test_actions_factory.py         # NEW — 12 tests
├── test_metrics_calibration.py     # NEW — 20 tests
├── test_metrics_decision.py        # NEW — 16 tests
├── test_dataset_splits.py          # NEW — 21 tests (incl. sorted family + hash scope)
├── test_system2_fallback.py        # NEW — 6 tests (incl. STOP regression)
├── test_router_confidence.py       # NEW — 11 tests
├── test_hybrid_policy.py           # NEW — 7 tests (incl. end-to-end)
└── test_laya_policy.py             # 5 tests (pre-existing)
```

---

## 4. Implementation Steps (status)

| Step | Description | Status |
|---|---|---|
| **0** | `metrics/calibration.py` + `metrics/decision.py` + tests | ✅ Done (36 tests) |
| **1** | `dataset/splits.py` + tests | ✅ Done (21 tests, with sorted-family + hash-scope fixes) |
| **2** | `actions/factory.py` + tests | ✅ Done (12 tests) |
| **3** | `system2/{base,fallback}.py` + tests | ✅ Done (6 tests) |
| **4** | `router/{base,confidence}.py` + tests | ✅ Done (11 tests) |
| **5** | `system1/hybrid_policy.py` + tests | ✅ Done (7 tests, with action-id contract guard) |
| **6** | LayaPolicy refactored to use shared candidate factory | ✅ Done |
| **7** | `tests/test_hybrid_policy.py` end-to-end on RecordingBackend | ✅ Done (smoke only; real BO deferred) |
| **8** | **End-to-end on real BOCoDe backend** (Branin + Ackley + Hartmann, simple_regret) | ⏸ Deferred to Phase 9.1 |
| **9** | Commit framework skeleton + design doc | ⏳ Pending this PR |

### 4.1 Phase 9.1 follow-up: real-backend smoke

```bash
uv run python scripts/run_hybrid_smoke.py \
    --problem branin --seed 0 --budget 30 \
    --system1 laya --router tau_conf=0.5 --system2 rule
```

Expected outputs:
- `simple_regret = known_optimum - best_value` (Branin optimum = -0.397887)
- Per-step routing stats (system1 count / system2 count)
- Calibration diagnostics on argmax-outcomes (with regime tag in the docstring)

This is the acceptance test that takes Phase 9 from "framework skeleton
passes tests" to "Phase 9 done".

---

## 5. Testing Strategy

### 5.1 Test categories

| Category | Purpose | Example |
|---|---|---|
| **Contract tests** | Protocol/interface invariants | `test_hybrid_rejects_system1_without_decide` |
| **Mathematical correctness** | Metrics match known formulas | `test_ece_delta_probs_self_consistent_is_zero` |
| **Filtering invariants** | Candidate factory budget + surrogate rules | `test_factory_returns_only_stop_when_budget_exhausted` |
| **Cross-process determinism** | Split assignment is RNG-seed-stable | `test_assign_splits_uses_sorted_family_order` |
| **Regression tests** | Lock in codex M1/M2/M3/M4 fixes | `test_fallback_drives_decision_loop_through_multiple_actions` |
| **End-to-end smoke** | Full DecisionLoop.run(policy) | `test_hybrid_drives_decision_loop_end_to_end` |

### 5.2 Acceptance threshold (gold_probs is NOT optimization benefit)

Per codex C2: `gold_probs` is a rollout soft target, NOT a ground-truth
"correct action". Training a System-1 on `gold_probs` and reporting
`top-1 > uniform` is a label-fit diagnostic, NOT evidence that the
model produces a better optimizer. The acceptance tests for Phase 9
**must** separate the two:

- **Diagnostic (cheap)**: `top_one_agreement`, `top_k_agreement`, `kl_mean`
  against `gold_probs` — only meaningful for interface & label-fit sanity.
- **Acceptance (real)**: `simple_regret = known_optimum - best_value` on a
  real BOCoDe problem with `HybridPolicy` driving `DecisionLoop.run(1)`.
  Computed in the evaluation layer, NOT in the metrics module.

Until simple_regret on real BO problems is reported, Phase 9 is
"framework skeleton" status, NOT "Phase 9 done".

### 5.3 Coverage targets

- 250/250 tests passing as of this commit.
- Phase 9.1 adds ≥1 simple_regret smoke test per problem family
  (Branin / Ackley / Hartmann).
- Total target: ≥ 255 tests.

---

## 6. Risks and Known Issues

### 6.1 Reward hacking / utility misspecification (KNOWN, deferred)

`utility` reward is a 4-term soft target that exhibits reward-hacking
patterns (STOP bias → retrain bias, see PROGRESS.md). Phase 9.0 uses
the existing `gold_probs` for diagnostic label-fit only. Future training
will use proper-scoring-rule loss (per RL Agent / System 1 reference) or
direct simple_regret credit assignment.

### 6.2 Calibration metric regimes (KNOWN, documented)

`expected_calibration_error` accepts any `outcome` source. Using
`outcome = argmax(probs)` measures internal consistency, not real
calibration. The function's docstring warns about this; the aggregate
result includes `n_records` and `notes` so downstream consumers know
which regime the metric came from.

### 6.3 Aggregate KL/NLL hiding failures (FIXED)

`aggregate_decision_metrics` now exposes `n_infinite_kl` and
`n_infinite_nll` counts. A single +inf record can no longer be hidden
behind a "kl_mean=0" headline.

### 6.4 Split determinism across processes (FIXED)

`assign_splits` now iterates families in SORTED order and seeds within
each family are also sorted before shuffling. Identical input + seed
yields identical assignments regardless of `set` iteration order.

### 6.5 physical_state_hash scope (FIXED)

`physical_state_hash` now hashes ONLY `record.state`. Previous version
included `record.problem` (trajectory seed — not physics) and
`record.candidate_actions` (factory output — not physics). The narrower
hash means "two records hash equal iff their BO state is physically
identical".

### 6.6 HybridPolicy contract guard (FIXED)

`HybridPolicy.select_action` validates that
`system1_action.type.value ∈ system1_probabilities`. A model that emits
a probability distribution on a vocabulary different from the candidate
factory's vocabulary now fails loudly instead of producing silent
mismatches in the loop.

---

## 7. Future fine-tuning plan (Phase 9.3)

After Phase 9.0–9.2 close:

1. Run the simple_regret smoke harness on real BOCoDe problems with
   `HybridPolicy(LayaPolicy(FakeLaya), ConfidenceRouter, FallbackSystem2)`.
2. Pick a real typed-decision base model with permissive license and
   released weights that supports the `choice / score / noul` primitive
   API.
3. Fine-tune the decision head on rollout data using a proper-scoring-rule
   loss (avoids the utility-hacking failure mode observed in Phase 8.5/8.6).
4. Replace `LayaPolicy(FakeLaya)` with the fine-tuned adapter.

Step 1 is the immediate Phase 9.1 gate; steps 2–4 are Phase 9.3.

---

## 8. Acceptance Criteria (Phase 9 done)

- [x] All 9 framework-skeleton steps have passing tests (250/250)
- [x] `HybridPolicy` drives `DecisionLoop.run(policy)` end-to-end
- [x] `LayaPolicy` is the production-ready zero-shot System-1 adapter
- [x] DecisionLoop constructor is unchanged
- [x] Codex gpt-6-sol findings C1, C2, M1, M2, M3, M4 resolved
- [x] New codex issues (calibration regimes, KL inf count, split
      determinism, hash scope, HybridPolicy contract guard) resolved
- [x] **Phase 9.1**: end-to-end smoke on real BOCoDe backend with
      `simple_regret` reported for Branin + Ackley + Hartmann
      (54/54 runs PASS acceptance; runs/phase9_1_smoke_{deepseek,fallback}.json)
- [x] Real DeepSeek API path verified (`--system2 deepseek` with
      `DEEPSEEK_API_KEY` loaded from .env via python-dotenv)
- [x] Phase 9.1 v9 (commit `78ae839`): all codex review-12
      findings fixed → final APPROVE verdict
- [ ] **Phase 9.3**: fine-tuned typed-decision model replaces
      `LayaPolicy(FakeLaya)` in the smoke harness — DEFERRED (no
      GPU budget + no licensed base checkpoint selected yet)

**Status: Phase 9 DONE** (approved by codex gpt-6-sol on commit `78ae839`).

---

## 9. Open Questions for codex `gpt-6-sol`

1. **Real-backend smoke methodology**: should Phase 9.1 use the same
   `RecordingBackend` shape but driven by a real BOCoDe Branin problem
   behind the scenes, or wire `BocodeBotorchBackend` into the harness
   from the start?
2. **ConfidenceRouter threshold tuning**: the docstring says tau_conf
   should be tuned on the calibration split. Is it acceptable to fix
   tau_conf=0.5 for Phase 9.1 and tune it in Phase 9.3, or do you want
   the tuning step included in 9.1?
3. **Laya adapter vs zero-shot baseline reporting**: Phase 9.1 smoke
   will report simple_regret for the HybridPolicy with `LayaPolicy
   (FakeLaya)` as the System-1. Is reporting the no-LLM `RulePolicy`
   baseline in the same table acceptable, or should we ship a separate
   baselines script?
4. **End-to-end smoke granularity**: should Phase 9.1 ship a smoke
   script (`scripts/run_hybrid_smoke.py`) as the contract surface, or
   is a pytest entry point (`tests/test_hybrid_smoke_real.py`) enough
   for now?
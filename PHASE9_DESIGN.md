# Phase 9: Hybrid Agent Framework Design

> **Status**: PLAN for external review
> **Date**: 2026-09-23
> **Reviewer**: codex `gpt-6-sol` (upgraded from `gpt-5.6-sol`)
> **Scope**: Phase 9 System-1 + Calibration + Router + System-2 stub
> **Reference**: [MASTER_PLAN.md §24](MASTER_PLAN.md) Hybrid Agent; §20 software architecture

---

## 1. Background

### 1.1 Current state (Phase 0–8 complete, Phase 8.6B pending commit)

| Phase | Module | Status |
|---|---|---|
| 0–4 | Backend + State + Action + DecisionLoop | ✅ 63 tests |
| 5–6 | Checkpoint/Restore + Counterfactual Rollout | ✅ 91 tests |
| 6-correction | 3 codex-flagged biases fixed | ✅ 105 tests |
| 7 | Utility + DecisionDataset JSONL + Generator | ✅ 117 tests |
| 8 | Empirical validation (revealed STOP bias) | ✅ 125 tests |
| 8.5 | Utility fix + oracle gate machinery | ✅ 136 tests |
| **8.6A** | **Scale wiring fix → Gate 3 PASS (+1.2 ~ +1.6 on 10-seed)** | **✅ 145 tests** |
| **8.6B** | **Gate 2/4 machinery + replay history metadata** | **✅ 7 tests, uncommitted** |

`runs/phase8_6b_h3_10seed.jsonl` empirical reality (per PROGRESS.md):
- Gate 2 (stable action preference): FAIL
- Gate 3 (early exploration mass): PASS (point estimate)
- Gate 4 (terminal optimization advantage): FAIL
- **Known issue**: utility misspecification → reward hacking (STOP bias → retrain bias) — **deferred to Phase 8.7+, not Phase 9 blocker**

### 1.2 Phase 9 goal (per MASTER_PLAN §23 + §24)

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
- Typed-decision System-1 interface + ≥1 working implementation
- Calibration metrics (ECE, Brier, NLL, reliability)
- Confidence-based Router
- System-2 stub (no real LLM call yet)
- End-to-end closed-loop eval entry point

### 1.3 Why typed decision primitives (the unifying contract)

Three reference implementations share one interface pattern — **typed decision primitives**:

| Primitive | Input | Output | Maps to our setting |
|---|---|---|---|
| `choice` | K candidates (K ∈ [2, 255]) | `{candidate: prob}` full distribution | **Select action** (5–7 actions) |
| `score` | 2–10 ordered levels | `{level: prob}` + expected value | **Confidence / urgency** |
| `noul` | boolean question | single P(true) | **Should escalate to System-2?** |

Reference implementations:
- **Jev** (TypeSafe, commercial API)
- **NanoJev** (TianyuCodings, MIT, Qwen3-0.6B backbone)
- **RL Agent / System 1** (arXiv:2503.23303, 2510.01237, Apache 2.0, ModernBERT-large backbone)

Writing our `TypedDecisionSystem1` Protocol **once** lets us swap any implementation later without refactoring.

### 1.4 Reference architecture choice: RL Agent / System 1

| Criterion | RL Agent | NanoJev | Jev API |
|---|---|---|---|
| License | **Apache 2.0** | MIT | Commercial |
| Fine-tunable | **Yes (base weights released)** | Yes (train from scratch) | **No** |
| Training paradigm | **RLCD (proper scoring rules)** | Supervised CE on gold_probs | N/A |
| Calibration built-in | **Yes (proper scoring rules enforce)** | Depends on training data | Black box |
| Backbone size | **421M (ModernBERT-large)** | 0.6B (Qwen3) | Unknown |
| Latency | **~35ms** | ~50–100ms | API latency |
| Reward-hack resistance | **High** (no dependency on utility proxy) | Low (CE optimizes gold_probs which is utility-derived) | N/A |

**Decision**: design against `RL Agent / System 1` as reference architecture, with `MockTypedSystem1` as Phase 9.1 placeholder. NanoJev remains as fallback if RLCD training proves unstable.

---

## 2. Interface Contracts

### 2.1 `TypedDecisionSystem1` Protocol

```python
from typing import Protocol
from dataclasses import dataclass

@dataclass(frozen=True)
class StateFeatures:
    """Canonical state encoding passed to System-1.
    Sub-states per MASTER_PLAN §7: sampling / surrogate / optimization / resource / history.
    """
    sampling: dict[str, float]
    surrogate: dict[str, float]
    optimization: dict[str, float]
    resource: dict[str, float]
    history: dict[str, float]
    raw: dict                       # original JSON state for free-form fields

class TypedDecisionSystem1(Protocol):
    """Reference contract: RL Agent / System 1 typed decision primitives.
    All outputs are FULL distributions, never argmax-only.
    """

    def predict_choice(
        self,
        state: StateFeatures,
        candidates: list[str],
    ) -> dict[str, float]:
        """Choice type: K-way probability distribution over candidates.

        Contract:
          - All probabilities ≥ 0
          - Probabilities sum to 1.0 within tolerance
          - Returns one entry per candidate in `candidates`
          - Candidates must be 2 ≤ K ≤ 255 (matches NanoJev/RL Agent limits)
        """
        ...

    def predict_score(
        self,
        state: StateFeatures,
        legend: dict[int, str],
    ) -> dict[int, float]:
        """Score type: probability distribution over ordered rubric levels.

        Contract:
          - Returns one entry per key in `legend`
          - Probabilities sum to 1.0
          - Returns expectation (float) via `predict_score_expectation`
        """
        ...

    def predict_score_expectation(
        self,
        state: StateFeatures,
        legend: dict[int, str],
    ) -> float:
        """Expected value E[level] under predict_score distribution."""
        ...

    def predict_noul(
        self,
        state: StateFeatures,
        question: str,
    ) -> float:
        """Noul type: P(true) for a boolean question.
        Used for: 'Should we escalate to System-2?'
        """
        ...
```

### 2.2 `System2Policy` Protocol (stub)

```python
class System2Policy(Protocol):
    """High-level reasoning fallback. Phase 9 ships as stub only."""

    def reason(
        self,
        state: StateFeatures,
        recent_history: list[str],
        candidates: list[str],
        remaining_budget: int,
    ) -> tuple[str, dict, float]:
        """Return (action_name, action_params, confidence).

        Phase 9.1: FallbackSystem2 returns ("stop", {}, 1.0) by default.
        Phase 11: replaced by Tau/LLM harness.
        """
        ...
```

### 2.3 `Router` Protocol

```python
class Router(Protocol):
    """Decides whether to use System-1 (fast) or System-2 (deep)."""

    def route(
        self,
        state: StateFeatures,
        system1_output: dict,            # TypedDecisionSystem1 outputs
    ) -> tuple[str, float]:             # ("system1" | "system2", confidence)
        ...

    def update(self, state: StateFeatures, action: str, reward: float) -> None:
        """Optional online learning of routing thresholds."""
        ...
```

Phase 9 ships `ConfidenceRouter`: `route` based on `predict_score` expectation (low confidence → System-2) and `predict_noul` (high escalate probability → System-2).

### 2.4 Calibration metrics

```python
def expected_calibration_error(
    probs: list[dict[str, float]],
    outcomes: list[str],
    n_bins: int = 15,
) -> float:
    """ECE: weighted avg |bin_confidence - bin_accuracy|."""
    ...

def brier_score(
    probs: dict[str, float],
    outcome: str,
    candidates: list[str],
) -> float:
    """Multi-class Brier: sum over candidates of (p_i - 1[outcome=i])^2."""
    ...

def negative_log_likelihood(
    probs: dict[str, float],
    outcome: str,
) -> float:
    """NLL = -log P(outcome)."""
    ...

def reliability_diagram_data(
    probs: list[dict[str, float]],
    outcomes: list[str],
    n_bins: int = 15,
) -> list[tuple[float, float, int]]:
    """Returns [(bin_confidence, bin_accuracy, bin_count), ...] for plotting."""
    ...
```

Per MASTER_PLAN §13: also need `C(s) = max_a P(a|s)` (confidence) and `H(s) = -Σ P(a|s) log P(a|s)` (entropy). These are derived utilities, not separate metrics.

---

## 3. Module Structure

### 3.1 New files

```
src/opt_agent/
├── system1/
│   ├── __init__.py
│   ├── typed_decision.py        # TypedDecisionSystem1 Protocol + StateFeatures
│   ├── mock_typed.py            # MockTypedSystem1 — uniform/zero baseline
│   ├── mlp_typed.py             # MLPTypedSystem1 — adapter from phase9_baselines.py
│   └── rl_agent.py              # Phase 9.3 only: fine-tuned RL Agent adapter
├── system2/
│   ├── __init__.py
│   ├── base.py                  # System2Policy Protocol
│   └── fallback.py              # FallbackSystem2 — returns ("stop", {}, 1.0)
├── router/
│   ├── __init__.py
│   ├── base.py                  # Router Protocol
│   └── confidence.py            # ConfidenceRouter — uses System-1 confidence/entropy/noul
└── metrics/
    ├── __init__.py
    ├── calibration.py           # ECE / Brier / NLL / reliability
    └── decision.py              # top-1, top-k, KL vs gold_probs

src/opt_agent/dataset/
└── splits.py                    # 5-way split: train / dev / calibration / test / ood
                                  # also: gold_probs_kind, physical_state_hash fields

src/opt_agent/loop/
└── decision_loop.py             # UPDATED: accepts (system1, router, system2) trio

tests/
├── test_system1_typed.py        # Contract tests for Protocol
├── test_system1_mock.py         # Mock implementation behavior
├── test_system1_mlp.py          # MLP training smoke
├── test_system2_fallback.py     # Stub behavior
├── test_router_confidence.py    # Routing thresholds
├── test_metrics_calibration.py  # ECE/Brier/NLL mathematical correctness
├── test_dataset_splits.py       # 5-way split integrity + hash uniqueness
└── test_decision_loop_hybrid.py # End-to-end: state → system1 → router → action
```

### 3.2 Module boundaries (research core vs engineering infrastructure)

Per MASTER_PLAN §28–29:

| Module | Boundary | Justification |
|---|---|---|
| `system1/` | **Research core** | Decision supervision + policy architecture |
| `system2/` | Research core | Reasoning strategy |
| `router/` | Research core | Selective reasoning (RQ3) |
| `metrics/calibration.py` | Research core | Confidence calibration is §13's RQ3 dependency |
| `dataset/splits.py` | Research core | OOD split is RQ2's transfer test |
| `loop/decision_loop.py` (update) | Research core | Closed-loop eval entry |
| `system1/rl_agent.py` (Phase 9.3) | Engineering infra (wrap) | Wrap external checkpoint |

---

## 4. Implementation Steps (priority order)

Each step ends with `pytest -q` passing and a commit.

### Step 1 — `metrics/calibration.py` + `tests/test_metrics_calibration.py`
- Pure functions, no I/O
- ECE / Brier / NLL / reliability + small synthetic tests
- Independent of everything else → unblocks router + system1 work
- ~50 LOC + ~80 LOC tests

### Step 2 — `dataset/splits.py` + `tests/test_dataset_splits.py`
- 5-way split (train / dev / calibration / test / ood)
- `physical_state_hash` (SHA256 over canonical state JSON)
- `gold_probs_kind` provenance field
- OOD split: by `problem.family` (held-out problem family)
- Wire into `dataset/generator.py` if not already
- ~80 LOC + ~100 LOC tests

### Step 3 — `system1/typed_decision.py` (Protocol + StateFeatures)
- Pure Protocol, no logic
- `StateFeatures` dataclass to canonicalize state extraction
- ~40 LOC + ~50 LOC tests

### Step 4 — `system1/mock_typed.py` + `tests/test_system1_mock.py`
- Uniform choice, mid-level score, neutral noul
- Deterministic via injected seed
- Verify Protocol contract compliance (probabilities sum to 1, etc.)
- ~50 LOC + ~80 LOC tests

### Step 5 — `system1/mlp_typed.py` + `tests/test_system1_mlp.py`
- Adapter wrapping `phase9_baselines.py` MLP
- `train(jsonl_path)` + `save(path)` + `load(path)`
- Smoke: train on Branin-only 10-seed dataset, verify top-1 agreement > uniform
- ~120 LOC + ~120 LOC tests

### Step 6 — `system2/base.py` + `system2/fallback.py` + `tests/test_system2_fallback.py`
- `FallbackSystem2` returns `("stop", {}, 1.0)`
- Phase 11 will swap in Tau harness — placeholder interface only
- ~30 LOC + ~30 LOC tests

### Step 7 — `router/confidence.py` + `tests/test_router_confidence.py`
- Threshold-based: `route` based on `predict_score_expectation < τ_conf` OR `predict_noul > τ_escalate`
- Configurable thresholds (not hard-coded)
- ~40 LOC + ~60 LOC tests

### Step 8 — `loop/decision_loop.py` upgrade + `tests/test_decision_loop_hybrid.py`
- `DecisionLoop(system1, router, system2)` constructor
- Per step: extract state → `system1.predict_*` → `router.route` → action
- Closed-loop run: returns `RunResult(state_trajectory, action_trajectory, simple_regret, calibration_stats)`
- **End-to-end smoke**: Branin 10-seed → 1 trajectory → assert `len(actions) > 0` and `simple_regret < 1.0` (well-defined vs known optimum -0.397887)
- ~80 LOC change + ~100 LOC tests

### Step 9 — Integration: commit Phase 8.6B + Phase 9 skeleton
- Two separate commits per work-flow rule
- Phase 8.6B first (already-tested)
- Phase 9 skeleton second

---

## 5. Testing Strategy

### 5.1 Test categories

| Category | Purpose | Example |
|---|---|---|
| **Contract tests** | Any `TypedDecisionSystem1` impl must satisfy | `test_predict_choice_probs_sum_to_one` (parametrized over impls) |
| **Mathematical correctness** | Calibration metrics match known formulas | `test_ece_uniform_distribution` (should be ~0) |
| **Smoke training** | MLP trains without NaN, top-1 agreement > uniform | `test_mlp_top1_above_uniform` |
| **Integration** | Full pipeline state → action works | `test_decision_loop_branin_smoke` |
| **Determinism** | Same seed → same output | `test_mlp_predict_deterministic` |

### 5.2 Coverage targets

- All Protocol methods: 100% covered
- Calibration metrics: edge cases (K=2, K=255, uniform, degenerate)
- Closed-loop test: 1 problem × 1 seed smoke, not full benchmark
- All Phase 9 tests must pass before any commit

### 5.3 Non-goals for Phase 9

- ❌ Real LLM System-2 (Phase 11)
- ❌ Engineering problem transfer (Phase 14)
- ❌ OOD detector sophistication (basic hash-based OOD split only)
- ❌ Multi-agent or multi-policy ensemble
- ❌ Online learning / production deployment

---

## 6. Risks and Known Issues

### 6.1 Reward hacking / utility misspecification (KNOWN, deferred)

**Issue**: utility misspecification (Phase 8.5 STOP bias → Phase 8.6A retrain bias) is a real research problem documented in PROGRESS.md. Naive CE training on `gold_probs` would propagate this bias to System-1.

**Mitigation**: Phase 9 System-1 design separates **reward source** from **decision model**:
- `MLPTypedSystem1.train()` accepts `reward_source: Literal["gold_probs", "simple_regret", "rlcd_proper_score"]`
- Default `reward_source="simple_regret"` once simple_regret is exposed in `RunResult` (Step 8)
- RLCD path (Phase 9.3) is reward-source-agnostic by design

**Status**: Known issue, NOT Phase 9 blocker. Phase 8.7+ work.

### 6.2 RL Agent base checkpoint assumptions

**Risk**: RL Agent was trained on human-labeled data (support triage, banking intents, etc.) — fine-tuning is necessary but training stability is unverified for our domain.

**Mitigation**: Phase 9.3 (RL Agent fine-tuning) gated on Phase 9.1–9.2 success; NanoJev fallback ready.

### 6.3 DecisionLoop backward compatibility

**Risk**: Updating `DecisionLoop` constructor may break Phase 4–6 tests.

**Mitigation**: Add `DecisionLoop` overload that accepts `policy` (legacy) OR `(system1, router, system2)` (new). Legacy path preserved.

### 6.4 Dataset split schema migration

**Risk**: Adding `split` / `gold_probs_kind` / `physical_state_hash` to `DecisionRecord` may invalidate existing JSONL files.

**Mitigation**: New fields are optional (default None / empty); existing JSONL loads correctly. Validation warns but does not fail.

---

## 7. Phase 9.3 Follow-up (RL Agent fine-tuning)

After Phase 9.1–9.2 skeleton works:

1. **Download RL Agent base checkpoint** from HuggingFace (Apache 2.0)
2. **Adapt `system1/rl_agent.py`**:
   - Wrap ModernBERT backbone + decision heads
   - State encoding: serialize `StateFeatures` to JSON, embed via ModernBERT tokenizer
   - Set attention over candidate [MASK] tokens
3. **Training loop**:
   - Loss: `RLCD(policy_gradient + proper_scoring_rules)` — uses simple regret per closed-loop run
   - Two-LR: backbone 1e-5, decision head 1e-4
   - Calibration: ECE / Brier / NLL on held-out calibration split
4. **Evaluation**:
   - Closed-loop: Branin + Ackley + Hartmann × 10 seeds × 3 horizons
   - Comparison: `MockTypedSystem1` baseline + `MLPTypedSystem1` + `RLAgentTypedSystem1`
   - Metrics: simple regret, ECE, NLL, LLM call rate (System-2 usage %)

---

## 8. Acceptance Criteria (Phase 9 done = all of these)

- [ ] All 9 implementation steps have passing tests
- [ ] `tests/test_decision_loop_hybrid.py` runs end-to-end on Branin 10-seed
- [ ] `MockTypedSystem1` and `MLPTypedSystem1` both satisfy `TypedDecisionSystem1` Protocol (verified by contract tests)
- [ ] DecisionLoop backward-compatible (Phase 4–6 tests still pass)
- [ ] Phase 8.6B committed separately
- [ ] Total test count: ≥ 165 (current 151 + ~14 new Phase 9 tests)
- [ ] External review by codex `gpt-6-sol` signed off
- [ ] PROGRESS.md updated with Phase 9 status

---

## 9. Open Questions for codex `gpt-6-sol`

1. **Is the Protocol interface set complete?** Should we add `predict_score_with_legend_validation` or similar safety wrappers?
2. **Mock vs real System-1 split**: do we want a separate `BaseTypedSystem1` ABC, or is duck-typing via Protocol enough?
3. **Calibration bin count**: 15 is the standard. Should we support per-record dynamic binning?
4. **Dataset split ratio**: train/dev/calibration/test/ood — any standard ratio (e.g., 60/10/10/10/10) or should we adopt NanoJev's exact split sizes?
5. **DecisionLoop return type**: `RunResult` with `simple_regret` field — is `simple_regret = known_optimum - best_value` (we know Branin optimum is -0.397887) the right semantics?
6. **Reward source default**: should `MLPTypedSystem1.train()` default to `"simple_regret"` (requires Phase 9.2 closed-loop runs) or `"gold_probs"` (works immediately but inherits hack)?
7. **Router threshold initial values**: what's a reasonable starting point for `τ_conf` and `τ_escalate`?

---

**End of plan. Awaiting codex `gpt-6-sol` review.**

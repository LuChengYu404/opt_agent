"""Phase 6 acceptance tests for CounterfactualRollout and to_gold_probs.

Per codex (gpt-5.6-sol) review 2025-09-23:
  - Same action types with different params don't collide in results dict.
  - Candidate order invariance.
  - Backend restored to base after every rollout.
  - Restoration after exceptions.
  - Atomic restore failures in nested checkpoint restore.
  - to_gold_probs rejects invalid temperature / empty results / invalid-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import RecordingBackend  # noqa: E402

from opt_agent.actions import ActionType, OptimizationAction  # noqa: E402
from opt_agent.backend.bocode_botorch import (  # noqa: E402
    BackendConfig,
    BocodeBotorchBackend,
)
from opt_agent.rollout import (  # noqa: E402
    Candidate,
    CounterfactualRollout,
    to_gold_probs,
)
from opt_agent.state import extract_state  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidate(action_id: str, action_type: ActionType, **params) -> Candidate:
    return Candidate(action_id, OptimizationAction(action_type, params))


def _brannin_backend(seed: int = 0, n_init: int = 5) -> BocodeBotorchBackend:
    be = BocodeBotorchBackend()
    be.initialize(BackendConfig(problem_id="Branin", seed=seed, n_init=n_init))
    return be


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------


def test_rollout_returns_one_result_per_candidate() -> None:
    be = RecordingBackend()
    state = extract_state(be, total_budget=20)
    candidates = [
        _candidate("g2", ActionType.GLOBAL_EXPLORATION, n_samples=2),
        _candidate("stop", ActionType.STOP),
        _candidate("retrain", ActionType.RETRAIN_SURROGATE),
    ]
    out = CounterfactualRollout(be, horizon=1).rollout(state, candidates)
    assert len(out.results) == 3
    assert [r.action_id for r in out.results] == ["g2", "stop", "retrain"]


def test_all_candidates_produce_finite_utilities() -> None:
    be = RecordingBackend()
    state = extract_state(be, total_budget=20)
    candidates = [
        _candidate("g2", ActionType.GLOBAL_EXPLORATION, n_samples=2),
        _candidate("g1", ActionType.GLOBAL_EXPLORATION, n_samples=1),
        _candidate("stop", ActionType.STOP),
    ]
    out = CounterfactualRollout(be, horizon=1).rollout(state, candidates)
    for r in out.results:
        assert r.status == "valid"
        assert isinstance(r.utility, float)
        assert r.utility == r.utility  # not NaN


def test_state_unchanged_after_rollout() -> None:
    be = RecordingBackend()
    state_before = extract_state(be, total_budget=20)
    history_before = be.get_history()

    candidates = [
        _candidate("g2", ActionType.GLOBAL_EXPLORATION, n_samples=2),
        _candidate("local", ActionType.LOCAL_EXPLOITATION, step=0.05),
    ]
    CounterfactualRollout(be, horizon=1).rollout(state_before, candidates)

    # Backend history unchanged.
    history_after = be.get_history()
    assert torch.equal(history_before.X, history_after.X)
    assert torch.equal(history_before.objectives, history_after.objectives)
    assert len(history_after.X) == 2


def test_rollout_restores_backend_when_candidate_raises() -> None:
    """If a candidate raises (e.g. invalid params) during execute_action,
    other candidates should still get evaluated, and backend is restored."""
    be = _brannin_backend()
    state = extract_state(be, total_budget=20)
    history_before = be.get_history()

    # Build candidates where one will raise: GLOBAL_EXPLORATION with n_samples
    # exceeding remaining budget → ValueError.
    candidates = [
        _candidate("ok", ActionType.GLOBAL_EXPLORATION, n_samples=1),
        _candidate("bad", ActionType.GLOBAL_EXPLORATION, n_samples=999),
        _candidate("stop", ActionType.STOP),
    ]
    out = CounterfactualRollout(be, horizon=1, max_rollout_steps=10).rollout(
        state, candidates
    )
    assert len(out.results) == 3
    bad_result = next(r for r in out.results if r.action_id == "bad")
    assert bad_result.status == "invalid"
    assert bad_result.error is not None

    # Backend must still be restored.
    history_after = be.get_history()
    assert torch.equal(history_before.X, history_after.X)


def test_rollout_is_candidate_order_invariant() -> None:
    """Same candidates in different orders must produce the same per-id results."""
    be = _brannin_backend()
    state = extract_state(be, total_budget=20)

    cands_ab = [
        _candidate("g1", ActionType.GLOBAL_EXPLORATION, n_samples=1),
        _candidate("g2", ActionType.GLOBAL_EXPLORATION, n_samples=2),
    ]
    cands_ba = list(reversed(cands_ab))

    out_ab = CounterfactualRollout(be, horizon=1).rollout(state, cands_ab)
    out_ba = CounterfactualRollout(be, horizon=1).rollout(state, cands_ba)

    ab = {r.action_id: r for r in out_ab.results}
    ba = {r.action_id: r for r in out_ba.results}
    assert ab.keys() == ba.keys()
    for key in ab:
        assert ab[key].delta_optimization == ba[key].delta_optimization
        assert ab[key].n_new_samples == ba[key].n_new_samples


def test_same_action_type_different_params_not_overwritten() -> None:
    be = RecordingBackend()
    state = extract_state(be, total_budget=20)
    candidates = [
        _candidate("g1", ActionType.GLOBAL_EXPLORATION, n_samples=1),
        _candidate("g3", ActionType.GLOBAL_EXPLORATION, n_samples=3),
    ]
    out = CounterfactualRollout(be, horizon=1).rollout(state, candidates)
    assert [r.action_id for r in out.results] == ["g1", "g3"]
    assert out.results[0].n_new_samples == 1
    assert out.results[1].n_new_samples == 3


# ---------------------------------------------------------------------------
# to_gold_probs
# ---------------------------------------------------------------------------


def test_to_gold_probs_sums_to_one_and_nonnegative() -> None:
    be = RecordingBackend()
    state = extract_state(be, total_budget=20)
    candidates = [
        _candidate("g2", ActionType.GLOBAL_EXPLORATION, n_samples=2),
        _candidate("stop", ActionType.STOP),
    ]
    out = CounterfactualRollout(be, horizon=1).rollout(state, candidates)
    probs = to_gold_probs(out.results)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert all(p >= 0 for p in probs.values())


def test_to_gold_probs_ordering_preserved() -> None:
    """Higher-utility action gets higher probability (monotone)."""
    be = RecordingBackend()
    state = extract_state(be, total_budget=20)
    candidates = [
        _candidate("g3", ActionType.GLOBAL_EXPLORATION, n_samples=3),
        _candidate("g1", ActionType.GLOBAL_EXPLORATION, n_samples=1),
    ]
    out = CounterfactualRollout(be, horizon=1).rollout(state, candidates)
    probs = to_gold_probs(out.results)
    # g1 should beat g3 in utility (fewer eval cost for similar improvement).
    assert probs["g1"] > probs["g3"]


def test_to_gold_probs_stop_is_dominant_at_zero_budget() -> None:
    """At zero remaining budget, STOP has cost=0 and is the only valid action."""
    be = RecordingBackend()
    # Burn through budget by appending samples directly (RecordingBackend has 2 init).
    be.evaluate(torch.tensor([[0.5, 0.5]], dtype=torch.double))  # 3 samples total
    # total_budget=3, remaining=0
    state = extract_state(be, total_budget=3)

    candidates = [
        _candidate("g1", ActionType.GLOBAL_EXPLORATION, n_samples=1),
        _candidate("stop", ActionType.STOP),
    ]
    # Note: g1 will be rejected by executor (budget=0). STOP is allowed.
    out = CounterfactualRollout(be, horizon=1, max_rollout_steps=10).rollout(
        state, candidates
    )
    probs = to_gold_probs(out.results)
    assert probs == {"stop": 1.0}


def test_to_gold_probs_rejects_nonpositive_temperature() -> None:
    be = RecordingBackend()
    state = extract_state(be, total_budget=20)
    candidates = [_candidate("stop", ActionType.STOP)]
    out = CounterfactualRollout(be, horizon=1).rollout(state, candidates)

    with pytest.raises(ValueError, match="temperature"):
        to_gold_probs(out.results, temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        to_gold_probs(out.results, temperature=-1.0)


def test_to_gold_probs_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        to_gold_probs(())


def test_to_gold_probs_rejects_all_invalid() -> None:
    be = RecordingBackend()
    # Burn through budget.
    be.evaluate(torch.tensor([[0.5, 0.5]], dtype=torch.double))
    # total_budget=3 → remaining=0 → every non-STOP action will be rejected.
    state = extract_state(be, total_budget=3)
    candidates = [
        _candidate("g1", ActionType.GLOBAL_EXPLORATION, n_samples=1),
    ]
    out = CounterfactualRollout(be, horizon=1, max_rollout_steps=10).rollout(
        state, candidates
    )
    with pytest.raises(ValueError, match="no valid candidate"):
        to_gold_probs(out.results)


# ---------------------------------------------------------------------------
# Multi-step (horizon > 1)
# ---------------------------------------------------------------------------


def test_horizon_greater_than_one_runs_continuation_policy() -> None:
    """With horizon=2, the first step is the candidate, second is from policy."""
    be = RecordingBackend()
    state = extract_state(be, total_budget=20)
    candidates = [
        _candidate("stop_now", ActionType.STOP),
    ]
    out = CounterfactualRollout(be, horizon=2).rollout(state, candidates)
    assert len(out.results) == 1
    # After STOP, the second step should not run (should_stop breaks the loop).
    assert out.results[0].n_new_samples == 0


def test_max_rollout_steps_cap_rejected() -> None:
    be = RecordingBackend()
    state = extract_state(be, total_budget=20)
    candidates = [
        _candidate(f"c{i}", ActionType.GLOBAL_EXPLORATION, n_samples=1)
        for i in range(5)
    ]
    with pytest.raises(ValueError, match="rollout step cap"):
        CounterfactualRollout(be, horizon=3, max_rollout_steps=10).rollout(
            state, candidates
        )


def test_empty_candidates_rejected() -> None:
    be = RecordingBackend()
    state = extract_state(be, total_budget=20)
    with pytest.raises(ValueError, match="must not be empty"):
        CounterfactualRollout(be, horizon=1).rollout(state, [])


def test_stale_state_rejected() -> None:
    be = RecordingBackend()
    state = extract_state(be, total_budget=20)
    # Mutate backend without passing new state.
    be.evaluate(torch.tensor([[0.5, 0.5]], dtype=torch.double))

    candidates = [_candidate("stop", ActionType.STOP)]
    with pytest.raises(ValueError, match="State does not describe"):
        CounterfactualRollout(be, horizon=1).rollout(state, candidates)


# ---------------------------------------------------------------------------
# Real Brannin end-to-end
# ---------------------------------------------------------------------------


def test_real_branin_rollout_finds_different_utilities() -> None:
    """End-to-end on Brannin-2D: different actions produce measurably different utilities."""
    be = _brannin_backend()
    state = extract_state(be, total_budget=30)

    candidates = [
        _candidate("global", ActionType.GLOBAL_EXPLORATION, n_samples=3),
        _candidate("surrogate", ActionType.SURROGATE_ACQUISITION),
        _candidate("retrain", ActionType.RETRAIN_SURROGATE),
        _candidate("stop", ActionType.STOP),
    ]
    out = CounterfactualRollout(be, horizon=1, max_rollout_steps=20).rollout(
        state, candidates
    )

    utilities = {r.action_id: r.utility for r in out.results}
    # STOP has zero cost + small Δopt, but Phase 8.5 adds a negative
    # w_stop_opportunity * remaining_budget_ratio penalty.
    assert utilities["stop"] <= 0
    # RETRAIN costs a fit (negative utility via w_fit * 1) but no eval.
    assert utilities["retrain"] < 0
    # SURROGATE_ACQUISITION should also have negative utility from the implicit
    # fit cost (Brannin was initialized without GP; first SURROGATE_ACQUISITION
    # triggers 1 fit).
    fits = {r.action_id: r.n_model_fits for r in out.results}
    assert fits["surrogate"] == 1
    assert fits["retrain"] == 1
    # h2: GLOBAL_EXPLORATION is greedy Thompson sampling — one lazy fit per
    # proposed point on a stale GP (n_samples=3, GP unfitted at rollout start
    # because Brannin was initialized without fit). Pre-h2 Sobol sampling
    # never fitted.
    assert fits["global"] == 3
    assert fits["stop"] == 0
    # Backend must be restored.
    assert len(be.get_history()) == 5

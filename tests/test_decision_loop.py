"""Phase 4 acceptance tests for DecisionLoop, RandomPolicy, RulePolicy.

Per codex (gpt-5.6-sol) review 2025-09-23: 17 tests covering manual step mode,
automatic run mode, two policies, external-mutation protection, termination
reasons, determinism, and a real-backend smoke test.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import FixedPolicy, RecordingBackend  # noqa: E402

from opt_agent.actions import ActionType, OptimizationAction  # noqa: E402
from opt_agent.backend.bocode_botorch import (  # noqa: E402
    BackendConfig,
    BocodeBotorchBackend,
)
from opt_agent.loop import (  # noqa: E402
    DecisionLoop,
    RandomPolicy,
    RulePolicy,
    TerminationReason,
)
from opt_agent.state import extract_state  # noqa: E402


# ---------------------------------------------------------------------------
# Manual step mode
# ---------------------------------------------------------------------------


def test_step_requires_reset() -> None:
    loop = DecisionLoop(RecordingBackend(), total_budget=10)
    with pytest.raises(RuntimeError, match="reset"):
        loop.step(OptimizationAction(ActionType.STOP))


def test_external_backend_mutation_is_rejected() -> None:
    backend = RecordingBackend()
    loop = DecisionLoop(backend, total_budget=10)
    loop.reset()

    backend.evaluate(torch.tensor([[0.2, 0.2]], dtype=torch.double))

    with pytest.raises(ValueError, match="outside DecisionLoop"):
        loop.step(OptimizationAction(ActionType.STOP))

    assert loop.steps == ()


def test_round_numbers_and_state_chain() -> None:
    loop = DecisionLoop(RecordingBackend(), total_budget=10)
    loop.reset()

    loop.step(OptimizationAction(ActionType.RETRAIN_SURROGATE))
    loop.step(OptimizationAction(ActionType.SURROGATE_ACQUISITION))

    assert [step.round for step in loop.steps] == [0, 1]
    assert loop.steps[0].next_state == loop.steps[1].state


def test_sample_delta_matches_action_result() -> None:
    loop = DecisionLoop(RecordingBackend(), total_budget=10)
    loop.reset()

    for action in [
        OptimizationAction(ActionType.RETRAIN_SURROGATE),
        OptimizationAction(ActionType.SURROGATE_ACQUISITION),
        OptimizationAction(ActionType.GLOBAL_EXPLORATION, {"n_samples": 2}),
    ]:
        next_state, result = loop.step(action)
        transition = loop.steps[-1]
        assert (
            next_state.sampling.n_samples - transition.state.sampling.n_samples
            == result.n_new_samples
        )


def test_last_action_is_propagated() -> None:
    loop = DecisionLoop(RecordingBackend(), total_budget=10)
    loop.reset()
    next_state, _ = loop.step(OptimizationAction(ActionType.SURROGATE_ACQUISITION))
    assert next_state.history.last_action == "surrogate_acquisition"


def test_failed_execution_does_not_commit_step() -> None:
    backend = RecordingBackend(fail_evaluate=True)
    loop = DecisionLoop(backend, total_budget=10)
    initial = loop.reset()

    with pytest.raises(RuntimeError, match="synthetic evaluation failure"):
        loop.step(OptimizationAction(ActionType.GLOBAL_EXPLORATION, {"n_samples": 1}))

    assert loop.steps == ()
    assert loop.state == initial


def test_reset_clears_bookkeeping_but_not_backend_history() -> None:
    backend = RecordingBackend()
    loop = DecisionLoop(backend, total_budget=10)
    loop.reset()
    loop.step(OptimizationAction(ActionType.SURROGATE_ACQUISITION))

    state = loop.reset()

    assert state.sampling.n_samples == 3  # backend history is preserved
    assert loop.steps == ()
    assert loop.termination_reason is None


# ---------------------------------------------------------------------------
# Automatic run mode + termination
# ---------------------------------------------------------------------------


def test_stop_action_terminates_loop() -> None:
    loop = DecisionLoop(RecordingBackend(), total_budget=10)
    trajectory = loop.run(FixedPolicy(OptimizationAction(ActionType.STOP)))

    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].result.should_stop
    assert trajectory.termination_reason == TerminationReason.STOP


def test_budget_exhaustion_terminates_loop() -> None:
    loop = DecisionLoop(RecordingBackend(), total_budget=4)
    policy = FixedPolicy(OptimizationAction(ActionType.SURROGATE_ACQUISITION))

    trajectory = loop.run(policy, max_rounds=10)

    assert len(trajectory.steps) == 2  # 2 → 3 → 4, then budget exhausted
    assert trajectory.final_state.sampling.n_samples == 4
    assert trajectory.termination_reason == TerminationReason.BUDGET_EXHAUSTED


def test_zero_sample_action_is_bounded_by_max_rounds() -> None:
    loop = DecisionLoop(RecordingBackend(), total_budget=10)
    policy = FixedPolicy(OptimizationAction(ActionType.RETRAIN_SURROGATE))

    trajectory = loop.run(policy, max_rounds=3)

    assert len(trajectory.steps) == 3
    assert trajectory.final_state.sampling.n_samples == 2  # no samples added
    assert trajectory.termination_reason == TerminationReason.MAX_ROUNDS


def test_max_rounds_zero_returns_initial_state() -> None:
    loop = DecisionLoop(RecordingBackend(), total_budget=10)
    policy = FixedPolicy(OptimizationAction(ActionType.STOP))

    trajectory = loop.run(policy, max_rounds=0)

    assert trajectory.steps == ()
    assert trajectory.final_state.sampling.n_samples == 2
    assert trajectory.termination_reason == TerminationReason.MAX_ROUNDS
    assert policy.calls == 0


def test_random_loop_terminates_under_hard_cap() -> None:
    loop = DecisionLoop(RecordingBackend(), total_budget=8)

    trajectory = loop.run(RandomPolicy(seed=0), max_rounds=20)

    assert len(trajectory.steps) <= 20
    assert trajectory.termination_reason in {
        TerminationReason.STOP,
        TerminationReason.BUDGET_EXHAUSTED,
        TerminationReason.MAX_ROUNDS,
    }


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def test_rule_policy_explores_after_three_stagnant_steps() -> None:
    state = extract_state(RecordingBackend(), total_budget=10)
    state = replace(
        state, optimization=replace(state.optimization, stagnation_steps=3)
    )

    action = RulePolicy(seed=7).select_action(state)

    assert action.type == ActionType.GLOBAL_EXPLORATION
    assert action.params["n_samples"] == 2
    assert "seed" not in action.params  # h3: AF sampling carries no seed


def test_rule_policy_stop_has_priority_when_budget_is_low() -> None:
    state = extract_state(RecordingBackend(), total_budget=6)  # remaining=4
    state = replace(
        state, optimization=replace(state.optimization, stagnation_steps=100)
    )

    action = RulePolicy().select_action(state)

    assert action.type == ActionType.STOP


def test_random_policy_reset_reproduces_action_sequence() -> None:
    state = extract_state(RecordingBackend(), total_budget=10)
    policy = RandomPolicy(seed=42)

    first = [policy.select_action(state) for _ in range(20)]
    policy.reset()
    second = [policy.select_action(state) for _ in range(20)]

    assert first == second


def test_random_policy_actions_fit_remaining_budget() -> None:
    state = extract_state(RecordingBackend(), total_budget=3)  # remaining=1
    policy = RandomPolicy(seed=0)

    for _ in range(100):
        action = policy.select_action(state)
        if action.type == ActionType.GLOBAL_EXPLORATION:
            assert action.params["n_samples"] == 1


# ---------------------------------------------------------------------------
# Real-backend smoke
# ---------------------------------------------------------------------------


def test_full_branin_rule_policy_smoke() -> None:
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(problem_id="Branin", seed=0, n_init=5))
    initial_best = extract_state(backend, total_budget=10).optimization.best_value

    trajectory = DecisionLoop(backend, total_budget=10).run(
        RulePolicy(seed=0), max_rounds=10
    )

    assert trajectory.steps
    assert trajectory.final_state.sampling.n_samples <= 10
    assert math.isfinite(trajectory.final_state.optimization.best_value)
    assert trajectory.final_state.optimization.best_value >= initial_best

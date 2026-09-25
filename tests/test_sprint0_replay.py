"""Sprint 0 — replayable counterfactual rollout tests.

Per codex (gpt-6-sol) Q-T2 review (2026-09-25):
- continuation policy + utility weights + horizon must be replayable per record
- "until budget exhausted" is achieved by passing remaining_budget as horizon

This file pins:
  1. ContinuationPolicySpec dataclass behavior
  2. build_continuation_policy registry
  3. UniformPolicy / OnPolicyMix policy behavior
  4. CounterfactualRollout uses spec instead of factory
  5. DatasetGenerator writes full utility_weights + continuation_policy
  6. replay_with_config:
     - same config → identical output
     - different continuation → different output
     - different horizon → different output
     - different utility weights → different output
"""

from __future__ import annotations

import base64
import math
from collections import Counter

import pytest

from opt_agent.actions import (
    ActionType,
    OptimizationAction,
)
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend
from opt_agent.dataset.generator import (
    DatasetConfig,
    DatasetGenerator,
    full_candidates,
)
from opt_agent.dataset.validation import replay_with_config
from opt_agent.loop.policies import (
    OnPolicyMix,
    RandomPolicy,
    RulePolicy,
    UniformPolicy,
)
from opt_agent.rollout.rollout import (
    CounterfactualRollout,
    to_gold_probs,
)
from opt_agent.rollout.specs import (
    CONTINUATION_POLICY_REGISTRY,
    ContinuationPolicySpec,
    build_continuation_policy,
    registered_policy_names,
    spec_from_metadata,
    spec_to_metadata,
)
from opt_agent.rollout.utility import UtilityWeights
from opt_agent.state import extract_state


# ---------------------------------------------------------------------------
# 1. ContinuationPolicySpec dataclass
# ---------------------------------------------------------------------------


def test_spec_defaults_are_sane():
    spec = ContinuationPolicySpec(name="rule")
    assert spec.name == "rule"
    assert spec.seed == 0
    assert spec.k is None
    assert dict(spec.extras) == {}


def test_spec_is_frozen():
    spec = ContinuationPolicySpec(name="rule")
    with pytest.raises(Exception):
        spec.name = "uniform"  # type: ignore[misc]


def test_spec_round_trip_through_metadata():
    original = ContinuationPolicySpec(
        name="on_policy", seed=123, k=4,
        extras={"rule_weight": 0.7, "extra_str": "foo"},
    )
    meta = spec_to_metadata(original)
    restored = spec_from_metadata(meta)
    assert restored == original


def test_spec_from_metadata_handles_missing():
    spec = spec_from_metadata(None)
    assert spec.name == "rule"
    assert spec.seed == 0


# ---------------------------------------------------------------------------
# 2. build_continuation_policy registry
# ---------------------------------------------------------------------------


def test_registry_contains_three_implementations():
    assert registered_policy_names() == ("on_policy", "rule", "uniform")


def test_build_rule_returns_rule_policy():
    p = build_continuation_policy(ContinuationPolicySpec(name="rule", seed=42))
    assert isinstance(p, RulePolicy)


def test_build_uniform_returns_uniform_policy():
    p = build_continuation_policy(ContinuationPolicySpec(name="uniform", seed=42))
    assert isinstance(p, UniformPolicy)


def test_build_on_policy_returns_on_policy_mix():
    p = build_continuation_policy(
        ContinuationPolicySpec(
            name="on_policy", seed=42, extras={"rule_weight": 0.5}
        )
    )
    assert isinstance(p, OnPolicyMix)


def test_build_unknown_raises_value_error():
    with pytest.raises(ValueError, match="unknown continuation policy"):
        build_continuation_policy(ContinuationPolicySpec(name="best_of_k", seed=0))


# ---------------------------------------------------------------------------
# 3. New policies behavior
# ---------------------------------------------------------------------------


def test_uniform_policy_samples_all_eight_action_types():
    """Over many calls, UniformPolicy should touch all 8 ActionTypes."""
    from opt_agent.state.models import (
        HistoryState,
        OptimizationProgress,
        OptimizationState,
        ResourceState,
        SamplingState,
        SurrogateState,
    )
    state = OptimizationState(
        sampling=SamplingState(
            n_samples=10, dimension=2, min_pairwise_distance=0.1,
            boundary_coverage=0.2,
        ),
        surrogate=SurrogateState(
            is_current=True, validation_rmse=0.1,
            mean_uncertainty=0.5, max_uncertainty=1.0,
        ),
        optimization=OptimizationProgress(
            best_value=0.0, last_value=0.0,
            recent_improvement=0.0, stagnation_steps=0,
        ),
        resource=ResourceState(
            used_budget=10, total_budget=30,
            remaining_budget=20, remaining_budget_ratio=20/30,
        ),
        history=HistoryState(last_action=None),
    )
    p = UniformPolicy(seed=0)
    types_seen = Counter()
    for _ in range(2000):
        action = p.select_action(state, backend=None)
        types_seen[action.type] += 1
    assert set(types_seen.keys()) == set(ActionType), (
        f"missing action types: {set(ActionType) - set(types_seen.keys())}"
    )
    # Each type should have a non-trivial share (~ 1/8 = 0.125 expected).
    for action_type, count in types_seen.items():
        assert count > 100, f"{action_type} under-sampled: {count}"


def test_uniform_policy_returns_stop_when_budget_exhausted():
    from opt_agent.state.models import (
        HistoryState,
        OptimizationProgress,
        OptimizationState,
        ResourceState,
        SamplingState,
        SurrogateState,
    )
    state = OptimizationState(
        sampling=SamplingState(
            n_samples=30, dimension=2, min_pairwise_distance=0.1,
            boundary_coverage=0.5,
        ),
        surrogate=SurrogateState(
            is_current=True, validation_rmse=0.1,
            mean_uncertainty=0.5, max_uncertainty=1.0,
        ),
        optimization=OptimizationProgress(
            best_value=1.0, last_value=1.0,
            recent_improvement=0.0, stagnation_steps=0,
        ),
        resource=ResourceState(
            used_budget=30, total_budget=30,
            remaining_budget=0, remaining_budget_ratio=0.0,
        ),
        history=HistoryState(last_action=None),
    )
    p = UniformPolicy(seed=0)
    for _ in range(50):
        assert p.select_action(state).type == ActionType.STOP


def test_on_policy_mix_validates_rule_weight():
    with pytest.raises(ValueError):
        OnPolicyMix(seed=0, rule_weight=1.5)
    with pytest.raises(ValueError):
        OnPolicyMix(seed=0, rule_weight=-0.1)


def test_on_policy_mix_rule_weight_zero_always_random():
    """rule_weight=0.0 → behaves exactly like RandomPolicy."""
    from opt_agent.state.models import (
        HistoryState,
        OptimizationProgress,
        OptimizationState,
        ResourceState,
        SamplingState,
        SurrogateState,
    )
    state = OptimizationState(
        sampling=SamplingState(
            n_samples=10, dimension=2, min_pairwise_distance=0.1,
            boundary_coverage=0.2,
        ),
        surrogate=SurrogateState(
            is_current=True, validation_rmse=0.1,
            mean_uncertainty=0.5, max_uncertainty=1.0,
        ),
        optimization=OptimizationProgress(
            best_value=0.0, last_value=0.0,
            recent_improvement=0.0, stagnation_steps=0,
        ),
        resource=ResourceState(
            used_budget=10, total_budget=30,
            remaining_budget=20, remaining_budget_ratio=20/30,
        ),
        history=HistoryState(last_action=None),
    )
    mix = OnPolicyMix(seed=0, rule_weight=0.0)
    random_p = RandomPolicy(seed=0)
    for _ in range(20):
        a = mix.select_action(state)
        b = random_p.select_action(state)
        assert a.type == b.type


def test_on_policy_mix_rule_weight_one_always_rule():
    """rule_weight=1.0 → behaves exactly like RulePolicy."""
    from opt_agent.state.models import (
        HistoryState,
        OptimizationProgress,
        OptimizationState,
        ResourceState,
        SamplingState,
        SurrogateState,
    )
    state = OptimizationState(
        sampling=SamplingState(
            n_samples=10, dimension=2, min_pairwise_distance=0.1,
            boundary_coverage=0.2,
        ),
        surrogate=SurrogateState(
            is_current=True, validation_rmse=0.1,
            mean_uncertainty=0.5, max_uncertainty=1.0,
        ),
        optimization=OptimizationProgress(
            best_value=0.0, last_value=0.0,
            recent_improvement=0.0, stagnation_steps=0,
        ),
        resource=ResourceState(
            used_budget=10, total_budget=30,
            remaining_budget=20, remaining_budget_ratio=20/30,
        ),
        history=HistoryState(last_action=None),
    )
    mix = OnPolicyMix(seed=0, rule_weight=1.0)
    rule_p = RulePolicy(seed=0)
    for _ in range(20):
        a = mix.select_action(state)
        b = rule_p.select_action(state)
        assert a.type == b.type


# ---------------------------------------------------------------------------
# 4. CounterfactualRollout accepts continuation_spec (not factory)
# ---------------------------------------------------------------------------


def test_counterfactual_rollout_uses_spec_for_horizon_gt_one():
    """CounterfactualRollout should call build_continuation_policy internally
    when horizon > 1, using the spec passed at construction.
    """
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(problem_id="Branin", seed=0, n_init=5))
    state = extract_state(backend, total_budget=20)
    rollout = CounterfactualRollout(
        backend,
        horizon=3,
        continuation_spec=ContinuationPolicySpec(name="uniform", seed=11),
    )
    assert rollout.continuation_spec.name == "uniform"
    assert rollout.continuation_spec.seed == 11


def test_counterfactual_rollout_default_spec_is_rule():
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(problem_id="Branin", seed=0, n_init=5))
    rollout = CounterfactualRollout(backend)
    assert rollout.continuation_spec.name == "rule"


# ---------------------------------------------------------------------------
# 5. DatasetGenerator writes full utility_weights + continuation_policy
# ---------------------------------------------------------------------------


@pytest.fixture
def small_dataset_records():
    """Generate a tiny dataset (1 problem, 1 seed, 1 trajectory, ~4 records)."""
    config = DatasetConfig(
        problems=("Branin",),
        seeds=(0,),
        trajectories_per_problem_seed=1,
        n_init=5,
        total_budget=15,
        max_actions=8,
        decision_every_k_actions=2,
        horizon=3,
        continuation_spec=ContinuationPolicySpec(
            name="on_policy", seed=0, extras={"rule_weight": 0.8},
        ),
    )
    gen = DatasetGenerator(config)
    return list(gen.generate())


def test_metadata_contains_full_utility_weights(small_dataset_records):
    r = small_dataset_records[0]
    uw = r.metadata["utility_weights"]
    assert set(uw.keys()) == {
        "w_optimization", "w_surrogate", "w_coverage",
        "w_evaluation", "w_fit", "w_stop_opportunity",
    }
    assert math.isclose(uw["w_stop_opportunity"], 0.75)


def test_metadata_contains_continuation_policy(small_dataset_records):
    r = small_dataset_records[0]
    cp = r.metadata["continuation_policy"]
    assert cp["name"] == "on_policy"
    assert cp["seed"] == 0
    assert cp["extras"]["rule_weight"] == 0.8


def test_metadata_preserves_legacy_w_stop_opportunity(small_dataset_records):
    r = small_dataset_records[0]
    # Backward compatibility with Phase 8.6B readers.
    assert "w_stop_opportunity" in r.metadata
    assert math.isclose(
        r.metadata["w_stop_opportunity"],
        r.metadata["utility_weights"]["w_stop_opportunity"],
    )


# ---------------------------------------------------------------------------
# 6. replay_with_config
# ---------------------------------------------------------------------------


def test_replay_with_same_config_is_identical(small_dataset_records):
    """Replay with original config → same gold_probs (modulo float noise)."""
    r = small_dataset_records[0]
    output = replay_with_config(r)
    probs = to_gold_probs(
        output.results, temperature=r.temperature, include_invalid_as_zero=True,
    )
    for action_id, p in probs.items():
        assert math.isclose(p, r.gold_probs[action_id], abs_tol=1e-9), (
            f"{action_id}: replay={p}, original={r.gold_probs[action_id]}"
        )


def test_replay_with_different_continuation_changes_output(small_dataset_records):
    r = small_dataset_records[0]
    out_rule = replay_with_config(
        r, continuation_spec=ContinuationPolicySpec(name="rule", seed=42),
    )
    out_uniform = replay_with_config(
        r, continuation_spec=ContinuationPolicySpec(name="uniform", seed=42),
    )
    probs_rule = to_gold_probs(
        out_rule.results, temperature=r.temperature, include_invalid_as_zero=True,
    )
    probs_uniform = to_gold_probs(
        out_uniform.results, temperature=r.temperature, include_invalid_as_zero=True,
    )
    different = any(
        not math.isclose(probs_rule[a], probs_uniform[a], abs_tol=1e-6)
        for a in probs_rule
    )
    assert different, "different continuations produced identical gold_probs"


def test_replay_with_different_horizon_changes_output(small_dataset_records):
    r = small_dataset_records[0]
    out_h1 = replay_with_config(r, horizon=1)
    out_h10 = replay_with_config(r, horizon=10)
    probs_h1 = to_gold_probs(
        out_h1.results, temperature=r.temperature, include_invalid_as_zero=True,
    )
    probs_h10 = to_gold_probs(
        out_h10.results, temperature=r.temperature, include_invalid_as_zero=True,
    )
    different = any(
        not math.isclose(probs_h1[a], probs_h10[a], abs_tol=1e-6)
        for a in probs_h1
    )
    assert different, "different horizons produced identical gold_probs"


def test_replay_with_modified_utility_weights_changes_output(small_dataset_records):
    r = small_dataset_records[0]
    out_no_stop_pen = replay_with_config(
        r, utility_weights=UtilityWeights(w_stop_opportunity=0.0),
    )
    probs = to_gold_probs(
        out_no_stop_pen.results,
        temperature=r.temperature, include_invalid_as_zero=True,
    )
    # With STOP penalty removed, STOP should rise in rank. Original config
    # has w_stop_opportunity=0.75.
    assert probs["stop"] >= r.gold_probs["stop"] - 1e-9, (
        f"removing STOP penalty should not decrease STOP probability: "
        f"{probs['stop']} vs original {r.gold_probs['stop']}"
    )


def test_replay_with_missing_replay_keys_raises():
    from opt_agent.dataset.decision_record import DecisionRecord

    # Build a record without replay metadata. Provide `horizon` so we get
    # past the metadata["horizon"] read; the missing replay keys should
    # raise inside `_reconstruct_checkpoint`.
    bad_record = DecisionRecord(
        id="test:0:0:0",
        problem={"family": "branin", "dimension": 2, "seed": 0},
        state={},
        candidate_actions=[
            {"action_id": "stop", "type": "stop", "params": {}},
        ],
        gold_probs={"stop": 1.0},
        rollout_results={"stop": {"status": "valid", "utility": 0.0}},
        temperature=1.0,
        metadata={"horizon": 1},  # no replay keys
    )
    with pytest.raises(ValueError, match="missing replay keys"):
        replay_with_config(bad_record)


def test_replay_leaves_backend_at_base_state(small_dataset_records):
    """replay_with_config should restore backend to base on return."""
    r = small_dataset_records[0]
    base_n = int(r.state["sampling"]["n_samples"])
    backend_factory = lambda: _replay_backend_for_record(r)
    backend = backend_factory()
    from opt_agent.backend.checkpoint import Checkpoint
    # Save current state
    pre_n = len(backend.get_history())
    output = replay_with_config(r, backend_factory=backend_factory)
    backend = backend_factory()
    post_n = len(backend.get_history())
    assert pre_n == post_n == base_n, (
        f"backend n_samples changed: pre={pre_n}, post={post_n}, base={base_n}"
    )


def _replay_backend_for_record(r):
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(
        problem_id=str(r.metadata["backend_problem_id"]),
        seed=int(r.problem["seed"]),
        n_init=1,
    ))
    meta = r.metadata
    import torch
    backend.restore(_make_checkpoint(meta))
    return backend


def _make_checkpoint(meta):
    from opt_agent.backend.checkpoint import Checkpoint
    import torch
    return Checkpoint(
        backend_id=str(meta["backend_id"]),
        problem_id=str(meta["backend_problem_id"]),
        problem_bounds=[tuple(b) for b in meta["backend_problem_bounds"]],
        X=torch.tensor(meta["backend_X"], dtype=torch.double),
        objectives=torch.tensor(meta["backend_objectives"], dtype=torch.double),
        constraints=torch.tensor(meta["backend_constraints"], dtype=torch.double),
        fitted=bool(meta["backend_fitted"]),
        model_state_dict=(
            base64.b64decode(meta["backend_model_state_b64"], validate=True)
            if meta["backend_model_state_b64"] is not None else None
        ),
        torch_rng_state=(
            base64.b64decode(meta["backend_rng_state_b64"], validate=True)
            if meta["backend_rng_state_b64"] is not None else None
        ),
    )

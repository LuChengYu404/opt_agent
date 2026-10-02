"""Phase 8.6B tests for Gate 2 (multi-seed stability) and Gate 4 (replayable oracle).

Per codex (gpt-5.6-sol) review 2025-09-23:
  - Gate 2 requires ≥10 seeds per problem with consistent top-action
    distribution across seeds.
  - Gate 4 needs trajectory replay infrastructure (Phase 8.6B added
    backend_X / backend_objectives / backend_constraints in metadata).
"""

from __future__ import annotations

import base64
import sys
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from opt_agent.actions import ActionType  # noqa: E402
from opt_agent.backend.bocode_botorch import (  # noqa: E402
    BackendConfig,
    BocodeBotorchBackend,
    EvaluationBatch,
)
from opt_agent.dataset import (  # noqa: E402
    DatasetConfig,
    DatasetGenerator,
    DecisionRecord,
    aggregate_mass,
    gate2_stability,
    oracle_test_from_record,
    write_jsonl,
)
from opt_agent.dataset.decision_record import _json_safe
from opt_agent.state import extract_state


def _make_replay_record(
    *, family="branin", seed=0, action_probs, record_id="test:0:0:0",
    total_budget=20, used_budget=5, last_action=None, remaining_budget_ratio=None,
):
    """Build a DecisionRecord with replay history populated (Phase 8.6B schema)."""
    n = used_budget
    X = [[0.5 * i, 7.5] for i in range(n)]
    Y = [[0.1 * i] for i in range(n)]
    C = [[] for _ in range(n)]
    rng_buffer = BytesIO()
    torch.save(torch.random.get_rng_state(), rng_buffer)
    cands = [{"action_id": k, "type": k, "params": {}} for k in action_probs]
    ratio = remaining_budget_ratio if remaining_budget_ratio is not None else (
        (total_budget - n) / total_budget
    )
    return DecisionRecord(
        id=record_id,
        problem={"family": family, "dimension": 2, "seed": seed},
        state={
            "sampling": {"n_samples": n, "dimension": 2,
                         "min_pairwise_distance": 1.0, "boundary_coverage": 0.0},
            "surrogate": {"is_current": False, "validation_rmse": float("nan"),
                          "mean_uncertainty": float("nan"),
                          "max_uncertainty": float("nan")},
            "optimization": {"best_value": max(y[0] for y in Y), "last_value": Y[-1][0],
                             "recent_improvement": 0.0, "stagnation_steps": 0},
            "resource": {"used_budget": n, "total_budget": total_budget,
                         "remaining_budget": total_budget - n,
                         "remaining_budget_ratio": ratio},
            "history": {"last_action": last_action},
        },
        candidate_actions=cands,
        gold_probs=action_probs,
        rollout_results={
            k: {"status": "valid", "utility": 0.0, "delta_optimization": 0.0,
                "delta_surrogate": float("nan"), "delta_coverage": 0.0,
                "n_new_samples": 0, "n_model_fits": 0, "error": None}
            for k in action_probs
        },
        temperature=1.0,
        metadata={
            "trajectory_id": 0, "step": 0, "run_seed": 0,
            "utility_version": "phase8.6b",
            # Handcrafted records must carry the current harness stamp to be
            # replayable (validation._assert_replayable_harness); real
            # datasets get this from versioning.stamp_metadata.
            "harness_version": "h3",
            "optimization_scale": 0.5, "evaluation_scale": 20.0,
            "fit_scale": 20.0, "horizon": 1, "w_stop_opportunity": 0.75,
            "backend_X": X, "backend_objectives": Y, "backend_constraints": C,
            "backend_id": "opt_agent.backend.bocode_botorch.BocodeBotorchBackend",
            "backend_problem_id": "Branin",
            "backend_problem_bounds": [(-5.0, 10.0), (0.0, 15.0)],
            "backend_fitted": False,
            "backend_model_state_b64": None,
            "backend_rng_state_b64": base64.b64encode(rng_buffer.getvalue()).decode("ascii"),
        },
    )


# ---------------------------------------------------------------------------
# Gate 2 — multi-seed stability
# ---------------------------------------------------------------------------


def test_gate2_requires_min_seeds() -> None:
    """Gate 2 fails if any problem has fewer than 10 independent seeds."""
    records = []
    for problem in ("branin", "hartmann"):
        for seed in range(3):
            probs = {
                "global_exploration": 0.5, "uncertainty_sampling": 0.2,
                "stop": 0.1, "retrain_surrogate": 0.1, "boundary_sampling": 0.1,
                "local_exploitation": 0.0, "error_guided_sampling": 0.0,
                "surrogate_acquisition": 0.0,
            }
            for step in range(2):
                ratio = 0.8 if step == 0 else 0.6
                records.append(_make_replay_record(
                    family=problem, seed=seed,
                    action_probs=probs, record_id=f"{problem}:{seed}:0:{step}",
                    remaining_budget_ratio=ratio,
                ))
    result = gate2_stability(records)
    assert result.total_seeds == 3
    assert not result.pass_


def test_gate2_passes_with_consistent_top_action() -> None:
    """10 seeds × 1 problem × consistent top action → gate 2 passes."""
    records = []
    for seed in range(10):
        # All seeds pick "global_exploration" as top in early phase.
        probs = {
            "global_exploration": 0.6, "uncertainty_sampling": 0.2,
            "stop": 0.1, "retrain_surrogate": 0.1,
            "local_exploitation": 0.0, "error_guided_sampling": 0.0,
            "boundary_sampling": 0.0, "surrogate_acquisition": 0.0,
        }
        records.append(_make_replay_record(
            family="branin", seed=seed,
            action_probs=probs, record_id=f"branin:{seed}:0:0",
        ))
    result = gate2_stability(records)
    assert result.total_seeds == 10
    assert result.pass_


def test_gate2_top_action_share_concentration() -> None:
    """Top action must dominate the seed-level histogram (>=0.6 share)."""
    # 10 seeds, mixed top action — fails concentration criterion.
    records = []
    for seed in range(10):
        # Alternate top actions across seeds.
        if seed % 2 == 0:
            top = "global_exploration"
        else:
            top = "retrain_surrogate"
        probs = {k: 0.05 for k in [
            "global_exploration", "local_exploitation",
            "uncertainty_sampling", "error_guided_sampling",
            "boundary_sampling", "surrogate_acquisition",
            "retrain_surrogate", "stop",
        ]}
        probs[top] = 0.6
        records.append(_make_replay_record(
            family="branin", seed=seed,
            action_probs=probs, record_id=f"branin:{seed}:0:0",
        ))
    result = gate2_stability(records)
    assert result.total_seeds == 10
    assert not result.pass_  # each seed is internally certain, but seeds split 50/50


# ---------------------------------------------------------------------------
# Gate 4 — replay infrastructure
# ---------------------------------------------------------------------------


def test_generator_writes_replay_history_to_metadata(tmp_path) -> None:
    """Phase 8.6B: generator must serialize backend_X / _objectives / _constraints
    into metadata for every record (so Gate 4 can replay)."""
    spec = DatasetConfig(
        problems=("Branin",), seeds=(0,), trajectories_per_problem_seed=1,
        n_init=3, total_budget=10, max_actions=4, decision_every_k_actions=2,
    )
    path = tmp_path / "ds.jsonl"
    write_jsonl(DatasetGenerator(spec).generate(), path)

    import json
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            meta = rec["metadata"]
            assert "backend_X" in meta
            assert "backend_objectives" in meta
            assert "backend_constraints" in meta
            assert len(meta["backend_X"]) == len(meta["backend_objectives"])
            assert len(meta["backend_objectives"]) == len(meta["backend_constraints"])
            assert meta["backend_rng_state_b64"]
            assert isinstance(meta["backend_fitted"], bool)


def test_oracle_test_from_record_replays_state() -> None:
    """Given a record with replay history, oracle_test_from_record reconstructs
    the backend and runs the oracle without raising."""
    probs = {
        "stop": 0.5, "global_exploration": 0.3, "uncertainty_sampling": 0.2,
        "local_exploitation": 0.0, "error_guided_sampling": 0.0,
        "boundary_sampling": 0.0, "retrain_surrogate": 0.0,
        "surrogate_acquisition": 0.0,
    }
    rec = _make_replay_record(
        action_probs=probs, record_id="replay:0:0:0",
        total_budget=20, used_budget=8,
    )
    # Small max_actions + 1 replicate to keep test fast.
    result = oracle_test_from_record(rec, random_replicates=1, max_actions=5)
    assert result.gold_action in ("stop", "global_exploration", "uncertainty_sampling")
    assert "stop" in result.terminal
    assert "uniform" in result.terminal


def test_oracle_replays_fitted_surrogate() -> None:
    """A saved fitted GP must remain fitted at the replayed decision point."""
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(problem_id="Branin", seed=2, n_init=5))
    backend.fit()
    state = extract_state(backend, total_budget=12)
    cp = backend.checkpoint()
    base = _make_replay_record(
        family="branin", seed=2, action_probs={"stop": 1.0},
        total_budget=12, used_budget=5,
    )
    rec = replace(
        base,
        state=_json_safe(state),
        metadata={
            **base.metadata,
            "backend_X": cp.X.tolist(),
            "backend_objectives": cp.objectives.tolist(),
            "backend_constraints": cp.constraints.tolist(),
            "backend_problem_bounds": cp.problem_bounds,
            "backend_fitted": True,
            "backend_model_state_b64": base64.b64encode(cp.model_state_dict).decode("ascii"),
            "backend_rng_state_b64": base64.b64encode(cp.torch_rng_state).decode("ascii"),
        },
    )
    result = oracle_test_from_record(rec, random_replicates=1, max_actions=4)
    assert result.terminal["gold"] == pytest.approx(state.optimization.best_value)


def test_oracle_test_from_record_rejects_missing_replay() -> None:
    """Records without backend_X in metadata raise ValueError (Phase 8.6B gate)."""
    probs = {"stop": 1.0, "global_exploration": 0.0,
             "uncertainty_sampling": 0.0, "local_exploitation": 0.0,
             "error_guided_sampling": 0.0, "boundary_sampling": 0.0,
             "retrain_surrogate": 0.0, "surrogate_acquisition": 0.0}
    rec = _make_replay_record(action_probs=probs, record_id="bad")
    # Strip the replay fields from the dict metadata.
    bad_meta = {k: v for k, v in rec.metadata.items()
               if k not in ("backend_X", "backend_objectives", "backend_constraints")}
    bad_record = DecisionRecord(
        id=rec.id, problem=rec.problem, state=rec.state,
        candidate_actions=rec.candidate_actions,
        gold_probs=rec.gold_probs,
        rollout_results=rec.rollout_results,
        temperature=rec.temperature, metadata=bad_meta,
    )
    with pytest.raises(ValueError, match="missing"):
        oracle_test_from_record(bad_record)


def test_oracle_replay_refuses_foreign_harness_stamp() -> None:
    """Round-1 D regression: executing a replay through the current executor
    is refused for known-past harness stamps unless explicitly acknowledged;
    unrecognized stamps are never replayable."""
    import pytest as _pytest

    rec = _make_replay_record(action_probs={"global_exploration": 1.0})
    rec.metadata["harness_version"] = "h2"
    with _pytest.raises(ValueError, match="harness"):
        oracle_test_from_record(rec, random_replicates=1, max_actions=5)
    result = oracle_test_from_record(
        rec, random_replicates=1, max_actions=5, allow_legacy_harness=True
    )
    assert result.gold_action is not None
    rec.metadata["harness_version"] = "h99"
    with _pytest.raises(ValueError, match="harness"):
        oracle_test_from_record(rec, allow_legacy_harness=True)

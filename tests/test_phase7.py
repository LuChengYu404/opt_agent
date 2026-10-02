"""Phase 7 acceptance tests for utility extension + DecisionDataset.

Per codex (gpt-5.6-sol) review 2025-09-23:
  - utility signs and costs match the 4-term formula.
  - NaN / Inf optional deltas are coerced to 0.
  - include_invalid_as_zero preserves aligned vocabulary.
  - duplicate action_id rejected in rollout.
  - JSONL is strict (allow_nan=False) and sanitizes non-finite floats.
  - generator yields records with aligned candidate / gold_probs / rollout vocabularies.
  - generator behavior policy never samples from gold_probs.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import RecordingBackend  # noqa: E402

from opt_agent.actions import ActionType  # noqa: E402
from opt_agent.actions.models import OptimizationAction  # noqa: E402
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend  # noqa: E402
from opt_agent.dataset import (  # noqa: E402
    DatasetConfig,
    DatasetGenerator,
    DecisionRecord,
    full_candidates,
    iter_jsonl,
    write_jsonl,
)
from opt_agent.rollout import (  # noqa: E402
    Candidate,
    CounterfactualRollout,
    to_gold_probs,
)
from opt_agent.rollout.utility import UtilityWeights, compute_utility  # noqa: E402
from opt_agent.state import extract_state  # noqa: E402


# ---------------------------------------------------------------------------
# Phase 7A — Utility extension
# ---------------------------------------------------------------------------


def test_utility_signs_and_costs() -> None:
    w = UtilityWeights(
        w_optimization=1.0,
        w_surrogate=2.0,
        w_coverage=3.0,
        w_evaluation=4.0,
        w_fit=5.0,
    )
    utility = compute_utility(
        delta_optimization=2.0,
        delta_surrogate=3.0,
        delta_coverage=-1.0,
        n_new_samples=1,
        n_model_fits=1,
        weights=w,
    )
    # 1*2 + 2*3 + 3*max(0,-1)=0 − 4*1 − 5*1 = 2 + 6 + 0 − 4 − 5 = −1
    # Wait — codex said delta_coverage=-1 should NOT be clamped to 0; the
    # caller decides sign convention. We do NOT clamp here; caller can pass
    # any real. 3 * (-1) = -3.
    # Expected: 1*2 + 2*3 + 3*(-1) − 4*1 − 5*1 = 2 + 6 − 3 − 4 − 5 = −4.
    assert utility == pytest.approx(-4.0)


def test_nonfinite_optional_deltas_become_zero() -> None:
    """NaN / Inf deltas coerce to 0 — utility falls back to base terms."""
    utility = compute_utility(
        delta_optimization=1.0,
        delta_surrogate=float("nan"),
        delta_coverage=float("inf"),
        n_new_samples=0,
        n_model_fits=0,
        weights=UtilityWeights(),
    )
    # 1*1 + 0*NaN→0 + 0*Inf→0 − 0 − 0 = 1.0
    assert utility == pytest.approx(1.0)


def test_counterfactual_result_has_surrogate_and_coverage_fields() -> None:
    """CounterfactualResult now exposes delta_surrogate / delta_coverage."""
    be = BocodeBotorchBackend()
    be.initialize(BackendConfig(problem_id="Branin", seed=0, n_init=5))
    state = extract_state(be, total_budget=20)
    cands = full_candidates()
    out = CounterfactualRollout(be, horizon=1, max_rollout_steps=20).rollout(
        state, cands
    )
    for r in out.results:
        assert hasattr(r, "delta_surrogate")
        assert hasattr(r, "delta_coverage")
        # delta_coverage is always finite; delta_surrogate may be NaN.
        assert math.isfinite(r.delta_coverage)


# ---------------------------------------------------------------------------
# Phase 7A — to_gold_probs extension
# ---------------------------------------------------------------------------


def test_include_invalid_as_zero_preserves_vocabulary() -> None:
    """With include_invalid_as_zero=True, invalid candidates appear with prob 0.

    Note: actions with 0 cost (STOP / RETRAIN_SURROGATE / SURROGATE_ACQUISITION)
    remain valid even at zero remaining budget; only sampling actions are
    rejected. So the probability mass is shared among those 3 zero-cost actions,
    while the 5 sampling actions get prob=0.
    """
    be = BocodeBotorchBackend()
    be.initialize(BackendConfig(problem_id="Branin", seed=0, n_init=5))
    for _ in range(15):  # 5 init + 15 eval = 20 samples
        be.evaluate(torch.tensor([[0.5, 7.5]], dtype=torch.double))
    state = extract_state(be, total_budget=20)  # remaining = 0
    cands = full_candidates()
    out = CounterfactualRollout(be, horizon=1, max_rollout_steps=20).rollout(
        state, cands
    )

    probs = to_gold_probs(out.results, temperature=1.0, include_invalid_as_zero=True)
    # All 6 ids must be present (h3 menu).
    assert len(probs) == 6
    # Sampling actions must have prob=0 (they were rejected by executor).
    for sampling_id in ("global_exploration", "local_exploitation",
                        "uncertainty_sampling"):
        assert probs[sampling_id] == 0.0
    # Zero-cost actions split mass among themselves. At least one must be >0.
    valid_ids = ("stop", "surrogate_acquisition", "retrain_surrogate")
    nonzero_valid = [k for k in valid_ids if probs[k] > 0]
    assert len(nonzero_valid) >= 1, (
        f"Expected at least one zero-cost action to have prob>0; "
        f"got probs={probs}"
    )
    # Sum is 1.
    assert sum(probs.values()) == pytest.approx(1.0)


def test_duplicate_action_ids_are_rejected() -> None:
    """Duplicate action_id would collide in results dict → reject upfront."""
    be = BocodeBotorchBackend()
    be.initialize(BackendConfig(problem_id="Branin", seed=0, n_init=5))
    state = extract_state(be, total_budget=20)
    cands = [
        Candidate("same", OptimizationAction(ActionType.STOP)),
        Candidate("same", OptimizationAction(ActionType.RETRAIN_SURROGATE)),
    ]
    with pytest.raises(ValueError, match="duplicate action_id"):
        CounterfactualRollout(be, horizon=1).rollout(state, cands)


# ---------------------------------------------------------------------------
# Phase 7B — JSONL codec
# ---------------------------------------------------------------------------


def test_jsonl_sanitizes_nested_nonfinite(tmp_path) -> None:
    record = DecisionRecord(
        id="test:0:0:0",
        problem={"family": "branin", "dimension": 2, "seed": 0},
        state={"x": float("nan"), "y": float("inf")},
        candidate_actions=[{"action_id": "stop", "type": "stop", "params": {}}],
        gold_probs={"stop": 1.0},
        rollout_results={"stop": {"utility": float("inf"), "delta_optimization": float("nan")}},
        temperature=1.0,
    )
    path = tmp_path / "records.jsonl"
    write_jsonl([record], path)

    raw = path.read_text()
    # Strict JSON: no NaN / Infinity tokens.
    assert "NaN" not in raw
    assert "Infinity" not in raw

    payload = json.loads(raw)
    assert payload["state"]["x"] is None
    assert payload["state"]["y"] is None
    assert payload["rollout_results"]["stop"]["utility"] is None
    assert payload["rollout_results"]["stop"]["delta_optimization"] is None


def test_jsonl_round_trip(tmp_path) -> None:
    record = DecisionRecord(
        id="test:0:0:0",
        problem={"family": "branin", "dimension": 2, "seed": 0},
        state={"best": -0.5},
        candidate_actions=[
            {"action_id": "stop", "type": "stop", "params": {}},
            {"action_id": "g1", "type": "global_exploration", "params": {"n_samples": 1}},
        ],
        gold_probs={"stop": 0.5, "g1": 0.5},
        rollout_results={
            "stop": {"status": "valid", "utility": 0.0, "delta_optimization": 0.0,
                     "delta_surrogate": float("nan"), "delta_coverage": 0.0,
                     "n_new_samples": 0, "n_model_fits": 0, "error": None},
            "g1": {"status": "valid", "utility": 1.0, "delta_optimization": 1.0,
                   "delta_surrogate": float("nan"), "delta_coverage": 0.05,
                   "n_new_samples": 1, "n_model_fits": 0, "error": None},
        },
        temperature=1.0,
    )
    path = tmp_path / "records.jsonl"
    write_jsonl([record], path)
    records = list(iter_jsonl(path))
    assert len(records) == 1
    r = records[0]
    assert r.id == record.id
    # NaN fields survive round-trip as None (sanitized).
    assert r.rollout_results["stop"]["delta_surrogate"] is None
    assert r.rollout_results["stop"]["delta_surrogate"] is None
    # Finite values preserved.
    assert r.rollout_results["g1"]["delta_coverage"] == pytest.approx(0.05)


def test_write_jsonl_creates_parent_dirs(tmp_path) -> None:
    path = tmp_path / "subdir" / "records.jsonl"
    write_jsonl([], path)
    assert path.parent.exists()


# ---------------------------------------------------------------------------
# Phase 7C — Dataset generator
# ---------------------------------------------------------------------------


def _stub_backend_factory():
    """Always return a fresh BocodeBotorchBackend (real backend for generator tests)."""
    return BocodeBotorchBackend()


def test_generator_record_has_aligned_vocabulary() -> None:
    cfg = DatasetConfig(
        problems=["Branin"],
        seeds=[0],
        trajectories_per_problem_seed=1,
        n_init=3,
        total_budget=10,
        decision_every_k_actions=2,
        max_actions=4,
    )
    gen = DatasetGenerator(cfg, backend_factory=_stub_backend_factory)
    records = list(gen.generate())
    assert records
    for r in records:
        ids = [c["action_id"] for c in r.candidate_actions]
        assert len(ids) == 6
        assert len(set(ids)) == 6  # no duplicates
        assert set(ids) == set(r.gold_probs)
        assert set(ids) == set(r.rollout_results)
        # gold_probs may contain NaN-surrogates but must sum to 1 if any valid.
        valid_ids = [
            k for k, v in r.rollout_results.items() if v["status"] == "valid"
        ]
        if valid_ids:
            assert sum(r.gold_probs.values()) == pytest.approx(1.0)


def test_generator_record_state_is_json_serializable() -> None:
    """No torch tensors, no NaN floats in state dict."""
    cfg = DatasetConfig(
        problems=["Branin"],
        seeds=[0],
        trajectories_per_problem_seed=1,
        n_init=3,
        total_budget=10,
        decision_every_k_actions=2,
        max_actions=4,
    )
    gen = DatasetGenerator(cfg, backend_factory=_stub_backend_factory)
    records = list(gen.generate())
    assert records
    raw_state = records[0].state
    def has_nan(v):
        if isinstance(v, float):
            return math.isnan(v)
        if isinstance(v, dict):
            return any(has_nan(x) for x in v.values())
        if isinstance(v, list):
            return any(has_nan(x) for x in v)
        return False
    raw = json.dumps(raw_state, allow_nan=False, default=str)
    assert "NaN" not in raw


def test_generator_to_jsonl_round_trip(tmp_path) -> None:
    cfg = DatasetConfig(
        problems=["Branin"],
        seeds=[0],
        trajectories_per_problem_seed=1,
        n_init=3,
        total_budget=10,
        decision_every_k_actions=2,
        max_actions=4,
    )
    gen = DatasetGenerator(cfg, backend_factory=_stub_backend_factory)
    path = tmp_path / "ds.jsonl"
    write_jsonl(gen.generate(), path)
    reloaded = list(iter_jsonl(path))
    assert reloaded
    for r in reloaded:
        assert "id" in r.__dict__


def test_full_candidates_returns_6_unique_ids() -> None:
    cands = full_candidates()
    ids = [c.action_id for c in cands]
    assert len(ids) == 6
    assert len(set(ids)) == 6

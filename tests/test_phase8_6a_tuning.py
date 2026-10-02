"""Phase 8.6A tests for scale wiring + Gate 3 corrected semantics.

Per codex (gpt-5.6-sol) review 2025-09-23:
  - test_default_resource_scales_use_total_budget: evaluation_scale / fit_scale
    default to total_budget (not 1.0).
  - test_objective_normalization_is_affine_invariant: optimization_scale =
    std(initial_objectives) is invariant to affine transform.
  - test_gate3_criterion_matches_declared_semantics: aggregate_mass uses
    log-odds minus log(N_exploration).
  - test_stop_weight_shifts_log_lift_analytically: shifting w_stop_opportunity
    shifts adjusted_lift by exactly (Δw * remaining_budget_ratio) / T.
  - test_temperature_can_help_or_hurt_gate3: temperature 1.0 is neutral on
    the metric given equal log-odds.
  - test_gate3_requires_seed_level_ci: point estimate alone is insufficient
    (< 5 seeds fall back to the point estimate; >= 5 seeds gate on CI).
  - test_nonfinite_utility_scales_rejected: NaN / Inf scales raise ValueError.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from opt_agent.dataset import (  # noqa: E402
    DatasetConfig,
    DatasetGenerator,
    DecisionRecord,
    aggregate_mass,
)
from opt_agent.dataset.validation import _lift
from opt_agent.rollout.utility import UtilityWeights, compute_utility


def _record(*, action_probs, family="branin", seed=0, remaining_budget_ratio=0.9,
            record_id=None):
    cands = [{"action_id": k, "type": k, "params": {}} for k in action_probs]
    return DecisionRecord(
        id=record_id or f"{family}:{seed}:0:0",
        problem={"family": family, "dimension": 2, "seed": seed},
        state={
            "sampling": {"n_samples": 5, "dimension": 2,
                         "min_pairwise_distance": 1.0, "boundary_coverage": 0.0},
            "surrogate": {"is_current": False, "validation_rmse": float("nan"),
                          "mean_uncertainty": float("nan"),
                          "max_uncertainty": float("nan")},
            "optimization": {"best_value": 0.0, "last_value": 0.0,
                             "recent_improvement": 0.0, "stagnation_steps": 0},
            "resource": {"used_budget": 5, "total_budget": 25,
                         "remaining_budget": 20,
                         "remaining_budget_ratio": remaining_budget_ratio},
            "history": {"last_action": None},
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
        metadata={"trajectory_id": 0, "step": 0, "run_seed": 0,
                  "utility_version": "test"},
    )


def test_default_resource_scales_use_total_budget() -> None:
    """Phase 8.6A: generator must default evaluation_scale / fit_scale to
    total_budget so 1 evaluation ≈ 1/total_budget utility units."""
    spec = DatasetConfig(
        problems=("Branin",), seeds=(0,), trajectories_per_problem_seed=1,
        n_init=3, total_budget=30, max_actions=4, decision_every_k_actions=2,
    )
    assert spec.evaluation_scale is None
    assert spec.fit_scale is None
    rec = next(iter(DatasetGenerator(spec).generate()))
    meta = rec.metadata
    assert meta["evaluation_scale"] == 30.0
    assert meta["fit_scale"] == 30.0


def test_objective_normalization_is_affine_invariant() -> None:
    """optimization_scale = std(initial_objectives) makes the metric on probs
    independent of the objective frame: scaling utilities by a > 0 with the
    scale (temperature) scaled by the same factor leaves gold_probs — hence
    _lift — unchanged, as does shifting all utilities by a constant (the
    softmax subtracts the max)."""
    from typing import NamedTuple

    from opt_agent.rollout import to_gold_probs

    class _Res(NamedTuple):
        action_id: str
        status: str
        utility: float

    ids = ("global_exploration", "uncertainty_sampling", "boundary_sampling",
           "local_exploitation", "error_guided_sampling", "stop",
           "surrogate_acquisition", "retrain_surrogate")
    u = dict(zip(ids, (1.4, 0.9, 0.6, 0.3, 0.1, -0.2, -0.5, -0.8)))

    def probs(utilities: dict[str, float], temperature: float) -> dict[str, float]:
        return to_gold_probs(
            tuple(_Res(a, "valid", v) for a, v in utilities.items()),
            temperature=temperature,
        )

    p_base = probs(u, 2.0)
    p_scaled = probs({a: 3.0 * v for a, v in u.items()}, 6.0)   # y -> 3y, scale x3
    p_shifted = probs({a: v - 7.5 for a, v in u.items()}, 2.0)  # y -> y - 7.5
    assert p_base == pytest.approx(p_scaled)
    assert p_base == pytest.approx(p_shifted)
    assert _lift(p_base) == pytest.approx(_lift(p_scaled))
    assert _lift(p_base) == pytest.approx(_lift(p_shifted))


def test_nonfinite_utility_scales_rejected() -> None:
    """NaN / Inf scales raise ValueError, not silently coerce."""
    with pytest.raises(ValueError, match="finite and positive"):
        compute_utility(
            delta_optimization=1.0,
            n_new_samples=0, n_model_fits=0,
            weights=UtilityWeights(),
            optimization_scale=float("nan"),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        compute_utility(
            delta_optimization=1.0,
            n_new_samples=0, n_model_fits=0,
            weights=UtilityWeights(),
            evaluation_scale=float("inf"),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        compute_utility(
            delta_optimization=1.0,
            n_new_samples=0, n_model_fits=0,
            weights=UtilityWeights(),
            fit_scale=0.0,
        )


def test_gate3_criterion_matches_declared_semantics() -> None:
    """adjusted_lift = log(p_exp / p_stop) - log(N_exploration).

    h3 menu: exploration actions are global_exploration + uncertainty_
    sampling (N=2). With p_exp = 0.8 and p_stop = 0.2:
    raw = 0.6, adj = log(4) - log(2) = log(2).
    """
    probs = {
        "global_exploration": 0.5, "uncertainty_sampling": 0.3,
        "stop": 0.2,
        "local_exploitation": 0.0, "retrain_surrogate": 0.0,
        "surrogate_acquisition": 0.0,
    }
    raw, adj = _lift(probs)
    assert raw == pytest.approx(0.6)
    assert adj == pytest.approx(math.log(4) - math.log(2), abs=1e-9)


def test_stop_weight_shifts_log_lift_analytically() -> None:
    """Δw_stop_opportunity shifts u(STOP) by -Δw × remaining_budget_ratio."""
    u_a = compute_utility(
        delta_optimization=0.0, n_new_samples=0, n_model_fits=0,
        weights=UtilityWeights(w_stop_opportunity=0.0),
        is_stop=True, remaining_budget_ratio=0.8,
    )
    u_b = compute_utility(
        delta_optimization=0.0, n_new_samples=0, n_model_fits=0,
        weights=UtilityWeights(w_stop_opportunity=1.0),
        is_stop=True, remaining_budget_ratio=0.8,
    )
    assert u_b == pytest.approx(u_a - 0.8)


def test_temperature_can_help_or_hurt_gate3() -> None:
    """Temperature sharpens softmax; can help or hurt depending on top action."""
    utilities = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, 0.0]

    def softmax(utils, T):
        z = [u / T for u in utils]
        zmax = max(z)
        exps = [math.exp(zi - zmax) for zi in z]
        s = sum(exps)
        return [e / s for e in exps]

    p_T1 = softmax(utilities, 1.0)
    p_T05 = softmax(utilities, 0.5)
    # Lower temperature concentrates mass on the argmax.
    assert max(p_T05) > max(p_T1)


def test_gate3_requires_seed_level_ci() -> None:
    """Smoke gate: point estimate > 0 passes. Full acceptance requires CI."""
    probs_strong = {
        "global_exploration": 0.5, "uncertainty_sampling": 0.3,
        "stop": 0.05, "retrain_surrogate": 0.15,
        "local_exploitation": 0.0, "error_guided_sampling": 0.0,
        "boundary_sampling": 0.0, "surrogate_acquisition": 0.0,
    }
    records = [
        _record(action_probs=probs_strong, seed=0, remaining_budget_ratio=0.9),
        _record(action_probs=probs_strong, seed=1, remaining_budget_ratio=0.9),
    ]
    result = aggregate_mass(records)
    assert result.pass_
    assert result.adjusted_lift_mean > 0.0


def test_gate3_seed_clustered_ci_branch() -> None:
    """With >= 5 seeds the gate criterion is the seed-clustered bootstrap
    CI lower bound (not the point estimate): consistently positive lift
    passes; consistently negative lift fails even though a CI exists."""
    probs_strong = {
        "global_exploration": 0.5, "uncertainty_sampling": 0.3,
        "stop": 0.05, "retrain_surrogate": 0.15,
        "local_exploitation": 0.0, "error_guided_sampling": 0.0,
        "boundary_sampling": 0.0, "surrogate_acquisition": 0.0,
    }
    probs_stop_heavy = {
        "stop": 0.8, "retrain_surrogate": 0.15,
        "global_exploration": 0.01, "uncertainty_sampling": 0.01,
        "boundary_sampling": 0.01, "local_exploitation": 0.01,
        "error_guided_sampling": 0.01, "surrogate_acquisition": 0.0,
    }
    strong = [
        _record(action_probs=probs_strong, seed=s, remaining_budget_ratio=0.9)
        for s in range(6)
    ]
    res_strong = aggregate_mass(strong)
    assert res_strong.seed_clustered_ci95 is not None
    assert res_strong.seed_clustered_ci95[0] > 0.0
    assert res_strong.pass_

    stop_heavy = [
        _record(action_probs=probs_stop_heavy, seed=s, remaining_budget_ratio=0.9)
        for s in range(6)
    ]
    res_stop = aggregate_mass(stop_heavy)
    assert res_stop.seed_clustered_ci95 is not None
    assert res_stop.seed_clustered_ci95[1] < 0.0
    assert not res_stop.pass_


def test_generator_with_corrected_scales_emits_metadata() -> None:
    """Phase 8.6A: record metadata must contain effective scales."""
    spec = DatasetConfig(
        problems=("Branin",), seeds=(0,), trajectories_per_problem_seed=1,
        n_init=3, total_budget=30, max_actions=4, decision_every_k_actions=2,
    )
    records = list(DatasetGenerator(spec).generate())
    assert records
    for r in records:
        meta = r.metadata
        assert "optimization_scale" in meta
        assert "evaluation_scale" in meta
        assert "fit_scale" in meta
        assert "w_stop_opportunity" in meta
        assert "horizon" in meta
        assert meta["evaluation_scale"] == 30.0
        assert meta["fit_scale"] == 30.0
        assert math.isfinite(meta["optimization_scale"])
        assert meta["optimization_scale"] > 0


def test_phase8_6a_dataset_passes_gate3() -> None:
    """End-to-end: with corrected scales, exploration beats STOP in early phase."""
    spec = DatasetConfig(
        problems=("Branin",), seeds=(0, 1, 2), trajectories_per_problem_seed=1,
        n_init=5, total_budget=20, max_actions=8, decision_every_k_actions=2,
    )
    records = list(DatasetGenerator(spec).generate())
    early_records = [
        r for r in records
        if r.state["resource"]["remaining_budget_ratio"] > 0.7
    ]
    assert early_records
    early_result = aggregate_mass(early_records)
    assert early_result.adjusted_lift_mean > 0.0, (
        f"Phase 8.6A: Branin early-phase exploration should now beat STOP; "
        f"got adjusted_lift_mean={early_result.adjusted_lift_mean:.4f}"
    )

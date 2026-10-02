"""Check the diagnostic's budget, replay, and random-stream guarantees."""
import torch
import pytest

from scripts.exp_attribution_suite import (
    SnapshotIOHBackend, _loop_task, branch, choices, new_backend, oracle_scores,
)
from opt_agent.actions import execute_action
from opt_agent.state import extract_state


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_terminal_branch_exhausts_evaluations_after_zero_eval_refit():
    be = new_backend({"family": "sixhumpcamel", "seed": 77})
    base = be.checkpoint()
    state = extract_state(be, total_budget=9)
    refit = choices(state, "original", 17)["retrain_surrogate"]
    result = branch(be, base, refit, budget=9, scale=1.0, kind="local", stream=101)
    assert result["terminal"]["n_evaluations"] == 9
    assert result["terminal"]["n_new"] == 4
    # h2 fit accounting: 1 explicit refit + 3 lazy fits. The continuation's
    # LOCAL_EXPLOITATION (PI) re-fits per evaluated point except the first
    # (the GP is fresh right after the refit). Pre-h2 this was 1 (Sobol /
    # perturbation samplers never fitted).
    assert result["terminal"]["n_fits"] == 4
    assert result["terminal"]["steps"] == 5
    assert result["short"]["n_evaluations"] == 7


def test_short_and_terminal_branches_have_identical_prefix():
    be = new_backend({"family": "sixhumpcamel", "seed": 78})
    base = be.checkpoint()
    state = extract_state(be, total_budget=12)
    first = choices(state, "unit", 18)["local_exploitation"]
    full = branch(be, base, first, budget=12, scale=1.0, kind="local", stream=102)
    short = branch(be, base, first, budget=12, scale=1.0, kind="local", stream=102, max_steps=3)
    assert full["short"] == short["terminal"]
    assert full["terminal"]["n_evaluations"] == 12
    assert short["terminal"]["n_evaluations"] == 8


def test_oracle_planning_restores_history_gp_and_rng():
    be = new_backend({"family": "sixhumpcamel", "seed": 79})
    be.fit()
    base = be.checkpoint()
    rng = torch.random.get_rng_state().clone()
    state = extract_state(be, total_budget=30)
    scores, cost = oracle_scores(be, state, menu="unit", kind="local", target="short",
                               scale=1.0, context={"family": "sixhumpcamel", "seed": 79}, decision=0)
    # h3: SAMPLING menu has 4 actions (was 6 under the h1 8-action menu).
    assert len(scores) == 4
    assert cost["planning_evaluations"] == 24
    assert torch.equal(be.get_history().X, base.X)
    assert torch.equal(be.get_history().objectives, base.objectives)
    assert be.is_fitted
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_ioh_snapshot_restores_fitted_model_after_sampling():
    be = new_backend({"fid": 6, "dim": 2, "instance": 9, "seed": 80})
    be.fit()
    base = be.checkpoint()
    X = be.get_history().X
    mean, std = be.predict(X)
    execute_action(be, extract_state(be, total_budget=30), choices(extract_state(be, total_budget=30), "unit", 19)["local_exploitation"])
    assert not be.is_fitted
    be.restore(base)
    restored_mean, restored_std = be.predict(X)
    assert torch.equal(mean, restored_mean)
    assert torch.equal(std, restored_std)
    assert len(be.get_history()) == 5


def test_unit_menu_and_full_loop_use_exact_budget():
    be = new_backend({"family": "sixhumpcamel", "seed": 81})
    state = extract_state(be, total_budget=30)
    unit = choices(state, "unit", 20)
    assert set(unit).isdisjoint({"stop", "retrain_surrogate"})
    assert all(a.params.get("n_samples", 1) == 1 for a in unit.values())
    task = {"key": "test", "context": {"family": "sixhumpcamel", "seed": 81},
            "policy": "local_exploitation", "scale": 1.0}
    first, second = _loop_task(task), _loop_task(task)
    assert first == second
    assert first["n_evaluations"] == 30
    assert len(first["actions"]) == 25


def test_normalization_ablation_maps_existing_bounds_to_unit_cube():
    original = new_backend({"family": "ackley", "seed": 82})
    normalized = new_backend({"family": "ackley", "seed": 82, "normalize_gp": True})
    assert torch.equal(original.get_history().X, normalized.get_history().X)
    normalized.fit()
    unit = normalized._model.input_transform(normalized.get_history().X)
    assert bool((unit >= 0).all()) and bool((unit <= 1).all())
    original.fit()
    assert not hasattr(original._model, "input_transform")

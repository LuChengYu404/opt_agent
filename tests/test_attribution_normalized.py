"""Corrected GP snapshots and matched unit/block attribution checks."""
from dataclasses import replace

import pytest
import torch

from scripts.exp_attribution_normalized import backend, branch, planning_scores
from opt_agent.actions import execute_action
from opt_agent.state import extract_state
from scripts.exp_attribution_suite import choices, SAMPLING


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


@pytest.mark.parametrize("normalized", [False, True])
def test_fitted_and_legacy_restore_preserve_predictions_and_setting(normalized):
    be = backend({"family": "sixhumpcamel", "seed": 711}, "normalized" if normalized else "raw")
    be.fit()
    cp = be.checkpoint()
    points = be.bounds.T.mean(dim=1).unsqueeze(0)
    expected = be.predict(points)
    # (legacy snapshots without a normalize_inputs stamp are no longer
    # restored — dropped in the 2026-10-01 slimming pass.)
    be.normalize_inputs = not normalized
    be.restore(cp)
    assert be.is_fitted
    assert be.normalize_inputs == normalized
    actual = be.predict(points)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


def test_unfitted_snapshot_preserves_normalization_setting():
    be = backend({"family": "sixhumpcamel", "seed": 711}, "normalized")
    cp = be.checkpoint()
    be.normalize_inputs = False
    be.restore(cp)
    assert be.normalize_inputs and not be.is_fitted
    be.fit()
    assert hasattr(be._model, "input_transform")


@pytest.mark.parametrize("action", ["uncertainty_sampling", "surrogate_acquisition"])
def test_all_gp_actions_use_normalization_and_restore_it_after_evaluation(action):
    be = backend({"family": "sixhumpcamel", "seed": 712}, "normalized")
    state = extract_state(be, total_budget=6)
    execute_action(be, state, choices(state, "unit", 39)[action])
    assert hasattr(be._model, "input_transform")
    cp = be.checkpoint()
    assert not cp.fitted and cp.normalize_inputs
    be.restore(cp)
    assert be.normalize_inputs


def test_blocks_share_prefix_and_fixed_intervention_is_identical():
    be = backend({"family": "sixhumpcamel", "seed": 714}, "normalized")
    be.fit()
    cp = be.checkpoint()
    args = dict(continuation="local_exploitation", stream=901, budget=10)
    unit = branch(be, cp, action="global_exploration", block=1, **args)
    block = branch(be, cp, action="global_exploration", block=5, **args)
    assert unit["first"] == block["first"]
    unit = branch(be, cp, action="local_exploitation", block=1, **args)
    block = branch(be, cp, action="local_exploitation", block=5, **args)
    assert unit == block and unit["n_new"] == 5


def test_planning_restores_normalized_gp_history_and_rng(monkeypatch):
    import scripts.exp_attribution_normalized as module
    real_branch = module.branch
    monkeypatch.setattr(module, "branch", lambda *a, **kw: real_branch(*a, **kw, budget=7))
    c = {"family": "sixhumpcamel", "seed": 714}
    be = backend(c, "normalized")
    be.fit()
    cp = be.checkpoint()
    predictions = be.predict(cp.X)
    rng = torch.random.get_rng_state().clone()
    scores, evaluations, fits = planning_scores(be, cp, c, 5, 1, "local_exploitation", replicas=1)
    # h3: SAMPLING menu has 4 actions (was 6 under the h1 8-action menu).
    assert len(scores) == 4 and evaluations == 8
    assert be.is_fitted and be.normalize_inputs
    assert torch.equal(be.get_history().X, cp.X)
    assert all(torch.equal(a, b) for a, b in zip(predictions, be.predict(cp.X)))
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_dataset_snapshot_roundtrip_preserves_unfitted_normalization():
    from types import SimpleNamespace
    from opt_agent.dataset.generator import DatasetConfig, DatasetGenerator, full_candidates
    from opt_agent.dataset.validation import _reconstruct_checkpoint
    be = backend({'family':'sixhumpcamel','seed':715}, 'normalized')
    gen = DatasetGenerator(DatasetConfig())
    record = gen._make_record(problem='SixHumpCamel', problem_seed=715,
        trajectory_id=0, step=0, run_seed=715, state=extract_state(be,total_budget=30),
        candidates=full_candidates(), output=SimpleNamespace(results=[]),
        probs={a:1/8 for a in (*SAMPLING,'retrain_surrogate','stop')},
        optimization_scale=1.0, evaluation_scale=1.0, fit_scale=1.0,
        replay_checkpoint=be.checkpoint())
    assert record.metadata['backend_normalize_inputs'] is True
    restored = backend({'family':'sixhumpcamel','seed':716}, 'raw')
    restored.restore(_reconstruct_checkpoint(record))
    assert restored.normalize_inputs and not restored.is_fitted
    del record.metadata['backend_normalize_inputs']
    assert _reconstruct_checkpoint(record).normalize_inputs is None

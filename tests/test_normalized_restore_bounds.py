"""A restored snapshot must carry its search box as well as its fitted GP."""
import pytest
import torch
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend


@pytest.mark.parametrize('initialized_elsewhere',[False,True])
def test_normalized_restore_sets_checkpoint_bounds(initialized_elsewhere):
    torch.set_num_threads(1)
    source=BocodeBotorchBackend(normalize_inputs=True)
    source.initialize(BackendConfig(problem_id='SixHumpCamel',seed=917,n_init=5))
    source.fit()
    cp=source.checkpoint()
    restored=BocodeBotorchBackend()
    if initialized_elsewhere:
        restored.initialize(BackendConfig(problem_id='Branin',seed=918,n_init=5))
    restored.restore(cp)
    assert restored.bounds is not None
    assert torch.equal(restored.bounds,source.bounds)
    assert restored.is_fitted and restored.normalize_inputs
    expected=source.predict(cp.X)
    actual=restored.predict(cp.X)
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))
    x=restored.propose()
    assert ((x>=source.bounds[0])&(x<=source.bounds[1])).all()

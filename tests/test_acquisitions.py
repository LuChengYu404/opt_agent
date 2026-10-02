"""LMABO AF portfolio registry tests (h2).

Covers: portfolio completeness + exploration/exploitation grouping
(arxiv_2602.07904 Appendix A, verified against the local PDF), unknown-name
rejection, backend propose() passthrough, and a real-backend sweep over all
12 AFs asserting finite, in-bounds, [1, d] proposals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.backend.acquisitions import (  # noqa: E402
    ACQUISITION_NAMES,
    ACQUISITION_SET,
    AF_DEFAULT,
    EXPLOITATIVE_AFS,
    EXPLORATORY_AFS,
    propose_candidate,
)
from opt_agent.backend.bocode_botorch import (  # noqa: E402
    BackendConfig,
    BocodeBotorchBackend,
)


@pytest.fixture(scope="module")
def fitted_branin():
    be = BocodeBotorchBackend()
    be.initialize(BackendConfig(problem_id="Branin", seed=0, n_init=5))
    be.fit()
    return be


def test_portfolio_matches_lmabo_appendix_a() -> None:
    assert len(ACQUISITION_NAMES) == 12
    assert ACQUISITION_SET == EXPLORATORY_AFS | EXPLOITATIVE_AFS
    assert not (EXPLORATORY_AFS & EXPLOITATIVE_AFS)
    # Paper's own grouping (Appendix A).
    assert EXPLORATORY_AFS == {
        "PosSTD", "UCB", "TS", "qKG", "qPES", "qMES", "qJES",
    }
    assert EXPLOITATIVE_AFS == {"PI", "LogPI", "EI", "LogEI", "PosMean"}
    # Default keeps the pre-portfolio (h1) trajectory behaviour.
    assert AF_DEFAULT == "LogEI"


@pytest.mark.parametrize("af", ACQUISITION_NAMES)
def test_propose_candidate_returns_in_bounds_point(fitted_branin, af) -> None:
    torch.manual_seed(7)
    best_f = float(fitted_branin.get_history().objectives.max())
    x = propose_candidate(
        af,
        model=fitted_branin._model,
        best_f=best_f,
        bounds=fitted_branin.bounds,
    )
    assert x.shape == (1, fitted_branin.bounds.shape[1])
    assert torch.isfinite(x).all()
    assert bool((x >= fitted_branin.bounds[0] - 1e-9).all())
    assert bool((x <= fitted_branin.bounds[1] + 1e-9).all())


def test_propose_candidate_rejects_unknown_name(fitted_branin) -> None:
    best_f = float(fitted_branin.get_history().objectives.max())
    with pytest.raises(ValueError, match="portfolio"):
        propose_candidate(
            "SuperEI",
            model=fitted_branin._model,
            best_f=best_f,
            bounds=fitted_branin.bounds,
        )



def test_backend_propose_with_restricted_bounds(fitted_branin) -> None:
    """The optional `bounds` override confines the maximizer (trust region)."""
    torch.manual_seed(13)
    center = torch.tensor([2.5, 7.5], dtype=torch.double)
    tr = torch.stack([center - 0.1, center + 0.1])  # [2, d]
    x = fitted_branin.propose(n=1, af="LogEI", bounds=tr)
    assert bool((x >= tr[0] - 1e-9).all())
    assert bool((x <= tr[1] + 1e-9).all())


# ---------------------------------------------------------------------------
# codex round-1 (finding C/E) regressions
# ---------------------------------------------------------------------------


def test_logei_default_path_matches_inline_reference(fitted_branin) -> None:
    """The registry's LogEI path is byte-identical to the pre-portfolio
    inline propose(): same acquisition, same optimizer settings, same RNG
    consumption."""
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.optim import optimize_acqf

    best_f = float(fitted_branin.get_history().objectives.max())

    def reference() -> torch.Tensor:
        ei = LogExpectedImprovement(model=fitted_branin._model, best_f=best_f)
        candidate, _ = optimize_acqf(
            acq_function=ei,
            bounds=fitted_branin.bounds,
            q=1,
            num_restarts=4,
            raw_samples=16,
        )
        return candidate.detach()

    rng = torch.random.get_rng_state().clone()
    x_ref = reference()
    torch.random.set_rng_state(rng)
    x_new = fitted_branin.propose(n=1)  # default af="LogEI"
    assert torch.equal(x_ref, x_new)


def test_ts_proposal_replays_after_checkpoint_restore() -> None:
    """TS randomness flows through the global torch RNG, so a checkpoint
    restore (history + GP + RNG) reproduces the exact proposal — the
    counterfactual-rollout fairness mechanism under the h2 sampler."""
    be = BocodeBotorchBackend()
    be.initialize(BackendConfig(problem_id="Branin", seed=11, n_init=5))
    be.fit()
    cp = be.checkpoint()
    x1 = be.propose(n=1, af="TS")
    be.evaluate(x1)
    be.restore(cp)
    x2 = be.propose(n=1, af="TS")
    assert torch.equal(x1, x2)


def test_pi_and_logei_propose_distinct_points(fitted_branin) -> None:
    """PI (rank by P(improve)) and LogEI (weight improvement magnitude) are
    distinct portfolio members: on this fitted Branin posterior their
    maximizers separate (PI exploits the incumbent basin, LogEI accepts
    uncertainty)."""
    torch.manual_seed(0)
    x_pi = fitted_branin.propose(n=1, af="PI")
    torch.manual_seed(0)
    x_ei = fitted_branin.propose(n=1, af="LogEI")
    assert torch.linalg.vector_norm(x_pi - x_ei).item() > 1e-3



def test_ioh_backend_checkpoint_roundtrip() -> None:
    """Round-2 MAJOR-2 regression + af forwarding: IOHBackend supports the
    Checkpoint protocol (history + fitted GP + RNG), enabling atomic greedy
    actions, and forwards `af` to the shared registry."""
    import ioh
    from scripts.ioh_backend import IOHBackend

    torch.manual_seed(29)
    be = IOHBackend()
    be.initialize(ioh.get_problem(fid=6, dimension=2, instance=1),
                  n_init=5, seed=0)
    be.fit()
    x_pi = be.propose(n=1, af="PI")  # registry forwarding (round-1 E)
    b = be.bounds
    assert bool((x_pi >= b[0] - 1e-9).all() and (x_pi <= b[1] + 1e-9).all())
    X = be.get_history().X
    mean, _ = be.predict(X)
    cp = be.checkpoint()
    be.evaluate(torch.tensor([[0.5, 0.5]], dtype=torch.double))
    be.restore(cp)
    assert len(be.get_history()) == 5
    mean2, _ = be.predict(X)
    assert torch.equal(mean, mean2)


def test_ioh_backend_multi_point_action_is_atomic() -> None:
    """Round-2 MAJOR-2 regression: with checkpoint support, a failing later
    greedy point on IOHBackend rolls back the earlier committed point."""
    import ioh
    from scripts.ioh_backend import IOHBackend

    from opt_agent.actions import execute_action
    from opt_agent.actions.models import ActionType, OptimizationAction
    from opt_agent.state import extract_state

    class FailSecondEvaluate(IOHBackend):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._n_eval = 0

        def evaluate(self, X):
            self._n_eval += 1
            if self._n_eval == 2:
                raise RuntimeError("injected second evaluation failure")
            return super().evaluate(X)

    torch.manual_seed(31)
    be = FailSecondEvaluate()
    be.initialize(ioh.get_problem(fid=6, dimension=2, instance=1),
                  n_init=5, seed=0)
    with pytest.raises(RuntimeError, match="injected second evaluation"):
        execute_action(
            be,
            extract_state(be, total_budget=20),
            OptimizationAction(ActionType.GLOBAL_EXPLORATION, {"n_samples": 2}),
        )
    assert len(be.get_history()) == 5

@pytest.mark.parametrize("af", ["qPES", "qMES", "qJES"])
def test_entropy_afs_normalized_consistent_and_physical(af) -> None:
    """Round-1 A + round-2 MAJOR-1 regression, consolidated: with
    normalize_inputs=True every entropy AF runs in the model's internal
    space (workspace posterior must equal the original posterior on BOTH a
    fresh fit and a restored train-mode GP) while returning points in the
    physical domain."""
    from opt_agent.backend.acquisitions import _entropy_workspace

    be = BocodeBotorchBackend(normalize_inputs=True)
    be.initialize(BackendConfig(problem_id="Branin", seed=13, n_init=6))
    be.fit()
    Xp = (torch.rand(5, 2, dtype=torch.double)
          * (be.bounds[1] - be.bounds[0]) + be.bounds[0]).clamp(
        min=be.bounds[0], max=be.bounds[1])

    def workspace_matches() -> None:
        m_af, b_af, _ = _entropy_workspace(be._model, be.bounds)
        Xn = be._model.input_transform(Xp)
        with torch.no_grad():
            ref = be._model.posterior(Xp).mean
            got = m_af.posterior(Xn).mean
        assert torch.allclose(ref, got, atol=1e-8)

    workspace_matches()  # fresh fit
    cp = be.checkpoint()
    torch.manual_seed(23)
    x = be.propose(n=1, af=af)  # no predict() first: restored train mode
    assert torch.isfinite(x).all()
    assert bool((x >= be.bounds[0] - 1e-9).all())
    assert bool((x <= be.bounds[1] + 1e-9).all())
    be.evaluate(x)
    be.restore(cp)
    # restore() leaves the GP in eval mode with normalized train inputs,
    # so the workspace's eval()-before-del ordering is exercised as a no-op.
    workspace_matches()

"""Guard the paired experiment against misaligned seeds and illegal actions."""
from __future__ import annotations

import pytest

from opt_agent.state.models import (
    HistoryState, OptimizationProgress, OptimizationState, ResourceState,
    SamplingState, SurrogateState,
)
from scripts.exp_rq1_rq2_closed_loop import RFPolicy, summarize
from scripts._shared_rq_features import ACTIONS


def _row(seed: int, policy: str, best: float, initial=None) -> dict:
    return {
        "family": "branin", "seed": seed, "budget": 10, "policy": policy,
        "initial_values": initial if initial is not None else [float(seed)],
        "initial_scale": 1.0, "terminal_best": best,
        "normalized_anytime_auc": best, "n_evaluations": 10,
        "actions": ["local_exploitation"],
    }


def test_summary_pairs_by_seed_even_when_rows_are_scrambled():
    rows = [
        _row(2, "rf_all", 12), _row(1, "surrogate", 4),
        _row(1, "rf_all", 5), _row(2, "surrogate", 10),
    ]
    paired = summarize(rows)["branin/budget10/rf_all"]["paired"]["surrogate"]
    assert paired["mean_raw_terminal_gain"] == pytest.approx(1.5)
    assert paired["wins_ties_losses"] == [2, 0, 0]


def test_summary_rejects_mismatched_initial_design():
    with pytest.raises(ValueError, match="initial observations"):
        summarize([
            _row(1, "rf_all", 5, initial=[1.0]),
            _row(1, "surrogate", 4, initial=[2.0]),
        ])


def test_rf_policy_selects_only_budget_valid_candidates():
    class Model:
        def __init__(self, score):
            self.score = score

        def predict(self, X):
            return [self.score]

    models = {a: Model(0.0) for a in ACTIONS}
    models["retrain_surrogate"] = Model(100.0)
    models["local_exploitation"] = Model(1.0)
    state = OptimizationState(
        sampling=SamplingState(5, 2, 0.2, 0.3),
        surrogate=SurrogateState(True, 0.1, 0.2, 0.3),
        optimization=OptimizationProgress(1.0, 1.0, 0.0, 1),
        resource=ResourceState(5, 10, 5, 0.5),
        history=HistoryState(None),
    )
    assert RFPolicy(models, 0).select_action(state).type.value == "local_exploitation"
    unfitted = OptimizationState(
        sampling=state.sampling,
        surrogate=SurrogateState(False, float("nan"), float("nan"), float("nan")),
        optimization=state.optimization,
        resource=state.resource,
        history=state.history,
    )
    assert RFPolicy(models, 0, allow_retrain=False).select_action(unfitted).type.value == "local_exploitation"

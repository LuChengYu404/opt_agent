"""Guard against counterfactual winner bias and mispaired causal contrasts."""
import numpy as np
import pytest

from scripts.analyze_attribution_suite import holm, loop_analysis, split_metrics


def test_split_replica_evaluation_exposes_a_noisy_oracle():
    # Two opposite lucky winners; evaluating each on the selection replicas
    # would report a spurious +2 advantage. Independent replicas reverse it.
    first = np.zeros((4, 6))
    second = np.zeros((4, 6))
    first[:2, 0], first[2:, 1] = 4, 4
    second[:2, 1], second[2:, 0] = 4, 4
    result = split_metrics([first, second])
    assert np.mean(result["split_heldout_gains"]) == pytest.approx(-2)
    assert result["top_choice_test_optimal_share"] == 0


def test_holm_controls_a_family_of_contrasts():
    assert holm({"a": .01, "b": .04, "c": .5}) == pytest.approx({"a": .03, "b": .08, "c": .5})


def loop_row(seed, policy, value, initial=None):
    return {"context": {"family": "branin", "seed": seed}, "policy": policy,
            "initial_values": [seed] if initial is None else initial,
            "terminal_best": value, "n_evaluations": 30, "unused_budget": 0,
            "n_fits": 0, "planning_evaluations": 0, "actions": ["local_exploitation"]}


def test_loop_contrasts_pair_by_context_not_row_order():
    rows = [loop_row(2, "oracle_terminal_local", 12), loop_row(1, "local_exploitation", 4),
            loop_row(1, "oracle_terminal_local", 5), loop_row(2, "local_exploitation", 10),
            loop_row(2, "schedule_0.4", 10), loop_row(1, "schedule_0.4", 4),
            loop_row(2, "oracle_short_local", 10), loop_row(1, "oracle_short_local", 4)]
    selections = {"branin": {"fixed": "local_exploitation", "schedule": "schedule_0.4"}}
    result = loop_analysis(rows, selections)
    assert result["branin"]["contrasts"]["oracle_terminal_local_minus_best_dev_fixed"]["mean"] == pytest.approx(1.5)


def test_loop_analysis_rejects_unequal_initial_observations():
    rows = [loop_row(1, "oracle_terminal_local", 5), loop_row(1, "local_exploitation", 4, initial=[2])]
    selections = {"branin": {"fixed": "local_exploitation", "schedule": "local_exploitation"}}
    with pytest.raises(ValueError, match="unequal initial"):
        loop_analysis(rows, selections)

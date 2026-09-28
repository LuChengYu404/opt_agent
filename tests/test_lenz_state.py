"""Lenz-style rich state + Jev-style presentation tests (2026-09-28).

Branch ``exp/lenz-state-jev-presentation``. Verifies:
  - extraction is an exact, pure-read materialization of (D_t, c_t, H_t)
    plus the frozen compressed summary;
  - the narrative contains the K/D/c/H sections;
  - the choice request is protocol-compatible with
    build_choice_request(narrative, executable) and round-trips through
    validate_choice_response.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from opt_agent.actions import candidate_actions_for_state  # noqa: E402
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend  # noqa: E402
from opt_agent.state import (  # noqa: E402
    LenzState,
    extract_lenz_state,
    extract_state,
)
from opt_agent.system1._prompt import (  # noqa: E402
    _executable_criterion,
    build_choice_request,
    validate_choice_response,
)
from opt_agent.system1.lenz_prompt import (  # noqa: E402
    build_lenz_choice_request,
    lenz_state_to_narrative,
)


@pytest.fixture(scope="module")
def branin_backend():
    be = BocodeBotorchBackend()
    be.initialize(BackendConfig(problem_id="Branin", seed=0, n_init=5))
    # Two extra Sobol-free evaluations so the history grows past the
    # initial design (deterministic points inside the Branin box).
    bounds = be.bounds
    extra = bounds[0] + torch.tensor([0.31, 0.47]) * (bounds[1] - bounds[0])
    be.evaluate(extra.unsqueeze(0))
    extra2 = bounds[0] + torch.tensor([0.83, 0.19]) * (bounds[1] - bounds[0])
    be.evaluate(extra2.unsqueeze(0))
    # Fit so the surrogate block holds real numbers (NaN != NaN would make
    # dataclass equality impossible) and the fitted=True path is covered.
    be.fit()
    return be


def test_extraction_materializes_the_full_tuple(branin_backend) -> None:
    history = branin_backend.get_history()
    lenz = extract_lenz_state(
        branin_backend, total_budget=20, decision_history=(None, None, "surrogate_acquisition")
    )

    # D_t: exact trial log, evaluation order, float equality.
    assert lenz.n_trials == len(history) == 7
    for trial, row, y in zip(
        lenz.trials, history.X.tolist(), history.objectives[:, 0].tolist()
    ):
        assert trial.x == tuple(row)
        assert trial.y == y

    # c_t: fixed Phase-1 configuration.
    assert lenz.config.problem == "Branin"
    assert lenz.config.dimension == 2
    assert lenz.config.surrogate == "SingleTaskGP"
    assert lenz.config.acquisition == "LogExpectedImprovement"
    assert lenz.config.objective_transform == (1.0, 0.0)
    assert lenz.config.surrogate_fitted == branin_backend.is_fitted
    lo, hi = branin_backend.bounds
    assert lenz.config.bounds == tuple(zip(lo.tolist(), hi.tolist()))

    # H_t is carried verbatim; summary.last_action mirrors its last entry.
    assert lenz.decision_history == (None, None, "surrogate_acquisition")
    assert lenz.summary.history.last_action == "surrogate_acquisition"

    # summary is exactly the frozen extraction.
    frozen = extract_state(
        branin_backend, total_budget=20, last_action="surrogate_acquisition"
    )
    assert lenz.summary == frozen

    # best_trial follows the MAX frame.
    assert lenz.best_trial().y == max(t.y for t in lenz.trials)


def test_extraction_is_pure_read(branin_backend) -> None:
    before = branin_backend.get_history().X.clone()
    fitted_before = branin_backend.is_fitted
    extract_lenz_state(branin_backend, total_budget=20)
    assert torch.equal(before, branin_backend.get_history().X)
    assert branin_backend.is_fitted == fitted_before


def test_decision_history_longer_than_history_rejected(branin_backend) -> None:
    with pytest.raises(ValueError, match="longer"):
        extract_lenz_state(
            branin_backend, total_budget=20,
            decision_history=tuple(["stop"] * 99),
        )


def test_narrative_has_all_four_sections(branin_backend) -> None:
    lenz = extract_lenz_state(
        branin_backend, total_budget=20,
        decision_history=(None, "global_exploration", "local_exploitation"),
    )
    text = lenz_state_to_narrative(lenz)

    # K_t — problem context.
    assert "'Branin'" in text and "2 continuous variables" in text
    # D_t — every trial coordinate and value, best marked.
    assert "Trial log (7 evaluations, best marked *):" in text
    for trial in lenz.trials:
        coords = ", ".join(f"{v:.6g}" for v in trial.x)
        assert f"f({coords}) = {trial.y:.6g}" in text
    assert "*" in text
    # c_t — configuration line.
    assert "Backend configuration:" in text
    assert "LogExpectedImprovement" in text
    # H_t — history with the initial-design convention.
    assert "Decision history: initial-design -> global_exploration -> local_exploitation" in text
    # Compressed summary tail identical to the frozen narrative.
    frozen_state_arg, _ = build_choice_request(
        lenz.summary, {}, state_format="narrative", criteria_format="executable"
    )
    assert frozen_state_arg in text


def test_choice_request_is_jev_protocol_compatible(branin_backend) -> None:
    lenz = extract_lenz_state(branin_backend, total_budget=20)
    choices = candidate_actions_for_state(lenz.summary, seed=0)

    state_arg, question = build_lenz_choice_request(lenz, choices)

    assert isinstance(state_arg, str)
    assert set(question) == {"next_action"}
    inner = question["next_action"]
    assert inner["type"] == "choice"
    assert set(inner["criteria"]) == set(choices)

    # Criteria and instructions are IDENTICAL to the frozen executable
    # format — only the state text is richer.
    frozen_arg, frozen_q = build_choice_request(
        lenz.summary, choices,
        state_format="narrative", criteria_format="executable",
    )
    assert inner["instructions"] == frozen_q["next_action"]["instructions"]
    for name, action in choices.items():
        assert inner["criteria"][name] == _executable_criterion(action, lenz.summary)
    assert state_arg != frozen_arg  # the whole point: richer state text

    # End-to-end: the standard parser accepts a synthetic answer.
    selected = next(iter(choices))
    probs = {k: (1.0 if k == selected else 0.0) for k in choices}
    raw = {"answers": {"next_action": {
        "type": "choice", "choice": selected, "probabilities": probs,
        "confidence": 0.9,
    }}}
    got_id, got_probs, got_conf = validate_choice_response(raw, choices)
    assert got_id == selected
    assert got_probs[selected] == pytest.approx(1.0)
    assert got_conf == pytest.approx(0.9)


def test_frozen_and_lenz_states_coexist(branin_backend) -> None:
    """LenzState is additive: the frozen tuple is untouched inside it."""
    lenz = extract_lenz_state(branin_backend, total_budget=20)
    assert isinstance(lenz, LenzState)
    # The frozen summary still satisfies its own invariants (budget math).
    s = lenz.summary
    assert s.resource.used_budget == lenz.n_trials
    assert s.resource.remaining_budget == 20 - lenz.n_trials


def test_lenz_state_from_record_roundtrip() -> None:
    """Rebuild from a generated record's replay metadata (JSONL path)."""
    from dataclasses import asdict

    from opt_agent.dataset import DatasetConfig, DatasetGenerator
    from opt_agent.dataset.decision_record import _json_safe
    from opt_agent.state import lenz_state_from_record

    spec = DatasetConfig(
        problems=("Branin",), seeds=(0,),
        trajectories_per_problem_seed=1, n_init=3, total_budget=12,
        max_actions=4, decision_every_k_actions=2,
    )
    record = next(iter(DatasetGenerator(spec).generate()))
    payload = _json_safe(record)
    lenz = lenz_state_from_record(payload)

    meta = payload["metadata"]
    assert lenz.n_trials == len(meta["backend_X"])
    for trial, xs, y in zip(
        lenz.trials, meta["backend_X"], meta["backend_objectives"]
    ):
        assert trial.x == tuple(xs)
        assert trial.y == y[0]
    assert lenz.config.problem == "Branin"
    assert lenz.config.objective_transform == (1.0, 0.0)
    assert lenz.config.surrogate_fitted == meta["backend_fitted"]
    # Frozen summary rebuilt from the record's state block.
    sd = payload["state"]
    assert lenz.summary.sampling.n_samples == sd["sampling"]["n_samples"]
    assert lenz.summary.resource.remaining_budget == sd["resource"]["remaining_budget"]
    assert lenz.summary.history.last_action == sd["history"].get("last_action")
    # H_t is truncated to last_action (dataset records carry nothing more).
    assert lenz.decision_history in ((), (sd["history"].get("last_action"),))


def test_llm_policy_accepts_lenz_state_with_injected_builder(branin_backend) -> None:
    """LLMSystem1Policy with request_builder=build_lenz_choice_request
    accepts a LenzState transparently (candidate extraction reads summary)."""
    from opt_agent.system1.llm_system1_policy import LLMSystem1Policy

    lenz = extract_lenz_state(branin_backend, total_budget=20)
    policy = LLMSystem1Policy(
        mode="mock", seed=0, request_builder=build_lenz_choice_request,
    )
    decision = policy.decide(lenz)
    assert decision.probabilities and abs(sum(decision.probabilities.values()) - 1.0) < 1e-6
    assert decision.model  # mock reports the heuristic model id

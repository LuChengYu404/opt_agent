"""Tests for JevStyleSystem1 (typed-decision model architecture)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.system1.typed_decision import (  # noqa: E402
    JevStyleSystem1,
    TypedDecisionConfig,
    gold_probs_to_target_tensor,
    state_to_text,
)


# ---------------------------------------------------------------------------
# Tiny config for fast tests
# ---------------------------------------------------------------------------


def _tiny_config() -> TypedDecisionConfig:
    """Override the encoder with the production default for unit tests.

    ModernBERT-base is already cached locally from the training script,
    so this avoids the HF download SSL retry that would otherwise add
    minutes to every test run.
    """
    return TypedDecisionConfig(
        encoder_name="answerdotai/ModernBERT-base",
        max_state_tokens=32,
    )


# ---------------------------------------------------------------------------
# state_to_text
# ---------------------------------------------------------------------------


def test_state_to_text_includes_all_subsystems() -> None:
    state = {
        "resource": {"remaining_budget": 5, "total_budget": 20},
        "sampling": {"n_samples": 3},
        "surrogate": {"is_current": True},
        "optimization": {
            "best_value": -0.4, "recent_improvement": 0.01,
            "stagnation_steps": 1,
        },
    }
    text = state_to_text(state)
    assert "remaining_budget=5/20" in text
    assert "n_samples=3" in text
    assert "surrogate_current=True" in text
    assert "best_value=-0.4" in text
    assert "stagnation_steps=1" in text


def test_state_to_text_handles_missing_fields() -> None:
    text = state_to_text({})
    assert "BO state:" in text
    # Missing fields render as "?" rather than raising.
    assert "remaining_budget=?/?" in text


# ---------------------------------------------------------------------------
# gold_probs_to_target_tensor
# ---------------------------------------------------------------------------


def test_gold_probs_to_target_renormalizes() -> None:
    cands = ["a", "b", "c"]
    target = gold_probs_to_target_tensor(
        {"a": 1.0, "b": 1.0, "c": 0.0}, cands,
    )
    assert pytest.approx(float(target.sum()), abs=1e-6) == 1.0
    assert target[0] == pytest.approx(0.5, abs=1e-6)
    assert target[2] == 0.0


def test_gold_probs_to_target_handles_missing_keys() -> None:
    """Candidates not in gold_probs get zero mass."""
    target = gold_probs_to_target_tensor(
        {"a": 1.0}, ["a", "b", "c"],
    )
    assert pytest.approx(float(target.sum()), abs=1e-6) == 1.0
    assert target[0] == 1.0
    assert target[1] == 0.0
    assert target[2] == 0.0


# ---------------------------------------------------------------------------
# JevStyleSystem1 forward pass
# ---------------------------------------------------------------------------


def test_jev_style_forward_returns_logits_and_confidence() -> None:
    """Forward pass produces per-candidate logits and a confidence scalar."""
    cfg = _tiny_config()
    model = JevStyleSystem1(cfg)
    state = {
        "resource": {"remaining_budget": 5, "total_budget": 20},
        "sampling": {"n_samples": 3},
        "surrogate": {"is_current": True},
        "optimization": {"best_value": -0.4, "recent_improvement": 0.01,
                         "stagnation_steps": 1},
    }
    candidates = ["global_exploration", "stop"]
    logits, confidence = model(state, candidates)
    assert logits.shape == (len(candidates),)
    assert 0.0 <= float(confidence.item()) <= 1.0


def test_jev_style_decide_distribution_is_probability_distribution() -> None:
    """decide_distribution returns a valid {candidate: prob} dict."""
    cfg = _tiny_config()
    model = JevStyleSystem1(cfg)
    state = {
        "resource": {"remaining_budget": 5, "total_budget": 20},
        "sampling": {"n_samples": 3},
        "surrogate": {"is_current": True},
        "optimization": {"best_value": -0.4, "recent_improvement": 0.01,
                         "stagnation_steps": 1},
    }
    candidates = ["global_exploration", "surrogate_acquisition", "stop"]
    probs, confidence = model.decide_distribution(state, candidates)
    assert set(probs.keys()) == set(candidates)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert pytest.approx(sum(probs.values()), abs=1e-5) == 1.0
    assert 0.0 <= confidence <= 1.0


def test_jev_style_handles_unknown_candidate_id() -> None:
    """An action id missing from the candidate pool falls back to
    embedding slot 0 (a learnable "unknown" token)."""
    cfg = _tiny_config()
    model = JevStyleSystem1(cfg)
    state = {"resource": {"remaining_budget": 5, "total_budget": 20},
             "sampling": {"n_samples": 3}, "surrogate": {"is_current": True},
             "optimization": {"best_value": -0.4}}
    # "fictional_action" is NOT in config.candidate_pool.
    candidates = ["fictional_action", "stop"]
    logits, _ = model(state, candidates)
    assert logits.shape == (2,)


def test_jev_style_is_eval_mode_by_default_after_init() -> None:
    """decide_distribution works without manually calling .eval() — the
    forward pass uses no_grad internally."""
    cfg = _tiny_config()
    model = JevStyleSystem1(cfg)
    state = {"resource": {"remaining_budget": 1, "total_budget": 10},
             "sampling": {"n_samples": 2}, "surrogate": {"is_current": True},
             "optimization": {"best_value": -0.1}}
    # No .eval() called — decide_distribution must still work.
    probs, _ = model.decide_distribution(state, ["stop", "global_exploration"])
    assert pytest.approx(sum(probs.values()), abs=1e-5) == 1.0
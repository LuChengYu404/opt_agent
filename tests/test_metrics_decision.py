"""Decision-quality metrics: top-1 / top-k agreement, KL, NLL vs gold_probs.

These measure agreement with rollout-derived soft labels (NOT terminal
optimization benefit — that lives in the evaluation layer via simple_regret).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.metrics import (  # noqa: E402
    aggregate_decision_metrics,
    kl_divergence,
    negative_log_likelihood_vs_gold,
    top_k_agreement,
    top_one_agreement,
)


def _delta(action: str, candidates: list[str]) -> dict[str, float]:
    return {a: (1.0 if a == action else 0.0) for a in candidates}


# ---------------------------------------------------------------------------
# top_one_agreement
# ---------------------------------------------------------------------------


def test_top_one_perfect_match() -> None:
    probs = [_delta("a", ["a", "b", "c"])]
    gold = [_delta("a", ["a", "b", "c"])]
    assert top_one_agreement(probs, gold) == 1.0


def test_top_one_full_mismatch() -> None:
    probs = [_delta("a", ["a", "b"])]
    gold = [_delta("b", ["a", "b"])]
    assert top_one_agreement(probs, gold) == 0.0


def test_top_one_partial_agreement() -> None:
    probs = [
        _delta("a", ["a", "b"]),
        _delta("b", ["a", "b"]),
        _delta("a", ["a", "b"]),
    ]
    gold = [
        _delta("a", ["a", "b"]),
        _delta("a", ["a", "b"]),
        _delta("b", ["a", "b"]),
    ]
    assert top_one_agreement(probs, gold) == pytest.approx(1.0 / 3.0)


def test_top_one_empty_returns_zero() -> None:
    assert top_one_agreement([], []) == 0.0


def test_top_one_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        top_one_agreement([{"a": 1.0}], [{"a": 1.0}, {"b": 1.0}])


# ---------------------------------------------------------------------------
# top_k_agreement
# ---------------------------------------------------------------------------


def test_top_k_agreement_monotonic_in_k() -> None:
    probs = [_delta("a", ["a", "b", "c"])]
    gold = [_delta("b", ["a", "b", "c"])]
    out = top_k_agreement(probs, gold, k_values=(1, 2, 3))
    assert out[1] == 0.0
    assert out[2] == 1.0
    assert out[3] == 1.0


def test_top_k_rejects_k_below_one() -> None:
    with pytest.raises(ValueError, match="k must be"):
        top_k_agreement(
            [{"a": 1.0}], [{"a": 1.0}], k_values=(0,)
        )


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------


def test_kl_identical_is_zero() -> None:
    p = {"a": 0.5, "b": 0.5}
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_kl_diverges_when_gold_has_zero_mass() -> None:
    """If model puts mass where gold has none → KL = +inf."""
    p = {"a": 0.5, "b": 0.5}
    q = {"a": 1.0, "b": 0.0}
    assert kl_divergence(p, q) == math.inf


def test_kl_uses_eps_to_avoid_log_zero_when_q_zero_and_p_zero() -> None:
    """If both are zero on a key, the key contributes 0 (no log)."""
    p = {"a": 1.0, "b": 0.0}
    q = {"a": 1.0, "b": 0.0}
    assert kl_divergence(p, q) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# NLL vs gold
# ---------------------------------------------------------------------------


def test_nll_vs_gold_uses_argmax() -> None:
    """Negative log likelihood at the gold's argmax action."""
    probs = {"a": 0.5, "b": 0.5}
    gold = {"a": 0.7, "b": 0.3}  # argmax = "a"
    assert negative_log_likelihood_vs_gold(probs, gold) == pytest.approx(math.log(2.0))


def test_nll_vs_gold_zero_mass_is_infinity() -> None:
    probs = {"a": 0.0, "b": 1.0}
    gold = {"a": 1.0, "b": 0.0}
    assert negative_log_likelihood_vs_gold(probs, gold) == math.inf


# ---------------------------------------------------------------------------
# aggregate_decision_metrics
# ---------------------------------------------------------------------------


def test_aggregate_handles_empty() -> None:
    result = aggregate_decision_metrics([], [])
    assert result.n_records == 0
    assert result.top_one_agreement == 0.0
    assert result.kl_mean == 0.0


def test_aggregate_perfect_alignment() -> None:
    probs = [_delta("a", ["a", "b"]), _delta("b", ["a", "b"])]
    gold = [_delta("a", ["a", "b"]), _delta("b", ["a", "b"])]
    result = aggregate_decision_metrics(probs, gold, k_values=(1, 2))
    assert result.top_one_agreement == 1.0
    assert result.top_k_agreements[1] == 1.0
    assert result.top_k_agreements[2] == 1.0
    assert result.kl_mean == pytest.approx(0.0, abs=1e-9)
    assert result.nll_mean == pytest.approx(0.0, abs=1e-9)
    assert result.n_records == 2


def test_aggregate_skips_infinite_in_mean() -> None:
    """Infinite KL/NLL should not poison the mean — finite values averaged.

    Record 1 has probs == gold (KL=0, NLL=0).
    Record 2 has probs on "b" while gold has zero mass on "b" → KL=inf, NLL=inf.
    Mean of finite KLs = 0.0 (only record 1 contributes); mean of finite
    NLLs = 0.0 likewise. The infinities are dropped, not propagated.
    """
    probs = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}]
    gold = [{"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0}]
    result = aggregate_decision_metrics(probs, gold, k_values=(1,))
    # Mean of finite KLs (just record 1's 0.0) = 0.0.
    assert result.kl_mean == pytest.approx(0.0)
    # Mean of finite NLLs (just record 1's 0.0) = 0.0.
    assert result.nll_mean == pytest.approx(0.0)
    assert result.top_one_agreement == 0.5  # 1 of 2 records agrees


def test_aggregate_all_infinite_returns_infinity() -> None:
    """When every record has +inf KL/NLL, the mean is +inf (no finite to average)."""
    probs = [{"a": 0.0, "b": 1.0}]
    gold = [{"a": 1.0, "b": 0.0}]
    result = aggregate_decision_metrics(probs, gold, k_values=(1,))
    assert math.isinf(result.kl_mean)
    assert math.isinf(result.nll_mean)
    assert result.n_records == 1

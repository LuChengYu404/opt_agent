"""Decision-quality metrics comparing a model's probabilities to a gold_probs
distribution (rollout-derived soft labels).

These measure label agreement, NOT terminal optimization benefit. The latter
is computed in the evaluation layer via `simple_regret` against a problem's
known optimum. Do not conflate agreement with optimization gain.

Per codex gpt-6-sol review 2026-09-23:
  - gold_probs is a rollout soft target; argmax(gold_probs) is not an
    independently observed "correct action" but a proxy.
  - Report agreement diagnostics separately from optimization benefit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DecisionResult:
    """Aggregate decision-quality metrics for a (probs, gold_probs) corpus.

    Attributes
    ----------
    top_one_agreement : float
        Fraction of records where argmax(probs) == argmax(gold_probs).
    top_k_agreements : dict[int, float]
        Per-k fraction of records where argmax(gold_probs) is in the top-k
        of probs. Keys are k values requested by the caller.
    kl_mean : float
        Mean KL(probs || gold_probs) over records; +inf if any P > 0 with
        Q == 0.
    nll_mean : float
        Mean -log P(argmax(gold_probs)) under probs.
    n_records : int
        Number of records included in the aggregate.
    """

    top_one_agreement: float
    top_k_agreements: dict[int, float]
    kl_mean: float
    nll_mean: float
    n_records: int


def _argmax(d: Mapping[str, float]) -> str:
    return max(d, key=d.get)


def top_one_agreement(
    probs: Sequence[Mapping[str, float]],
    gold_probs: Sequence[Mapping[str, float]],
) -> float:
    """Fraction of records where argmax(probs) == argmax(gold_probs)."""
    if not probs:
        return 0.0
    if len(probs) != len(gold_probs):
        raise ValueError(
            f"length mismatch: {len(probs)} vs {len(gold_probs)}"
        )
    hits = sum(
        1 for p, g in zip(probs, gold_probs) if _argmax(p) == _argmax(g)
    )
    return hits / len(probs)


def top_k_agreement(
    probs: Sequence[Mapping[str, float]],
    gold_probs: Sequence[Mapping[str, float]],
    k_values: Sequence[int] = (1, 2, 3, 5),
) -> dict[int, float]:
    """Per-k fraction of records where argmax(gold) is in top-k of probs."""
    if len(probs) != len(gold_probs):
        raise ValueError(
            f"length mismatch: {len(probs)} vs {len(gold_probs)}"
        )
    out: dict[int, float] = {}
    for k in k_values:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        hits = 0
        for p, g in zip(probs, gold_probs):
            ranked = sorted(p, key=lambda a: -p[a])
            if _argmax(g) in ranked[:k]:
                hits += 1
        out[k] = hits / len(probs) if probs else 0.0
    return out


def kl_divergence(
    probs: Mapping[str, float],
    gold_probs: Mapping[str, float],
    eps: float = 1e-12,
) -> float:
    """KL(probs || gold_probs) over the union of keys.

    Returns +inf if the model assigns positive mass to any key where gold
    assigns explicit zero. Missing keys in gold are smoothed with `eps` so
    a missing category does not blow up the divergence.
    """
    keys = set(probs) | set(gold_probs)
    kl = 0.0
    for k in keys:
        p = max(float(probs.get(k, 0.0)), 0.0)
        if p <= 0:
            continue
        if k in gold_probs:
            q = float(gold_probs[k])
            if q <= 0.0:
                return math.inf
            kl += p * math.log(p / q)
        else:
            # Key absent from gold: smooth with eps.
            kl += p * math.log(p / eps)
    return kl


def negative_log_likelihood_vs_gold(
    probs: Mapping[str, float], gold_probs: Mapping[str, float],
) -> float:
    """NLL of the model's probs under the gold's argmax outcome.

    Equivalent to `negative_log_likelihood(probs, argmax(gold_probs))` but
    inlined for clarity at the call site.
    """
    chosen = _argmax(gold_probs)
    p = float(probs.get(chosen, 0.0))
    if p <= 0.0:
        return math.inf
    return -math.log(p)


def aggregate_decision_metrics(
    probs: Sequence[Mapping[str, float]],
    gold_probs: Sequence[Mapping[str, float]],
    k_values: Sequence[int] = (1, 2, 3, 5),
) -> DecisionResult:
    """One-call summary over a (probs, gold_probs) corpus."""
    if len(probs) != len(gold_probs):
        raise ValueError(
            f"length mismatch: {len(probs)} vs {len(gold_probs)}"
        )
    if not probs:
        return DecisionResult(0.0, {}, 0.0, 0.0, 0)
    kls = [kl_divergence(p, g) for p, g in zip(probs, gold_probs)]
    # Filter infinities for the mean; report inf separately if any.
    finite_kls = [k for k in kls if math.isfinite(k)]
    kl_mean = sum(finite_kls) / len(finite_kls) if finite_kls else math.inf
    nlls = [
        negative_log_likelihood_vs_gold(p, g)
        for p, g in zip(probs, gold_probs)
    ]
    finite_nlls = [n for n in nlls if math.isfinite(n)]
    nll_mean = sum(finite_nlls) / len(finite_nlls) if finite_nlls else math.inf
    return DecisionResult(
        top_one_agreement=top_one_agreement(probs, gold_probs),
        top_k_agreements=top_k_agreement(probs, gold_probs, k_values),
        kl_mean=kl_mean,
        nll_mean=nll_mean,
        n_records=len(probs),
    )

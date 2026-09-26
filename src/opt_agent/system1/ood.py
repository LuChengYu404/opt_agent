"""Out-of-distribution (OOD) detection on OptimizationState.

Per MASTER_PLAN §13 ("System-1 knows when it is unreliable"), we want
the router to escalate when the current BO state looks unlike the
states the policy was trained on. This module implements a simple
z-score detector on a fixed set of budget-normalized features:

  - n_samples / total_budget
  - stagnation_steps / max(n_samples, 1)
  - remaining_budget_ratio
  - is_current (0/1)
  - recent_improvement / max(|best_value|, 1)   (relative improvement)

These four ratios are roughly comparable across problem families and
budgets. `recent_improvement / max(|best_value|, 1)` is bounded because
relative improvement is what matters for BO decisions, not absolute
objective value (which is problem-dependent).

OOD score
---------
The detector returns:
  score(state) in [0, 1] — fraction of features whose |z-score| exceeds
                            a per-feature threshold.
  is_ood(state) — boolean above a configurable cutoff (default 0.5).
  reason(state) — human-readable breakdown for diagnostics.

The detector is fit on the calibration split of a DecisionRecord corpus
(see `dataset.splits`). At fit time, NaN values are dropped and the
per-feature mean / std are cached.

Feature: relative_improvement
-----------------------------
`relative_improvement = ri / max(|best_value|, 1.0)`. For states where
|best_value| >= 1, this rescales the per-step improvement so the same
relative progress across two problems maps to the same feature value.
For |best_value| < 1 (early BO before much signal), the floor is 1.0
to avoid division by tiny numbers; the feature is then close to `ri`,
which is acceptable because BO progress at the very start of a run is
inherently high-variance and the OOD detector treats it as "noisy
signal" rather than a precise invariant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from opt_agent.state.models import OptimizationState
from opt_agent.system1._prompt import _finite


_FEATURES: tuple[str, ...] = (
    "sample_fraction",
    "stagnation_per_sample",
    "remaining_budget_ratio",
    "is_current",
    "relative_improvement",
)


@dataclass(frozen=True)
class FeatureStats:
    """Per-feature mean and std across the calibration corpus.

    `count` is the number of records used (NaN features are dropped).
    """

    means: dict[str, float]
    stds: dict[str, float]
    count: int


@dataclass(frozen=True)
class OODScore:
    score: float            # in [0, 1]
    is_ood: bool
    z_per_feature: dict[str, float]
    above_threshold: dict[str, bool]
    reason: str


class StateOODDetector:
    """Z-score OOD detector on a fixed feature set over OptimizationState.

    Per codex gpt-6-sol review 2026-09-23: keep the feature set small and
    budget-normalized so the detector can be fit on one calibration
    split and applied across problem families. Avoid features whose
    scale depends on the unknown optimum (e.g. raw `best_value`).
    """

    def __init__(
        self,
        *,
        z_threshold: float = 2.0,
        cutoff: float = 0.5,
    ) -> None:
        if z_threshold <= 0:
            raise ValueError(f"z_threshold must be > 0; got {z_threshold}")
        if not (0.0 <= cutoff <= 1.0):
            raise ValueError(f"cutoff must lie in [0, 1]; got {cutoff}")
        self.z_threshold = z_threshold
        self.cutoff = cutoff
        self._stats: FeatureStats | None = None

    @staticmethod
    def features(state: OptimizationState) -> dict[str, float | None]:
        """Extract the OOD feature dict from a state.

        Returns `None` for finite-violating entries so the fit step can
        skip them.
        """
        n = max(int(state.resource.total_budget), 1)
        n_samples = int(state.sampling.n_samples)
        stagnation = int(state.optimization.stagnation_steps)
        bv = float(state.optimization.best_value)
        ri = float(state.optimization.recent_improvement)
        # `relative_improvement` rescales the per-step improvement so
        # the same relative progress maps to the same feature value
        # across problems whose best_value magnitude differs. The
        # max(|bv|, 1.0) floor prevents tiny-magnitude early-BO values
        # from blowing up the ratio; this is acceptable because early
        # BO progress is inherently high-variance and the detector
        # treats it as noisy signal rather than a precise invariant.
        denom = max(abs(bv), 1.0)
        rel_improvement = ri / denom
        return {
            "sample_fraction": _finite(n_samples / n),
            "stagnation_per_sample": _finite(
                stagnation / max(n_samples, 1)
            ),
            "remaining_budget_ratio": _finite(
                float(state.resource.remaining_budget_ratio)
            ),
            "is_current": _finite(
                1.0 if state.surrogate.is_current else 0.0
            ),
            "relative_improvement": _finite(rel_improvement),
        }

    def fit(self, states: Sequence[OptimizationState]) -> FeatureStats:
        """Compute per-feature mean and std across a corpus."""
        per_feature: dict[str, list[float]] = {f: [] for f in _FEATURES}
        for s in states:
            fmap = self.features(s)
            for f, v in fmap.items():
                if v is None:
                    continue
                per_feature[f].append(v)
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for f in _FEATURES:
            vals = per_feature[f]
            if not vals:
                # No valid samples for this feature: use 0/1 prior.
                means[f] = 0.0
                stds[f] = 1.0
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(var) if var > 0 else 1.0
            means[f] = mean
            stds[f] = std
        self._stats = FeatureStats(means=means, stds=stds, count=len(states))
        return self._stats

    @property
    def stats(self) -> FeatureStats | None:
        return self._stats

    def score(self, state: OptimizationState) -> OODScore:
        if self._stats is None:
            raise RuntimeError("StateOODDetector.score called before fit()")
        fmap = self.features(state)
        z_per_feature: dict[str, float] = {}
        above: dict[str, bool] = {}
        reasons: list[str] = []
        for f in _FEATURES:
            v = fmap.get(f)
            if v is None:
                z_per_feature[f] = 0.0
                above[f] = False
                continue
            mean = self._stats.means[f]
            std = self._stats.stds[f]
            z = (v - mean) / std if std > 0 else 0.0
            z_per_feature[f] = z
            is_out = abs(z) > self.z_threshold
            above[f] = is_out
            if is_out:
                reasons.append(f"{f}={v:.3f} (z={z:+.2f})")
        # Score: fraction of features above threshold.
        if not _FEATURES:
            score_val = 0.0
        else:
            score_val = sum(above.values()) / len(_FEATURES)
        return OODScore(
            score=score_val,
            is_ood=score_val > self.cutoff,
            z_per_feature=z_per_feature,
            above_threshold=above,
            reason=(
                f"out-of-distribution features ({int(score_val * len(_FEATURES))}/{len(_FEATURES)}): "
                + "; ".join(reasons)
                if reasons
                else "all features within calibration distribution"
            ),
        )
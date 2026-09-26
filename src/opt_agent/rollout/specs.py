"""Continuation policy specs (Sprint 0).

Per codex (gpt-6-sol) Q-T2 review of DISCUSS.md (2026-09-25):
- Continuation policy name + seed + (optionally) extras must be in dataset
  metadata so that any record can be replayed with a different continuation.
- Per codex Q-T1, horizon is an int; "until budget exhausted" is achieved by
  passing `state.resource.remaining_budget` as the horizon value at the call
  site — `CounterfactualRollout` itself only sees an int.

Sprint 0 implements 3 policies: `rule` / `uniform` / `on_policy`. The spec
dataclass also accepts `best_of_k` / `laya_zeroshot` names for forward
compatibility, but `build_continuation_policy` raises if asked to construct
one — they are deferred to a later sprint per the user's Decision 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ContinuationPolicySpec:
    """Specification of a continuation policy.

    `name` must be a key in `CONTINUATION_POLICY_REGISTRY`. The factory
    receives the full spec and returns a fresh `PolicyLike` instance per call
    (so per-call RNG state is independent).

    `seed` is the canonical seed passed to the policy's constructor.

    `k` is reserved for the future `best_of_k` policy (not implemented in
    Sprint 0).

    `extras` holds policy-specific kwargs (e.g. `rule_weight` for the
    `on_policy` mix). Default empty mapping.
    """
    name: str
    seed: int = 0
    k: int | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry — lazy imports inside factories to avoid circular dependency with
# opt_agent.loop.policies (which in turn doesn't import this module).
# ---------------------------------------------------------------------------


def _make_rule(spec: ContinuationPolicySpec) -> Any:
    from opt_agent.loop.policies import RulePolicy
    return RulePolicy(seed=spec.seed)


def _make_uniform(spec: ContinuationPolicySpec) -> Any:
    from opt_agent.loop.policies import UniformPolicy
    return UniformPolicy(seed=spec.seed)


def _make_on_policy(spec: ContinuationPolicySpec) -> Any:
    from opt_agent.loop.policies import OnPolicyMix
    rule_weight = float(spec.extras.get("rule_weight", 0.8))
    return OnPolicyMix(seed=spec.seed, rule_weight=rule_weight)


CONTINUATION_POLICY_REGISTRY: dict[str, Callable[[ContinuationPolicySpec], Any]] = {
    "rule": _make_rule,
    "uniform": _make_uniform,
    "on_policy": _make_on_policy,
}


def registered_policy_names() -> tuple[str, ...]:
    """Return the registered continuation policy names (sorted)."""
    return tuple(sorted(CONTINUATION_POLICY_REGISTRY))


def build_continuation_policy(spec: ContinuationPolicySpec) -> Any:
    """Build a fresh `PolicyLike` from a `ContinuationPolicySpec`.

    Raises
    ------
    ValueError
        If `spec.name` is not in the registry. The error message lists the
        registered names so the caller can fix the config.
    """
    if spec.name not in CONTINUATION_POLICY_REGISTRY:
        raise ValueError(
            f"unknown continuation policy {spec.name!r}; "
            f"registered: {sorted(CONTINUATION_POLICY_REGISTRY)}"
        )
    return CONTINUATION_POLICY_REGISTRY[spec.name](spec)


def spec_to_metadata(spec: ContinuationPolicySpec) -> dict[str, Any]:
    """Serialize a spec to a JSON-safe dict (for dataset metadata)."""
    return {
        "name": str(spec.name),
        "seed": int(spec.seed),
        "k": spec.k,
        "extras": {str(k): _json_safe(v) for k, v in spec.extras.items()},
    }


def spec_from_metadata(meta: Mapping[str, Any] | None) -> ContinuationPolicySpec:
    """Reconstruct a spec from metadata dict. Falls back to rule/seed=0."""
    if meta is None:
        return ContinuationPolicySpec(name="rule", seed=0)
    return ContinuationPolicySpec(
        name=str(meta.get("name", "rule")),
        seed=int(meta.get("seed", 0)),
        k=meta.get("k"),
        extras=dict(meta.get("extras") or {}),
    )


def _json_safe(value: Any) -> Any:
    """Best-effort JSON-safe coercion. Delegates to
    ``opt_agent.dataset.decision_record._json_safe`` (the comprehensive
    version that walks dataclasses / enums / mappings); a deferred
    import avoids a circular module dependency at import time.
    """
    from opt_agent.dataset.decision_record import _json_safe as _impl
    return _impl(value)

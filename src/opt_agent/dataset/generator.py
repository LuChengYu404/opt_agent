"""Optimization Decision Dataset Generator (Phase 7 starter, Phase 8 main).

Per codex (gpt-5.6-sol) review 2025-09-23:
  - Behavior policy (RulePolicy + random mix) is INDEPENDENT of gold_probs;
    never sample from gold_probs during trajectory walking.
  - Decision points are extracted at every Nth step (sparse by default).
  - Candidates are the FULL action vocabulary (8 actions) per record; ids
    are deterministic to ensure aligned vocabulary across records.
  - `derive_run_seed` uses SHA-256 of problem name (NOT Python's built-in
    `hash()`, which varies with PYTHONHASHSEED across processes).
"""

from __future__ import annotations

import base64
import hashlib
import math
import random
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Iterator

from opt_agent.actions import (
    ActionType,
    OptimizationAction,
    execute_action,
)
from opt_agent.actions.factory import ALL_CANDIDATE_SPECS
from opt_agent.actions.models import ActionResult
from opt_agent.backend.bocode_botorch import BackendConfig, UnifiedBackend
from opt_agent.backend.checkpoint import Checkpoint
from opt_agent.dataset.decision_record import DecisionRecord, _json_safe
from opt_agent.loop.policies import RandomPolicy, RulePolicy
from opt_agent.rollout import (
    Candidate,
    CounterfactualRollout,
    to_gold_probs,
)
from opt_agent.rollout.specs import ContinuationPolicySpec, spec_to_metadata
from opt_agent.rollout.utility import UtilityWeights
from opt_agent.state import OptimizationState, extract_state


def full_candidates(seed: int = 0) -> list[Candidate]:
    """Return the unfiltered 8-action candidate set used by the dataset.

    Per codex M1: this reads `actions.factory.ALL_CANDIDATE_SPECS` — the
    SINGLE source of truth for the action vocabulary. The runtime factory
    (`actions.candidate_actions_for_state`) filters this same list by
    state. There is no second 8-action list anywhere in the codebase.
    """
    out: list[Candidate] = []
    for action_id, action_type, params in ALL_CANDIDATE_SPECS:
        merged = dict(params)
        if "seed" in params or action_type in {
            ActionType.GLOBAL_EXPLORATION, ActionType.LOCAL_EXPLOITATION,
            ActionType.UNCERTAINTY_SAMPLING, ActionType.ERROR_GUIDED_SAMPLING,
            ActionType.BOUNDARY_SAMPLING,
        }:
            merged["seed"] = seed
        out.append(Candidate(
            action_id=action_id,
            action=OptimizationAction(action_type, merged),
        ))
    return out


def derive_seed(*parts: int) -> int:
    """Deterministic 31-bit seed from multiple integers."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.to_bytes(8, "little", signed=False))
        h.update(b"|")
    return int.from_bytes(h.digest()[:4], "little") & 0x7FFFFFFF


# Domain-separation salt for the long-horizon quota subsample (2026-09-27).
# Keeps the k-subset choice independent from the candidate-param seed stream.
_SUBSAMPLE_SALT = 918273645


def subsample_candidates(
    candidates: list[Candidate],
    k: int,
    seed: int,
) -> list[Candidate]:
    """Deterministically keep k of `candidates` (order-preserving).

    The subset is drawn with `random.Random(seed)` so the same
    (run_seed, step, salt) triple always yields the same subset — this keeps
    partial records replayable and keeps affine-transform pairs aligned
    (Gate 5 compares records by id with the same run_seed/step).
    """
    if not 1 <= k <= len(candidates):
        raise ValueError(
            f"subsample size must be in [1, {len(candidates)}]; got k={k}"
        )
    if k == len(candidates):
        return list(candidates)
    rng = random.Random(seed)
    chosen = set(rng.sample(range(len(candidates)), k))
    return [c for i, c in enumerate(candidates) if i in chosen]


def _average_rollouts(outputs: list) -> "RolloutOutput":
    """Merge replicate rollouts of the SAME candidate subset (P1b denoising).

    Per action: utility/deltas/counts are means over the replicas where the
    action was valid; the per-replica utilities and their std are stashed in
    ``metrics['utility_replicas']`` / ``metrics['utility_std']``. An action
    valid in no replica is emitted as-is from the first output (invalid).
    T2: averaging leaves the expected ranking unchanged and shrinks utility
    noise as 1/sqrt(R).
    """
    import statistics as _stats

    from opt_agent.rollout import CounterfactualResult, RolloutOutput

    base = outputs[0]
    merged: list[CounterfactualResult] = []
    for first in base.results:
        replicas = [
            r for out in outputs
            for r in out.results
            if r.action_id == first.action_id and r.status == "valid"
        ]
        if not replicas:
            merged.append(first)
            continue

        def _mean(attr: str) -> float:
            vals = [getattr(r, attr) for r in replicas]
            return float(_stats.fmean(vals))

        utilities = [r.utility for r in replicas]
        metrics = dict(first.metrics)
        metrics["utility_replicas"] = utilities
        metrics["utility_std"] = (
            float(_stats.pstdev(utilities)) if len(utilities) > 1 else 0.0
        )
        merged.append(CounterfactualResult(
            action_id=first.action_id,
            action=first.action,
            status="valid",
            n_new_samples=_mean("n_new_samples"),
            n_model_fits=_mean("n_model_fits"),
            delta_optimization=_mean("delta_optimization"),
            delta_surrogate=_mean("delta_surrogate"),
            delta_coverage=_mean("delta_coverage"),
            utility=float(_stats.fmean(utilities)),
            metrics=metrics,
            trajectory_tail=first.trajectory_tail,
            error=None,
        ))
    return RolloutOutput(s_t=base.s_t, results=tuple(merged))


def derive_run_seed(problem: str, problem_seed: int, trajectory_id: int) -> int:
    """Process-stable seed combining (problem_name, problem_seed, trajectory_id).

    Uses SHA-256 of the problem name to avoid Python's built-in `hash()`
    which depends on PYTHONHASHSEED and varies across processes.
    """
    digest = hashlib.sha256(problem.encode("utf-8")).digest()
    problem_key = int.from_bytes(digest[:8], "little")
    return derive_seed(problem_seed, trajectory_id, problem_key)


@dataclass(frozen=True)
class DatasetConfig:
    problems: tuple[str, ...] = ("Branin", "Hartmann", "Ackley")
    seeds: tuple[int, ...] = (0,)
    n_init: int = 5
    total_budget: int = 30
    trajectories_per_problem_seed: int = 1
    decision_every_k_actions: int = 2  # extract decision every 2 actions
    max_actions: int = 25
    horizon: int = 1
    utility_weights: UtilityWeights = field(default_factory=UtilityWeights)
    temperature: float = 1.0
    random_action_rate: float = 0.2  # 20% random, 80% rule
    # Phase 8.6A oracle-gate fix (per codex 2025-09-23):
    # - evaluation_scale / fit_scale default to None → generator sets
    #   evaluation_scale = fit_scale = total_budget (so 1 evaluation ≈
    #   1/total_budget utility units, comparable to Δopt).
    # - optimization_scale default to None → generator computes
    #   std(initial_objectives) per trajectory for cross-problem invariance.
    optimization_scale: float | None = None
    evaluation_scale: float | None = None
    fit_scale: float | None = None
    # Phase 9 Gate 5 — affine rescaling of the objective (codex §27).
    # Applied at write-time inside BocodeBotorchBackend so GP fits, state
    # extractor, optimizer, and counterfactual rollouts all see the
    # transformed scale consistently. Default (1.0, 0.0) is the identity.
    objective_scale: float = 1.0
    objective_shift: float = 0.0
    # Long-horizon quota control (2026-09-27): when set to k < |vocab|, each
    # decision point rolls out only a deterministic k-subset of the candidate
    # vocabulary at the configured horizon. Rollout cost scales as k/|vocab|.
    # `candidate_actions` still lists the full vocabulary; the record's
    # `gold_probs`/`rollout_results` cover only the rolled-out subset and
    # metadata["partial_rollout"] records which ids were rolled out. Pairwise
    # consumers (contrastive distillation) are unaffected by construction;
    # softmax-target consumers must read `partial_rollout` first (renormalizing
    # a subset softmax to the full vocabulary would inflate the top gap).
    rollout_action_subsample: int | None = None
    # "fixed": use `horizon` for every decision point (historical default).
    # "remaining": per decision point, roll out until the budget is exhausted
    # (horizon = remaining_budget) — the true long-horizon semantics from the
    # Phase 8.6C horizon diagnosis. metadata["horizon"] records the actual
    # per-record value; metadata["horizon_mode"] records the mode.
    horizon_mode: str = "fixed"
    # P1b denoising (2026-09-27, per the noise-vs-misspecification diagnostic
    # `exp_teacher_noise_diagnostic.py`: single-rollout labels are
    # NOISE-dominated — inter-roll gap spread 0.084 > median stored gap 0.05,
    # above-median-gap flip rate 0.74). When R > 1, each decision point's
    # rollout is repeated R times with fresh candidate/continuation seeds and
    # the per-action utilities are AVERAGED (T2: noise shrinks as 1/sqrt(R);
    # the expected ranking is unchanged). rollout_results carry the mean
    # utility plus `utility_replicas` and `utility_std` for downstream
    # uncertainty accounting.
    rollout_replicates: int = 1
    # Sprint 0 (per codex Q-T2 2026-09-25): continuation policy spec, used by
    # `CounterfactualRollout` for follow-up steps and recorded in metadata
    # so records can be replayed with a different continuation. Default:
    # "on_policy" matching the dataset's behavior mix.
    continuation_spec: ContinuationPolicySpec = field(
        default_factory=lambda: ContinuationPolicySpec(
            name="on_policy",
            seed=0,
            extras={"rule_weight": 0.8},
        )
    )

    def __post_init__(self) -> None:
        """Validate Phase 9 Gate 5 affine params eagerly.

        `objective_scale` must be finite and strictly positive; `objective_shift`
        must be finite. Rejecting at config construction time (rather than
        deep inside generate()) gives a clearer error and avoids a partially-
        constructed DatasetGenerator if downstream code re-uses the factory.
        """
        scale = float(self.objective_scale)
        shift = float(self.objective_shift)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"DatasetConfig.objective_scale must be finite and > 0; got {scale!r}"
            )
        if not math.isfinite(shift):
            raise ValueError(
                f"DatasetConfig.objective_shift must be finite; got {shift!r}"
            )
        k = self.rollout_action_subsample
        if k is not None and not 1 <= k <= len(ALL_CANDIDATE_SPECS):
            raise ValueError(
                f"rollout_action_subsample must be None or in "
                f"[1, {len(ALL_CANDIDATE_SPECS)}]; got {k!r}"
            )
        if self.horizon_mode not in ("fixed", "remaining"):
            raise ValueError(
                f"horizon_mode must be 'fixed' or 'remaining'; "
                f"got {self.horizon_mode!r}"
            )
        if self.horizon_mode == "fixed" and self.horizon < 1:
            raise ValueError(f"horizon must be >= 1 in fixed mode; got {self.horizon}")
        if not 1 <= self.rollout_replicates <= 8:
            raise ValueError(
                f"rollout_replicates must be in [1, 8]; "
                f"got {self.rollout_replicates}"
            )


BackendFactory = Callable[[], UnifiedBackend]


class DatasetGenerator:
    def __init__(
        self,
        config: DatasetConfig,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        self.config = config
        # Default factory: real BocodeBotorchBackend configured with the
        # affine objective transform from DatasetConfig. Validates scale/shift
        # eagerly so misuse raises at __init__ time, not deep inside generate().
        if backend_factory is None:
            scale = float(config.objective_scale)
            shift = float(config.objective_shift)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(
                    f"DatasetConfig.objective_scale must be finite and > 0; got {scale!r}"
                )
            if not math.isfinite(shift):
                raise ValueError(
                    f"DatasetConfig.objective_shift must be finite; got {shift!r}"
                )

            def _factory() -> UnifiedBackend:
                from opt_agent.backend.bocode_botorch import BocodeBotorchBackend
                return BocodeBotorchBackend(
                    objective_scale=scale, objective_shift=shift
                )

            self.backend_factory = _factory
        else:
            self.backend_factory = backend_factory

    @staticmethod
    def _default_factory() -> UnifiedBackend:
        from opt_agent.backend.bocode_botorch import BocodeBotorchBackend
        # Threaded from DatasetConfig in __init__ via closure set in
        # `_make_default_factory`; bare default below for tests/CLI that
        # use the static API directly without going through __init__.
        return BocodeBotorchBackend()

    def generate(self) -> Iterator[DecisionRecord]:
        for problem in self.config.problems:
            for problem_seed in self.config.seeds:
                for trajectory_id in range(self.config.trajectories_per_problem_seed):
                    yield from self._one_trajectory(problem, problem_seed, trajectory_id)

    def _one_trajectory(
        self,
        problem: str,
        problem_seed: int,
        trajectory_id: int,
    ) -> Iterator[DecisionRecord]:
        run_seed = derive_run_seed(problem, problem_seed, trajectory_id)
        backend = self.backend_factory()
        backend.initialize(BackendConfig(
            problem_id=problem,
            seed=problem_seed,
            n_init=self.config.n_init,
        ))
        # Phase 8.6A: per-trajectory scales — defaults to budget-based for
        # resource costs and std(initial_objectives) for Δopt normalization.
        initial_y = backend.get_history().objectives[:, 0]
        obj_std = float(initial_y.std(correction=0).item())
        if not math.isfinite(obj_std) or obj_std <= 0:
            obj_std = 1.0
        optimization_scale = (
            obj_std if self.config.optimization_scale is None
            else float(self.config.optimization_scale)
        )
        evaluation_scale = (
            float(self.config.total_budget)
            if self.config.evaluation_scale is None
            else float(self.config.evaluation_scale)
        )
        fit_scale = (
            float(self.config.total_budget)
            if self.config.fit_scale is None
            else float(self.config.fit_scale)
        )

        rule = RulePolicy(seed=run_seed)
        random_policy = RandomPolicy(seed=run_seed)
        behavior_rng = random.Random(run_seed)
        last_action_name: str | None = None  # Phase 8.5 fix: track for state.history.last_action

        for step in range(self.config.max_actions):
            state = extract_state(
                backend,
                total_budget=self.config.total_budget,
                last_action=last_action_name,
            )

            # Decision point?
            if step % self.config.decision_every_k_actions == 0:
                remaining = int(state.resource.remaining_budget)
                h_step = (
                    max(1, remaining)
                    if self.config.horizon_mode == "remaining"
                    else self.config.horizon
                )
                if self.config.horizon_mode == "remaining" and remaining <= 0:
                    break  # nothing left to decide; budget exhausted
                cands = full_candidates(seed=derive_seed(run_seed, step))
                k = self.config.rollout_action_subsample
                if k is not None and k < len(cands):
                    rolled = subsample_candidates(
                        cands, k, derive_seed(run_seed, step, _SUBSAMPLE_SALT)
                    )
                else:
                    rolled = cands
                # Sprint 0 fix: scale the rollout cap to rolled candidates ×
                # horizon so long-horizon runs (e.g. horizon=total_budget)
                # don't hit the default cap of 200.
                rollout_cap = max(200, len(rolled) * h_step + 32)
                R = self.config.rollout_replicates
                subset_seed = derive_seed(run_seed, step, _SUBSAMPLE_SALT)
                outputs = []
                for rep in range(R):
                    # Same rolled subset every replicate (deterministic salt);
                    # fresh action seeds and continuation seed per replicate.
                    rep_cands = (
                        cands if R == 1
                        else full_candidates(seed=derive_seed(run_seed, step, rep))
                    )
                    rep_rolled = rolled if R == 1 else subsample_candidates(
                        rep_cands, len(rolled), subset_seed
                    )
                    outputs.append(CounterfactualRollout(
                        backend,
                        horizon=h_step,
                        utility_weights=self.config.utility_weights,
                        continuation_spec=(
                            self.config.continuation_spec if R == 1 else
                            replace(
                                self.config.continuation_spec,
                                seed=derive_seed(
                                    int(self.config.continuation_spec.seed),
                                    rep, 424242,
                                ),
                            )
                        ),
                        max_rollout_steps=rollout_cap,
                        optimization_scale=optimization_scale,
                        evaluation_scale=evaluation_scale,
                        fit_scale=fit_scale,
                    ).rollout(state, rep_rolled))
                output = outputs[0] if R == 1 else _average_rollouts(outputs)
                probs = to_gold_probs(
                    output.results,
                    temperature=self.config.temperature,
                    include_invalid_as_zero=True,
                )
                partial_meta: dict | None = None
                if rolled is not cands:
                    partial_meta = {
                        "k": int(k),
                        "action_ids": [c.action_id for c in rolled],
                        "note": (
                            "gold_probs and rollout_results cover only these "
                            "action_ids; do not renormalize to the full "
                            "vocabulary. Pairwise (contrastive) consumers are "
                            "unaffected by construction."
                        ),
                    }
                # Preserve the complete decision point, including a fitted GP
                # and RNG state. History alone cannot faithfully replay it.
                replay_checkpoint = backend.checkpoint()
                yield self._make_record(
                    problem=problem,
                    problem_seed=problem_seed,
                    trajectory_id=trajectory_id,
                    step=step,
                    run_seed=run_seed,
                    state=state,
                    candidates=cands,
                    output=output,
                    probs=probs,
                    optimization_scale=optimization_scale,
                    evaluation_scale=evaluation_scale,
                    fit_scale=fit_scale,
                    replay_checkpoint=replay_checkpoint,
                    partial_rollout=partial_meta,
                    horizon=h_step,
                )

            # Behavior policy — NEVER samples from gold_probs.
            if behavior_rng.random() < self.config.random_action_rate:
                action = random_policy.select_action(state, backend)
            else:
                action = rule.select_action(state, backend)

            result: ActionResult = execute_action(backend, state, action)
            last_action_name = action.type.value
            if result.should_stop:
                break

    def _make_record(
        self,
        *,
        problem: str,
        problem_seed: int,
        trajectory_id: int,
        step: int,
        run_seed: int,
        state: OptimizationState,
        candidates: list[Candidate],
        output,
        probs: dict[str, float],
        optimization_scale: float,
        evaluation_scale: float,
        fit_scale: float,
        replay_checkpoint: Checkpoint | None = None,
        partial_rollout: dict | None = None,
        horizon: int | None = None,
    ) -> DecisionRecord:
        # Phase 8.6B: serialize enough backend state for exact replay.
        replay_meta: dict = {}
        if replay_checkpoint is not None:
            cp = replay_checkpoint
            replay_meta = {
                "backend_X": cp.X.detach().cpu().tolist(),
                "backend_objectives": cp.objectives.detach().cpu().tolist(),
                "backend_constraints": cp.constraints.detach().cpu().tolist(),
                "backend_id": cp.backend_id,
                "backend_problem_id": cp.problem_id,
                "backend_problem_bounds": cp.problem_bounds,
                "backend_fitted": cp.fitted,
                "backend_model_state_b64": (
                    base64.b64encode(cp.model_state_dict).decode("ascii")
                    if cp.model_state_dict is not None else None
                ),
                "backend_rng_state_b64": (
                    base64.b64encode(cp.torch_rng_state).decode("ascii")
                    if cp.torch_rng_state is not None else None
                ),
            }
        return DecisionRecord(
            id=f"{problem.lower()}:{problem_seed}:{trajectory_id}:{step}",
            problem={
                "family": problem.lower(),
                "dimension": int(state.sampling.dimension),
                "seed": int(problem_seed),
            },
            state=_json_safe(state),
            candidate_actions=[
                {
                    "action_id": c.action_id,
                    "type": c.action.type.value,
                    "params": dict(c.action.params),
                }
                for c in candidates
            ],
            gold_probs=probs,
            rollout_results={
                r.action_id: {
                    "status": r.status,
                    "utility": r.utility,
                    "delta_optimization": r.delta_optimization,
                    "delta_surrogate": r.delta_surrogate,
                    "delta_coverage": r.delta_coverage,
                    "n_new_samples": r.n_new_samples,
                    "n_model_fits": r.n_model_fits,
                    "error": r.error,
                    **(
                        {"utility_replicas": r.metrics["utility_replicas"],
                         "utility_std": r.metrics["utility_std"]}
                        if "utility_replicas" in r.metrics else {}
                    ),
                }
                for r in output.results
            },
            temperature=float(self.config.temperature),
            metadata={
                "trajectory_id": int(trajectory_id),
                "step": int(step),
                "run_seed": int(run_seed),
                "utility_version": "phase8.6b",
                "optimization_scale": float(optimization_scale),
                "evaluation_scale": float(evaluation_scale),
                "fit_scale": float(fit_scale),
                "horizon": int(
                    self.config.horizon if horizon is None else horizon
                ),
                "horizon_mode": str(self.config.horizon_mode),
                # Phase 9 Gate 5 — affine objective transform (so replay
                # infrastructure can reconstruct the same transformed backend).
                "objective_scale": float(self.config.objective_scale),
                "objective_shift": float(self.config.objective_shift),
                # Sprint 0: full utility_weights (not just w_stop_opportunity)
                # and continuation_policy spec, so any record is replayable
                # with different config (per codex Q-T2 2026-09-25).
                "utility_weights": {
                    "w_optimization": float(self.config.utility_weights.w_optimization),
                    "w_surrogate": float(self.config.utility_weights.w_surrogate),
                    "w_coverage": float(self.config.utility_weights.w_coverage),
                    "w_evaluation": float(self.config.utility_weights.w_evaluation),
                    "w_fit": float(self.config.utility_weights.w_fit),
                    "w_stop_opportunity": float(self.config.utility_weights.w_stop_opportunity),
                },
                "continuation_policy": spec_to_metadata(self.config.continuation_spec),
                # Keep w_stop_opportunity for backward compatibility with
                # Phase 8.6B readers. New code should use utility_weights.
                "w_stop_opportunity": float(self.config.utility_weights.w_stop_opportunity),
                # Long-horizon quota (2026-09-27): present only for records
                # generated with rollout_action_subsample < |vocab|.
                **({"partial_rollout": partial_rollout} if partial_rollout else {}),
                **replay_meta,
            },
        )

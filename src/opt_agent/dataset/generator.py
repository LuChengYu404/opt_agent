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
from opt_agent.rollout.utility import UtilityWeights
from opt_agent.state import OptimizationState, extract_state


# Phase 7 default: full action vocabulary with sensible default params.
# id mirrors action type so records align across calls.
_PHASE7_CANDIDATE_SPECS: list[tuple[str, ActionType, dict]] = [
    ("global_exploration",  ActionType.GLOBAL_EXPLORATION,    {"n_samples": 2}),
    ("local_exploitation",  ActionType.LOCAL_EXPLOITATION,    {"step": 0.1}),
    ("uncertainty_sampling", ActionType.UNCERTAINTY_SAMPLING, {"n_samples": 1}),
    ("error_guided_sampling", ActionType.ERROR_GUIDED_SAMPLING, {"n_samples": 1}),
    ("boundary_sampling",   ActionType.BOUNDARY_SAMPLING,     {"n_samples": 2, "tol": 0.05}),
    ("surrogate_acquisition", ActionType.SURROGATE_ACQUISITION, {}),
    ("retrain_surrogate",   ActionType.RETRAIN_SURROGATE,     {}),
    ("stop",                ActionType.STOP,                  {}),
]


def full_candidates(seed: int = 0) -> list[Candidate]:
    """Return the Phase 7 default 8-action candidate set.

    Only actions that accept `seed` (per `_ALLOWED_PARAMS`) get it. Other
    actions (STOP / RETRAIN_SURROGATE / SURROGATE_ACQUISITION) ignore seed.
    """
    _SEED_ACCEPTING = {
        ActionType.GLOBAL_EXPLORATION,
        ActionType.LOCAL_EXPLOITATION,
        ActionType.UNCERTAINTY_SAMPLING,
        ActionType.ERROR_GUIDED_SAMPLING,
        ActionType.BOUNDARY_SAMPLING,
    }
    out: list[Candidate] = []
    for action_id, action_type, params in _PHASE7_CANDIDATE_SPECS:
        merged = dict(params)
        if action_type in _SEED_ACCEPTING:
            merged["seed"] = seed
        out.append(Candidate(action_id=action_id, action=OptimizationAction(action_type, merged)))
    return out


def derive_seed(*parts: int) -> int:
    """Deterministic 31-bit seed from multiple integers."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.to_bytes(8, "little", signed=False))
        h.update(b"|")
    return int.from_bytes(h.digest()[:4], "little") & 0x7FFFFFFF


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


BackendFactory = Callable[[], UnifiedBackend]


class DatasetGenerator:
    def __init__(
        self,
        config: DatasetConfig,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        self.config = config
        # Default factory: real BocodeBotorchBackend.
        self.backend_factory: BackendFactory = backend_factory or self._default_factory

    @staticmethod
    def _default_factory() -> UnifiedBackend:
        from opt_agent.backend.bocode_botorch import BocodeBotorchBackend
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
                cands = full_candidates(seed=derive_seed(run_seed, step))
                output = CounterfactualRollout(
                    backend,
                    horizon=self.config.horizon,
                    utility_weights=self.config.utility_weights,
                    optimization_scale=optimization_scale,
                    evaluation_scale=evaluation_scale,
                    fit_scale=fit_scale,
                ).rollout(state, cands)
                probs = to_gold_probs(
                    output.results,
                    temperature=self.config.temperature,
                    include_invalid_as_zero=True,
                )
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
                "horizon": int(self.config.horizon),
                "w_stop_opportunity": float(self.config.utility_weights.w_stop_opportunity),
                **replay_meta,
            },
        )

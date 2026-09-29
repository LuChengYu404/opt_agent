"""Full-loop learned-System-1 / real-System-2 routing cost frontier.

Train RF on rollout seeds 0–15, calibrate choice probabilities on 16–19,
then evaluate independent BO trajectories. Every arm uses the same System-2
prompt/model. Use ``--system2 mock`` only for wiring checks; those results
are never evidence of an LLM performance/cost frontier.

Real example (requires DEEPSEEK_API_KEY):
  uv run python scripts/exp_rq3_router_sweep.py --system2 real \
      --families branin rastrigin --seeds 100 101 102 \
      --policies rf llm confidence_0.5 confidence_0.75 random_0.5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _shared_rq_features import ACTIONS, FAMILIES, compute_family_scales, extract_features  # noqa: E402
from exp_rq1_rq2_closed_loop import PROBLEM_IDS, fit_models  # noqa: E402
from exp_rq2_1_cross_family_transfer import build_dataset  # noqa: E402
from opt_agent.actions.factory import candidate_actions_for_state  # noqa: E402
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend  # noqa: E402
from opt_agent.dataset import read_jsonl  # noqa: E402
from opt_agent.dataset.validation import phase_of_ratio  # noqa: E402
from opt_agent.loop import DecisionLoop  # noqa: E402
from opt_agent.metrics.temperature import TemperatureCalibrator, fit_temperature  # noqa: E402
from opt_agent.router import ConfidenceRouter, Route  # noqa: E402
from opt_agent.system1 import HybridPolicy  # noqa: E402
from opt_agent.system2 import FallbackSystem2  # noqa: E402
from opt_agent.system2.llm_deepseek import LLMSystem2  # noqa: E402


POLICIES = (
    "rf", "llm", "confidence_0.25", "confidence_0.5", "confidence_0.75",
    "confidence_0.9", "random_0.25", "random_0.5", "random_0.75",
)
_WORKER_CONTEXT: dict = {}


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    top = max(scores.values())
    exp = {key: math.exp(value - top) for key, value in scores.items()}
    total = sum(exp.values())
    return {key: value / total for key, value in exp.items()}


class RFDecider:
    def __init__(self, models: dict, calibrator: TemperatureCalibrator,
                 seed: int) -> None:
        self.models, self.calibrator, self.seed = models, calibrator, seed

    def decide(self, state):
        choices = candidate_actions_for_state(state, seed=self.seed)
        phase = phase_of_ratio(state.resource.remaining_budget_ratio)
        x = extract_features(asdict(state), phase)
        scores = {action: float(self.models[action].predict([x])[0])
                  for action in choices}
        probabilities = self.calibrator.transform(_softmax(scores))
        action_id = max(probabilities, key=probabilities.get)
        return SimpleNamespace(action=choices[action_id], probabilities=probabilities)

    def select_action(self, state):
        return self.decide(state).action


class DirectSystem2:
    def __init__(self, system2) -> None:
        self.system2 = system2

    def select_action(self, state):
        return self.system2.reason(state).action


class RandomRouter:
    def __init__(self, rate: float, seed: int) -> None:
        self.rate, self.rng = rate, random.Random(seed)

    def route(self, probabilities: dict[str, float]) -> Route:
        name = "system2" if self.rng.random() < self.rate else "system1"
        return Route(name=name, confidence=max(probabilities.values()),
                     reason=f"random rate={self.rate}")

    def update(self, probabilities, chosen, reward) -> None:
        return None


def prepare_model(dataset: Path):
    records = [r.__dict__ for r in read_jsonl(dataset, version_policy="warn")]
    train = [r for r in records if int(r["problem"]["seed"]) <= 15]
    calibration = [r for r in records if 16 <= int(r["problem"]["seed"]) <= 19]
    if not train or not calibration:
        raise ValueError("need rollout training seeds 0–15 and calibration seeds 16–19")
    family_scales = compute_family_scales(train)
    X_train, Y_train, fi_train, _, _ = build_dataset(train, family_scales=family_scales)
    X_cal, Y_cal, _, _, _ = build_dataset(calibration, family_scales=family_scales)
    models, n_train = fit_models(X_train, Y_train, fi_train, None)
    probs, outcomes = [], []
    for x, y in zip(X_cal, Y_cal):
        scores = {action: float(models[action].predict([x])[0]) for action in ACTIONS}
        probs.append(_softmax(scores))
        outcomes.append(ACTIONS[max(range(len(ACTIONS)), key=lambda j: y[j])])
    result = fit_temperature(probs, outcomes)
    calibrator = TemperatureCalibrator(result.temperature)
    return models, calibrator, {
        "n_training_records": n_train, "n_calibration_records": len(X_cal),
        "temperature": result.temperature, "nll_before": result.nll_before,
        "nll_after": result.nll_after,
        "calibration_target": "argmax short-horizon rollout utility, not terminal success",
    }


def _make_system2(mode: str, seed: int):
    if mode == "mock":
        return FallbackSystem2()
    return LLMSystem2(candidate_seed=seed)


def run_one(family: str, seed: int, budget: int, policy_name: str,
            models: dict, calibrator: TemperatureCalibrator, mode: str) -> dict:
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(problem_id=PROBLEM_IDS[family], seed=seed, n_init=5))
    initial = backend.get_history().objectives[:, 0].tolist()
    system2 = _make_system2(mode, seed)
    decider = RFDecider(models, calibrator, seed)
    if policy_name == "rf":
        policy = decider
    elif policy_name == "llm":
        policy = DirectSystem2(system2)
    elif policy_name.startswith("confidence_"):
        threshold = float(policy_name.split("_", 1)[1])
        policy = HybridPolicy(decider, ConfidenceRouter(tau_conf=threshold), system2)
    elif policy_name.startswith("random_"):
        rate = float(policy_name.split("_", 1)[1])
        policy = HybridPolicy(decider, RandomRouter(rate, seed), system2)
    else:
        raise ValueError(policy_name)
    start = time.monotonic()
    trajectory = DecisionLoop(backend, total_budget=budget).run(
        policy, max_rounds=budget + 5,
    )
    elapsed = time.monotonic() - start
    history = backend.get_history()
    values = history.objectives[:, 0].tolist()
    if len(values) > budget:
        raise ValueError("objective budget exceeded")
    if isinstance(policy, HybridPolicy):
        routes = [step.route.name for step in policy.history]
    else:
        routes = ["system1" if policy_name == "rf" else "system2"] * len(trajectory.steps)
    llm_calls = int(getattr(system2, "call_count", routes.count("system2")))
    if mode == "real" and llm_calls != routes.count("system2"):
        raise ValueError("recorded LLM call count differs from routing history")
    return {
        "family": family, "seed": seed, "budget": budget, "policy": policy_name,
        "system2_mode": mode, "initial_values": initial,
        "terminal_best": max(values), "n_evaluations": len(values),
        "n_steps": len(trajectory.steps), "termination": trajectory.termination_reason.value,
        "actions": [step.action.type.value for step in trajectory.steps],
        "routes": routes, "llm_calls": llm_calls,
        "llm_tokens": int(getattr(system2, "total_tokens", 0)),
        "llm_fallbacks": int(getattr(system2, "fallback_count", 0)),
        "elapsed_s": elapsed,
    }


def _run_worker(task: tuple[str, int, int, str, str]) -> dict:
    torch.set_num_threads(1)
    family, seed, budget, policy, mode = task
    return run_one(family, seed, budget, policy, _WORKER_CONTEXT["models"],
                   _WORKER_CONTEXT["calibrator"], mode)


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["policy"])].append(row)
    out = {}
    for (family, policy), group in sorted(groups.items()):
        out[f"{family}/{policy}"] = {
            "n_seeds": len(group),
            "mean_terminal_best": statistics.fmean(r["terminal_best"] for r in group),
            "mean_llm_calls": statistics.fmean(r["llm_calls"] for r in group),
            "mean_llm_tokens": statistics.fmean(r["llm_tokens"] for r in group),
            "mean_elapsed_s": statistics.fmean(r["elapsed_s"] for r in group),
            "total_fallbacks": sum(r["llm_fallbacks"] for r in group),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "runs/phase8_6d_h3_20seed_5family.jsonl")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--system2", choices=("real", "mock"), default="real")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/exp_rq3_router_sweep.json")
    args = parser.parse_args()
    if args.budget <= 5 or args.workers < 1:
        parser.error("budget must exceed five initial points and workers must be positive")
    if args.system2 == "real" and not os.environ.get("DEEPSEEK_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    if args.system2 == "real" and not os.environ.get("DEEPSEEK_API_KEY"):
        parser.error("DEEPSEEK_API_KEY is required for a real cost-frontier run")
    torch.set_num_threads(1)
    models, calibrator, training = prepare_model(args.dataset)
    config = {
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "training": training, "families": args.families, "seeds": args.seeds,
        "budget": args.budget, "policies": args.policies,
        "system2_mode": args.system2,
        "system2_model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
                         if args.system2 == "real" else "rule-fallback-mock",
    }
    rows = []
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous["config"] != config:
            parser.error("existing output has different config")
        rows = previous["runs"]
    done = {(r["family"], r["seed"], r["policy"]) for r in rows}
    tasks = [(family, seed, args.budget, policy, args.system2)
             for family in args.families for seed in args.seeds
             for policy in args.policies if (family, seed, policy) not in done]

    def save_row(row):
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps({"config": config, "runs": rows}, allow_nan=False) + "\n")
        temp.replace(args.output)
        print(f"{row['family']} s{row['seed']} {row['policy']} "
              f"best={row['terminal_best']:.3f} calls={row['llm_calls']} "
              f"tokens={row['llm_tokens']}", flush=True)

    if args.workers == 1:
        for task in tasks:
            save_row(run_one(*task[:4], models, calibrator, task[4]))
    else:
        if "fork" not in multiprocessing.get_all_start_methods():
            parser.error("parallel mode requires fork for copy-on-write RF models")
        global _WORKER_CONTEXT
        _WORKER_CONTEXT = {"models": models, "calibrator": calibrator}
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("fork")) as pool:
            futures = [pool.submit(_run_worker, task) for task in tasks]
            for future in as_completed(futures):
                save_row(future.result())
    family_order = {family: i for i, family in enumerate(args.families)}
    policy_order = {policy: i for i, policy in enumerate(args.policies)}
    rows.sort(key=lambda r: (family_order[r["family"]], r["seed"], policy_order[r["policy"]]))
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps({
        "config": config, "runs": rows, "summary": summarize(rows),
    }, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output)


if __name__ == "__main__":
    main()

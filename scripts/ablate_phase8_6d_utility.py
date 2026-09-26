"""Ablate utility weights on saved Phase 8.6D branches and replay first actions.

All variants rank the same saved counterfactual branches. Terminal outcomes
use the same RulePolicy continuation and baseline arms as Gate 4.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.actions import ActionType, OptimizationAction  # noqa: E402
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend  # noqa: E402
from opt_agent.dataset.decision_record import iter_jsonl  # noqa: E402
from opt_agent.dataset.generator import derive_seed  # noqa: E402
from opt_agent.dataset.validation import _reconstruct_checkpoint, _run_arm, replay_with_config  # noqa: E402
from opt_agent.loop.policies import RulePolicy  # noqa: E402
from opt_agent.rollout.specs import ContinuationPolicySpec  # noqa: E402
from opt_agent.rollout.utility import UtilityWeights, compute_utility  # noqa: E402
from opt_agent.state import extract_state  # noqa: E402


VARIANTS = {
    "current": UtilityWeights(),
    "no_eval": UtilityWeights(w_evaluation=0),
    "no_fit": UtilityWeights(w_fit=0),
    "no_stop_pen": UtilityWeights(w_stop_opportunity=0),
    "no_cost": UtilityWeights(w_evaluation=0, w_fit=0),
    "terminal_only": UtilityWeights(
        w_evaluation=0, w_fit=0, w_stop_opportunity=0
    ),
}


def _mean_ci(values: list[float]) -> dict:
    rng = random.Random(0)
    n = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(2000)
    )
    return {
        "mean": statistics.fmean(values),
        "ci95": [means[50], means[1950]],
        "positive_seeds": sum(v > 0 for v in values),
    }


def _rank(record, weights: UtilityWeights) -> str:
    resource = record.state["resource"]
    meta = record.metadata
    utilities = {}
    for action_id, result in record.rollout_results.items():
        if result["status"] != "valid":
            continue
        utilities[action_id] = compute_utility(
            delta_optimization=result["delta_optimization"],
            delta_surrogate=(
                result["delta_surrogate"]
                if result["delta_surrogate"] is not None else math.nan
            ),
            delta_coverage=(
                result["delta_coverage"]
                if result["delta_coverage"] is not None else math.nan
            ),
            n_new_samples=result["n_new_samples"],
            n_model_fits=result["n_model_fits"],
            weights=weights,
            optimization_scale=meta["optimization_scale"],
            evaluation_scale=meta["evaluation_scale"],
            fit_scale=meta["fit_scale"],
            is_stop=action_id == "stop",
            remaining_budget_ratio=(
                resource["remaining_budget"] - result["n_new_samples"]
            ) / resource["total_budget"],
        )
        if weights == VARIANTS["current"] and not math.isclose(
            utilities[action_id], result["utility"], rel_tol=1e-9, abs_tol=1e-9
        ):
            raise AssertionError(f"utility mismatch: {record.id} {action_id}")
    return max(utilities, key=utilities.get)


def _terminal_for_actions(record, action_ids: set[str]) -> dict[str, dict]:
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(
        problem_id=record.metadata["backend_problem_id"],
        seed=int(record.problem["seed"]),
        n_init=1,
    ))
    base = _reconstruct_checkpoint(record)
    specs = {a["action_id"]: a for a in record.candidate_actions}
    budget = int(record.state["resource"]["total_budget"])
    initial_remaining = int(record.state["resource"]["remaining_budget"])
    continuation = RulePolicy(seed=derive_seed(int(record.metadata["run_seed"]), 1))
    outcomes = {}
    for action_id in sorted(action_ids):
        backend.restore(base)
        state = extract_state(
            backend,
            total_budget=budget,
            last_action=record.state["history"].get("last_action"),
        )
        spec = specs[action_id]
        action = OptimizationAction(ActionType(spec["type"]), dict(spec["params"]))
        best, remaining = _run_arm(
            backend,
            initial_state=state,
            first_action=action,
            choose_next=continuation.select_action,
            total_budget=budget,
            max_actions=30,
        )
        outcomes[action_id] = {
            "terminal_best": best,
            "new_evaluations": initial_remaining - remaining,
        }
    backend.restore(base)
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("runs/phase8_6d_hfull_10seed.jsonl"))
    parser.add_argument("--gate-report", type=Path, default=Path("runs/phase8_6d_report.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase8_6d_utility_ablation.json"))
    parser.add_argument("--label-continuation", choices=("saved", "rule"), default="saved")
    parser.add_argument("--limit", type=int, default=0, help="First N records for a smoke run")
    args = parser.parse_args()

    records = [r for r in iter_jsonl(args.dataset) if r.metadata["step"] == 0]
    records.sort(key=lambda r: (r.problem["family"], r.problem["seed"]))
    if args.limit:
        records = records[:args.limit]
    gate_rows = {r["record_id"]: r for r in json.loads(args.gate_report.read_text())["gate4"]["rows"]}
    rows = []
    for i, record in enumerate(records, 1):
        label_record = record
        if args.label_continuation == "rule":
            output = replay_with_config(
                record,
                continuation_spec=ContinuationPolicySpec(name="rule", seed=0),
                horizon=int(record.state["resource"]["remaining_budget"]),
            )
            label_record = replace(record, rollout_results={
                result.action_id: {
                    "status": result.status,
                    "utility": result.utility,
                    "delta_optimization": result.delta_optimization,
                    "delta_surrogate": result.delta_surrogate,
                    "delta_coverage": result.delta_coverage,
                    "n_new_samples": result.n_new_samples,
                    "n_model_fits": result.n_model_fits,
                }
                for result in output.results
            })
        labels = {name: _rank(label_record, weights) for name, weights in VARIANTS.items()}
        original_top = max(record.gold_probs, key=record.gold_probs.get)
        if args.label_continuation == "saved" and labels["current"] != original_top:
            raise AssertionError(f"saved top action mismatch: {record.id}")
        gate = gate_rows[record.id]
        outcomes = _terminal_for_actions(record, set(labels.values()))
        if original_top in outcomes and not math.isclose(
            outcomes[original_top]["terminal_best"],
            gate["terminal"]["gold"], rel_tol=1e-9, abs_tol=1e-9,
        ):
            raise AssertionError(f"Gate 4 replay mismatch: {record.id}")
        rows.append({
            "record_id": record.id,
            "family": record.problem["family"],
            "seed": int(record.problem["seed"]),
            "objective_scale": record.metadata["optimization_scale"],
            "baselines": {k: gate["terminal"][k] for k in ("stop", "uniform", "behavior")},
            "labels": labels,
            "outcomes": outcomes,
        })
        print(f"{i}/{len(records)} {record.id} {labels}", flush=True)

    summary = defaultdict(dict)
    for family in sorted({r["family"] for r in rows}):
        group = [r for r in rows if r["family"] == family]
        for name in VARIANTS:
            terminal = [r["outcomes"][r["labels"][name]] for r in group]
            summary[family][name] = {
                "n_seeds": len(group),
                "stop_first": sum(r["labels"][name] == "stop" for r in group),
                "top_actions": dict(Counter(r["labels"][name] for r in group)),
                "mean_new_evaluations": statistics.fmean(t["new_evaluations"] for t in terminal),
                "mean_terminal_best": statistics.fmean(t["terminal_best"] for t in terminal),
                "gain": {
                    baseline: _mean_ci([
                        (t["terminal_best"] - r["baselines"][baseline]) / r["objective_scale"]
                        for r, t in zip(group, terminal, strict=True)
                    ])
                    for baseline in ("stop", "uniform", "behavior")
                },
            }
    payload = {
        "dataset": str(args.dataset),
        "gate_report": str(args.gate_report),
        "continuation_for_terminal": "rule",
        "label_continuation": (
            "on_policy (saved branches)" if args.label_continuation == "saved"
            else "rule (replayed branches)"
        ),
        "variants": {k: vars(v) for k, v in VARIANTS.items()},
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

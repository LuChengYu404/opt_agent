"""Sprint 1 driver — Q-S5 control-variable diagnostic (per codex Q-T1/Q-T2).

Decisions (user-approved 2026-09-25):
  D1 = A1 (1-step Δbest only as third utility variant)
  D2 = β  (paired oracle test only when top action changes)
  D3 = ii (sample 30 records, balanced per problem × phase)
  D4 = JSON + Markdown

Usage
-----
    uv run python scripts/diagnose_qs5.py \\
        --dataset runs/phase8_6b_h3_10seed.jsonl \\
        --output runs/qs5_diagnosis \\
        --sample-size 30 \\
        --seed 0

Outputs:
  <output>.json   — full per-(record, config) data + aggregates
  <output>.md     — human-readable summary with per-config tables
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from opt_agent.actions import ActionType, OptimizationAction
from opt_agent.backend.bocode_botorch import BackendConfig, BocodeBotorchBackend
from opt_agent.backend.checkpoint import Checkpoint
from opt_agent.dataset.decision_record import DecisionRecord, iter_jsonl
from opt_agent.dataset.validation import (
    _reconstruct_checkpoint,
    _candidates_from_record,
    oracle_test,
    replay_with_config,
)
from opt_agent.rollout.rollout import Candidate, to_gold_probs
from opt_agent.rollout.specs import ContinuationPolicySpec
from opt_agent.rollout.utility import UtilityWeights
from opt_agent.state import extract_state


# ---------------------------------------------------------------------------
# Config definitions (Q-S5 control-variable matrix)
# ---------------------------------------------------------------------------


# Three utility variants:
#   original       — current Phase 8.6A 4-term utility
#   no_stop_pen    — drop w_stop_opportunity to 0
#   delta_best     — A1: utility = Δbest only, all other terms 0
UTILITY_VARIANTS: dict[str, UtilityWeights] = {
    "original": UtilityWeights(),
    "no_stop_pen": UtilityWeights(w_stop_opportunity=0.0),
    "delta_best": UtilityWeights(
        w_optimization=1.0, w_surrogate=0.0, w_coverage=0.0,
        w_evaluation=0.0, w_fit=0.0, w_stop_opportunity=0.0,
    ),
}

# Five horizon levels. "full" is replaced at runtime with the per-record
# remaining_budget value (the user's Decision 2 (b)).
HORIZON_LEVELS: list[int | str] = [1, 3, 5, 10, "full"]

# Continuation policy fixed at "rule" per Q-S5 partial-factorial Decision 4 (Z).
CONTINUATION: ContinuationPolicySpec = ContinuationPolicySpec(name="rule", seed=0)


def config_names() -> list[str]:
    """Return the 15 config names as `{utility}_h{horizon}`."""
    return [
        f"{util}_h{h}"
        for util in UTILITY_VARIANTS.keys()
        for h in HORIZON_LEVELS
    ]


# ---------------------------------------------------------------------------
# Phase bucketing (mirrors dataset.validation.phase_of)
# ---------------------------------------------------------------------------


def phase_of(record: DecisionRecord) -> str:
    ratio = float(record.state["resource"]["remaining_budget_ratio"])
    if ratio == 0:
        return "terminal"
    if ratio > 0.7:
        return "early"
    if ratio < 0.3:
        return "late"
    return "mid"


# ---------------------------------------------------------------------------
# Balanced sampling (per problem × phase)
# ---------------------------------------------------------------------------


def sample_records(
    records: list[DecisionRecord], n: int, seed: int
) -> list[DecisionRecord]:
    """Sample `n` records balanced by (problem, phase).

    Round-robin over cells, falling back to whatever remains when a cell
    empties. Phases with very few records may be slightly over- or
    under-represented vs. perfect balance — the goal is to have at least
    one record per (problem, phase) cell for per-cell aggregation.
    """
    rng = random.Random(seed)
    by_cell: dict[tuple[str, str], list[DecisionRecord]] = defaultdict(list)
    for r in records:
        by_cell[(r.problem["family"], phase_of(r))].append(r)
    cells = [lst for lst in by_cell.values() if lst]
    if not cells:
        return []
    rng.shuffle(cells)
    sampled: list[DecisionRecord] = []
    while len(sampled) < n and any(cells):
        for cell in cells:
            if cell and len(sampled) < n:
                idx = rng.randrange(len(cell))
                sampled.append(cell.pop(idx))
    return sampled


# ---------------------------------------------------------------------------
# Single (record, config) execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunResult:
    record_id: str
    problem: str
    phase: str
    config: str
    utility_variant: str
    horizon: int            # resolved int (after "full" → remaining_budget)
    new_top_action: str
    original_top_action: str
    changed: bool
    new_probs: dict[str, float]
    oracle: dict[str, Any] | None = None  # filled only when changed=True


def _oracle_for_changed_top(
    record: DecisionRecord, new_top_action_id: str, args: argparse.Namespace
) -> dict[str, Any] | None:
    """Run paired full-budget oracle test with new top action.

    Compares new_top vs STOP (and uniform/behavior baselines) using a fixed
    RulePolicy continuation. Returns the gains dict from `oracle_test`.
    Skips if the record is missing replay metadata.
    """
    meta = record.metadata
    required = ("backend_X", "backend_objectives", "backend_problem_id", "backend_rng_state_b64")
    if any(k not in meta for k in required):
        return {"error": "missing replay metadata"}

    problem_id = str(meta["backend_problem_id"])
    # Phase 9 Gate 5 — thread affine transform from metadata so that any
    # evaluate() calls after restore() append transformed values to the
    # (already-transformed) history. Without this, oracle_test would mix
    # raw and transformed objectives — silent corruption.
    from opt_agent.dataset.validation import _affine_kwargs_from_metadata
    affine_kwargs = _affine_kwargs_from_metadata(meta)
    backend = BocodeBotorchBackend(**affine_kwargs)
    try:
        backend.initialize(BackendConfig(
            problem_id=problem_id,
            seed=int(record.problem["seed"]),
            n_init=1,
        ))
    except Exception as exc:
        return {"error": f"backend init failed: {exc}"}

    base = _reconstruct_checkpoint(record)
    try:
        backend.restore(base)
        state = extract_state(
            backend,
            total_budget=int(record.state["resource"]["total_budget"]),
            last_action=record.state["history"].get("last_action"),
        )
        oracle = oracle_test(
            backend,
            state,
            record,
            objective_scale=float(meta.get("optimization_scale", 1.0)),
            random_replicates=args.oracle_replicates,
            max_actions=args.oracle_max_actions,
        )
        # Note: oracle_test uses the record's ORIGINAL gold_probs to pick
        # the gold arm. For our diagnostic we want to compare the NEW top
        # action against STOP/uniform/behavior. We achieve this by injecting
        # the new top into record.gold_probs temporarily — this is safe
        # because the record is a local copy.
        from dataclasses import replace
        rec_with_new_gold = replace(
            record,
            gold_probs={**record.gold_probs, new_top_action_id: 1.0},
        )
        # Suppress all others by zeroing:
        rec_with_new_gold = replace(
            rec_with_new_gold,
            gold_probs={k: (1.0 if k == new_top_action_id else 0.0)
                       for k in record.gold_probs},
        )
        oracle_new = oracle_test(
            backend,
            state,
            rec_with_new_gold,
            objective_scale=float(meta.get("optimization_scale", 1.0)),
            random_replicates=args.oracle_replicates,
            max_actions=args.oracle_max_actions,
        )
        return {
            "original_top_terminal": oracle.terminal,
            "new_top_terminal": oracle_new.terminal,
            "original_top_gains": oracle.gains,
            "new_top_gains": oracle_new.gains,
        }
    except Exception as exc:
        return {"error": f"oracle failed: {exc}"}
    finally:
        backend.restore(base)


def run_one(
    record: DecisionRecord, utility_name: str, horizon_spec: int | str,
    args: argparse.Namespace,
) -> RunResult:
    weights = UTILITY_VARIANTS[utility_name]
    # Resolve "full" → remaining_budget at runtime (user Decision 2 (b)).
    if horizon_spec == "full":
        remaining = int(record.state["resource"]["remaining_budget"])
        resolved_horizon = max(1, remaining)
    else:
        resolved_horizon = int(horizon_spec)

    try:
        output = replay_with_config(
            record,
            utility_weights=weights,
            horizon=resolved_horizon,
            continuation_spec=CONTINUATION,
        )
        new_probs = to_gold_probs(
            output.results, temperature=record.temperature,
            include_invalid_as_zero=True,
        )
    except Exception as exc:
        # On failure, return a sentinel result so we know which (record, config) failed.
        return RunResult(
            record_id=record.id,
            problem=record.problem["family"],
            phase=phase_of(record),
            config=f"{utility_name}_h{horizon_spec}",
            utility_variant=utility_name,
            horizon=resolved_horizon,
            new_top_action="ERROR",
            original_top_action=max(record.gold_probs, key=record.gold_probs.get),
            changed=False,
            new_probs={},
        )

    new_top = max(new_probs, key=new_probs.get)
    original_top = max(record.gold_probs, key=record.gold_probs.get)
    changed = new_top != original_top

    oracle = None
    if changed:
        oracle = _oracle_for_changed_top(record, new_top, args)

    return RunResult(
        record_id=record.id,
        problem=record.problem["family"],
        phase=phase_of(record),
        config=f"{utility_name}_h{horizon_spec}",
        utility_variant=utility_name,
        horizon=resolved_horizon,
        new_top_action=new_top,
        original_top_action=original_top,
        changed=changed,
        new_probs=new_probs,
        oracle=oracle,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class ConfigSummary:
    config: str
    n_records: int = 0
    n_changed: int = 0
    top1_agreement_with_original: float = 0.0
    mean_kl_vs_original: float = 0.0
    changed_records: list[str] = field(default_factory=list)
    # Oracle statistics (only over changed records):
    n_oracle_evaluated: int = 0
    n_oracle_pass_gold_vs_stop: int = 0  # new > stop in terminal best_value
    mean_gain_new_vs_stop: float = 0.0
    mean_gain_new_vs_behavior: float = 0.0
    mean_gain_original_vs_stop: float = 0.0


def _safe_kl(p: dict[str, float], q: dict[str, float]) -> float:
    """Symmetric KL on aligned vocabulary, eps-smoothed."""
    keys = set(p) | set(q)
    eps = 1e-9
    total = 0.0
    for k in keys:
        pk = max(p.get(k, 0.0), eps)
        qk = max(q.get(k, 0.0), eps)
        total += pk * math.log(pk / qk)
    return total


def aggregate(results: list[RunResult]) -> dict[str, Any]:
    """Build per-config and per-(problem, phase, config) summaries."""
    by_config: dict[str, list[RunResult]] = defaultdict(list)
    by_ppc: dict[tuple[str, str, str], list[RunResult]] = defaultdict(list)
    for r in results:
        by_config[r.config].append(r)
        by_ppc[(r.problem, r.phase, r.config)].append(r)

    def _config_summary(name: str, rs: list[RunResult]) -> ConfigSummary:
        s = ConfigSummary(config=name)
        s.n_records = len(rs)
        s.n_changed = sum(1 for r in rs if r.changed)
        s.top1_agreement_with_original = (
            sum(1 for r in rs if r.new_top_action == r.original_top_action) / len(rs)
            if rs else 0.0
        )
        # KL averaged across records with finite new_probs.
        kls = []
        for r in rs:
            if r.new_probs:
                # We need original probs for this; reconstruct from record.
                pass
        # We can't compute KL here without original_probs. Defer to caller.
        s.changed_records = [r.record_id for r in rs if r.changed]
        oracle_gains_vs_stop = []
        oracle_gains_vs_behavior = []
        original_gains_vs_stop = []
        oracle_pass = 0
        oracle_n = 0
        for r in rs:
            if r.oracle is None or "error" in r.oracle or "new_top_gains" not in r.oracle:
                continue
            oracle_n += 1
            new_g = r.oracle["new_top_gains"]
            orig_g = r.oracle["original_top_gains"]
            if "stop" in new_g:
                oracle_gains_vs_stop.append(float(new_g["stop"]))
            if "behavior" in new_g:
                oracle_gains_vs_behavior.append(float(new_g["behavior"]))
            if "stop" in orig_g:
                original_gains_vs_stop.append(float(orig_g["stop"]))
            if oracle_gains_vs_stop and oracle_gains_vs_stop[-1] > 0:
                oracle_pass += 1
        s.n_oracle_evaluated = oracle_n
        s.n_oracle_pass_gold_vs_stop = oracle_pass
        s.mean_gain_new_vs_stop = (
            statistics.fmean(oracle_gains_vs_stop) if oracle_gains_vs_stop else 0.0
        )
        s.mean_gain_new_vs_behavior = (
            statistics.fmean(oracle_gains_vs_behavior) if oracle_gains_vs_behavior else 0.0
        )
        s.mean_gain_original_vs_stop = (
            statistics.fmean(original_gains_vs_stop) if original_gains_vs_stop else 0.0
        )
        return s

    config_summaries = {name: _config_summary(name, rs) for name, rs in by_config.items()}

    # Per-(problem, phase, config): top-1 agreement.
    ppc_summaries: list[dict[str, Any]] = []
    for (problem, phase, config), rs in sorted(by_ppc.items()):
        if not rs:
            continue
        n = len(rs)
        n_agree = sum(1 for r in rs if r.new_top_action == r.original_top_action)
        ppc_summaries.append({
            "problem": problem,
            "phase": phase,
            "config": config,
            "n_records": n,
            "top1_agreement": n_agree / n,
        })

    return {
        "per_config": config_summaries,
        "per_problem_phase_config": ppc_summaries,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def format_markdown(
    args: argparse.Namespace, sampled: list[DecisionRecord],
    results: list[RunResult], summary: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Q-S5 Diagnostic Report")
    lines.append("")
    lines.append(f"- **Dataset**: `{args.dataset}`")
    lines.append(f"- **Records sampled**: {len(sampled)} (seed={args.seed})")
    lines.append(f"- **Configs**: {len(results) // max(1, len(sampled))} per record × {len(sampled)} records = {len(results)} runs")
    lines.append(f"- **Continuation policy**: `{CONTINUATION.name}` (fixed per Q-S5 Decision 4-Z)")
    lines.append("")
    lines.append("## Per-config summary")
    lines.append("")
    lines.append("| Config | N | Changed | Top-1 Agree | Oracle N | Mean gain vs STOP (new) | Mean gain vs STOP (orig) | Oracle Pass (new>STOP) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name in sorted(summary["per_config"].keys()):
        s = summary["per_config"][name]
        lines.append(
            f"| `{name}` | {s.n_records} | {s.n_changed} | "
            f"{s.top1_agreement_with_original:.2f} | {s.n_oracle_evaluated} | "
            f"{s.mean_gain_new_vs_stop:+.4f} | {s.mean_gain_original_vs_stop:+.4f} | "
            f"{s.n_oracle_pass_gold_vs_stop}/{s.n_oracle_evaluated} |"
        )
    lines.append("")
    lines.append("**Reading the table**:")
    lines.append("- `Top-1 Agree` = fraction of records where the new config picks the same top action as the original.")
    lines.append("- `Mean gain vs STOP (new)` = mean normalized (gold_new − stop) / scale over records where top action changed.")
    lines.append("- `Mean gain vs STOP (orig)` = same but using original gold_probs.")
    lines.append("- `Oracle Pass (new>STOP)` = count of changed records where new top action terminal best_value > STOP.")
    lines.append("")
    lines.append("## Per-(problem, phase, config) top-1 agreement")
    lines.append("")
    lines.append("| Problem | Phase | Config | N | Top-1 Agree |")
    lines.append("|---|---|---|---|---|")
    for row in summary["per_problem_phase_config"]:
        lines.append(
            f"| `{row['problem']}` | {row['phase']} | `{row['config']}` | "
            f"{row['n_records']} | {row['top1_agreement']:.2f} |"
        )
    lines.append("")
    lines.append("## Conclusions (heuristic — codex review required for final word)")
    lines.append("")
    best_by_agree = max(
        summary["per_config"].values(),
        key=lambda s: (s.top1_agreement_with_original, -s.n_changed),
        default=None,
    )
    best_by_oracle = max(
        summary["per_config"].values(),
        key=lambda s: s.mean_gain_new_vs_stop,
        default=None,
    )
    if best_by_agree:
        lines.append(f"- **Most label-consistent config**: `{best_by_agree.config}` "
                     f"(top-1 agreement {best_by_agree.top1_agreement_with_original:.2f}).")
    if best_by_oracle:
        lines.append(f"- **Best oracle gain (when changed)**: `{best_by_oracle.config}` "
                     f"(mean gain vs STOP = {best_by_oracle.mean_gain_new_vs_stop:+.4f}).")
    n_total_changed = sum(s.n_changed for s in summary["per_config"].values())
    n_total = sum(s.n_records for s in summary["per_config"].values())
    lines.append(f"- **Total (record, config) pairs**: {n_total}, "
                 f"of which {n_total_changed} had top-action changes "
                 f"({100 * n_total_changed / max(1, n_total):.1f}%).")
    lines.append("")
    lines.append("If most configs change few top actions AND the original top still loses to STOP, "
                 "the counterfactual teacher is fundamentally mis-specified (Q-S5 verdict: utility not just "
                 "wrong but possibly *mis-aimed*). Switch route per Q-Route recommendation.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("runs/phase8_6b_h3_10seed.jsonl"))
    p.add_argument("--output", type=Path, default=Path("runs/qs5_diagnosis"))
    p.add_argument("--sample-size", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--oracle-replicates", type=int, default=3)
    p.add_argument("--oracle-max-actions", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[diagnose_qs5] Loading dataset: {args.dataset}")
    records = list(iter_jsonl(args.dataset))
    print(f"[diagnose_qs5] Loaded {len(records)} records")

    sampled = sample_records(records, args.sample_size, args.seed)
    print(f"[diagnose_qs5] Sampled {len(sampled)} records (seed={args.seed})")

    configs = [(util, h) for util in UTILITY_VARIANTS for h in HORIZON_LEVELS]
    print(f"[diagnose_qs5] Running {len(configs)} configs × {len(sampled)} records "
          f"= {len(configs) * len(sampled)} (record, config) pairs")

    results: list[RunResult] = []
    n_total = len(configs) * len(sampled)
    done = 0
    for record in sampled:
        for util, h in configs:
            res = run_one(record, util, h, args)
            results.append(res)
            done += 1
            if done % 30 == 0 or done == n_total:
                print(f"[diagnose_qs5] progress: {done}/{n_total}")

    print("[diagnose_qs5] Aggregating")
    summary = aggregate(results)

    json_payload: dict[str, Any] = {
        "metadata": {
            "dataset": str(args.dataset),
            "n_records_total": len(records),
            "n_records_sampled": len(sampled),
            "sample_seed": args.seed,
            "utility_variants": list(UTILITY_VARIANTS.keys()),
            "horizon_levels": [str(h) for h in HORIZON_LEVELS],
            "continuation_policy": CONTINUATION.name,
            "oracle_replicates": args.oracle_replicates,
            "oracle_max_actions": args.oracle_max_actions,
        },
        "per_record_config": [asdict(r) for r in results],
        "summary": {
            "per_config": {
                name: asdict(s) for name, s in summary["per_config"].items()
            },
            "per_problem_phase_config": summary["per_problem_phase_config"],
        },
    }

    json_path = args.output.with_suffix(".json")
    md_path = args.output.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_payload, indent=2, allow_nan=False))
    md_path.write_text(format_markdown(args, sampled, results, summary))
    print(f"[diagnose_qs5] Wrote {json_path} ({json_path.stat().st_size:,} bytes)")
    print(f"[diagnose_qs5] Wrote {md_path}")


if __name__ == "__main__":
    main()

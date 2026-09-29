"""Long-horizon decision-dataset generator with rollout-quota control (2026-09-27).

Motivation
----------
PROGRESS.md (EXP-RQ1.2 naming correction) records that true full-horizon
labels — horizon = remaining_budget, the semantics the Phase 8.6C horizon
diagnosis identified as decisive — were "not in scope for RQ1; cost ≈ 90
minutes" for a 226-record 3-family dataset. The blocker is the rollout quota:
8 candidates × horizon-H branches per decision point, each step a GP-bound
backend operation.

This driver makes that quota controllable along two orthogonal axes:

  --horizon remaining     per-decision-point horizon = remaining_budget
                         (true long-horizon labels)
  --subsample-k K         roll out only K of the 8 candidates per decision
                         point (deterministic subset; quota scales K/8).
                         Records carry metadata["partial_rollout"]; pairwise
                         (contrastive) supervision is unaffected.

and operational axes:

  --workers W             process-parallel over (problem, seed) trajectories
  --resume                append-only; trajectories whose records already
                         exist in the output are skipped
  cost accounting         per-record backend cost units
                         (Σ n_new_samples + n_model_fits over rolled-out
                         branches) are summarized at the end

Usage
-----
    .venv/bin/python scripts/generate_dataset_longhorizon.py \
        runs/longhorizon_pilot.jsonl \
        --horizon remaining --subsample-k 4 --seeds 3 --workers 6

    .venv/bin/python scripts/generate_dataset_longhorizon.py \
        runs/longhorizon_full8.jsonl \
        --horizon remaining --subsample-k 8 --seeds 10 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.dataset import (  # noqa: E402
    DatasetConfig,
    DatasetGenerator,
    iter_jsonl,
    write_jsonl,
)

PROBLEMS_5FAMILY = ("Branin", "Hartmann", "Ackley", "SixHumpCamel", "Rastrigin")


def _expected_resume_meta(spec) -> dict:
    """Metadata projection of the REQUESTED config, for resume comparison
    (codex final-audit-2): same-stamp resumes with a different horizon
    mode / utility weights / continuation policy / quota must not append.
    Fields not serialized into record metadata (decision_every_k,
    replicates) cannot be compared here — the refusal message says so."""
    from opt_agent.rollout.specs import spec_to_metadata

    return {
        "horizon_mode": str(spec.horizon_mode),
        "utility_weights": {
            "w_optimization": float(spec.utility_weights.w_optimization),
            "w_surrogate": float(spec.utility_weights.w_surrogate),
            "w_coverage": float(spec.utility_weights.w_coverage),
            "w_evaluation": float(spec.utility_weights.w_evaluation),
            "w_fit": float(spec.utility_weights.w_fit),
            "w_stop_opportunity": float(
                spec.utility_weights.w_stop_opportunity),
        },
        "continuation_policy": spec_to_metadata(spec.continuation_spec),
        "expects_partial_rollout": spec.rollout_action_subsample < 8,
    }


def _existing_trajectory_keys(path: Path, spec=None) -> set[tuple[str, int, int]]:
    """(family, problem_seed, trajectory_id) triples already on disk.

    Resume guard (codex final audit B2): the existing file's version stamps
    must be exactly the CURRENT ones before new records are appended —
    otherwise resuming after a version change would silently create a
    mixed-version artifact. Raises SystemExit on incompatibility.
    """
    from opt_agent.dataset.versioning import (
        HARNESS_VERSION, UTILITY_VERSION, check_record_versions)

    keys: set[tuple[str, int, int]] = set()
    if not path.exists():
        return keys
    stamps_seen: set[tuple] = set()
    for rec in iter_jsonl(path):
        meta = rec.metadata
        stamps_seen.add((meta.get("harness_version"),
                         meta.get("utility_version")))
        keys.add((
            rec.problem["family"],
            int(rec.problem["seed"]),
            int(meta.get("trajectory_id", 0)),
        ))
    if len(stamps_seen) > 1:
        raise SystemExit(
            f"{path} already mixes version stamps "
            f"{sorted(map(str, stamps_seen))}; refusing to resume into a "
            f"mixed-version artifact — write a new output file")
    if stamps_seen and stamps_seen.pop() != (HARNESS_VERSION,
                                             UTILITY_VERSION):
        raise SystemExit(
            f"{path} was written under different version stamps than the "
            f"current code ({HARNESS_VERSION}/{UTILITY_VERSION}); refusing "
            f"to append — write a new versioned output file")
    if spec is not None and keys:
        expected = _expected_resume_meta(spec)
        first_meta = _first_record_meta(path)
        mismatched = [
            k for k in ("horizon_mode", "utility_weights",
                        "continuation_policy")
            if k in first_meta and first_meta[k] != expected[k]
        ]
        has_partial = "partial_rollout" in first_meta
        if has_partial != expected["expects_partial_rollout"]:
            mismatched.append("partial_rollout (quota)")
        if mismatched:
            raise SystemExit(
                f"{path} was generated with a different configuration on "
                f"{mismatched}; refusing to append under the requested "
                f"config — write a new output file. (Fields not recorded "
                f"in record metadata — decision_every_k, replicates — "
                f"cannot be compared; verify them manually.)")
    return keys


def _first_record_meta(path: Path) -> dict:
    for rec in iter_jsonl(path):
        return rec.metadata
    return {}


def _config_from_task(cfg: dict) -> "DatasetConfig":
    """Rebuild DatasetConfig from its asdict() form (nested dataclasses back)."""
    from opt_agent.rollout.specs import ContinuationPolicySpec
    from opt_agent.rollout.utility import UtilityWeights

    fixed = dict(cfg)
    if isinstance(fixed.get("utility_weights"), dict):
        fixed["utility_weights"] = UtilityWeights(**fixed["utility_weights"])
    if isinstance(fixed.get("continuation_spec"), dict):
        fixed["continuation_spec"] = ContinuationPolicySpec(
            **fixed["continuation_spec"]
        )
    return DatasetConfig(**fixed)


def _run_trajectory(task: dict) -> list:
    """Worker: generate all records for one (problem, seed, trajectory_id).

    Pin torch to a single thread: GP fitting is the bottleneck, and with
    many process workers the default (all cores) oversubscribes the machine.
    """
    import torch

    torch.set_num_threads(1)
    from dataclasses import replace as _replace

    base = _config_from_task(task["config"])
    spec = _replace(
        base,
        problems=(task["family"],),
        seeds=(task["seed"],),
        trajectories_per_problem_seed=1,
    )
    return list(DatasetGenerator(spec).generate())


def _record_line(rec) -> str:
    from opt_agent.dataset.decision_record import _json_safe

    return json.dumps(_json_safe(rec), sort_keys=True) + "\n"


def _cost_units(records: list) -> int:
    """Backend cost proxy: total new samples + GP fits across rolled branches."""
    total = 0
    for rec in records:
        for r in rec.rollout_results.values():
            total += int(r.get("n_new_samples", 0)) + int(r.get("n_model_fits", 0))
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--horizon", default="remaining",
                        help="'remaining' (to budget exhaustion) or a fixed int")
    parser.add_argument("--subsample-k", type=int, default=4,
                        help="Candidates rolled out per decision point (1-8)")
    parser.add_argument("--replicates", type=int, default=1,
                        help="Rollout replicates per decision point; utilities averaged (P1b denoising)")
    parser.add_argument("--continuation", default="on_policy", choices=["on_policy", "rule"],
                        help="Rollout continuation policy (P1b alignment: rule matches Gate 4)")
    parser.add_argument("--terminal-utility", action="store_true",
                        help="Pure terminal-delta-best utility (w_opt=1, cost/stop terms zeroed)")
    parser.add_argument("--problems", nargs="+", default=list(PROBLEMS_5FAMILY))
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of problem seeds (0..N-1)")
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--max-actions", type=int, default=25)
    parser.add_argument("--decision-every-k", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="Skip trajectories whose records already exist")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output (ignored with --resume)")
    args = parser.parse_args()

    if args.horizon == "remaining":
        horizon, horizon_mode = 1, "remaining"
    else:
        try:
            horizon = int(args.horizon)
        except ValueError:
            parser.error(f"--horizon must be 'remaining' or an int; got {args.horizon!r}")
        horizon_mode = "fixed"

    if args.output.exists() and not args.resume and not args.force:
        print(f"error: {args.output} exists; pass --force or --resume", file=sys.stderr)
        return 1

    from opt_agent.rollout.specs import ContinuationPolicySpec
    from opt_agent.rollout.utility import UtilityWeights

    weights = UtilityWeights()
    if args.terminal_utility:
        weights = UtilityWeights(
            w_optimization=1.0, w_surrogate=0.0, w_coverage=0.0,
            w_evaluation=0.0, w_fit=0.0, w_stop_opportunity=0.0,
        )
    cont = ContinuationPolicySpec(
        name=args.continuation, seed=0,
        extras={"rule_weight": 0.8} if args.continuation == "on_policy" else {},
    )
    spec = replace(
        DatasetConfig(),
        problems=tuple(args.problems),
        continuation_spec=cont,
        utility_weights=weights,
        seeds=tuple(range(args.seeds)),
        total_budget=args.budget,
        max_actions=args.max_actions,
        decision_every_k_actions=args.decision_every_k,
        horizon=horizon,
        horizon_mode=horizon_mode,
        rollout_action_subsample=args.subsample_k,
        rollout_replicates=args.replicates,
    )

    done_keys = (_existing_trajectory_keys(args.output, spec)
                 if args.resume else set())
    tasks = [
        {"family": p, "seed": s, "trajectory_id": t, "config": asdict(spec)}
        for p in spec.problems
        for s in spec.seeds
        for t in range(spec.trajectories_per_problem_seed)
        if (p.lower(), s, t) not in done_keys
    ]
    skipped = len(done_keys)

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    collected: list = []
    t0 = time.time()
    n_tasks = len(tasks)
    if tasks:
        import multiprocessing

        # spawn: workers import cleanly (no fork-safety issues with torch
        # state); each worker pins itself to 1 torch thread.
        ctx = multiprocessing.get_context("spawn")
        # Serialise the already-present records into tmp first when resuming.
        if args.resume and args.output.exists():
            tmp.write_text(args.output.read_text(), encoding="utf-8")
        else:
            tmp.write_text("", encoding="utf-8")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {pool.submit(_run_trajectory, t): t for t in tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                recs = fut.result()
                collected.extend(recs)
                with tmp.open("a", encoding="utf-8") as fh:
                    for rec in recs:
                        fh.write(_record_line(rec))
                done = sum(1 for f in futures if f.done())
                print(
                    f"[{done}/{n_tasks}] {task['family']} seed={task['seed']} "
                    f"→ {len(recs)} records",
                    flush=True,
                )
    else:
        tmp.write_text("", encoding="utf-8")

    wall = time.time() - t0
    tmp.replace(args.output)

    partial = sum(1 for r in collected if r.metadata.get("partial_rollout"))
    summary = {
        "output": str(args.output),
        "horizon_mode": horizon_mode,
        "subsample_k": args.subsample_k,
        "problems": list(spec.problems),
        "seeds": args.seeds,
        "records_new": len(collected),
        "records_partial": partial,
        "trajectories_skipped_resume": skipped,
        "cost_units_new": _cost_units(collected),
        "cost_units_per_record": (
            round(_cost_units(collected) / len(collected), 2) if collected else 0.0
        ),
        "wall_clock_s": round(wall, 1),
        "quot_note": (
            f"quota per decision point ≈ {args.subsample_k} × remaining_budget "
            f"backend steps (vs 8 × remaining_budget at full rollout)"
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

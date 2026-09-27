"""P1a — teacher label noise vs misspecification diagnostic (2026-09-27).

Gate 4 has failed at every horizon (h3, Phase 8.6D hfull, k=4 quota). Before
choosing a fix (replicate-and-average denoising vs target redefinition),
separate the two candidate causes:

  NOISE      — rollout utilities are stochastic (candidate seeds, continuation
               seeds, GP refits); sign(U_a − U_b) flips under re-rolls.
  MISSPEC    — the utility is a deterministic-but-wrong functional of the
               trajectory; re-rolls agree with each other and with the stored
               labels, but the labels still fail Gate 4.

Protocol: sample records from the k=4 long-horizon dataset (stratified by
family), restore each record's replayed backend checkpoint, re-run the SAME
rolled-out candidate actions R times with fresh seeds, and measure:

  - pair-sign agreement between stored labels and each re-roll,
    stratified by stored |ΔU| quartile (below/above median);
  - argmax stability across re-rolls;
  - utility gap scale (median |ΔU| vs the inter-roll std of ΔU).

Verdict heuristic (predeclared): flip rate on pairs ABOVE the median gap
≥ 0.25 ⇒ NOISE-dominated (denoise first); ≤ 0.10 ⇒ MISSPEC-dominated
(redefine the target); in between ⇒ MIXED.

Runs one record per parallel CPU worker (spawn).

Usage:
    .venv/bin/python scripts/exp_teacher_noise_diagnostic.py --repeats 3 --workers 12
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.dataset import iter_jsonl  # noqa: E402


def _reroll_record(task: dict) -> dict | None:
    """Restore a record's backend; re-roll its rolled-out actions R times."""
    import torch

    torch.set_num_threads(1)
    from opt_agent.actions.models import ActionType, OptimizationAction
    from opt_agent.backend.bocode_botorch import (
        BackendConfig,
        BocodeBotorchBackend,
    )
    from opt_agent.dataset.generator import derive_seed
    from opt_agent.dataset.validation import _reconstruct_checkpoint
    from opt_agent.rollout import Candidate, CounterfactualRollout
    from opt_agent.rollout.specs import ContinuationPolicySpec
    from opt_agent.state import extract_state

    rec = task["record"]
    cp = _reconstruct_checkpoint(rec)
    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(
        problem_id=cp.problem_id, seed=int(rec.problem["seed"]), n_init=1,
    ))
    backend.restore(cp)
    state = extract_state(
        backend,
        total_budget=int(rec.state["resource"]["total_budget"]),
        last_action=rec.state["history"].get("last_action"),
    )

    horizon = int(rec.metadata.get("horizon", 1))
    md = rec.metadata
    run_seed = int(md["run_seed"])
    rerun: list[dict[str, float]] = []
    for rep in range(task["repeats"]):
        cands = []
        for spec in rec.candidate_actions:
            if spec["action_id"] not in rec.rollout_results:
                continue
            merged = dict(spec["params"])
            if "seed" in spec["params"]:
                merged["seed"] = derive_seed(
                    run_seed, rep, 987654321,
                    abs(hash(spec["action_id"])) & 0xFFFF,
                )
            cands.append(Candidate(
                action_id=spec["action_id"],
                action=OptimizationAction(ActionType(spec["type"]), merged),
            ))
        cont = ContinuationPolicySpec(
            name=md.get("continuation_policy", {}).get("name", "on_policy"),
            seed=derive_seed(run_seed, rep, 555555),
            extras=md.get("continuation_policy", {}).get("extras", {}),
        )
        out = CounterfactualRollout(
            backend,
            horizon=max(1, horizon),
            continuation_spec=cont,
            max_rollout_steps=max(200, len(cands) * horizon + 32),
            optimization_scale=float(md.get("optimization_scale", 1.0)),
            evaluation_scale=float(md.get("evaluation_scale", 30.0)),
            fit_scale=float(md.get("fit_scale", 30.0)),
        ).rollout(state, cands)
        rerun.append({
            r.action_id: float(r.utility)
            for r in out.results if r.status == "valid"
        })

    stored = {
        a: float(cell["utility"])
        for a, cell in rec.rollout_results.items()
        if cell.get("status") == "valid"
    }
    flips, gaps, spread = [], [], []
    actions = sorted(stored)
    for i, a in enumerate(actions):
        for b in actions[i + 1:]:
            g = stored[a] - stored[b]
            if abs(g) <= 1e-12:
                continue
            per_roll = [r[a] - r[b] for r in rerun if a in r and b in r]
            agree = all(d * g > 0 for d in per_roll)
            flips.append(0 if agree else 1)
            gaps.append(abs(g))
            if len(per_roll) >= 2:
                spread.append(statistics.pstdev(per_roll))
    argmax_stable = len({
        max(r, key=lambda a: r[a]) for r in rerun if r
    }) == 1
    return {
        "id": rec.id,
        "family": rec.problem["family"],
        "horizon": horizon,
        "n_actions": len(stored),
        "n_pairs": len(flips),
        "flip_rate": (sum(flips) / len(flips)) if flips else None,
        "median_gap": statistics.median(gaps) if gaps else None,
        "median_gap_spread": statistics.median(spread) if spread else None,
        "argmax_stable_across_rerolls": argmax_stable,
        "flips": flips,
        "gaps": gaps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path,
                        default=Path("runs/longhorizon_k4_20seed_5family.jsonl"))
    parser.add_argument("--per-family", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path,
                        default=Path("runs/exp_teacher_noise_diagnostic.json"))
    args = parser.parse_args()

    records = list(iter_jsonl(args.dataset))
    rng = random.Random(args.seed)
    by_fam: dict[str, list] = {}
    for r in records:
        by_fam.setdefault(r.problem["family"], []).append(r)
    sample = []
    for fam in sorted(by_fam):
        pool = by_fam[fam]
        sample.extend(rng.sample(pool, min(args.per_family, len(pool))))
    tasks = [{"record": r, "repeats": args.repeats} for r in sample]
    print(f"{len(tasks)} records × {args.repeats} re-rolls", flush=True)

    results = []
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(_reroll_record, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r is None:
                continue
            results.append(r)
            if i % 10 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] flip_rate={r['flip_rate']}",
                      flush=True)

    all_flips = [f for r in results for f in r["flips"]]
    all_gaps = [g for r in results for g in r["gaps"]]
    order = sorted(range(len(all_gaps)), key=lambda i: all_gaps[i])
    half = len(order) // 2
    below = [all_flips[i] for i in order[:half]]
    above = [all_flips[i] for i in order[half:]]
    above_rate = statistics.fmean(above) if above else None
    verdict = None
    if above_rate is not None:
        if above_rate >= 0.25:
            verdict = "NOISE_DOMINATED"
        elif above_rate <= 0.10:
            verdict = "MISSPEC_DOMINATED"
        else:
            verdict = "MIXED"
    summary = {
        "n_records": len(results),
        "repeats": args.repeats,
        "mean_flip_rate": (statistics.fmean(all_flips)
                           if all_flips else None),
        "flip_rate_below_median_gap": (statistics.fmean(below)
                                       if below else None),
        "flip_rate_above_median_gap": above_rate,
        "median_stored_gap": (statistics.median(all_gaps)
                              if all_gaps else None),
        "median_interroll_gap_spread": statistics.median([
            r["median_gap_spread"] for r in results
            if r["median_gap_spread"] is not None
        ]) if any(r["median_gap_spread"] is not None
                  for r in results) else None,
        "argmax_stable_share": (
            statistics.fmean([r["argmax_stable_across_rerolls"]
                              for r in results]) if results else None
        ),
        "verdict": verdict,
        "verdict_rule": ("flip rate on pairs above the median stored gap: "
                         ">=0.25 NOISE; <=0.10 MISSPEC; else MIXED"),
    }
    args.json_out.write_text(
        json.dumps({"summary": summary, "records": results}, indent=2,
                   sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

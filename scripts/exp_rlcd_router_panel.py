"""EXP-RLCD router panel — does calibrated confidence pay? (2026-09-27)

The journal P1 gap ("缺一个赢的 regime") and P3's model point at the same
test: EXP-RQ3's router failed because the RF softmax max never exceeded
0.25 (no signal); the proper-score arm is the first System-1 whose
reported distribution is trained to be calibrated against the teacher
(RLCD_THEORY R1). If its confidence carries routing-relevant information,
routing low-confidence decisions to a STRONG fixed System-2 should recover
best-of-both (Local-like behavior where Local wins, Surrogate where
Surrogate wins) — a positive regime for the calibrated-System-1 story.

Design (predeclared):
  - System-1 arms: `proper_only`, `bt_only` (forced System-1, from the h3
    checkpoints `runs/rlcd_laya_proper_v1`, `runs/rlcd_laya_bt_v1`).
  - System-2: AlwaysSurrogate (the strongest fixed control; also P3's
    stronger-System-2 model) wrapped as FallbackSystem2.
  - Routed arms: `proper_q{20,40,60}`, `bt_q{20,40,60}` — ConfidenceRouter
    with tau = the p-th percentile of that arm's max-prob distribution,
    measured on the CALIBRATION slice (h3 problem seeds 16–17) and frozen:
    route the p% least-confident decisions to System-2.
  - Reference arm: `surrogate` (System-2 alone).
  - Panel: 5 BOCoDe families × fresh seeds 300–309 × budget 30; paired
    backend seeds across arms. Parallel workers (spawn).

  Success criterion (predeclared): a routed arm beats BOTH its pure arm
  and `surrogate` with record... with seed-clustered CI excluding 0 on at
  least one family, without a CI-detectable loss to the better pure arm
  on any family.

Usage:
    .venv/bin/python scripts/exp_rlcd_router_panel.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _rlcd_common import bootstrap_ci as _bootstrap_ci  # noqa: E402
from _rlcd_common import make_always_surrogate  # noqa: E402

CHECKPOINTS = {
    "proper": "runs/rlcd_laya_proper_v1",
    "bt": "runs/rlcd_laya_bt_v1",
}
QUANTILES = (0.20, 0.40, 0.60)
PROBLEMS = ("Branin", "Ackley", "Hartmann", "SixHumpCamel", "Rastrigin")


def _confidence_percentiles() -> dict[str, dict[float, float]]:
    """Per arm, the p-th percentile of max prob over the calib slice.

    Calib slice = h3 records with problem seeds 16–17 (the same split
    convention as the calibration ablation). CPU inference is sufficient.
    """
    import torch

    from opt_agent.dataset import read_jsonl
    from opt_agent.system1.rlcd import (
        RLCDConfig,
        _collate,
        build_item,
        load_laya_for_training,
    )

    records = [
        r for r in read_jsonl(
            Path("runs/phase8_6d_h3_20seed_5family.jsonl"),
            version_policy="warn")
        if 16 <= int(r.problem["seed"]) < 18
    ]
    out: dict[str, dict[float, float]] = {}
    for arm, path in CHECKPOINTS.items():
        model, tok, _cfg, _ = load_laya_for_training(path)
        model.eval()
        confs = []
        device = next(model.parameters()).device
        with torch.no_grad():
            for r in records:
                item = build_item(r, tok, max_len=512, head_max_len=192)
                if item is None:
                    continue
                batch = _collate([item], tok.pad_token_id)
                tensors = {k: v.to(device) for k, v in batch.items()
                           if torch.is_tensor(v)}
                logits, _ = model(
                    tensors["input_ids"], tensors["attention_mask"],
                    tensors["marker_pos"], tensors["marker_mask"],
                    tensors["qtype"],
                )
                confs.append(float(torch.softmax(logits[0].float().cpu(), -1).max()))
        del model
        out[arm] = {
            q: sorted(confs)[min(len(confs) - 1, int(q * len(confs)))]
            for q in QUANTILES
        }
        print(f"[thresholds:{arm}] n={len(confs)} "
              f"taus={ {q: round(v, 4) for q, v in out[arm].items()} }",
              flush=True)
    return out


def _run_one(task: dict) -> dict:
    """Worker: one (problem, seed, arm) trajectory."""
    import torch

    torch.set_num_threads(1)
    from opt_agent.backend.bocode_botorch import (
        BackendConfig,
        BocodeBotorchBackend,
    )
    from opt_agent.loop import DecisionLoop
    from opt_agent.router.confidence import ConfidenceRouter
    from opt_agent.system1 import HybridPolicy, LayaPolicy
    from opt_agent.system2.fallback import FallbackSystem2

    problem, seed, budget, arm = (
        task["problem"], task["seed"], task["budget"], task["arm"]
    )
    if arm == "surrogate":
        policy = make_always_surrogate(seed=seed)
        route_counts = None
    else:
        if arm.endswith("_only"):
            base, routed = arm[:-len("_only")], ""
        else:
            base, _, routed = arm.partition("_q")
        system1 = LayaPolicy(model_id=CHECKPOINTS[base], seed=seed)
        if routed:
            tau = task["taus"][base][float(routed) / 100]
            policy = HybridPolicy(
                system1=system1,
                router=ConfidenceRouter(tau_conf=tau),
                system2=FallbackSystem2(
                    baseline=make_always_surrogate(seed=seed)  # type: ignore[arg-type]
                ),
            )
        else:
            policy = HybridPolicy(
                system1=system1,
                router=ConfidenceRouter(tau_conf=0.0),  # forced System-1
                system2=FallbackSystem2(
                    baseline=make_always_surrogate(seed=seed)  # type: ignore[arg-type]
                ),
            )
        route_counts = None  # filled after the run from policy.history

    backend = BocodeBotorchBackend()
    backend.initialize(BackendConfig(problem_id=problem, seed=seed, n_init=5))
    initial = backend.get_history().objectives.flatten()
    scale = float(initial.std(correction=0)) or 1.0
    trajectory = DecisionLoop(backend, total_budget=budget).run(
        policy, max_rounds=40
    )
    history = backend.get_history()
    actions = [step.action.type.value for step in trajectory.steps]
    if route_counts is None and hasattr(policy, "history"):
        route_counts = dict(Counter(
            step.route.name for step in policy.history  # type: ignore[attr-defined]
        ))
    return {
        "problem": problem, "seed": seed, "budget": budget, "arm": arm,
        "scale": scale,
        "terminal_best": float(history.objectives.max()),
        "new_evaluations": len(history) - 5,
        "termination": trajectory.termination_reason.value,
        "actions": actions,
        "route_counts": route_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(300, 310)))
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/exp_rlcd_router_panel.json"))
    args = parser.parse_args()

    taus = _confidence_percentiles()
    arms = (["surrogate", "proper_only", "bt_only"]
            + [f"proper_q{int(q*100)}" for q in QUANTILES]
            + [f"bt_q{int(q*100)}" for q in QUANTILES])
    tasks = [
        {"problem": p, "seed": s, "budget": args.budget, "arm": arm,
         "taus": taus}
        for p in PROBLEMS
        for s in args.seeds
        for arm in arms
    ]
    print(f"{len(tasks)} trajectories, {len(arms)} arms, "
          f"{args.workers} workers", flush=True)

    rows: list[dict] = []
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            row = fut.result()
            rows.append(row)
            if i % 20 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] {t['problem']} seed={t['seed']} "
                      f"{t['arm']}", flush=True)

    by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by[(row["problem"], row["arm"])].append(row)

    summary: dict = {}
    comparisons = []
    for (problem, arm), group in sorted(by.items()):
        summary[f"{problem}/{arm}"] = {
            "n": len(group),
            "mean_best": statistics.fmean(r["terminal_best"] for r in group),
            "route_counts": (
                dict(sum((Counter(r["route_counts"]) for r in group
                          if r["route_counts"]), Counter()))
                or None
            ),
        }
    # Predeclared contrasts: routed vs both pure arms and surrogate.
    for problem in PROBLEMS:
        for base in ("proper", "bt"):
            for q in QUANTILES:
                arm = f"{base}_q{int(q*100)}"
                for ref in (f"{base}_only", "surrogate"):
                    a = {r["seed"]: r for r in by[(problem, arm)]}
                    b = {r["seed"]: r for r in by[(problem, ref)]}
                    diffs = [
                        (a[s]["terminal_best"] - b[s]["terminal_best"])
                        / a[s]["scale"]
                        for s in sorted(set(a) & set(b))
                    ]
                    ci = _bootstrap_ci(diffs) if len(diffs) >= 5 else None
                    comparisons.append({
                        "problem": problem, "arm": arm, "vs": ref,
                        "mean_normalized_gain": statistics.fmean(diffs),
                        "ci95": ci,
                        "ci_excludes_zero": (ci is not None and
                                             (ci[0] > 0 or ci[1] < 0)),
                    })

    args.output.write_text(
        json.dumps({"taus": taus, "arms": arms, "rows": rows,
                    "summary": summary, "comparisons": comparisons},
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparisons, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

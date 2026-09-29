"""RLCD hyperparameter sweep — parallel GPU utilization (2026-09-27).

Runs a small grid per arm (proper_score: w_sph × lr; bt_pairs: β × margin)
as concurrent single-GPU worker processes (two by default — the RTX 3090
comfortably holds two batch-8 ModernBERT-large trainings). Each run trains
with save=False (no 1.6 GB writes) and reports decision-level val metrics;
best config per arm is printed. CPU stays free for dataset generation /
closed-loop workers.

Usage:
    .venv/bin/python scripts/sweep_rlcd.py --workers 2
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.dataset import read_jsonl  # noqa: E402
from opt_agent.system1.rlcd import RLCDConfig, train_rlcd  # noqa: E402


def _one(task: dict) -> dict:
    import time

    import torch

    torch.set_num_threads(4)
    torch.manual_seed(0)
    records = read_jsonl(Path(task["dataset"]), version_policy="warn")
    train = [r for r in records if int(r.problem["seed"]) < task["train_max"]]
    val = [r for r in records if int(r.problem["seed"]) >= task["val_min"]]
    cfg = RLCDConfig(save=False, output_dir="/tmp/rlcd_sweep_unused", **task["cfg"])
    start = time.perf_counter()
    rep = train_rlcd(train, val, cfg)
    return {
        "tag": task["tag"],
        "objective": task["cfg"]["objective"],
        "overrides": task["overrides"],
        "best_val_pairwise_accuracy": rep["best_val_pairwise_accuracy"],
        "final_val": rep["final_val"],
        "train_wall_s": round(time.perf_counter() - start, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path,
                        default=Path("runs/phase8_6d_h3_20seed_5family.jsonl"))
    parser.add_argument("--train-max", type=int, default=16)
    parser.add_argument("--val-min", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent GPU training processes")
    parser.add_argument("--json-out", type=Path,
                        default=Path("runs/exp_rlcd_sweep.json"))
    args = parser.parse_args()

    tasks: list[dict] = []
    for w_sph, lr in itertools.product((0.25, 0.5, 1.0), (5e-6, 1e-5, 2e-5)):
        tasks.append({
            "tag": f"proper_wsph{w_sph}_lr{lr:g}",
            "dataset": str(args.dataset),
            "train_max": args.train_max, "val_min": args.val_min,
            "cfg": {
                "objective": "proper_score",
                "num_epochs": args.epochs,
                "learning_rate": lr,
                "w_sph": w_sph,
            },
            "overrides": {"w_sph": w_sph, "lr": lr},
        })
    for beta, margin in itertools.product((0.5, 1.0, 2.0), (0.0, 0.05)):
        tasks.append({
            "tag": f"bt_beta{beta:g}_margin{margin:g}",
            "dataset": str(args.dataset),
            "train_max": args.train_max, "val_min": args.val_min,
            "cfg": {
                "objective": "bt_pairs",
                "num_epochs": args.epochs,
                "learning_rate": 1e-5,
                "beta": beta,
                "margin": margin,
            },
            "overrides": {"beta": beta, "margin": margin},
        })

    results = []
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")  # CUDA requires spawn
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(_one, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            print(
                f"[sweep] {r['tag']}: pair_acc={r['best_val_pairwise_accuracy']:.3f} "
                f"log_score={r['final_val'].get('mean_log_score')}",
                flush=True,
            )

    results.sort(key=lambda r: -r["best_val_pairwise_accuracy"])
    args.json_out.write_text(json.dumps(results, indent=2, sort_keys=True),
                             encoding="utf-8")
    for arm in ("proper_score", "bt_pairs"):
        arm_rows = [r for r in results if r["objective"] == arm]
        if arm_rows:
            best = arm_rows[0]
            print(f"BEST {arm}: {best['tag']} "
                  f"pair_acc={best['best_val_pairwise_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""EXP-RLCD-LOO — leave-one-family-out transfer of fine-tuned Laya (2026-09-27).

Journal-level RQ2 evidence for the fine-tuning line: can a Laya System-1
fine-tuned (RLCD proper-score / BT pairs) on FOUR families transfer to an
unseen fifth, at the decision level?

Design: 5 folds. Each fold trains both arms on the other four families
(early stopping on a held-out seed slice OF THE TRAIN FAMILIES) and
evaluates on all records of the held-out family: pairwise accuracy,
top-1 agreement, mean log score vs that family's teacher labels. The RF
baseline is retrained per fold on raw rollout utilities (decision-level
metrics are invariant to per-record monotone utility transforms, so no
cross-family scale pooling is needed).

Runs fold-tasks as spawned GPU worker processes (two by default).

Usage:
    .venv/bin/python scripts/exp_rlcd_loo_transfer.py --workers 2
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _rlcd_common import (  # noqa: E402
    per_record_laya,
    raw_gold as _raw_gold,
    rf_pairwise_and_top1 as _rf_pairwise_and_top1,
)

from opt_agent.dataset import read_jsonl  # noqa: E402


def _eval_arm(model, tok, records) -> dict:
    import torch

    per_rec = per_record_laya(model, tok, records,
                              "cuda" if torch.cuda.is_available() else "cpu")
    n_pairs = sum(v[1] for v in per_rec.values())
    return {
        "pairs": n_pairs,
        "pairwise_accuracy": (sum(v[0] for v in per_rec.values()) / n_pairs)
                             if n_pairs else 0.0,
        "top1_agreement": (sum(v[2] for v in per_rec.values()) / len(per_rec))
                          if per_rec else 0.0,
        "mean_log_score": (
            sum(v[3] for v in per_rec.values() if v[3] is not None)
            / max(1, sum(1 for v in per_rec.values() if v[3] is not None))
        ) if any(v[3] is not None for v in per_rec.values()) else None,
        "records": len(per_rec),
    }


def _rf_fold(train_records, test_records) -> dict:
    from sklearn.ensemble import RandomForestRegressor

    from _shared_rq_features import ACTIONS, extract_features
    from opt_agent.dataset import phase_of_ratio

    def feats(r):
        return extract_features(
            r.state, phase_of_ratio(
                float(r.state["resource"]["remaining_budget_ratio"])
            )
        )

    # Train only on full-width records (all 8 actions valid) so the target
    # matrix is rectangular; test-side metrics are width-agnostic.
    X, Y = [], []
    n_train_skipped = 0
    for r in train_records:
        gold = _raw_gold(r)
        if gold is None:
            continue
        if set(gold) != set(ACTIONS):
            n_train_skipped += 1
            continue
        X.append(feats(r))
        Y.append([gold[a] for a in ACTIONS])
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=0, n_jobs=8,
    )
    rf.fit(X, Y)
    correct = total = top1 = n = 0
    for r in test_records:
        gold = _raw_gold(r)
        if gold is None:
            continue
        pred_vec = rf.predict([feats(r)])[0]
        pred = dict(zip(ACTIONS, pred_vec))
        c, t, m = _rf_pairwise_and_top1(pred, gold)
        correct += c
        total += t
        top1 += m
        n += 1
    return {
        "pairwise_accuracy": correct / total if total else 0.0,
        "top1_agreement": top1 / n if n else 0.0,
        "records": n,
        "train_records_full_width": len(X),
        "train_records_skipped_partial": n_train_skipped,
    }


def _run_fold(task: dict) -> dict:
    """Train both arms on 4 families; evaluate on the held-out family."""
    import shutil

    import torch

    torch.set_num_threads(4)
    from opt_agent.system1.rlcd import RLCDConfig, train_rlcd

    records = read_jsonl(Path(task["dataset"]), version_policy="warn")
    held_out = task["family"]
    train_pool = [r for r in records if r.problem["family"] != held_out]
    test = [r for r in records if r.problem["family"] == held_out]
    # Early-stopping slice from the TRAIN families (seeds >= 16 of pool).
    tr = [r for r in train_pool if int(r.problem["seed"]) < 16]
    va = [r for r in train_pool if int(r.problem["seed"]) >= 16]

    out = {"family": held_out, "test_records": len(test)}
    for objective in ("proper_score", "bt_pairs"):
        tmp_dir = Path(f"runs/.rlcd_loo_tmp/{held_out}_{objective}")
        cfg = RLCDConfig(
            output_dir=str(tmp_dir), objective=objective,
            num_epochs=task["epochs"],
        )
        train_rlcd(tr, va, cfg)
        import laya

        agent = laya.load(str(tmp_dir))
        out[objective] = _eval_arm(agent.model, agent.tok, test)
        del agent
        torch.cuda.empty_cache()
        shutil.rmtree(tmp_dir, ignore_errors=True)
    out["rf"] = _rf_fold(train_pool, test)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path,
                        default=Path("runs/phase8_6d_h3_20seed_5family.jsonl"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--json-out", type=Path,
                        default=Path("runs/exp_rlcd_loo_transfer.json"))
    args = parser.parse_args()

    families = sorted({r.problem["family"] for r in read_jsonl(args.dataset, version_policy="warn")})
    tasks = [
        {"family": f, "dataset": str(args.dataset), "epochs": args.epochs}
        for f in families
    ]
    results = []
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(_run_fold, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            for arm in ("proper_score", "bt_pairs", "rf"):
                m = r[arm]
                print(
                    f"[loo:{r['family']}] {arm:<13} "
                    f"pair_acc={m['pairwise_accuracy']:.3f} "
                    f"top1={m['top1_agreement']:.3f}"
                    + (f" log_score={m['mean_log_score']:.3f}"
                       if m.get("mean_log_score") is not None else ""),
                    flush=True,
                )

    results.sort(key=lambda r: r["family"])
    agg = {}
    for arm in ("proper_score", "bt_pairs", "rf"):
        accs = [r[arm]["pairwise_accuracy"] for r in results]
        tops = [r[arm]["top1_agreement"] for r in results]
        agg[arm] = {
            "mean_pairwise_accuracy": sum(accs) / len(accs),
            "mean_top1_agreement": sum(tops) / len(tops),
        }
        ls = [r[arm]["mean_log_score"] for r in results
              if r[arm].get("mean_log_score") is not None]
        if ls:
            agg[arm]["mean_log_score"] = sum(ls) / len(ls)
    args.json_out.write_text(
        json.dumps({"folds": results, "aggregate": agg}, indent=2,
                   sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(agg, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

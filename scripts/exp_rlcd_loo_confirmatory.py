"""EXP-RLCD-LOO confirmatory rerun — per-record CIs, dual training seeds.

Upgrades the exploratory E5 (five-fold leave-one-family-out, per-fold
aggregates) to the journal's confirmatory standard:

  - per-record predictions logged for every arm;
  - TWO training seeds per fold per Laya arm (per-record results averaged
    over seeds for the CI, both reported);
  - record-clustered paired bootstrap CIs for bt/proper vs RF on each
    held-out family and pooled;
  - RF retrained per fold on full-width raw-utility records.

Usage:
    .venv/bin/python scripts/exp_rlcd_loo_confirmatory.py --workers 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _rlcd_common import (  # noqa: E402
    ACTIONS_T,
    bootstrap_ci as _bootstrap_ci,
    paired_accuracy_diffs as _paired_diffs,
    per_record_laya as _per_record_laya,
    raw_gold as _raw_gold,
    rf_pairwise_and_top1,
)

from opt_agent.dataset import read_jsonl  # noqa: E402

def _per_record_laya_compat(model, tok, records, device_str):
    from _rlcd_common import per_record_laya
    return per_record_laya(model, tok, records, device_str)


def _rf_per_record(train_records, test_records) -> dict[str, tuple]:
    from sklearn.ensemble import RandomForestRegressor

    from _shared_rq_features import extract_features
    from opt_agent.dataset import phase_of_ratio

    def feats(r):
        return extract_features(
            r.state, phase_of_ratio(
                float(r.state["resource"]["remaining_budget_ratio"])
            )
        )

    X, Y = [], []
    for r in train_records:
        gold = _raw_gold(r)
        if gold is None or set(gold) != set(ACTIONS_T):
            continue
        X.append(feats(r))
        Y.append([gold[a] for a in ACTIONS_T])
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=0, n_jobs=8,
    )
    rf.fit(X, Y)
    out: dict[str, tuple] = {}
    for r in test_records:
        gold = _raw_gold(r)
        if gold is None:
            continue
        pred = dict(zip(ACTIONS_T, rf.predict([feats(r)])[0]))
        correct, total, top1 = rf_pairwise_and_top1(pred, gold)
        out[r.id] = (correct, total, top1, None)
    return out


def _run_fold(task: dict) -> dict:
    import shutil

    import torch

    torch.set_num_threads(4)
    from opt_agent.system1.rlcd import RLCDConfig, train_rlcd

    records = read_jsonl(Path(task["dataset"]), version_policy="warn")
    held_out = task["family"]
    train_pool = [r for r in records if r.problem["family"] != held_out]
    test = [r for r in records if r.problem["family"] == held_out]
    # Split the 20-seed pool by seed RANK (16 smallest -> train, rest ->
    # early-stop val) so seed-offset datasets (e.g. the s100-119
    # replication stream) split the same way; on the original seeds 0-19
    # this is exactly `< 16` / `>= 16`.
    pool_seeds = sorted({int(r.problem["seed"]) for r in train_pool})
    cut = pool_seeds[15] if len(pool_seeds) >= 16 else pool_seeds[-1]
    tr = [r for r in train_pool if int(r.problem["seed"]) <= cut]
    va = [r for r in train_pool if int(r.problem["seed"]) > cut]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = {"family": held_out, "test_records": len(test), "arms": {}}
    for objective in ("proper_score", "bt_pairs"):
        seed_rows = []
        for tseed in task["train_seeds"]:
            tmp = Path(f"runs/.rlcd_loo_c/{held_out}_{objective}_s{tseed}")
            cfg = RLCDConfig(
                output_dir=str(tmp), objective=objective,
                num_epochs=task["epochs"], seed=tseed,
            )
            torch.manual_seed(tseed)
            train_rlcd(tr, va, cfg)
            import laya

            agent = laya.load(str(tmp))
            per_rec = _per_record_laya_compat(agent.model, agent.tok, test, device)
            seed_rows.append(per_rec)
            del agent
            torch.cuda.empty_cache()
            shutil.rmtree(tmp, ignore_errors=True)
        # Average per-record counts over training seeds.
        ids = sorted({i for row in seed_rows for i in row})
        merged = {}
        for i in ids:
            rows = [row[i] for row in seed_rows if i in row]
            merged[i] = (
                statistics.fmean(r[0] for r in rows),
                statistics.fmean(r[1] for r in rows),
                statistics.fmean(r[2] for r in rows),
                (statistics.fmean([r[3] for r in rows if r[3] is not None])
                 if any(r[3] is not None for r in rows) else None),
            )
        out["arms"][objective] = {
            "n_train_seeds": len(seed_rows),
            "per_seed_acc": [
                (sum(v[0] for v in row.values()) /
                 max(1, sum(v[1] for v in row.values())))
                for row in seed_rows
            ],
            "per_record": {k: list(v) for k, v in merged.items()},
        }
    rf_rec = _rf_per_record(train_pool, test)
    out["arms"]["rf"] = {
        "per_record": {k: list(v) for k, v in rf_rec.items()},
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path,
                        default=Path("runs/phase8_6d_h3_20seed_5family.jsonl"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--json-out", type=Path,
                        default=Path("runs/exp_rlcd_loo_confirmatory.json"))
    args = parser.parse_args()

    families = sorted({r.problem["family"] for r in read_jsonl(args.dataset, version_policy="warn")})
    tasks = [{"family": f, "dataset": str(args.dataset),
              "epochs": args.epochs, "train_seeds": args.train_seeds}
             for f in families]
    results = []
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = {pool.submit(_run_fold, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            for arm in ("proper_score", "bt_pairs"):
                accs = r["arms"][arm]["per_seed_acc"]
                rf = r["arms"]["rf"]["per_record"]
                rf_acc = (sum(v[0] for v in rf.values())
                          / max(1, sum(v[1] for v in rf.values())))
                print(f"[loo-c:{r['family']}] {arm:<13} "
                      f"seeds={[round(a,3) for a in accs]} "
                      f"rf={rf_acc:.3f}", flush=True)

    # Pooled + per-fold record-clustered CIs (bt/proper vs rf).
    contrasts = {}
    pooled = {"bt_pairs": [], "proper_score": []}
    for r in results:
        rf = {k: tuple(v) for k, v in r["arms"]["rf"]["per_record"].items()}
        for arm in ("bt_pairs", "proper_score"):
            laya = {k: tuple(v) for k, v in r["arms"][arm]["per_record"].items()}
            diffs = _paired_diffs(laya, rf)
            ci = _bootstrap_ci(diffs)
            contrasts[f"{r['family']}/{arm}_vs_rf"] = {
                "n_records": len(diffs),
                "mean_per_record_diff": statistics.fmean(diffs),
                "ci95": ci,
                "ci_excludes_zero": ci[0] > 0 or ci[1] < 0,
            }
            pooled[arm].extend(diffs)
    for arm, diffs in pooled.items():
        ci = _bootstrap_ci(diffs)
        contrasts[f"POOLED/{arm}_vs_rf"] = {
            "n_records": len(diffs),
            "mean_per_record_diff": statistics.fmean(diffs),
            "ci95": ci,
            "ci_excludes_zero": ci[0] > 0 or ci[1] < 0,
        }

    args.json_out.write_text(
        json.dumps({"folds": results, "contrasts": contrasts}, indent=2,
                   sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(contrasts, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

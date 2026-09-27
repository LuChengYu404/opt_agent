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
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from opt_agent.dataset import iter_jsonl  # noqa: E402

ACTIONS_T = ("global_exploration", "local_exploitation", "uncertainty_sampling",
             "error_guided_sampling", "boundary_sampling", "surrogate_acquisition",
             "retrain_surrogate", "stop")


def _raw_gold(record) -> dict[str, float] | None:
    gold = {}
    for a, cell in record.rollout_results.items():
        if cell.get("status") != "valid":
            continue
        u = cell.get("utility")
        if isinstance(u, (int, float)) and u == u:
            gold[a] = float(u)
    return gold if len(gold) >= 2 else None


def _per_record_laya(model, tok, records, device_str: str) -> dict[str, list]:
    """record_id → [(correct, total, top1, log_score)] (one per record)."""
    import torch

    from opt_agent.system1.rlcd import (  # noqa: F401
        RLCDConfig,
        _collate,
        build_item,
    )

    device = torch.device(device_str)
    model.to(device).eval()
    out: dict[str, list] = {}
    with torch.no_grad():
        for r in records:
            item = build_item(r, tok, max_len=512, head_max_len=192)
            if item is None or _raw_gold(r) is None:
                continue
            batch = _collate([item], tok.pad_token_id)
            tensors = {k: v.to(device) for k, v in batch.items()
                       if torch.is_tensor(v)}
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits, _ = model(
                    tensors["input_ids"], tensors["attention_mask"],
                    tensors["marker_pos"], tensors["marker_mask"],
                    tensors["qtype"],
                )
            row = logits[0].float().cpu()
            index_of = {a: i for i, a in enumerate(item["option_ids"])}
            gold = _raw_gold(r)
            gold_top = max(gold, key=lambda a: gold[a])
            model_top = max((a for a in gold if a in index_of),
                            key=lambda a: float(row[index_of[a]]))
            # item["pairs"] carries option indices already.
            correct = sum(int(row[i] > row[j]) for i, j, _ in item["pairs"])
            log_score = None
            if item.get("target") is not None:
                logq = torch.log_softmax(row, -1).clamp_min(-9.21)
                t = torch.tensor(item["target"], dtype=logq.dtype)
                log_score = float((t * logq).sum())
            out[r.id] = (correct, len(item["pairs"]),
                         int(model_top == gold_top), log_score)
    model.to("cpu")
    return out


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
        actions = sorted(gold)
        correct = total = 0
        for i, a in enumerate(actions):
            for b in actions[i + 1:]:
                g = gold[a] - gold[b]
                if abs(g) <= 1e-12:
                    continue
                total += 1
                correct += int((pred[a] - pred[b] > 0) == (g > 0))
        top1 = int(max(actions, key=lambda a: pred[a])
                   == max(actions, key=lambda a: gold[a]))
        out[r.id] = (correct, total, top1, None)
    return out


def _bootstrap_ci(diffs: list[float], repetitions: int = 2000,
                  seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0)
    means = sorted(
        statistics.fmean(diffs[rng.randrange(n)] for _ in range(n))
        for _ in range(repetitions)
    )
    return means[int(0.025 * repetitions)], means[int(0.975 * repetitions)]


def _paired_diffs(a: dict[str, tuple], b: dict[str, tuple]) -> list[float]:
    """Record-clustered accuracy diffs: per-record correct-rate difference."""
    ids = sorted(set(a) & set(b))
    out = []
    for i in ids:
        ca, ta = a[i][0], a[i][1]
        cb, tb = b[i][0], b[i][1]
        if ta == 0 or tb == 0:
            continue
        out.append(ca / ta - cb / tb)
    return out


def _run_fold(task: dict) -> dict:
    import shutil

    import torch

    torch.set_num_threads(4)
    from opt_agent.system1.rlcd import RLCDConfig, train_rlcd

    records = [r for r in iter_jsonl(Path(task["dataset"]))]
    held_out = task["family"]
    train_pool = [r for r in records if r.problem["family"] != held_out]
    test = [r for r in records if r.problem["family"] == held_out]
    tr = [r for r in train_pool if int(r.problem["seed"]) < 16]
    va = [r for r in train_pool if int(r.problem["seed"]) >= 16]
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
            per_rec = _per_record_laya(agent.model, agent.tok, test, device)
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

    families = sorted({r.problem["family"] for r in iter_jsonl(args.dataset)})
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

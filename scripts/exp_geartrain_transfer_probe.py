"""P3 — numerical → engineering transfer probe on GearTrain (2026-09-27).

Journal P0 gap ("≥1 real engineering task"): GearTrain is the only BOCoDe
engineering problem passing the backend's Phase-1 gates (single-objective,
unconstrained, d=4). This probe asks whether System-1 models trained on the
FIVE SYNTHETIC families rank correctly on an engineering problem's decision
states — the RQ2.3 "numerical → engineering" decision-level probe.

Arms (no retraining): zero-shot Laya, the h3 fine-tuned proper/bt
checkpoints, and the RF trained on the synthetic families (raw utilities;
decision-level metrics are invariant to per-record monotone transforms).

Usage:
    .venv/bin/python scripts/exp_geartrain_transfer_probe.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from opt_agent.dataset import iter_jsonl  # noqa: E402
from opt_agent.system1.rlcd import (  # noqa: E402
    RLCDConfig,
    _collate,
    build_item,
    load_laya_for_training,
)

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


def _laya_arm(model, tok, records, device) -> dict:
    model.to(device).eval()
    n_pairs = n_correct = n_top1 = n = 0
    with torch.no_grad():
        for r in records:
            item = build_item(r, tok, max_len=512, head_max_len=192)
            gold = _raw_gold(r)
            if item is None or gold is None:
                continue
            batch = _collate([item], tok.pad_token_id)
            tensors = {k: v.to(device) for k, v in batch.items()
                       if torch.is_tensor(v)}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits, _ = model(
                    tensors["input_ids"], tensors["attention_mask"],
                    tensors["marker_pos"], tensors["marker_mask"],
                    tensors["qtype"],
                )
            row = logits[0].float().cpu()
            index_of = {a: i for i, a in enumerate(item["option_ids"])}
            n += 1
            n_top1 += int(
                max(gold, key=lambda a: float(row[index_of[a]]))
                == max(gold, key=lambda a: gold[a])
            )
            for i, j, _ in item["pairs"]:
                n_pairs += 1
                n_correct += int(row[i] > row[j])
    model.to("cpu")
    return {"records": n, "pairs": n_pairs,
            "pairwise_accuracy": n_correct / n_pairs if n_pairs else 0.0,
            "top1_agreement": n_top1 / n if n else 0.0}


def _rf_arm(train_records, test_records) -> dict:
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
    n_pairs = n_correct = n_top1 = n = 0
    for r in test_records:
        gold = _raw_gold(r)
        if gold is None:
            continue
        pred = dict(zip(ACTIONS_T, rf.predict([feats(r)])[0]))
        actions = sorted(gold)
        n += 1
        n_top1 += int(max(actions, key=lambda a: pred[a])
                      == max(actions, key=lambda a: gold[a]))
        for i, a in enumerate(actions):
            for b in actions[i + 1:]:
                g = gold[a] - gold[b]
                if abs(g) <= 1e-12:
                    continue
                n_pairs += 1
                n_correct += int((pred[a] - pred[b] > 0) == (g > 0))
    return {"records": n, "pairs": n_pairs,
            "pairwise_accuracy": n_correct / n_pairs if n_pairs else 0.0,
            "top1_agreement": n_top1 / n if n else 0.0}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = list(iter_jsonl(Path("runs/geartrain_h3_10seed.jsonl")))
    train = [r for r in iter_jsonl(Path("runs/phase8_6d_h3_20seed_5family.jsonl"))]
    print(f"geartrain records: {len(test)}", flush=True)

    out: dict = {}
    model, tok, _cfg, _ = load_laya_for_training("/home/lcy/.cache/opt_agent_v3/laya")
    out["zero_shot"] = _laya_arm(model, tok, test, device)
    del model
    torch.cuda.empty_cache()
    for name, path in (("proper_h3", "runs/rlcd_laya_proper_v1"),
                       ("bt_h3", "runs/rlcd_laya_bt_v1")):
        m, t2, _c, _ = load_laya_for_training(path)
        out[name] = _laya_arm(m, t2, test, device)
        del m
        torch.cuda.empty_cache()
    out["rf_synthetic"] = _rf_arm(train, test)

    Path("runs/exp_geartrain_transfer_probe.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for k, v in out.items():
        print(f"{k:<14} pair_acc={v['pairwise_accuracy']:.3f} "
              f"top1={v['top1_agreement']:.3f} (n={v['records']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

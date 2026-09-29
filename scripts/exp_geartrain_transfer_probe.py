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

from _rlcd_common import (  # noqa: E402
    ACTIONS_T,
    per_record_laya,
    raw_gold as _raw_gold,
    rf_pairwise_and_top1,
)

from opt_agent.dataset import read_jsonl  # noqa: E402
from opt_agent.system1.rlcd import (  # noqa: E402
    RLCDConfig,
    _collate,
    build_item,
    load_laya_for_training,
)

def _laya_arm(model, tok, records, device) -> dict:
    per_rec = per_record_laya(model, tok, records, str(device))
    n_pairs = sum(v[1] for v in per_rec.values())
    return {"records": len(per_rec), "pairs": n_pairs,
            "pairwise_accuracy": (sum(v[0] for v in per_rec.values()) / n_pairs)
                                 if n_pairs else 0.0,
            "top1_agreement": (sum(v[2] for v in per_rec.values()) / len(per_rec))
                              if per_rec else 0.0}


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
        c, tt, m = rf_pairwise_and_top1(pred, gold)
        n += 1
        n_top1 += m
        n_pairs += tt
        n_correct += c
    return {"records": n, "pairs": n_pairs,
            "pairwise_accuracy": n_correct / n_pairs if n_pairs else 0.0,
            "top1_agreement": n_top1 / n if n else 0.0}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from opt_agent.dataset.versioning import assert_combinable
    test = read_jsonl(Path("runs/geartrain_h3_10seed.jsonl"),
                      version_policy="warn")
    train = read_jsonl(Path("runs/phase8_6d_h3_20seed_5family.jsonl"),
                       version_policy="warn")
    # Cross-dataset consumption: CHECK the combined pool without an
    # opt-in — identical stamps must pass on their own; a cross-version
    # mix raises/warns and only a justified same-data A/B note may opt in
    # (codex final-audit-2).
    assert_combinable(
        {"geartrain_h3_10seed.jsonl": [r.metadata for r in test],
         "phase8_6d_h3_20seed_5family.jsonl": [r.metadata for r in train]},
        "warn")
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

"""Paired bootstrap CIs for the EXP-RLCD decision-level comparison (2026-09-27).

Re-evaluates zero-shot Laya, the RLCD fine-tuned checkpoint, and the RF
utility baseline on the same held-out records, computes per-record pairwise
correct counts, and bootstraps (clustered by record) CIs for the pairwise
accuracy differences. Cluster level = record (each record contributes its
own pair set; resampling records keeps within-record pair correlation).

Usage:
    .venv/bin/python scripts/analyze_rlcd_ci.py
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _rlcd_common import per_record_laya  # noqa: E402

from opt_agent.dataset import iter_jsonl, phase_of_ratio  # noqa: E402
from opt_agent.system1.rlcd import (  # noqa: E402
    RLCDConfig,
    build_item,
    load_laya_for_training,
)
from train_rlcd_laya import fit_rf, rf_per_record  # noqa: E402


def paired_bootstrap(
    per_rec_a: dict[str, tuple[int, int, int, float]],
    per_rec_b: dict[str, tuple[int, int, int, float]],
    *,
    repetitions: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for accuracy(A) − accuracy(B), clustered by record.

    Each resample recomputes both arms' correct counts AND pair totals from
    the resampled records, so each arm is normalized by its own denominator
    (arm pair totals need not match across arms).
    """
    ids = sorted(set(per_rec_a) & set(per_rec_b))
    rng = random.Random(seed)
    n = len(ids)
    diffs = []
    for _ in range(repetitions):
        ca = cb = ta = tb = 0
        for _ in range(n):
            rid = ids[rng.randrange(n)]
            ca += per_rec_a[rid][0]
            ta += per_rec_a[rid][1]
            cb += per_rec_b[rid][0]
            tb += per_rec_b[rid][1]
        diffs.append(ca / max(1, ta) - cb / max(1, tb))
    diffs.sort()
    return diffs[int(0.025 * repetitions)], diffs[int(0.975 * repetitions)], (
        sum(d > 0 for d in diffs) / repetitions
    )


def main() -> int:
    dataset = Path("runs/phase8_6d_h3_20seed_5family.jsonl")
    train_records = [r for r in iter_jsonl(dataset) if int(r.problem["seed"]) < 16]
    records = [r for r in iter_jsonl(dataset) if int(r.problem["seed"]) >= 16]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg = RLCDConfig(device=str(device), batch_size=8)

    model, tok, _cfg, _ = load_laya_for_training("/home/lcy/.cache/opt_agent_v3/laya")
    model.to(device)
    zero = per_record_laya(model, tok, records, str(device))
    model.to("cpu"); del model

    import laya

    arms = {}
    for name, path in (
        ("proper", "runs/rlcd_laya_proper_v1"),
        ("bt", "runs/rlcd_laya_bt_v1"),
    ):
        if not Path(path).is_dir():
            print(f"[skip] {name}: {path} not found")
            continue
        a = laya.load(path)
        a.model.to(device)
        arms[name] = per_record_laya(a.model, a.tok, records, str(device))
        a.model.to("cpu")

    rf_model, rf_scales = fit_rf([asdict(r) for r in train_records])
    rf = rf_per_record(rf_model, rf_scales, [asdict(r) for r in records])

    def _acc(d):
        c = sum(v[0] for v in d.values()); t = sum(v[1] for v in d.values())
        return c / t if t else 0.0

    def _top1(d):
        return (sum(v[2] for v in d.values()) / len(d)) if d else 0.0

    def _logs(d):
        vals = [v[3] for v in d.values()
                if v[3] is not None and v[3] == v[3]]  # skip None/NaN
        return sum(vals) / len(vals) if vals else None

    result = {
        "n_records": len(next(iter(arms.values()), zero)),
        "pairs": sum(v[1] for v in zero.values()),
        "acc": {"zero": _acc(zero), "rf": _acc(rf),
                **{k: _acc(v) for k, v in arms.items()}},
        "top1": {"zero": _top1(zero), "rf": _top1(rf),
                 **{k: _top1(v) for k, v in arms.items()}},
        "mean_log_score": {"zero": _logs(zero),
                           **{k: _logs(v) for k, v in arms.items()}},
        "ci": {
            **{f"{k}_minus_zero": paired_bootstrap(v, zero)
               for k, v in arms.items()},
            **{f"{k}_minus_rf": paired_bootstrap(v, rf)
               for k, v in arms.items()},
            "rf_minus_zero": paired_bootstrap(rf, zero),
            **({f"proper_minus_bt": paired_bootstrap(arms["proper"], arms["bt"])}
               if "proper" in arms and "bt" in arms else {}),
        },
    }
    Path("runs/exp_rlcd_ci.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

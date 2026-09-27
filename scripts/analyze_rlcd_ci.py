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

from opt_agent.dataset import iter_jsonl, phase_of_ratio  # noqa: E402
from opt_agent.system1.rlcd import (  # noqa: E402
    RLCDConfig,
    build_item,
    load_laya_for_training,
)
from train_rlcd_laya import fit_rf, rf_per_record  # noqa: E402


def _per_record_counts(model, tok, records, cfg, device) -> dict[str, tuple[int, int, int]]:
    """record_id → (correct_pairs, total_pairs, top1_match)."""
    model.eval()
    out: dict[str, tuple[int, int, int]] = {}
    with torch.no_grad():
        for r in records:
            item = build_item(r, tok, max_len=cfg.max_len, head_max_len=cfg.head_max_len)
            if item is None:
                continue
            logits = _forward_one(model, item, tok, device, cfg)
            index_of = {a: i for i, a in enumerate(item["option_ids"])}
            gold_top = max(r.gold_probs, key=lambda a: r.gold_probs[a])
            model_top = max(index_of, key=lambda a: float(logits[index_of[a]]))
            correct = sum(
                int(logits[i] > logits[j]) for i, j, _ in item["pairs"]
            )
            out[r.id] = (correct, len(item["pairs"]), int(model_top == gold_top))
    return out


def _forward_one(model, item, tok, device, cfg):
    from opt_agent.system1.rlcd import _collate

    batch = _collate([item], tok.pad_token_id)
    tensors = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        logits, _ = model(
            tensors["input_ids"], tensors["attention_mask"],
            tensors["marker_pos"], tensors["marker_mask"], tensors["qtype"],
        )
    return logits[0].float()


def paired_bootstrap(
    per_rec_a: dict[str, tuple[int, int, int]],
    per_rec_b: dict[str, tuple[int, int, int]],
    *,
    repetitions: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for accuracy(A) − accuracy(B), clustered by record."""
    ids = sorted(set(per_rec_a) & set(per_rec_b))
    rng = random.Random(seed)
    n = len(ids)
    diffs = []
    for _ in range(repetitions):
        ca = cb = ta = tb = 0
        for _ in range(n):
            rid = ids[rng.randrange(n)]
            ca += per_rec_a[rid][0]; ta += per_rec_a[rid][2]
            cb += per_rec_b[rid][0]; tb += per_rec_b[rid][2]
        diffs.append(ca / max(1, sum(per_rec_a[i][1] for i in ids)) -
                     cb / max(1, sum(per_rec_b[i][1] for i in ids)))
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
    zero = _per_record_counts(model, tok, records, eval_cfg, device)
    model.to("cpu"); del model

    ft_agent = None
    import laya

    ft_agent = laya.load("runs/rlcd_laya_v1")
    ft_agent.model.to(device)
    ft = _per_record_counts(ft_agent.model, ft_agent.tok, records, eval_cfg, device)
    ft_agent.model.to("cpu")

    rf_model, rf_scales = fit_rf([asdict(r) for r in train_records])
    rf = rf_per_record(rf_model, rf_scales, [asdict(r) for r in records])

    def _acc(d):
        c = sum(v[0] for v in d.values()); t = sum(v[1] for v in d.values())
        return c / t if t else 0.0

    def _top1(d):
        return (sum(v[2] for v in d.values()) / len(d)) if d else 0.0

    result = {
        "n_records": len(ft),
        "pairs": sum(v[1] for v in ft.values()),
        "acc": {"zero": _acc(zero), "ft": _acc(ft), "rf": _acc(rf)},
        "top1": {"zero": _top1(zero), "ft": _top1(ft), "rf": _top1(rf)},
        "ci": {
            "ft_minus_zero": paired_bootstrap(ft, zero),
            "ft_minus_rf": paired_bootstrap(ft, rf),
            "rf_minus_zero": paired_bootstrap(rf, zero),
        },
    }
    Path("runs/exp_rlcd_ci.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

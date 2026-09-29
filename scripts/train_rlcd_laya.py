"""EXP-RLCD — fine-tune Laya on the counterfactual rollout teacher (branch
exp/rlcd-laya-finetune, 2026-09-27; objective reference corrected same day).

**RLCD here means the Jev/Laya-lineage method**: "Reinforcement Learning
for Calibrated Decisions" — RL whose reward is a strictly proper scoring
rule, shipped in the laya SDK as `laya.common.proper_reward` (log score +
0.5·spherical; RPS on ordinal questions). No method paper exists (per
arXiv:2609.24574 and independent tests); this driver uses the SDK's exact
implementation. It is NOT the identically-named contrastive-distillation
paper of Yang et al. (ICLR 2024).

This driver trains both fine-tuning arms and compares four models on
held-out seeds (default: train problem-seeds 0–15, validate 16–19 of the
canonical 20-seed 5-family dataset):

    zero-shot Laya | RLCD proper-score | BT pairwise (baseline arm) | RF

Metrics: pairwise ranking accuracy, top-1 agreement with gold argmax, and
mean log score (distribution recovery). The RF baseline mirrors the
workshop outline's multi-output RandomForestRegressor (family-median
rescaled utilities, `_shared_rq_features`).

Usage:
    .venv/bin/python scripts/train_rlcd_laya.py \
        --dataset runs/phase8_6d_h3_20seed_5family.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _rlcd_common import rf_pairwise_and_top1 as _rf_pairwise_and_top1  # noqa: E402,F401

from opt_agent.dataset import phase_of_ratio, read_jsonl  # noqa: E402
from opt_agent.system1.rlcd import (  # noqa: E402
    RLCDConfig,
    evaluate_ranking,
    load_laya_for_training,
    train_rlcd,
)
from _shared_rq_features import (  # noqa: E402
    extract_features,
    extract_utilities,
    compute_family_scales,
)


def fit_rf(train_dicts: list[dict]):
    """Fit the multi-output RF utility regressor; return (rf, family_scales)."""
    from sklearn.ensemble import RandomForestRegressor

    scales = compute_family_scales(train_dicts)
    X, Y = [], []
    for r in train_dicts:
        utils = extract_utilities(r, scales)
        if utils is None:
            continue
        phase = phase_of_ratio(
            float(r["state"]["resource"]["remaining_budget_ratio"])
        )
        X.append(extract_features(r["state"], phase))
        Y.append([utils[a] for a in sorted(utils)])
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=0, n_jobs=-1
    )
    rf.fit(X, Y)
    return rf, scales


def rf_per_record(rf, scales, val_dicts: list[dict]) -> dict[str, tuple[int, int, int]]:
    """record id → (correct_pairs, total_pairs, top1_match) for the RF."""
    out: dict[str, tuple[int, int, int]] = {}
    for r in val_dicts:
        gold = extract_utilities(r, scales)
        if gold is None:
            continue
        phase = phase_of_ratio(
            float(r["state"]["resource"]["remaining_budget_ratio"])
        )
        pred_vec = rf.predict([extract_features(r["state"], phase)])[0]
        pred = dict(zip(sorted(gold), pred_vec))
        c, t, m = _rf_pairwise_and_top1(pred, gold)
        out[r["id"]] = (c, t, m)
    return out


def train_and_eval_rf(train_dicts: list[dict], val_dicts: list[dict]) -> dict:
    """Multi-output RF utility regressor — the current 'semantic routing'
    System-1 baseline (RandomForest), trained and evaluated on the same
    split and metrics as the RLCD model."""
    rf, scales = fit_rf(train_dicts)
    per_rec = rf_per_record(rf, scales, val_dicts)
    correct = sum(v[0] for v in per_rec.values())
    total = sum(v[1] for v in per_rec.values())
    top1 = sum(v[2] for v in per_rec.values())
    n_rec = len(per_rec)
    return {
        "train_records_used": None,  # filled by caller when convenient
        "val_records": n_rec,
        "pairs": total,
        "pairwise_accuracy": correct / total if total else 0.0,
        "top1_agreement": top1 / n_rec if n_rec else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path,
                        default=Path("runs/phase8_6d_h3_20seed_5family.jsonl"))
    parser.add_argument("--model-dir", default="/home/lcy/.cache/opt_agent_v3/laya")
    parser.add_argument("--output-dir", default="runs/rlcd_laya_proper_v1",
                        help="Checkpoint dir for the RLCD proper-score arm")
    parser.add_argument("--bt-output-dir", default="runs/rlcd_laya_bt_v1",
                        help="Checkpoint dir for the BT pairwise baseline arm")
    parser.add_argument("--skip-bt", action="store_true",
                        help="Skip the BT baseline arm (proper-score only)")
    parser.add_argument("--train-seed-max", type=int, default=16,
                        help="Train on problem seeds [0, N)")
    parser.add_argument("--val-seed-min", type=int, default=16,
                        help="Validate on problem seeds [N, ∞)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-rf", action="store_true")
    parser.add_argument("--json-out", type=Path, default=Path("runs/exp_rlcd_report.json"))
    parser.add_argument("--report-out", type=Path, default=Path("runs/EXP_RLCD_REPORT.md"))
    parser.add_argument("--version-policy", choices=["allow", "warn", "strict"],
                        default="warn",
                        help="Dataset version pooling policy at the loading boundary (guide 4.5/6.6)")
    args = parser.parse_args()

    records = read_jsonl(args.dataset, version_policy=args.version_policy)
    train_records = [
        r for r in records if int(r.problem["seed"]) < args.train_seed_max
    ]
    val_records = [
        r for r in records if int(r.problem["seed"]) >= args.val_seed_min
    ]
    print(f"records: {len(records)} total, {len(train_records)} train, "
          f"{len(val_records)} val", flush=True)

    # --- Zero-shot baseline -------------------------------------------------
    t0 = time.time()
    model, tok, cfg, _agent = load_laya_for_training(args.model_dir)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu"
        else "cpu"
    )
    model.to(device)
    eval_cfg = RLCDConfig(model_dir=args.model_dir, device=str(device),
                          batch_size=args.batch_size)
    model.eval()
    zero = evaluate_ranking(model, tok, val_records, eval_cfg, device)
    print(f"[zero-shot] {json.dumps(zero)} ({time.time()-t0:.0f}s)", flush=True)

    # --- Fine-tune arms ------------------------------------------------------
    def _train_arm(objective: str, output_dir: str) -> tuple[dict, dict]:
        config = RLCDConfig(
            model_dir=args.model_dir,
            output_dir=output_dir,
            objective=objective,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            beta=args.beta,
            max_pairs_per_record=args.max_pairs,
            device=str(device),
        )
        t = time.time()
        rep = train_rlcd(train_records, val_records, config)
        rep["wall_clock_s"] = round(time.time() - t, 1)
        import laya

        arm_agent = laya.load(output_dir)
        arm_agent.model.to(device)
        metrics = evaluate_ranking(
            arm_agent.model, arm_agent.tok, val_records, eval_cfg, device
        )
        arm_agent.model.to("cpu")
        return rep, metrics

    proper_report, proper_metrics = _train_arm("proper_score", args.output_dir)
    if args.skip_bt:
        bt_report = bt_metrics = None
    else:
        bt_report, bt_metrics = _train_arm("bt_pairs", args.bt_output_dir)

    # --- RF baseline ---------------------------------------------------------
    rf = None
    if not args.skip_rf:
        t2 = time.time()
        rf_model, rf_scales = fit_rf([asdict(r) for r in train_records])
        per_rec = rf_per_record(rf_model, rf_scales, [asdict(r) for r in val_records])
        correct = sum(v[0] for v in per_rec.values())
        total = sum(v[1] for v in per_rec.values())
        top1 = sum(v[2] for v in per_rec.values())
        rf = {
            "val_records": len(per_rec),
            "pairs": total,
            "pairwise_accuracy": correct / total if total else 0.0,
            "top1_agreement": top1 / len(per_rec) if per_rec else 0.0,
            "wall_clock_s": round(time.time() - t2, 1),
        }
        print(f"[rf] {json.dumps(rf)}", flush=True)

    result = {
        "dataset": str(args.dataset),
        "split": {
            "train_seed_range": f"[0, {args.train_seed_max})",
            "val_seed_range": f"[{args.val_seed_min}, ∞)",
            "train_records": len(train_records),
            "val_records": len(val_records),
        },
        "zero_shot_laya": zero,
        "rlcd_proper_score": proper_metrics,
        "rlcd_proper_score_train": proper_report,
        "bt_pairs_baseline": bt_metrics,
        "bt_pairs_train": bt_report,
        "rf_baseline": rf,
        "method": (
            "RLCD (Jev/Laya lineage: Reinforcement Learning for Calibrated "
            "Decisions) — strictly proper scoring rule reward (the SDK's own "
            "proper_reward: log + 0.5·spherical) of the reported option "
            "distribution vs the counterfactual teacher gold_probs; "
            "bt_pairs arm = Bradley-Terry pairwise baseline (DPO-family). "
            "NOT Yang et al. ICLR 2024 contrastive distillation."
        ),
    }
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True),
                             encoding="utf-8")

    def _ls(x):
        return "—" if x is None or x.get("mean_log_score") is None else (
            f"{x['mean_log_score']:.4f}"
        )

    lines = [
        "# EXP-RLCD — RLCD fine-tuning of Laya on the counterfactual teacher",
        "",
        "Date: 2026-09-27 (objective reference corrected same day) · "
        "branch `exp/rlcd-laya-finetune`",
        f"Dataset: `{args.dataset}` ({len(records)} records; "
        f"{len(train_records)} train / {len(val_records)} val by problem seed)",
        "",
        "**RLCD = the Jev/Laya-lineage method** (\"Reinforcement Learning for",
        "Calibrated Decisions\"): reward = strictly proper scoring rule (SDK's",
        "`proper_reward`: log score + 0.5·spherical). No method paper exists;",
        "the implementation is the one shipped with the checkpoint. NOT the",
        "identically-named Yang et al. ICLR 2024 paper. Theory:",
        "`docs/RLCD_THEORY.md`.",
        "",
        "| Model | Pairwise acc | Top-1 agreement | Mean log score | Pairs | Records |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Laya zero-shot | {zero['pairwise_accuracy']:.3f} | "
        f"{zero['top1_agreement']:.3f} | {_ls(zero)} | {zero['pairs']} | "
        f"{zero['records']} |",
    ]
    if proper_metrics:
        lines.append(
            f"| **Laya + RLCD (proper score)** | "
            f"{proper_metrics['pairwise_accuracy']:.3f} | "
            f"{proper_metrics['top1_agreement']:.3f} | {_ls(proper_metrics)} | "
            f"{proper_metrics['pairs']} | {proper_metrics['records']} |"
        )
    if bt_metrics:
        lines.append(
            f"| Laya + BT pairs (baseline arm) | "
            f"{bt_metrics['pairwise_accuracy']:.3f} | "
            f"{bt_metrics['top1_agreement']:.3f} | {_ls(bt_metrics)} | "
            f"{bt_metrics['pairs']} | {bt_metrics['records']} |"
        )
    if rf:
        lines.append(
            f"| RandomForest (utilities) | {rf['pairwise_accuracy']:.3f} | "
            f"{rf['top1_agreement']:.3f} | — | {rf['pairs']} | "
            f"{rf['val_records']} |"
        )
    lines += [
        "",
        f"RLCD arm: {proper_report['train_pairs']} pairs / "
        f"{proper_report['train_items']} records seen; "
        f"{proper_report['wall_clock_s']}s on {device}.",
        f"Saved checkpoints: `{args.output_dir}` (proper-score arm)"
        + (f", `{args.bt_output_dir}` (BT arm)" if bt_metrics else "")
        + " (loadable via `LayaPolicy(model_id=...)`).",
        "",
        "Caveats: decision-level metrics only (teacher recovery), not",
        "closed-loop optimization performance; same-family split by seed",
        "(cross-family transfer is RQ2 and not claimed here); value capped",
        "by the adaptive-value ceiling D (RLCD_THEORY T4).",
        "",
    ]
    args.report_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.json_out} and {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

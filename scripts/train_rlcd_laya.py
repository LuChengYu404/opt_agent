"""EXP-RLCD — fine-tune Laya on counterfactual rollout pairs (branch
exp/rlcd-laya-finetune, 2026-09-27).

Reference method: Yang et al., "RLCD: Reinforcement Learning from
Contrastive Distillation", ICLR 2024 (arXiv:2307.12950). This driver runs
the CD (contrastive-distillation) stage of RLCD adapted to the optimization
decision task: contrastive pairs come from the counterfactual rollout
teacher U(a|s); Laya's per-option scorer is fine-tuned with a Bradley-Terry
objective; the closed-loop RL stage is designed in docs/RLCD_THEORY.md §5.

Decision-level comparison on held-out seeds (default: train problem-seeds
0–15, validate 16–19 of the canonical 20-seed 5-family dataset):

    zero-shot Laya  vs  RLCD fine-tuned Laya  vs  RandomForest baseline

Metrics: pairwise ranking accuracy against the teacher's utility ordering,
and top-1 agreement with the record's gold argmax. The RF baseline mirrors
the workshop outline's multi-output RandomForestRegressor trained on the
same records' utilities (family-median rescaled, features from
`_shared_rq_features`).

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

from opt_agent.dataset import iter_jsonl, phase_of_ratio  # noqa: E402
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


def _rf_pairwise_and_top1(pred: dict[str, float], gold: dict[str, float]) -> tuple[int, int, int]:
    """(correct, total, top1_match) for one record's predicted utilities."""
    actions = sorted(set(pred) & set(gold))
    correct = 0
    total = 0
    for i, a in enumerate(actions):
        for b in actions[i + 1:]:
            g = gold[a] - gold[b]
            if abs(g) <= 1e-12:
                continue
            total += 1
            p = pred[a] - pred[b]
            correct += int((p > 0) == (g > 0))
    top = int(
        max(actions, key=lambda a: pred[a]) == max(actions, key=lambda a: gold[a])
    )
    return correct, total, top


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
    parser.add_argument("--output-dir", default="runs/rlcd_laya_v1")
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
    args = parser.parse_args()

    records = list(iter_jsonl(args.dataset))
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

    # --- RLCD contrastive-distillation stage --------------------------------
    config = RLCDConfig(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        beta=args.beta,
        max_pairs_per_record=args.max_pairs,
        device=str(device),
    )
    t1 = time.time()
    train_report = train_rlcd(train_records, val_records, config)
    train_wall = time.time() - t1

    # --- Fine-tuned eval through the SAVED checkpoint (proves loadability) --
    import laya

    ft_agent = laya.load(args.output_dir)
    ft_agent.model.to(device)
    ft = evaluate_ranking(ft_agent.model, ft_agent.tok, val_records, eval_cfg, device)
    ft_agent.model.to("cpu")

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
        "rlcd_finetuned": ft,
        "rlcd_train": train_report,
        "rf_baseline": rf,
        "train_wall_clock_s": round(train_wall, 1),
        "method": (
            "RLCD CD stage (Yang et al., ICLR 2024, arXiv:2307.12950) adapted: "
            "pairs from counterfactual rollout utilities; Bradley-Terry loss on "
            "Laya per-option scorer logits"
        ),
    }
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True),
                             encoding="utf-8")

    lines = [
        "# EXP-RLCD — RLCD fine-tuning of Laya on counterfactual pairs",
        "",
        f"Date: 2026-09-27 · branch `exp/rlcd-laya-finetune`",
        f"Dataset: `{args.dataset}` ({len(records)} records; "
        f"{len(train_records)} train / {len(val_records)} val by problem seed)",
        "",
        "Method: contrastive pairs (a⁺ ≻ a⁻ ⟺ U(a⁺|s) > U(a⁻|s)) from the",
        "counterfactual rollout teacher; Bradley-Terry loss −log σ(β(z⁺−z⁻))",
        "on Laya's per-option scorer. Theory: `docs/RLCD_THEORY.md`.",
        "",
        "| Model | Pairwise acc | Top-1 agreement | Pairs | Records |",
        "|---|---:|---:|---:|---:|",
        f"| Laya zero-shot | {zero['pairwise_accuracy']:.3f} | "
        f"{zero['top1_agreement']:.3f} | {zero['pairs']} | {zero['records']} |",
        f"| Laya + RLCD (CD) | {ft['pairwise_accuracy']:.3f} | "
        f"{ft['top1_agreement']:.3f} | {ft['pairs']} | {ft['records']} |",
    ]
    if rf:
        lines.append(
            f"| RandomForest (utilities) | {rf['pairwise_accuracy']:.3f} | "
            f"{rf['top1_agreement']:.3f} | {rf['pairs']} | {rf['val_records']} |"
        )
    lines += [
        "",
        f"Training: {train_report['train_pairs']} pairs over "
        f"{train_report['train_items']} usable records; "
        f"{train_wall:.0f}s on {device}.",
        "Saved checkpoint: `" + args.output_dir + "` "
        "(loadable via `LayaPolicy(model_id=...)`).",
        "",
        "Caveats: decision-level metrics only (teacher-ranking recovery),",
        "not closed-loop optimization performance; same-family split by seed",
        "(cross-family transfer is RQ2 and not claimed here).",
        "",
    ]
    args.report_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.json_out} and {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

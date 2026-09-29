"""EXP-RLCD calibration ablation — proper-score training vs post-hoc
temperature scaling of the BT arm (2026-09-27).

Closes the sharpest reviewer risk for the calibration claim: Laya's
lineage ships post-hoc temperature scaling, so "BT + fitted temperature"
is the obvious cheap alternative to proper-score training. If it recovers
the proper arm's distribution metric, the R1 story weakens to "any
objective + calibration step works"; if it does not, training-time
properness has content beyond post-hoc rescaling.

Protocol: honest split — temperature fitted on calib records (problem
seeds 16–17 of the h3 dataset), ALL metrics reported on the held-out
records (seeds 18–19). Metrics: mean log score vs the soft teacher
(the training-relevant calibration), NLL of the teacher's argmax
(decision calibration), ECE against the teacher's argmax (15-bin),
plus the proper arm with its own fitted temperature as the control.

Usage:
    .venv/bin/python scripts/exp_rlcd_calibration_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.dataset import read_jsonl  # noqa: E402
from opt_agent.system1.rlcd import (  # noqa: E402
    RLCDConfig,
    _collate,
    build_item,
    load_laya_for_training,
)

FLOOR = -9.21  # same log floor as laya.common.proper_reward


def _records_logits(model, tok, records, device):
    """Yield (item, logits) per record with supervision."""
    cfg = RLCDConfig(device=str(device), batch_size=8)
    out = []
    with torch.no_grad():
        for r in records:
            item = build_item(r, tok, max_len=cfg.max_len,
                              head_max_len=cfg.head_max_len)
            if item is None or item.get("target") is None:
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
            out.append((item, logits[0].float().cpu()))
    return out


def _metrics(pairs, temperature: float = 1.0) -> dict:
    """log score vs soft teacher; NLL + ECE vs teacher argmax."""
    logs, nlls, confs, corrects = [], [], [], []
    for item, logits in pairs:
        z = logits / temperature
        logq = torch.log_softmax(z, dim=-1).clamp_min(FLOOR)
        q = torch.exp(logq)
        q = q / q.sum()
        t = torch.tensor(item["target"])
        logs.append(float((t * torch.log(q.clamp_min(1e-12)).clamp_min(FLOOR)).sum()))
        top = int(t.argmax())
        nlls.append(-float(torch.log(q[top].clamp_min(1e-12))))
        conf = float(q.max())
        confs.append(conf)
        corrects.append(int(q.argmax() == top))
    # 15-bin ECE vs teacher-argmax correctness + discrimination (routing-
    # relevant): confidence gap between correct/incorrect argmaxes and the
    # area under the selective risk curve (AURC; lower is better).
    bins = 15
    ece = 0.0
    n = len(confs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i in range(n) if lo <= confs[i] < hi or
               (b == bins - 1 and confs[i] == 1.0)]
        if not idx:
            continue
        acc = sum(corrects[i] for i in idx) / len(idx)
        conf = sum(confs[i] for i in idx) / len(idx)
        ece += len(idx) / n * abs(acc - conf)
    conf_ok = [c for c, k in zip(confs, corrects) if k]
    conf_bad = [c for c, k in zip(confs, corrects) if not k]
    order = sorted(range(n), key=lambda i: -confs[i])
    risks, wrong = [], 0
    for rank, i in enumerate(order, 1):
        wrong += 1 - corrects[i]
        risks.append(wrong / rank)
    aurc = sum(risks) / n if risks else 0.0
    return {
        "mean_log_score": sum(logs) / len(logs),
        "mean_nll_argmax": sum(nlls) / len(nlls),
        "ece15_argmax": ece,
        "mean_max_prob": sum(confs) / n,
        "argmax_accuracy": sum(corrects) / n,
        "conf_when_correct": (sum(conf_ok) / len(conf_ok)) if conf_ok else None,
        "conf_when_incorrect": (sum(conf_bad) / len(conf_bad)) if conf_bad else None,
        "conf_gap": ((sum(conf_ok) / len(conf_ok) - sum(conf_bad) / len(conf_bad))
                     if conf_ok and conf_bad else None),
        "aurc": aurc,
    }


def _fit_temperature(pairs) -> float:
    """1-D ternary-search minimizer of NLL vs the soft teacher."""
    def nll(T: float) -> float:
        vals = []
        for item, logits in pairs:
            logq = torch.log_softmax(logits / T, dim=-1).clamp_min(FLOOR)
            t = torch.tensor(item["target"])
            vals.append(-float((t * logq).sum()))
        return sum(vals) / len(vals)

    lo, hi = 0.05, 20.0
    for _ in range(60):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = [r for r in iter_jsonl(Path("runs/phase8_6d_h3_20seed_5family.jsonl"))
               if int(r.problem["seed"]) >= 16]
    calib = [r for r in records if int(r.problem["seed"]) < 18]
    held = [r for r in records if int(r.problem["seed"]) >= 18]
    print(f"calib={len(calib)} held={len(held)}", flush=True)

    arms = {}
    for name, path in (("bt", "runs/rlcd_laya_bt_v1"),
                       ("proper", "runs/rlcd_laya_proper_v1")):
        model, tok, _cfg, _ = load_laya_for_training(path)
        model.to(device).eval()
        arms[name] = _records_logits(model, tok, calib + held, device)
        model.to("cpu")
        del model
        torch.cuda.empty_cache()
    # _records_logits iterates calib + held in order but skips unusable
    # records; recover the split point via record ids.
    calib_ids = {r.id for r in calib}
    calib_n = sum(1 for item, _ in arms["bt"] if item["record_id"] in calib_ids)

    result = {"n_calib": calib_n, "n_held": len(arms["bt"]) - calib_n,
              "arms": {}}
    for name in ("bt", "proper"):
        cal_pairs = arms[name][:calib_n]
        held_pairs = arms[name][calib_n:]
        T = _fit_temperature(cal_pairs)
        result["arms"][name] = {
            "fitted_temperature_on_calib": T,
            "raw": _metrics(held_pairs),
            "tempered": _metrics(held_pairs, temperature=T),
        }
        print(f"[{name}] T*={T:.3f} "
              f"log_score {result['arms'][name]['raw']['mean_log_score']:.4f} "
              f"-> {result['arms'][name]['tempered']['mean_log_score']:.4f} "
              f"| ECE {result['arms'][name]['raw']['ece15_argmax']:.4f} "
              f"-> {result['arms'][name]['tempered']['ece15_argmax']:.4f}",
              flush=True)

    Path("runs/exp_rlcd_calibration_ablation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("wrote runs/exp_rlcd_calibration_ablation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

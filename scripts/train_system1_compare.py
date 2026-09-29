"""Train the real typed-decision System-1 with a seed-disjoint validation set.

Targets are deliberately named: ``rule_full_budget`` is a stable imitation
baseline; ``no_fit`` is the exploratory Phase 8.6D rollout utility ablation.
Neither target is treated as ground-truth optimal action supervision.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.ablate_phase8_6d_utility import _rank  # noqa: E402
from opt_agent.actions.factory import ALL_CANDIDATE_SPECS  # noqa: E402
from opt_agent.dataset.decision_record import read_jsonl  # noqa: E402
from opt_agent.rollout.utility import UtilityWeights  # noqa: E402
from opt_agent.system1.typed_decision import JevStyleSystem1, TypedDecisionConfig  # noqa: E402


ACTIONS = [spec[0] for spec in ALL_CANDIDATE_SPECS]


def target_action(record, target: str) -> str:
    if target == "no_fit":
        return _rank(record, UtilityWeights(w_fit=0))
    if target == "rule_full_budget":
        state = record.state
        if state["resource"]["remaining_budget"] <= 0:
            return "stop"
        if state["optimization"]["stagnation_steps"] >= 3 and state["resource"]["remaining_budget"] >= 2:
            return "global_exploration"
        return "surrogate_acquisition"
    raise ValueError(target)


def evaluate(model, rows, device):
    model.eval()
    losses, correct = [], 0
    with torch.no_grad():
        for record, label in rows:
            logits, _ = model(record.state, ACTIONS)
            target = torch.tensor([ACTIONS.index(label)], device=device)
            losses.append(float(F.cross_entropy(logits.unsqueeze(0), target)))
            correct += ACTIONS[int(logits.argmax())] == label
    return {"cross_entropy": sum(losses) / len(losses), "top1": correct / len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/phase8_6d_hfull_10seed.jsonl"))
    parser.add_argument("--target", choices=("no_fit", "rule_full_budget"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")
    random.seed(20260926)
    torch.manual_seed(20260926)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = read_jsonl(args.input)
    train = [(r, target_action(r, args.target)) for r in records if int(r.problem["seed"]) < 8]
    valid = [(r, target_action(r, args.target)) for r in records if int(r.problem["seed"]) in (8, 9)]
    if not train or not valid:
        raise ValueError("need training seeds 0-7 and validation seeds 8-9")
    print(json.dumps({
        "target": args.target, "device": str(device), "train": len(train), "valid": len(valid),
        "train_labels": dict(Counter(label for _, label in train)),
        "valid_labels": dict(Counter(label for _, label in valid)),
    }, sort_keys=True), flush=True)
    config = TypedDecisionConfig(learning_rate=args.lr, batch_size=8, num_epochs=args.epochs)
    model = JevStyleSystem1(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        random.Random(20260926 + epoch).shuffle(train)
        model.train()
        batch_losses = []
        for start in range(0, len(train), config.batch_size):
            batch = train[start:start + config.batch_size]
            logits = torch.stack([model(record.state, ACTIONS)[0] for record, _ in batch])
            targets = torch.tensor([ACTIONS.index(label) for _, label in batch], device=device)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        metrics = evaluate(model, valid, device)
        metrics.update(epoch=epoch, train_cross_entropy=sum(batch_losses) / len(batch_losses))
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        if metrics["cross_entropy"] < best_loss - 1e-4:
            best_loss, best_epoch, stale = metrics["cross_entropy"], epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(), "config": asdict(config),
                "target": args.target, "source": str(args.input), "train_seeds": list(range(8)),
                "validation_seeds": [8, 9], "best_epoch": best_epoch,
                "validation": {k: metrics[k] for k in ("cross_entropy", "top1")},
                "n_train": len(train), "n_validation": len(valid), "history": history,
            }, args.output)
        else:
            stale += 1
            if stale >= args.patience:
                break
    print(f"saved={args.output} best_epoch={best_epoch} validation_ce={best_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()

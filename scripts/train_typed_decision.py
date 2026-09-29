"""Phase 9.3: fine-tune a Jev-style typed-decision model on rollout data.

Per MASTER_PLAN §26 (NanoJev-style fine-tuning) and codex gpt-6-sol
review 2026-09-23 ("the gold_probs label is a rollout soft target, not
ground truth; use proper-scoring-rule loss"):

  - Train on `runs/phase8_6b_h3_10seed.jsonl` (or similar Phase 8.6B
    corpus).
  - Loss: KL(gold_probs || pred_probs) per record — a proper
    scoring rule on the soft target distribution. Equivalent to
    cross-entropy when gold_probs is treated as a teacher distribution.
  - Architecture: `JevStyleSystem1` (ModernBERT-base encoder + set
    attention over candidate embeddings).
  - Optimizer: AdamW, lr=1e-5 (BERT fine-tuning default).
  - Epochs: 5 (small dataset, RTX 3090, ~minutes per epoch).
  - Save checkpoint to `runs/typed_decision_v1.pt`.

Usage:
  uv run python scripts/train_typed_decision.py \
      --input  runs/phase8_6b_h3_10seed.jsonl \
      --output runs/typed_decision_v1.pt \
      --epochs 5

Then plug the trained model into the smoke harness via a thin adapter
(see scripts/run_hybrid_smoke.py — `trained-decision` policy kind).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.actions.factory import ALL_CANDIDATE_SPECS  # noqa: E402
from opt_agent.system1.typed_decision import (  # noqa: E402
    JevStyleSystem1,
    TypedDecisionConfig,
    gold_probs_to_target_tensor,
)


def _iter_records(path: Path) -> Iterator[dict]:
    """Checked loader boundary: training reads go through read_jsonl with
    an explicit version policy (codex final-audit-2). Malformed lines now
    raise instead of being silently skipped."""
    from opt_agent.dataset.decision_record import read_jsonl

    for record in read_jsonl(path, version_policy="warn"):
        yield dict(record.__dict__)


def _prepare_batch(
    records: list[dict], device: torch.device,
) -> tuple[list[dict], torch.Tensor]:
    """Stack per-record target tensors for batched KL loss."""
    states = [r["state"] for r in records]
    candidates_list = []
    targets_list = []
    for r in records:
        # Use the factory's full vocabulary for evaluation alignment;
        # mask absent candidates so they don't pull probability mass.
        cands = [c[0] for c in ALL_CANDIDATE_SPECS]
        candidates_list.append(cands)
        targets_list.append(
            gold_probs_to_target_tensor(
                r["gold_probs"], cands, device=device,
            )
        )
    return states, torch.stack(targets_list)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("runs/phase8_6b_h3_10seed.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/typed_decision_v1.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the dataset size (for quick smoke runs).")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1

    config = TypedDecisionConfig(
        learning_rate=args.lr, batch_size=args.batch_size,
        num_epochs=args.epochs,
    )
    print(f"Loading model {config.encoder_name} ...")
    model = JevStyleSystem1(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Load records.
    records = list(_iter_records(args.input))
    if args.limit is not None:
        records = records[:args.limit]
    print(f"Loaded {len(records)} records from {args.input}")

    if not records:
        print("ERROR: no training records available", file=sys.stderr)
        return 1

    history: list[dict] = []

    for epoch in range(args.epochs):
        model.train()
        # Shuffle each epoch.
        import random
        random.Random(epoch).shuffle(records)
        total_loss = 0.0
        n_batches = 0
        t0 = time.monotonic()
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            states, targets = _prepare_batch(batch, args.device)
            # Per-record forward (variable candidate subsets not needed
            # here — we use the full 8-candidate vocabulary for every
            # record, which simplifies batching at the cost of slight
            # distribution shift between train and inference).
            cands = [c[0] for c in ALL_CANDIDATE_SPECS]
            logits_list = []
            for s in states:
                logits, _ = model(s, cands)
                logits_list.append(logits)
            logits_batch = torch.stack(logits_list, dim=0)
            log_probs = F.log_softmax(logits_batch, dim=-1)
            loss = F.kl_div(log_probs, targets, reduction="batchmean")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.monotonic() - t0
        print(f"epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.4f}  ({elapsed:.1f}s)")
        history.append({"epoch": epoch + 1, "loss": avg_loss, "seconds": elapsed})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "history": history,
        "n_records": len(records),
    }, args.output)
    print(f"\nSaved checkpoint to {args.output}")
    print(f"Final loss: {history[-1]['loss']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
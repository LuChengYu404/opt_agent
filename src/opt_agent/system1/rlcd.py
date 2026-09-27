"""RLCD — fine-tuning Laya on the counterfactual rollout teacher.

Branch ``exp/rlcd-laya-finetune`` (2026-09-27; objective corrected same day
after verifying the term's origin).

**Name disambiguation (verified 2026-09-27).** In the Jev/Laya line of
research (typed, non-autoregressive System-1 decision models — the
"TypeSafe-style" family), **RLCD = "Reinforcement Learning for Calibrated
Decisions"**: reinforcement learning whose reward is a *strictly proper
scoring rule*, so the optimum of the expected reward is the true outcome
distribution — calibration by construction. Laya ships this objective in
its SDK as ``laya.common.proper_reward`` (log score + 0.5·spherical − RPS
on ordinal questions) with ``td_lambda_targets`` for multi-turn credit
assignment; no method paper has been published for it (noted by
arXiv:2609.24574 and independent third-party tests). This is UNRELATED to
the identically abbreviated "RLCD: Reinforcement Learning from Contrastive
Distillation" of Yang et al., ICLR 2024 (arXiv:2307.12950), which trains
preference models on prompt-induced contrastive pairs.

This module implements the Laya meaning:

- **`objective="proper_score"` (RLCD proper)** — maximize Laya's own
  strictly proper scoring rule (the exact ``proper_reward`` from the SDK)
  of the reported option distribution against the counterfactual teacher
  distribution ``gold_probs``. When the target distribution is known, the
  RL objective reduces to its differentiable form (reward = proper score;
  no sampling needed); the RL form — scoring the *realized* outcome, with
  TD(λ) credit assignment — is the closed-loop stage (RLCD_THEORY R4).
- **`objective="bt_pairs"` (baseline arm)** — Bradley-Terry pairwise loss
  on utility-ordered action pairs (classic paired-comparison / DPO-family
  distillation; identifies the teacher *ranking* only). Kept as the
  comparison arm because it is the estimator that remains unbiased under
  action-subsampled rollout quotas (RLCD_THEORY T3).

Public API:
    ContrastivePair     — (record_id, pos_action, neg_action, margin)
    mine_pairs          — extract pairs from a DecisionRecord
    bradley_terry_loss  — −log σ(β(z⁺ − z⁻)), batched over record pairs
    proper_score_loss   — −proper_reward(softmax(z), gold_probs) (RLCD)
    build_item          — record → Laya model input item + pairs + target
    RLCDConfig          — training hyperparameters (incl. objective)
    load_laya_for_training — Agent → (trainable model, tokenizer, cfg)
    train_rlcd          — fine-tune with early stopping on val ranking
    save_laya_checkpoint— write a directory loadable by laya.Agent / LayaPolicy
    evaluate_ranking    — pairwise accuracy, top-1 agreement, mean log score
"""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from opt_agent.dataset.decision_record import DecisionRecord
from opt_agent.system1._prompt import build_choice_request


# ---------------------------------------------------------------------------
# Pair mining
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContrastivePair:
    """One contrastive supervision event: `pos` beat `neg` at `record`'s state."""

    record_id: str
    pos: str
    neg: str
    margin: float  # U(pos) − U(neg) under the rollout teacher


def mine_pairs(
    record: DecisionRecord,
    *,
    margin: float = 0.0,
    max_pairs: int | None = None,
    rng: random.Random | None = None,
) -> list[ContrastivePair]:
    """All (or sampled) strict utility-order pairs among valid rolled branches.

    Works identically on full-vocabulary and partial (quota-subsampled)
    records — pairs are mined from whatever `rollout_results` contains.
    Invalid branches and non-finite utilities are skipped. Ties within
    1e-12 produce no pair (a tie carries no ordering information).
    """
    valid: list[tuple[str, float]] = []
    for action_id, res in record.rollout_results.items():
        if res.get("status") != "valid":
            continue
        u = float(res.get("utility", float("nan")))
        if math.isfinite(u):
            valid.append((action_id, u))
    valid.sort(key=lambda kv: (-kv[1], kv[0]))
    pairs: list[ContrastivePair] = []
    for i, (a_hi, u_hi) in enumerate(valid):
        for a_lo, u_lo in valid[i + 1:]:
            m = u_hi - u_lo
            if m <= 1e-12 or m < margin:
                continue
            pairs.append(ContrastivePair(record.id, a_hi, a_lo, m))
    if max_pairs is not None and len(pairs) > max_pairs:
        rng = rng or random.Random(0)
        pairs = rng.sample(pairs, max_pairs)
        pairs.sort(key=lambda p: -p.margin)
    return pairs


# ---------------------------------------------------------------------------
# Sequence / loss plumbing (Laya-side)
# ---------------------------------------------------------------------------


def _record_choice_parts(record: DecisionRecord) -> tuple[dict, dict]:
    """(state_arg, internal question dict) using the deployment prompt shape.

    Goes through the same `build_choice_request` helper the deployed
    LayaPolicy uses (structured state, legacy criteria — the prompt shape
    behind `runs/phase9_5_laya_zero_shot_5x5.json`), so the fine-tuned model
    is trained on its deployment input distribution. Option order follows
    `record.candidate_actions` (criteria dict insertion order).
    """
    from opt_agent.actions import ActionType, OptimizationAction

    choices = {
        spec["action_id"]: OptimizationAction(
            ActionType(spec["type"]), dict(spec["params"])
        )
        for spec in record.candidate_actions
    }
    state_arg, question = build_choice_request(
        record.state,  # type: ignore[arg-type]
        choices,
        state_format="structured",
        criteria_format="legacy",
    )
    inner = question["next_action"]
    return state_arg, {"t": inner["type"], "ins": inner["instructions"],
                       "crit": inner["criteria"]}


def build_item(
    record: DecisionRecord,
    tokenizer,
    *,
    max_len: int = 512,
    head_max_len: int = 192,
) -> dict | None:
    """Build one Laya training item for a record: ids/markers/qtype + target.

    The item carries both supervision forms:
      - ``pairs``: option-index pairs for the BT baseline arm (utility order);
      - ``target``: the teacher distribution ``gold_probs`` aligned to
        ``option_ids`` order (renormalized over the offered options), for the
        RLCD proper-score objective.

    Returns None when options truncate past `head_max_len` (indices could
    misalign) or when the record yields no supervision (no pair AND no
    nonzero-mass target).
    """
    from laya.common import QTYPES, build_sequence, render_options

    state_arg, q = _record_choice_parts(record)
    ids, markers = build_sequence(tokenizer, state_arg, q, max_len, head_max_len)
    option_ids = [opt.split(":")[0] for opt in render_options(q)]
    if len(markers) != len(option_ids):
        return None
    index_of = {a: i for i, a in enumerate(option_ids)}
    pairs = [
        (index_of[p.pos], index_of[p.neg], p.margin)
        for p in mine_pairs(record)
        if p.pos in index_of and p.neg in index_of
    ]
    offered = [a for a in option_ids if a in record.gold_probs]
    mass = sum(record.gold_probs[a] for a in offered)
    target = (
        [
            (record.gold_probs[a] / mass) if a in record.gold_probs else 0.0
            for a in option_ids
        ]
        if offered and mass > 0 else None
    )
    if not pairs and target is None:
        return None
    return {
        "ids": ids,
        "markers": markers,
        "qtype": QTYPES[q["t"]],
        "pairs": pairs,
        "record_id": record.id,
        "option_ids": option_ids,
        "target": target,
    }


def bradley_terry_loss(
    logits: torch.Tensor,
    items: Sequence[Mapping],
    *,
    beta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Bradley-Terry (RLCD preference-model) loss over a batch of items.

    For each item's pair (i⁺, i⁻):  ℓ = −log σ(β·(z[i⁺] − z[i⁻]))
    (equivalently softplus(−β·Δz)). The population minimizer is any scoring
    whose differences recover the teacher's pairwise preference
    probabilities — i.e. the utility ordering (RLCD_THEORY T1/T2).
    """
    losses: list[torch.Tensor] = []
    correct = 0
    total = 0
    for row, item in zip(logits, items):
        for i_pos, i_neg, _m in item["pairs"]:
            diff = beta * (row[int(i_pos)] - row[int(i_neg)])
            losses.append(F.softplus(-diff))
            correct += int(diff.item() > 0)
            total += 1
    if not losses:
        return logits.sum() * 0.0, {"pairs": 0, "pair_acc": 0.0}
    return torch.stack(losses).mean(), {
        "pairs": total, "pair_acc": correct / total,
    }


def proper_score_loss(
    logits: torch.Tensor,
    items: Sequence[Mapping],
    *,
    qtype_ids: torch.Tensor | None = None,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """RLCD objective: −(strictly proper scoring rule reward).

    Uses the exact ``laya.common.proper_reward`` implementation shipped with
    the checkpoint's SDK (log score + ``w_sph``·spherical score; the RPS
    term is masked to ordinal ``score``-type questions and therefore inert
    for our ``choice`` decisions). The reported distribution is
    ``softmax(logits)``; the target is the teacher distribution carried by
    each item (``item["target"]``, gold_probs aligned to option order).

    Strict properness: the expected reward is uniquely maximized when the
    reported distribution equals the target — so the population optimum
    recovers the teacher *distribution*, not just its ranking
    (RLCD_THEORY R1). Loss is the negated reward averaged over the batch.
    """
    from laya.common import proper_reward

    rewards: list[torch.Tensor] = []
    nlls: list[float] = []
    for row, item in zip(logits, items):
        if item.get("target") is None:
            continue
        q = torch.softmax(row, dim=-1).unsqueeze(0)          # (1, K)
        t = torch.tensor(item["target"], dtype=q.dtype,
                         device=q.device).unsqueeze(0)        # (1, K)
        qt = torch.zeros((), device=q.device)                 # choice qtype id = 0
        mask = torch.ones_like(q, dtype=q.dtype)
        r = proper_reward(q, t, qt.unsqueeze(0), mask, w_sph=w_sph, w_rps=w_rps)
        rewards.append(r.mean())
        with torch.no_grad():
            nlls.append(float(-torch.log(
                q.detach().clamp_min(1e-12)[0, torch.argmax(t, dim=-1).item()]
            )))
    if not rewards:
        return logits.sum() * 0.0, {"items": 0, "mean_nll": 0.0}
    loss = -torch.stack(rewards).mean()
    return loss, {"items": len(rewards), "mean_nll": sum(nlls) / len(nlls)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RLCDConfig:
    model_dir: str = "/home/lcy/.cache/opt_agent_v3/laya"
    output_dir: str = "runs/rlcd_laya_v1"
    objective: str = "proper_score"  # "proper_score" (RLCD) | "bt_pairs" (baseline)
    learning_rate: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 3
    beta: float = 1.0          # BT inverse-temperature on logit differences
    margin: float = 0.0        # drop pairs with utility gap below this
    w_sph: float = 0.5         # spherical-score weight in proper_score (SDK default)
    max_pairs_per_record: int | None = None
    weight_decay: float = 0.01
    warmup_frac: float = 0.1   # fraction of total steps for linear warmup
    seed: int = 0
    device: str = "cuda"
    amp_dtype: str = "bfloat16"
    max_len: int = 512
    head_max_len: int = 192
    min_lr_frac: float = 0.1   # cosine floor as a fraction of peak lr
    grad_clip: float = 1.0
    save: bool = True          # False for sweep runs (skip 1.6 GB writes)

    def __post_init__(self) -> None:
        if self.objective not in ("proper_score", "bt_pairs"):
            raise ValueError(
                f"objective must be 'proper_score' or 'bt_pairs'; "
                f"got {self.objective!r}"
            )


def load_laya_for_training(model_dir: str):
    """Instantiate laya.Agent on a checkpoint; return trainable parts.

    Reuses the SDK's own loading path (tokenizer config fixups, strict
    weight verification, device fallback) instead of reimplementing it.
    """
    import laya

    agent = laya.load(model_dir)
    agent.model.train()
    return agent.model, agent.tok, agent.cfg, agent


def _collate(items: list[Mapping], pad_id: int) -> dict:
    from laya.common import collate_items

    return collate_items([list(items)], pad_id)


def _forward(model, batch_items: list[Mapping], tok, device, cfg: RLCDConfig):
    batch = _collate(batch_items, tok.pad_token_id)
    tensors = {
        k: v.to(device)
        for k, v in batch.items()
        if torch.is_tensor(v)
    }
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16,
        enabled=device.type == "cuda",
    ):
        logits, _ = model(
            tensors["input_ids"], tensors["attention_mask"],
            tensors["marker_pos"], tensors["marker_mask"], tensors["qtype"],
        )
    return logits.float()


def _iter_minibatches(items: list[dict], batch_size: int, rng: random.Random):
    order = list(range(len(items)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [items[i] for i in order[start:start + batch_size]]


@torch.no_grad()
def evaluate_ranking(model, tok, records, cfg: RLCDConfig, device) -> dict:
    """Pairwise accuracy + top-1 agreement + mean log score vs the teacher.

    Top-1 agreement compares argmax(model logits) with argmax(gold_probs)
    over the same option vocabulary. For partial-rollout records the
    comparison covers the rolled-out subset (gold_probs' support).
    ``mean_log_score`` is the distribution-recovery metric: the mean of
    Σ_a target(a)·log q(a|s) with q = softmax(logits), floored at
    log(1e-4) exactly like ``laya.common.proper_reward`` so the number is
    comparable with the RLCD training reward's log-score component.
    """
    was_training = model.training
    model.eval()
    n_pairs = 0
    n_correct = 0
    n_records = 0
    n_top1 = 0
    log_scores: list[float] = []
    for start in range(0, len(records), cfg.batch_size):
        chunk = records[start:start + cfg.batch_size]
        kept: list[tuple[dict, DecisionRecord]] = []
        for r in chunk:
            item = build_item(
                r, tok, max_len=cfg.max_len, head_max_len=cfg.head_max_len
            )
            if item is not None:
                if cfg.max_pairs_per_record is not None:
                    item = {**item, "pairs": item["pairs"][: cfg.max_pairs_per_record]}
                kept.append((item, r))
        if not kept:
            continue
        logits = _forward(model, [it for it, _ in kept], tok, device, cfg)
        for row, (item, rec) in zip(logits, kept):
            index_of = {a: i for i, a in enumerate(item["option_ids"])}
            gold_top = max(rec.gold_probs, key=lambda a: rec.gold_probs[a])
            model_top = max(
                (a for a in rec.gold_probs if a in index_of),
                key=lambda a: float(row[index_of[a]]),
            )
            n_records += 1
            n_top1 += int(model_top == gold_top)
            if item.get("target") is not None:
                logq = torch.log_softmax(row, dim=-1).clamp_min(-9.21)
                t = torch.tensor(item["target"], dtype=logq.dtype, device=logq.device)
                log_scores.append(float((t * logq).sum()))
            for i_pos, i_neg, _m in item["pairs"]:
                n_pairs += 1
                n_correct += int(row[int(i_pos)] > row[int(i_neg)])
    if was_training:
        model.train()
    return {
        "pairs": n_pairs,
        "pairwise_accuracy": (n_correct / n_pairs) if n_pairs else 0.0,
        "top1_agreement": (n_top1 / n_records) if n_records else 0.0,
        "mean_log_score": (
            sum(log_scores) / len(log_scores) if log_scores else None
        ),
        "records": n_records,
    }


def train_rlcd(
    train_records: list[DecisionRecord],
    val_records: list[DecisionRecord],
    config: RLCDConfig,
) -> dict:
    """Contrastive-distillation fine-tune of Laya on rollout-utility pairs.

    Early stopping selects the epoch with the best validation pairwise
    accuracy; the best weights are restored and saved to
    `config.output_dir` as a laya-loadable checkpoint. Returns a report
    dict (also written to `<output_dir>/rlcd_train_report.json`).
    """
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    rng = random.Random(config.seed)

    model, tok, cfg, _agent = load_laya_for_training(config.model_dir)
    device = torch.device(
        config.device if torch.cuda.is_available() or config.device == "cpu"
        else "cpu"
    )
    model.to(device)

    def _build(records: list[DecisionRecord]) -> list[dict]:
        out = []
        for r in records:
            item = build_item(
                r, tok, max_len=config.max_len, head_max_len=config.head_max_len
            )
            if item is None:
                continue
            if config.max_pairs_per_record is not None:
                item = {**item, "pairs": item["pairs"][: config.max_pairs_per_record]}
            out.append(item)
        return out

    train_items = _build(train_records)
    n_train_pairs = sum(len(it["pairs"]) for it in train_items)
    history: list[dict] = []

    optim = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = max(
        1, config.num_epochs * math.ceil(len(train_items) / config.batch_size)
    )
    warmup = max(1, int(config.warmup_frac * total_steps))

    def _lr_at(step: int) -> float:
        if step < warmup:
            return config.learning_rate * step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        frac = config.min_lr_frac + (1 - config.min_lr_frac) * 0.5 * (
            1 + math.cos(math.pi * prog)
        )
        return config.learning_rate * frac

    best_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    step = 0
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_pairs = 0
        epoch_acc_sum = 0.0
        epoch_acc_batches = 0
        n_batches = 0
        for batch_items in _iter_minibatches(train_items, config.batch_size, rng):
            for g in optim.param_groups:
                g["lr"] = _lr_at(step)
            optim.zero_grad(set_to_none=True)
            logits = _forward(model, batch_items, tok, device, config)
            if config.objective == "proper_score":
                loss, stats = proper_score_loss(
                    logits, batch_items, w_sph=config.w_sph
                )
            else:
                loss, stats = bradley_terry_loss(
                    logits, batch_items, beta=config.beta
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optim.step()
            step += 1
            epoch_loss += float(loss.item())
            epoch_pairs += stats.get("pairs", stats.get("items", 0))
            if "pair_acc" in stats:
                epoch_acc_sum += stats["pair_acc"]
                epoch_acc_batches += 1
            n_batches += 1
        val = evaluate_ranking(model, tok, val_records, config, device)
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / max(1, n_batches),
            "train_pair_acc": (
                epoch_acc_sum / epoch_acc_batches if epoch_acc_batches else None
            ),
            "val_pairwise_accuracy": val["pairwise_accuracy"],
            "val_top1_agreement": val["top1_agreement"],
            "val_mean_log_score": val["mean_log_score"],
            "val_pairs": val["pairs"],
        })
        print(
            f"[rlcd:{config.objective}] epoch {epoch}: "
            f"loss={history[-1]['train_loss']:.4f} "
            f"val_pair_acc={val['pairwise_accuracy']:.3f} "
            f"val_top1={val['top1_agreement']:.3f} "
            f"val_logscore={val['mean_log_score']:.4f}",
            flush=True,
        )
        if val["pairwise_accuracy"] > best_acc:
            best_acc = val["pairwise_accuracy"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    final = evaluate_ranking(model, tok, val_records, config, device)
    if config.save:
        save_laya_checkpoint(config.model_dir, config.output_dir, best_state, cfg)

    report = {
        "config": {
            "model_dir": config.model_dir,
            "output_dir": config.output_dir,
            "objective": config.objective,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "beta": config.beta,
            "margin": config.margin,
            "max_pairs_per_record": config.max_pairs_per_record,
        },
        "train_records": len(train_records),
        "train_items": len(train_items),
        "train_pairs": n_train_pairs,
        "val_records": len(val_records),
        "best_val_pairwise_accuracy": best_acc,
        "final_val": final,
        "history": history,
    }
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rlcd_train_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def save_laya_checkpoint(
    source_dir: str, output_dir: str, state_dict: Mapping, cfg: Mapping
) -> None:
    """Write a directory loadable by `laya.Agent(path)` / `LayaPolicy(model_id=path)`."""
    from safetensors.torch import save_file

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(source_dir)
    for name in ("tokenizer", "encoder"):
        if (src / name).is_dir() and not (out / name).exists():
            shutil.copytree(src / name, out / name)
    cfg_src = src / "rl_agent_config.json"
    if cfg_src.exists():
        shutil.copy(cfg_src, out / "rl_agent_config.json")
    save_file(
        {k: v.contiguous() for k, v in state_dict.items()},
        str(out / "model.safetensors"),
    )

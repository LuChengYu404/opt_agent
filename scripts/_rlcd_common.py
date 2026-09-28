"""Shared helpers for the RLCD experiment scripts (2026-09-28 cleanup).

Consolidates pieces that were copy-pasted across the RLCD-family scripts
(`train_rlcd_laya.py`, `analyze_rlcd_ci.py`, `exp_rlcd_loo_transfer.py`,
`exp_rlcd_loo_confirmatory.py`, `exp_geartrain_transfer_probe.py`,
`eval_laya_closed_loop.py`, `exp_rlcd_router_panel.py`):

  - ``RuleFullBudget`` / ``AlwaysSurrogatePolicy`` — closed-loop baselines
  - ``bootstrap_ci`` — seed/record-clustered percentile bootstrap
  - ``raw_gold`` — valid-action utility dict from a DecisionRecord
    (decision-level metrics are invariant to per-record monotone
    transforms, so no cross-family scale pooling is needed)
  - ``rf_pairwise_and_top1`` — per-record RF ranking metrics
  - ``per_record_laya`` — per-record Laya ranking metrics (correct pairs,
    total pairs, top-1 match, mean log score with the proper_reward floor)

Only ACTIVE scripts import from here; archived one-shot experiment scripts
(exp_rq*, exp_rq0_5_*, diagnose_*, analyze_rq3_*) intentionally keep their
frozen copies so their historical behavior stays reproducible.
"""

from __future__ import annotations

import random
import statistics
from typing import Mapping

FLOOR = -9.21  # log floor matching laya.common.proper_reward

ACTIONS_T = (
    "global_exploration", "local_exploitation", "uncertainty_sampling",
    "error_guided_sampling", "boundary_sampling", "surrogate_acquisition",
    "retrain_surrogate", "stop",
)


def _policies():
    """Lazy imports to keep this module importable without torch."""
    from opt_agent.actions import candidate_actions_for_state
    from opt_agent.loop.policies import RulePolicy

    class RuleFullBudget(RulePolicy):
        LOW_BUDGET_THRESHOLD = 0

    class AlwaysSurrogatePolicy:
        """Always pick SURROGATE_ACQUISITION — the 'smart BO baseline'."""

        def __init__(self, seed: int = 0) -> None:
            self.seed = seed

        def select_action(self, state, backend=None):
            choices = candidate_actions_for_state(state, seed=self.seed)
            if "surrogate_acquisition" in choices:
                return choices["surrogate_acquisition"]
            return next(iter(choices.values()))

    return RuleFullBudget, AlwaysSurrogatePolicy


def make_rule_full_budget(seed: int = 0):
    RuleFullBudget, _ = _policies()
    return RuleFullBudget(seed=seed)


def make_always_surrogate(seed: int = 0):
    _, AlwaysSurrogatePolicy = _policies()
    return AlwaysSurrogatePolicy(seed=seed)


def bootstrap_ci(values: list[float], repetitions: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile-bootstrap 95% CI for the mean of `values`."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(repetitions)
    )
    return means[int(0.025 * repetitions)], means[int(0.975 * repetitions)]


def raw_gold(record) -> dict[str, float] | None:
    """Valid-action utility dict; None if fewer than 2 usable actions."""
    gold: dict[str, float] = {}
    for a, cell in record.rollout_results.items():
        if cell.get("status") != "valid":
            continue
        u = cell.get("utility")
        if isinstance(u, (int, float)) and u == u:  # NaN check
            gold[a] = float(u)
    return gold if len(gold) >= 2 else None


def rf_pairwise_and_top1(pred: Mapping[str, float],
                         gold: Mapping[str, float]) -> tuple[int, int, int]:
    """(correct, total, top1_match) for one record's predicted utilities."""
    actions = sorted(set(pred) & set(gold))
    correct = total = 0
    for i, a in enumerate(actions):
        for b in actions[i + 1:]:
            g = gold[a] - gold[b]
            if abs(g) <= 1e-12:
                continue
            total += 1
            correct += int((pred[a] - pred[b] > 0) == (g > 0))
    top = int(max(actions, key=lambda a: pred[a])
              == max(actions, key=lambda a: gold[a]))
    return correct, total, top


def per_record_laya(model, tok, records, device_str: str) -> dict[str, tuple]:
    """record_id → (correct, total, top1, log_score) for a Laya model.

    Single-record batches; CPU or CUDA per `device_str`. Uses the same
    prompt/item builder as training (`rlcd.build_item`) and the
    proper_reward log floor for comparability with training rewards.
    """
    import torch

    from opt_agent.system1.rlcd import _collate, build_item

    device = torch.device(device_str)
    was_training = model.training
    model.to(device).eval()
    out: dict[str, tuple] = {}
    with torch.no_grad():
        for r in records:
            item = build_item(r, tok, max_len=512, head_max_len=192)
            gold = raw_gold(r)
            if item is None or gold is None:
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
            row = logits[0].float().cpu()
            index_of = {a: i for i, a in enumerate(item["option_ids"])}
            correct = sum(int(row[i] > row[j]) for i, j, _ in item["pairs"])
            log_score = None
            if item.get("target") is not None:
                logq = torch.log_softmax(row, -1).clamp_min(FLOOR)
                t = torch.tensor(item["target"], dtype=logq.dtype)
                log_score = float((t * logq).sum())
            out[r.id] = (
                correct, len(item["pairs"]),
                int(max(gold, key=lambda a: float(row[index_of[a]]))
                    == max(gold, key=lambda a: gold[a])),
                log_score,
            )
    if was_training:
        model.train()
    model.to("cpu")
    return out


def paired_accuracy_diffs(a: Mapping[str, tuple],
                          b: Mapping[str, tuple]) -> list[float]:
    """Record-clustered per-record accuracy differences (a − b)."""
    ids = sorted(set(a) & set(b))
    out = []
    for i in ids:
        ca, ta = a[i][0], a[i][1]
        cb, tb = b[i][0], b[i][1]
        if ta == 0 or tb == 0:
            continue
        out.append(ca / ta - cb / tb)
    return out

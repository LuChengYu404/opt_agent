"""RLCD module tests (branch exp/rlcd-laya-finetune, 2026-09-27).

Layered to keep wall-clock low:
  - mine_pairs / bradley_terry_loss: pure logic, no laya import.
  - build_item: real cached laya tokenizer (fast, CPU) over synthetic records.
  - gradient flow through a tiny stub DecisionModel (no pretrained weights).

The full fine-tune path (real ModernBERT-large checkpoint) is exercised by
`scripts/train_rlcd_laya.py`, not by unit tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.dataset.decision_record import DecisionRecord
from opt_agent.system1.rlcd import (
    ContrastivePair,
    bradley_terry_loss,
    mine_pairs,
)

LAYA_CACHE = Path("/home/lcy/.cache/opt_agent_v3/laya")


def _record(
    utilities: dict[str, float],
    *,
    partial: bool = False,
    invalid: set[str] | None = None,
) -> DecisionRecord:
    """Synthetic record with the FULL 8-action vocabulary as candidates but
    rollout_results only for `utilities` (mimics partial quota records)."""
    all_ids = (
        "global_exploration", "local_exploitation", "uncertainty_sampling",
        "error_guided_sampling", "boundary_sampling", "surrogate_acquisition",
        "retrain_surrogate", "stop",
    )
    invalid = invalid or set()
    rr = {
        a: {"status": "invalid" if a in invalid else "valid", "utility": u,
            "delta_optimization": 0.0, "n_new_samples": 1, "n_model_fits": 0,
            "delta_surrogate": None, "delta_coverage": 0.0, "error": None}
        for a, u in utilities.items()
    }
    # Descending utilities paired with descending softmax masses, so the
    # highest-utility action carries the highest gold probability (matches
    # to_gold_probs orientation).
    z = torch.tensor(sorted(utilities.values(), reverse=True)).softmax(-1).tolist()
    order = sorted(utilities, key=lambda a: -utilities[a])
    gold = {a: p for a, p in zip(order, z)}
    return DecisionRecord(
        id="synthetic:0:0:0",
        problem={"family": "branin", "dimension": 2, "seed": 0},
        state={
            "sampling": {"n_samples": 6, "dimension": 2, "boundary_coverage": 0.1,
                         "min_pairwise_distance": 0.2},
            "surrogate": {"is_current": True, "validation_rmse": 0.1,
                          "mean_uncertainty": 0.2, "max_uncertainty": 0.4},
            "optimization": {"best_value": -1.0, "recent_improvement": 0.01,
                             "stagnation_steps": 1, "last_value": -0.9},
            "resource": {"total_budget": 30, "remaining_budget": 24,
                         "remaining_budget_ratio": 0.8},
            "history": {"last_action": None, "recent_actions": [],
                        "recent_improvement": 0.01, "failure_count": 0},
        },
        candidate_actions=[
            {"action_id": a, "type": "global_exploration", "params": {"n_samples": 2}}
            for a in all_ids
        ],
        gold_probs=gold,
        rollout_results=rr,
        temperature=1.0,
        metadata={"partial_rollout": {"k": len(utilities),
                                      "action_ids": list(utilities)}} if partial else {},
    )


# ---------------------------------------------------------------------------
# mine_pairs
# ---------------------------------------------------------------------------


def test_mine_pairs_orders_by_utility() -> None:
    rec = _record({"local_exploitation": 0.5, "stop": -0.5, "retrain_surrogate": 0.0})
    pairs = mine_pairs(rec)
    assert [(p.pos, p.neg) for p in pairs] == [
        ("local_exploitation", "retrain_surrogate"),
        ("local_exploitation", "stop"),
        ("retrain_surrogate", "stop"),
    ]
    assert pairs[0].margin == pytest.approx(0.5)


def test_mine_pairs_skips_invalid_and_nonfinite() -> None:
    rec = _record(
        {"local_exploitation": 0.5, "stop": -0.5, "boundary_sampling": float("nan")},
        invalid={"stop"},
    )
    # `stop` invalid and `boundary_sampling` non-finite → only one valid
    # action remains → no pair.
    assert mine_pairs(rec) == []
    # Control: with all three valid, both pairs appear.
    ok = _record({"local_exploitation": 0.5, "stop": -0.5, "boundary_sampling": 0.2})
    assert [(p.pos, p.neg) for p in mine_pairs(ok)] == [
        ("local_exploitation", "boundary_sampling"),
        ("local_exploitation", "stop"),
        ("boundary_sampling", "stop"),
    ]


def test_mine_pairs_margin_filter_and_cap() -> None:
    rec = _record({"local_exploitation": 0.10, "uncertainty_sampling": 0.05,
                   "surrogate_acquisition": 0.00})
    # margins: lu=.05, ls=.10, us=.05 → only the ≥.06 margin keeps local>surrogate
    assert len(mine_pairs(rec, margin=0.06)) == 1
    assert len(mine_pairs(rec, max_pairs=1)) == 1


def test_mine_pairs_ties_produce_no_pair() -> None:
    rec = _record({"local_exploitation": 0.3, "uncertainty_sampling": 0.3})
    assert mine_pairs(rec) == []


def test_mine_pairs_partial_records_are_first_class() -> None:
    """The quota property: k-of-8 records yield the same pairs over their
    rolled-out subset as a full record restricted to that subset."""
    keep = {"local_exploitation", "uncertainty_sampling"}
    full = _record({"local_exploitation": 1.0, "uncertainty_sampling": 0.5,
                    "surrogate_acquisition": 0.0})
    part = _record({"local_exploitation": 1.0, "uncertainty_sampling": 0.5},
                   partial=True)
    full_subset = [p for p in mine_pairs(full) if {p.pos, p.neg} <= keep]
    assert [(p.pos, p.neg) for p in full_subset] == [
        (p.pos, p.neg) for p in mine_pairs(part)
    ]


# ---------------------------------------------------------------------------
# bradley_terry_loss
# ---------------------------------------------------------------------------


def test_bradley_terry_loss_zero_when_perfectly_ordered() -> None:
    logits = torch.tensor([[10.0, 0.0, -10.0]])
    items = [{"pairs": [(0, 1, 0.5), (0, 2, 0.7), (1, 2, 0.2)]}]
    loss, stats = bradley_terry_loss(logits, items)
    assert stats["pairs"] == 3
    assert stats["pair_acc"] == 1.0
    assert float(loss) == pytest.approx(0.0, abs=1e-3)  # softplus(−10) ≈ 4.5e-5


def test_bradley_terry_loss_gradient_moves_toward_teacher_order() -> None:
    z = torch.zeros(3, requires_grad=True)
    items = [{"pairs": [(0, 1, 1.0), (1, 2, 1.0)]}]
    for _ in range(200):
        loss, _ = bradley_terry_loss(z.unsqueeze(0), items, beta=2.0)
        z.grad = None
        loss.backward()
        with torch.no_grad():
            z -= 0.1 * z.grad
    assert z[0] > z[1] > z[2]


def test_bradley_terry_loss_empty_batch() -> None:
    logits = torch.zeros(2, 3)
    loss, stats = bradley_terry_loss(logits, [{"pairs": []}, {"pairs": []}])
    assert stats["pairs"] == 0
    assert float(loss) == 0.0


# ---------------------------------------------------------------------------
# proper_score_loss (the RLCD objective proper)
# ---------------------------------------------------------------------------


def test_proper_score_loss_prefers_matching_distribution() -> None:
    from opt_agent.system1.rlcd import proper_score_loss

    target = [0.7, 0.2, 0.1]
    logits_match = torch.log(torch.tensor([0.7, 0.2, 0.1]))
    logits_off = torch.log(torch.tensor([0.1, 0.2, 0.7]))
    items = [{"target": target, "pairs": [], "option_ids": ["a", "b", "c"]}]
    loss_match, _ = proper_score_loss(logits_match.unsqueeze(0), items)
    loss_off, _ = proper_score_loss(logits_off.unsqueeze(0), items)
    # Strictly proper: the matching distribution must score strictly higher
    # (lower loss) than any other report.
    assert float(loss_match) < float(loss_off)


def test_proper_score_loss_gradient_flow() -> None:
    from opt_agent.system1.rlcd import proper_score_loss

    z = torch.zeros(3, requires_grad=True)
    items = [{"target": [0.6, 0.3, 0.1], "pairs": [], "option_ids": list("abc")}]
    loss, stats = proper_score_loss(z.unsqueeze(0), items)
    loss.backward()
    assert stats["items"] == 1
    assert z.grad is not None and float(z.grad.abs().sum()) > 0


def test_proper_score_loss_empty_targets() -> None:
    from opt_agent.system1.rlcd import proper_score_loss

    logits = torch.zeros(2, 3)
    loss, stats = proper_score_loss(
        logits, [{"target": None, "pairs": []}, {"target": None, "pairs": []}]
    )
    assert stats["items"] == 0
    assert float(loss) == 0.0


def test_rlcd_config_objective_validation() -> None:
    from opt_agent.system1.rlcd import RLCDConfig

    RLCDConfig(objective="proper_score")  # ok
    RLCDConfig(objective="bt_pairs")      # ok
    with pytest.raises(ValueError, match="objective"):
        RLCDConfig(objective="contrastive_distillation")


# ---------------------------------------------------------------------------
# build_item (real tokenizer, synthetic record)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (LAYA_CACHE / "tokenizer").is_dir(),
                    reason="cached laya tokenizer not present")
class TestBuildItem:
    @pytest.fixture(scope="class")
    def tok(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(LAYA_CACHE / "tokenizer"))

    def test_item_aligns_markers_options_and_pairs(self, tok) -> None:
        from opt_agent.system1.rlcd import build_item

        rec = _record({"local_exploitation": 0.5, "stop": -0.5,
                       "retrain_surrogate": 0.0})
        item = build_item(rec, tok)
        assert item is not None
        # Full 8-action vocabulary is offered; 3 have pairs.
        assert len(item["markers"]) == 8 == len(item["option_ids"])
        index_of = {a: i for i, a in enumerate(item["option_ids"])}
        # item["pairs"] already carries option indices.
        assert {(i, j) for i, j, _ in item["pairs"]} == {
            (index_of["local_exploitation"], index_of["retrain_surrogate"]),
            (index_of["local_exploitation"], index_of["stop"]),
            (index_of["retrain_surrogate"], index_of["stop"]),
        }
        assert item["qtype"] == 0  # choice

    def test_item_carries_target_and_skips_only_unsupervised(self, tok) -> None:
        from opt_agent.system1.rlcd import build_item

        rec = _record({"local_exploitation": 0.5, "uncertainty_sampling": 0.5})
        item = build_item(rec, tok)  # tie → no pairs, but target remains
        assert item is not None
        assert item["pairs"] == []
        assert item["target"] is not None
        assert sum(item["target"]) == pytest.approx(1.0)
        # Target mass only on offered actions with gold mass.
        idx = {a: i for i, a in enumerate(item["option_ids"])}
        assert item["target"][idx["stop"]] == 0.0


# ---------------------------------------------------------------------------
# Gradient flow through a stub DecisionModel (no pretrained weights)
# ---------------------------------------------------------------------------


class _StubEncoder(torch.nn.Module):
    """Smallest stand-in satisfying DecisionModel's encoder contract."""

    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(100, d)
        self.config = type("C", (), {"hidden_size": d})()

    def forward(self, input_ids=None, attention_mask=None, **_):
        h = self.emb(input_ids)
        return type("Out", (), {"last_hidden_state": h})()


def _stub_model():
    from laya.common import DecisionModel

    return DecisionModel(_StubEncoder(), head_layers=1, n_act=2)


def test_stub_model_loss_backward_flow() -> None:
    from opt_agent.system1.rlcd import bradley_terry_loss

    torch.manual_seed(0)
    model = _stub_model()
    n, L, K = 2, 12, 4
    batch = {
        "input_ids": torch.randint(0, 100, (n, L)),
        "attention_mask": torch.ones(n, L, dtype=torch.long),
        "marker_pos": torch.tensor([[2, 5, 8, 11], [2, 5, 8, 11]]),
        "marker_mask": torch.ones(n, K, dtype=torch.bool),
        "qtype": torch.zeros(n, dtype=torch.long),
    }
    items = [{"pairs": [(0, 1, 1.0), (2, 3, 0.5)]}, {"pairs": [(3, 0, 0.2)]}]
    logits, _ = model(**batch)
    loss, stats = bradley_terry_loss(logits, items)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    assert stats["pairs"] == 3

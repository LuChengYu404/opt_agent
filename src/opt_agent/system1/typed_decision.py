"""Jev-style typed-decision model.

Per MASTER_PLAN §12 + §26, Phase 9.3 fine-tunes a typed-decision
model on rollout data. The model exposes a `decide(state) →
{action: prob}` interface compatible with `HybridPolicy`.

Architecture (simplified from NanoJev-style set attention):

  1. Encoder (ModernBERT-base by default) embeds a textual
     serialization of the BO state into a single state vector h_s.
  2. A learned candidate embedding table maps each candidate action
     id to a vector h_c_i.
  3. Per-candidate score = dot(h_s, h_c_i) / sqrt(d_model) — softmax
     over candidates gives the choice distribution.
  4. A sigmoid head predicts a confidence scalar in [0, 1] for
     downstream routing.

The model is trained on Phase 8.6B rollout data with KL divergence
against `gold_probs`. Training script lives in
`scripts/train_typed_decision.py`.

Per codex gpt-6-sol review of earlier phases: the action_id vocabulary
comes from `actions.factory.ALL_CANDIDATE_SPECS` (single source of
truth). We dynamically compute candidate embeddings for whichever
subset of the 8-action vocabulary the state-filtered factory returns.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _candidate_pool_from_factory() -> tuple[str, ...]:
    """Derive the candidate-pool vocabulary from ``ALL_CANDIDATE_SPECS`` so
    adding a new ``ActionType`` to the factory is automatically reflected
    in the typed-decision model. Deferred import avoids a circular module
    dependency at import time."""
    from opt_agent.actions.factory import ALL_CANDIDATE_SPECS
    return tuple(spec[0] for spec in ALL_CANDIDATE_SPECS)


@dataclass(frozen=True)
class TypedDecisionConfig:
    """Hyperparameters + identity for the typed-decision model."""

    encoder_name: str = "answerdotai/ModernBERT-base"
    max_state_tokens: int = 128
    max_candidates: int = 8
    hidden_dim: int = 768
    learning_rate: float = 1e-5
    batch_size: int = 16
    num_epochs: int = 5
    candidate_pool: tuple[str, ...] = field(default_factory=_candidate_pool_from_factory)


class JevStyleSystem1(nn.Module):
    """Encoder + set-attention decision head for the choice primitive.

    For a given state, returns:
      - `logits[candidate_id]`: pre-softmax score for each candidate.
      - `confidence`: scalar in [0, 1] (sigmoid head).
    """

    def __init__(self, config: TypedDecisionConfig) -> None:
        super().__init__()
        self.config = config
        from transformers import AutoModel, AutoTokenizer
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)
        # Learned candidate embeddings (one row per candidate id).
        self.candidate_emb = nn.Embedding(
            num_embeddings=len(config.candidate_pool),
            embedding_dim=config.hidden_dim,
        )
        # Confidence head: state vector → scalar in [0, 1].
        self.confidence_head = nn.Linear(config.hidden_dim, 1)

    def _state_text(self, state) -> str:
        """Serialize a BO state into a compact text prompt.

        Accepts either a Mapping (dict-like) or an `OptimizationState`
        dataclass instance (the runtime path). At training time we
        pass a dict (from `record["state"]`); at inference time we
        pass the dataclass.
        """
        if hasattr(state, "resource"):  # OptimizationState dataclass
            return (
                f"BO state: "
                f"remaining_budget={state.resource.remaining_budget}/{state.resource.total_budget}; "
                f"n_samples={state.sampling.n_samples}; "
                f"surrogate_current={state.surrogate.is_current}; "
                f"best_value={state.optimization.best_value}; "
                f"recent_improvement={state.optimization.recent_improvement}; "
                f"stagnation_steps={state.optimization.stagnation_steps}"
            )
        # Mapping (dict) path — used at training time.
        sub = state.get("resource", {})
        sample = state.get("sampling", {})
        opt = state.get("optimization", {})
        surrogate = state.get("surrogate", {})
        return (
            f"BO state: "
            f"remaining_budget={sub.get('remaining_budget', '?')}/{sub.get('total_budget', '?')}; "
            f"n_samples={sample.get('n_samples', '?')}; "
            f"surrogate_current={surrogate.get('is_current', '?')}; "
            f"best_value={opt.get('best_value', '?')}; "
            f"recent_improvement={opt.get('recent_improvement', '?')}; "
            f"stagnation_steps={opt.get('stagnation_steps', '?')}"
        )

    def _candidate_text(self, candidate_id: str) -> str:
        return f"action: {candidate_id}"

    def encode_state(self, state) -> torch.Tensor:
        """Return the [CLS] pooled state vector, shape (hidden_dim,)."""
        text = self._state_text(state)
        enc = self.tokenizer(
            text,
            max_length=self.config.max_state_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        device = next(self.encoder.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.encoder(**enc)
        return out.last_hidden_state[:, 0, :].squeeze(0)

    def forward(
        self,
        state,
        candidates: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        logits : (len(candidates),)
            Pre-softmax scores, one per candidate.
        confidence : ()
            Scalar in [0, 1].
        """
        h_state = self.encode_state(state)  # (hidden_dim,)
        # Map candidate id → index in the candidate pool.
        pool = list(self.config.candidate_pool)
        # If a candidate id is missing from the pool, fall back to
        # embedding index 0 (a learnable "unknown" slot — initialized
        # randomly; the model's training will never see this row in
        # practice because ALL_CANDIDATE_SPECS is the single source).
        indices = torch.tensor(
            [pool.index(c) if c in pool else 0 for c in candidates],
            dtype=torch.long,
            device=h_state.device,
        )
        cand_emb = self.candidate_emb(indices)  # (K, hidden_dim)
        # Scaled dot-product per candidate.
        scale = 1.0 / math.sqrt(self.config.hidden_dim)
        logits = (h_state.unsqueeze(0) * cand_emb).sum(dim=-1) * scale
        confidence = torch.sigmoid(self.confidence_head(h_state))
        return logits, confidence

    def decide_distribution(
        self,
        state: Mapping,
        candidates: Sequence[str],
    ) -> dict[str, float]:
        """Convenience: softmax over candidates → {candidate_id: prob}."""
        with torch.no_grad():
            logits, conf = self.forward(state, candidates)
            probs = F.softmax(logits, dim=-1).cpu().tolist()
        return dict(zip(candidates, probs)), float(conf.item())


def state_to_text(state) -> str:
    """Serialize a BO state to compact text. Accepts either an
    ``OptimizationState`` dataclass or a Mapping (for training-time
    records loaded from JSONL).
    """
    if hasattr(state, "resource"):  # OptimizationState dataclass
        return (
            f"BO state: "
            f"remaining_budget={state.resource.remaining_budget}/{state.resource.total_budget}; "
            f"n_samples={state.sampling.n_samples}; "
            f"surrogate_current={state.surrogate.is_current}; "
            f"best_value={state.optimization.best_value}; "
            f"recent_improvement={state.optimization.recent_improvement}; "
            f"stagnation_steps={state.optimization.stagnation_steps}"
        )
    sub = state.get("resource", {})
    sample = state.get("sampling", {})
    opt = state.get("optimization", {})
    surrogate = state.get("surrogate", {})
    return (
        f"BO state: "
        f"remaining_budget={sub.get('remaining_budget', '?')}/{sub.get('total_budget', '?')}; "
        f"n_samples={sample.get('n_samples', '?')}; "
        f"surrogate_current={surrogate.get('is_current', '?')}; "
        f"best_value={opt.get('best_value', '?')}; "
        f"recent_improvement={opt.get('recent_improvement', '?')}; "
        f"stagnation_steps={opt.get('stagnation_steps', '?')}"
    )


def gold_probs_to_target_tensor(
    gold_probs: Mapping[str, float],
    candidates: Sequence[str],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Reorder gold_probs to match the model's candidate ordering."""
    out = torch.zeros(len(candidates), dtype=torch.float32, device=device)
    for i, c in enumerate(candidates):
        out[i] = float(gold_probs.get(c, 0.0))
    s = out.sum().item()
    if s > 0:
        out = out / s
    return out
"""Adapter that wraps a trained JevStyleSystem1 checkpoint as a System-1
policy compatible with `HybridPolicy`.

Per MASTER_PLAN §12 + §26: Phase 9.3 fine-tunes a typed-decision
model on Phase 8.6B rollout data, then plugs the trained checkpoint
back into the decision loop via `HybridPolicy`. This module is the
bridge.

Usage:
    policy = TypedDecisionPolicy.from_checkpoint("runs/typed_decision_v1.pt")
    decision = policy.decide(state)
    hybrid = HybridPolicy(system1=policy, router=ConfidenceRouter(...),
                          system2=FallbackSystem2())
    DecisionLoop(backend, total_budget=...).run(hybrid)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from opt_agent.actions import OptimizationAction
from opt_agent.actions.factory import candidate_actions_for_state
from opt_agent.state import OptimizationState
from opt_agent.system1.typed_decision import (
    JevStyleSystem1,
    TypedDecisionConfig,
)


@dataclass(frozen=True)
class TypedDecisionOutput:
    """The decide() return value, compatible with HybridPolicy.

    HybridPolicy calls `decide(state)` and reads `.action` (an
    OptimizationAction) and `.probabilities` (a dict[str, float]).
    """

    action: OptimizationAction
    probabilities: dict[str, float]
    confidence: float


class TypedDecisionPolicy:
    """Loads a trained checkpoint and exposes a decide(state) interface.

    The adapter does NOT modify the underlying module — it calls
    `module.decide_distribution(state, candidates)` and constructs an
    OptimizationAction from the argmax candidate.
    """

    def __init__(
        self,
        model: JevStyleSystem1,
        config: TypedDecisionConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 0,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        self.seed = seed

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
        seed: int = 0,
    ) -> "TypedDecisionPolicy":
        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        config = TypedDecisionConfig(**ckpt["config"])
        model = JevStyleSystem1(config)
        model.load_state_dict(ckpt["model_state_dict"])
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(model, config, device=device, seed=seed)

    def _candidate_pool(self, state: OptimizationState) -> list[str]:
        """Return only actions that can execute in the current state."""
        return list(candidate_actions_for_state(state, seed=self.seed))

    def decide(self, state: OptimizationState) -> TypedDecisionOutput:
        available = candidate_actions_for_state(state, seed=self.seed)
        candidates = list(available)
        probs, confidence = self.model.decide_distribution(state, candidates)
        # Reuse the factory action so required parameters and budget caps
        # survive the argmax decision.
        chosen_id = max(probs, key=probs.get)
        action = available[chosen_id]
        return TypedDecisionOutput(
            action=action, probabilities=probs, confidence=confidence,
        )

    def reset(self) -> None:
        # Trained model has no internal state.
        return None

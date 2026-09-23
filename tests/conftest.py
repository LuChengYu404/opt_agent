"""Shared pytest fixtures and stubs for opt_agent tests.

`RecordingBackend` was originally in tests/test_action_executor.py. Codex review
(2025-09-23) suggested moving it here so test_decision_loop.py can reuse it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_agent.backend.bocode_botorch import (  # noqa: E402
    EvaluationBatch,
    UnifiedBackend,
)


class RecordingBackend(UnifiedBackend):
    """Stub backend that records all calls and returns synthetic data.

    Initial history = 2 points at [[0,0],[1,1]] with objectives [-0.5, -1.0].
    Sobol step from seed=0 with d=2 yields candidate [0.5, 0.5].
    """

    def __init__(
        self,
        fitted: bool = False,
        fail_evaluate: bool = False,
        propose_value: torch.Tensor | None = None,
    ) -> None:
        X = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        obj = torch.tensor([[-0.5], [-1.0]], dtype=torch.double)
        self._history = EvaluationBatch(
            X=X,
            objectives=obj,
            constraints=torch.zeros((2, 0), dtype=torch.double),
            feasible=torch.ones((2, 1), dtype=torch.bool),
        )
        self._fitted = fitted
        self._fail_evaluate = fail_evaluate
        self.fit_calls = 0
        self.propose_calls = 0
        self.evaluate_calls = 0
        self.last_evaluated: torch.Tensor | None = None
        self._propose_value = propose_value or torch.tensor(
            [[0.25, 0.25]], dtype=torch.double
        )

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def bounds(self) -> torch.Tensor:
        return torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    def initialize(self, config):
        raise NotImplementedError

    def evaluate(self, X: torch.Tensor) -> EvaluationBatch:
        self.evaluate_calls += 1
        if self._fail_evaluate:
            raise RuntimeError("synthetic evaluation failure")
        n = X.shape[0]
        new_X = torch.cat([self._history.X, X], dim=0)
        new_obj = torch.cat(
            [self._history.objectives, X.sum(dim=-1, keepdim=True) / 2], dim=0
        )
        new_cons = torch.zeros((new_X.shape[0], 0), dtype=torch.double)
        new_feas = torch.ones((new_X.shape[0], 1), dtype=torch.bool)
        self._history = EvaluationBatch(
            X=new_X, objectives=new_obj, constraints=new_cons, feasible=new_feas
        )
        self._fitted = False
        self.last_evaluated = X
        return EvaluationBatch(
            X=X.clone(),
            objectives=new_obj[-n:],
            constraints=new_cons[-n:],
            feasible=new_feas[-n:],
        )

    def fit(self) -> None:
        self.fit_calls += 1
        self._fitted = True

    def propose(self, n: int = 1) -> torch.Tensor:
        self.propose_calls += 1
        # Return a slightly perturbed point each call so different candidates
        # don't all evaluate the same point.
        base = self._propose_value.clone()
        if self.propose_calls > 1:
            base = base + torch.tensor(
                [[0.01 * self.propose_calls, 0.0]], dtype=torch.double
            )
            base = base.clamp(min=0.0, max=1.0)
        return base

    def get_history(self) -> EvaluationBatch:
        return self._history

    def predict(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = X.shape[0]
        mean = X.sum(dim=-1, keepdim=True) / 2  # matches synthetic objective
        std = torch.full((n, 1), 0.1, dtype=torch.double)
        return mean, std

    def checkpoint(self):
        from opt_agent.backend.checkpoint import Checkpoint
        return Checkpoint(
            backend_id=type(self).__module__ + "." + type(self).__name__,
            problem_id="Synthetic2D",
            problem_bounds=[(0.0, 1.0), (0.0, 1.0)],
            X=self._history.X.detach().clone(),
            objectives=self._history.objectives.detach().clone(),
            constraints=self._history.constraints.detach().clone(),
            fitted=self._fitted,
            model_state_dict=None,
            torch_rng_state=None,
        )

    def restore(self, cp) -> None:
        from opt_agent.backend.checkpoint import Checkpoint
        if not isinstance(cp, Checkpoint):
            raise ValueError("restore expects Checkpoint")
        expected = type(self).__module__ + "." + type(self).__name__
        if cp.backend_id != expected:
            raise ValueError(f"backend_id mismatch: {cp.backend_id} vs {expected}")
        self._history = EvaluationBatch(
            X=cp.X.detach().clone(),
            objectives=cp.objectives.detach().clone(),
            constraints=cp.constraints.detach().clone(),
            feasible=torch.ones((cp.X.shape[0], 1), dtype=torch.bool),
        )
        # RecordingBackend doesn't store a GP model; restore just sets fitted=False.
        self._fitted = False


class FixedPolicy:
    """Returns a pre-set action. Tracks call count."""

    def __init__(self, action) -> None:
        self.action = action
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def select_action(self, state, backend=None):
        self.calls += 1
        return self.action

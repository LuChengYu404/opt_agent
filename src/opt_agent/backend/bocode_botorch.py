"""Unified backend interface + BocodeBotorchBackend (Phase 1 corrected).

Correction history (2025-09-23, per codex gpt-5.6-sol review):
  - BOCoDe uses MAXIMIZATION convention (Brannin returns -0.397887 at known
    global minima). best_f = history.objectives.max(), not min().
  - `acquisition` field removed from BackendConfig (fixed to LogEI for Phase 1).
  - `bounds` removed from BackendConfig (set internally from problem; not user
    input in Phase 1).
  - UnifiedBackend uses abc.ABC + @abstractmethod.
  - get_history() returns cloned tensors so callers cannot mutate internal state.
  - propose(n != 1) raises (LogExpectedImprovement analytic only supports q=1).
  - Gating uses allowlist (Branin/Hartmann/Ackley) + structural assertions.

Design follows:
  - MASTER_PLAN.md §6 (Unified Backend)
  - project-opt-agent-v3-phase0-review / -phase1 (Codex review 2025-09-23)
  - project-opt-agent-v3-bocode-direction (BOCoDe MAXIMIZE convention)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Public data containers
# ---------------------------------------------------------------------------


@dataclass
class EvaluationBatch:
    """A batch of evaluated design points.

    Convention: BOCoDe returns objectives in a MAXIMIZATION frame
    (project-opt-agent-v3-bocode-direction). `objectives.max()` is the incumbent.

    Attributes
    ----------
    X : Tensor, shape [n, d]
        Sampled points.
    objectives : Tensor, shape [n, m]
        Objective values (maximize).
    constraints : Tensor, shape [n, c]
        Constraint values; feasible iff all constraints <= 0.
    feasible : Tensor, shape [n, 1], bool
        Derived: all constraints <= 0 per row.
    """
    X: torch.Tensor
    objectives: torch.Tensor
    constraints: torch.Tensor
    feasible: torch.Tensor

    def __len__(self) -> int:
        return int(self.X.shape[0])


@dataclass
class BackendConfig:
    """Static config for one optimization run.

    Phase 1: acquisition is fixed (LogExpectedImprovement); bounds are
    resolved from the problem. Neither is user-controllable yet.
    """
    problem_id: str = "Branin"
    seed: int = 0
    n_init: int = 5


# ---------------------------------------------------------------------------
# Backend interface (minimal, Phase 1)
# ---------------------------------------------------------------------------


class UnifiedBackend(ABC):
    """Minimal interface — Phase 1.

    Concrete backend: BocodeBotorchBackend.
    """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the surrogate model is currently fitted to the latest history."""
        ...

    @abstractmethod
    def initialize(self, config: BackendConfig) -> EvaluationBatch:
        ...

    @abstractmethod
    def evaluate(self, X: torch.Tensor) -> EvaluationBatch:
        ...

    @abstractmethod
    def fit(self) -> None:
        ...

    @abstractmethod
    def propose(self, n: int = 1) -> torch.Tensor:
        ...

    @abstractmethod
    def get_history(self) -> EvaluationBatch:
        ...


# ---------------------------------------------------------------------------
# Phase-1 concrete backend
# ---------------------------------------------------------------------------


class BocodeBotorchBackend(UnifiedBackend):
    """BOCoDe problem + BoTorch GP + LogExpectedImprovement.

    Phase-1 constraints (validated by tests):
      - Single-objective only (n_obj == 1).
      - Unconstrained only (n_constr == 0).
      - Continuous box-bounded only.
      - Problems limited to allowlist {Branin, Hartmann, Ackley}. Other problems
        are rejected to avoid admitting categorical/integer/noisy BOCoDe
        problems that the SingleTaskGP + optimize_acqf stack cannot handle.

    Phase 1 explicitly does NOT support:
      - Multi-objective (raises ValueError).
      - Constrained problems (raises ValueError).
      - Checkpoint / restore (Phase 5).
      - Configurable acquisition / surrogate (fixed LogExpectedImprovement +
        SingleTaskGP).
      - Batch proposals (propose(n=1) only).
    """

    _ALLOWED_PROBLEMS = ("Branin", "Hartmann", "Ackley")

    def __init__(self) -> None:
        self.config: Optional[BackendConfig] = None
        self._problem = None
        self._history_X: Optional[torch.Tensor] = None
        self._history_obj: Optional[torch.Tensor] = None
        self._history_cons: Optional[torch.Tensor] = None
        self._model = None
        self._mll = None
        self._fitted: bool = False

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _bounds_to_tensor(bounds) -> torch.Tensor:
        """Convert BOCoDe bounds (list of (lo, hi) tuples, length d) to [2, d] tensor."""
        lo = torch.tensor([float(b[0]) for b in bounds], dtype=torch.double)
        hi = torch.tensor([float(b[1]) for b in bounds], dtype=torch.double)
        return torch.stack([lo, hi], dim=0)

    def _assert_phase1_compatible(self, prob) -> None:
        """Raise if the problem violates Phase-1 restrictions.

        Both name allowlist and structural (n_obj, n_constr) gates apply.
        """
        cls_name = type(prob).__name__
        if cls_name not in self._ALLOWED_PROBLEMS:
            raise ValueError(
                f"Phase 1 backend allowlist is {self._ALLOWED_PROBLEMS}; "
                f"got '{cls_name}'. Extend the allowlist only after verifying "
                f"the backend handles the new problem type correctly."
            )
        d = len(prob.bounds)
        mid = torch.tensor(
            [(b[0] + b[1]) / 2 for b in prob.bounds], dtype=torch.double
        ).unsqueeze(0)
        obj, cons = prob.evaluate(mid)
        if obj.shape[-1] != 1:
            raise ValueError(
                f"Phase 1 backend requires n_obj == 1 (single-objective); "
                f"got n_obj={obj.shape[-1]} for '{cls_name}'."
            )
        if cons.shape[-1] != 0:
            raise ValueError(
                f"Phase 1 backend requires n_constr == 0 (unconstrained); "
                f"got n_constr={cons.shape[-1]} for '{cls_name}'."
            )

    # -- UnifiedBackend API -------------------------------------------------

    def initialize(self, config: BackendConfig) -> EvaluationBatch:
        import bocode  # bocode uses __getattr__ magic; do NOT `from bocode import get_problem`

        self.config = config
        cls = bocode.get_problem(config.problem_id)
        prob = cls()
        self._assert_phase1_compatible(prob)

        self._problem = prob
        bounds_t = self._bounds_to_tensor(prob.bounds)
        d = bounds_t.shape[1]

        torch.manual_seed(config.seed)
        from torch.quasirandom import SobolEngine
        sobol = SobolEngine(dimension=d, scramble=True, seed=config.seed)
        unit = sobol.draw(config.n_init).to(dtype=torch.double)
        X0 = bounds_t[0] + unit * (bounds_t[1] - bounds_t[0])

        with torch.no_grad():
            obj0, cons0 = prob.evaluate(X0)

        self._history_X = X0.detach().clone()
        self._history_obj = obj0.detach().clone()
        self._history_cons = cons0.detach().clone()
        self._fitted = False
        return self.get_history()

    def evaluate(self, X: torch.Tensor) -> EvaluationBatch:
        if self._problem is None:
            raise RuntimeError("Backend not initialized — call initialize() first.")
        if X.ndim != 2:
            raise ValueError(f"evaluate() expects [n, d] tensor; got shape {tuple(X.shape)}")
        if X.shape[1] != self._history_X.shape[1]:
            raise ValueError(
                f"evaluate() expects d={self._history_X.shape[1]}; got d={X.shape[1]}"
            )
        if not torch.isfinite(X).all():
            raise ValueError("evaluate() received non-finite values in X")
        with torch.no_grad():
            obj, cons = self._problem.evaluate(X)
        self._history_X = torch.cat([self._history_X, X.detach()], dim=0)
        self._history_obj = torch.cat([self._history_obj, obj.detach()], dim=0)
        self._history_cons = torch.cat([self._history_cons, cons.detach()], dim=0)
        self._fitted = False  # invalidate fitted surrogate
        return self._slice_batch(X.detach(), obj.detach(), cons.detach())

    def fit(self) -> None:
        """Fit a SingleTaskGP over current history using BoTorch."""
        if self._history_X is None:
            raise RuntimeError("Backend not initialized — call initialize() first.")

        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from gpytorch.mlls import ExactMarginalLogLikelihood

        X = self._history_X.detach().clone()
        Y = self._history_obj.detach().clone()

        self._model = SingleTaskGP(X, Y)
        self._mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
        try:
            fit_gpytorch_mll(self._mll)
            self._fitted = True
        except Exception:
            # Clear partial model so propose() remains unavailable.
            self._model = None
            self._mll = None
            self._fitted = False
            raise

    def propose(self, n: int = 1) -> torch.Tensor:
        """Use LogEI to propose n new points (returned, but not evaluated).

        Phase 1 only supports n=1 because analytic LogExpectedImprovement is
        a single-point acquisition function.
        """
        if not self._fitted:
            raise RuntimeError("Backend not fitted — call fit() first.")
        if n != 1:
            raise ValueError(
                f"Phase 1 propose() only supports n=1; got n={n}. "
                "Use qLogExpectedImprovement for batch proposals."
            )
        from botorch.acquisition.analytic import LogExpectedImprovement
        from botorch.optim import optimize_acqf

        assert self._model is not None and self._problem is not None
        bounds_t = self._bounds_to_tensor(self._problem.bounds)

        best_f = self._history_obj.max().item()  # MAXIMIZE convention
        ei = LogExpectedImprovement(model=self._model, best_f=best_f)
        candidate, _ = optimize_acqf(
            acq_function=ei,
            bounds=bounds_t,
            q=1,
            num_restarts=4,
            raw_samples=16,
        )
        return candidate.detach()

    def get_history(self) -> EvaluationBatch:
        """Return a cloned history. Callers cannot mutate backend state."""
        if self._history_X is None:
            raise RuntimeError("Backend not initialized — call initialize() first.")
        return self._slice_batch(
            self._history_X.clone(),
            self._history_obj.clone(),
            self._history_cons.clone(),
        )

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def bounds(self) -> Optional[torch.Tensor]:
        if self._problem is None:
            return None
        return self._bounds_to_tensor(self._problem.bounds)

    # -- internal -----------------------------------------------------------

    def _slice_batch(
        self, X: torch.Tensor, obj: torch.Tensor, cons: torch.Tensor
    ) -> EvaluationBatch:
        if cons.shape[-1] > 0:
            feasible = (cons <= 0).all(dim=-1, keepdim=True)
        else:
            feasible = torch.ones((X.shape[0], 1), dtype=torch.bool, device=X.device)
        return EvaluationBatch(X=X, objectives=obj, constraints=cons, feasible=feasible)

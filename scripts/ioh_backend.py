"""Minimal IOH backend adapter — exposes IOH problems through the BocodeBotorchBackend interface.

Implements just enough of BocodeBotorchBackend's surface to drive DecisionLoop:
- initialize(BackendConfig)
- evaluate(X)
- fit()
- propose(n=1)
- predict(X)
- is_fitted / bounds / history properties

We use IOH problems (Rastrigin, Sphere, Ellipsoid, BuecheRastrigin, LinearSlope)
at arbitrary dimensions, which gives the dimension-crossed benchmark
that EXP-RQ0.5 / Codex #1 needs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

# We implement a minimal Backend-like object directly; not subclassing
# BocodeBotorchBackend because the interface differs slightly.
# All methods used by DecisionLoop.run + executor are implemented.


@dataclass
class IOHBackend:
    """Minimal BO backend around an IOH problem."""

    problem: any = field(default=None, init=False)
    bounds_lb: np.ndarray = field(default=None, init=False)
    bounds_ub: np.ndarray = field(default=None, init=False)
    dim: int = field(default=0, init=False)
    _history_X: Optional[torch.Tensor] = field(default=None, init=False)
    _history_y: Optional[torch.Tensor] = field(default=None, init=False)
    _fitted_X: Optional[torch.Tensor] = field(default=None, init=False)
    _fitted_y: Optional[torch.Tensor] = field(default=None, init=False)
    _fitted: bool = field(default=False, init=False)
    _counter: int = field(default=0, init=False)

    def initialize(self, problem, n_init: int = 5, seed: int = 0) -> None:
        """Initialize with an ioh problem. n_init and seed are honored.

        Convention: IOH minimizes, but v3 framework assumes maximization.
        We negate IOH values at storage time so `history.objectives.max()`
        gives the best-seen (negated-min = max of negated). The LLM sees
        correct maximization values; local_exploitation's argmax works.
        """
        self.problem = problem
        self.dim = int(problem.meta_data.n_variables)
        # bounds.lb / bounds.ub may be scalars or per-dim arrays.
        lb_arr = np.asarray(problem.bounds.lb).flatten()
        ub_arr = np.asarray(problem.bounds.ub).flatten()
        if lb_arr.size == 1:
            lb = float(lb_arr[0])
            ub = float(ub_arr[0])
            self.bounds_lb = np.full(self.dim, lb)
            self.bounds_ub = np.full(self.dim, ub)
        else:
            assert lb_arr.size == self.dim and ub_arr.size == self.dim
            self.bounds_lb = lb_arr.astype(float)
            self.bounds_ub = ub_arr.astype(float)
        # Sobol-like initial points (using torch.quasirandom).
        from torch.quasirandom import SobolEngine
        torch.manual_seed(seed)
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=seed)
        unit = sobol.draw(n_init).to(dtype=torch.double)
        X0 = torch.tensor(self.bounds_lb, dtype=torch.double) + unit * (
            torch.tensor(self.bounds_ub - self.bounds_lb, dtype=torch.double)
        )
        # Evaluate IOH problems in batch (ioh supports vectorized).
        with torch.no_grad():
            raw_y0 = torch.tensor(
                [self.problem(x.cpu().numpy()) for x in X0], dtype=torch.double
            ).reshape(-1, 1)
        # NEGATE so that maximization semantics hold throughout the
        # downstream state extraction / LLM payload.
        y0 = -raw_y0
        self._history_X = X0.detach().clone()
        self._history_y = y0.detach().clone()
        self._fitted = False
        self._counter = 0

    def evaluate(self, X: torch.Tensor):
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"shape mismatch: {tuple(X.shape)} vs dim={self.dim}")
        if not torch.isfinite(X).all():
            raise ValueError("non-finite X")
        with torch.no_grad():
            raw_y = torch.tensor(
                [self.problem(x.cpu().numpy()) for x in X], dtype=torch.double
            ).reshape(-1, 1)
        # NEGATE for maximization semantics.
        new_y = -raw_y
        self._history_X = torch.cat([self._history_X, X.detach()], dim=0)
        self._history_y = torch.cat([self._history_y, new_y.detach()], dim=0)
        self._fitted = False
        return new_y.detach()

    def fit(self) -> None:
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from botorch.fit import fit_gpytorch_mll

        X = self._history_X.detach().clone()
        # history_y already stores -IOH_y; GP fits a maximization model
        # directly on it. No negation needed here.
        Y = self._history_y.detach().clone()
        # Normalize inputs to [0, 1] unit cube (BoTorch best practice).
        # NOTE: Normalize transform state is not preserved through our
        # custom checkpoint/restore; defer normalization until checkpoint
        # boundary handling is updated.
        self._model = SingleTaskGP(X, Y)
        self._mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
        try:
            fit_gpytorch_mll(self._mll)
            self._fitted = True
        except Exception:
            self._fitted = False
            raise

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def propose(self, n: int = 1) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("Backend not fitted — call fit() first.")
        if n != 1:
            raise ValueError("IOHBackend only supports n=1 propose.")
        from botorch.acquisition.analytic import LogExpectedImprovement
        from botorch.optim import optimize_acqf

        assert self._fitted
        bounds_t = torch.stack([
            torch.tensor(self.bounds_lb, dtype=torch.double),
            torch.tensor(self.bounds_ub, dtype=torch.double),
        ])
        # history_y stores -IOH_y; max is the best (negated-min).
        best_f = float(self._history_y.max().item())
        ei = LogExpectedImprovement(model=self._model, best_f=best_f)
        candidate, _ = optimize_acqf(
            acq_function=ei, bounds=bounds_t, q=1,
            num_restarts=4, raw_samples=16,
        )
        return candidate.detach()

    def predict(self, X: torch.Tensor):
        if not self._fitted:
            raise RuntimeError("not fitted")
        if X.ndim != 2:
            raise ValueError(f"predict expects [n, d]; got {tuple(X.shape)}")
        self._model.eval()
        with torch.no_grad():
            posterior = self._model.posterior(X)
            mean = posterior.mean
            var = posterior.variance
            if var.ndim == 1:
                var = var.unsqueeze(-1)
            std = var.clamp(min=0.0).sqrt()
        # GP is fit on -IOH_y; posterior.mean is on the negated scale
        # (= v3 maximizing scale). Return as-is.
        return mean.detach(), std.detach()

    @property
    def bounds(self):
        return torch.stack([
            torch.tensor(self.bounds_lb, dtype=torch.double),
            torch.tensor(self.bounds_ub, dtype=torch.double),
        ])

    def get_history(self):
        class _Batch:
            def __init__(self, X, y):
                self.X = X
                self.objectives = y
            def __len__(self):
                return self.X.shape[0]
        return _Batch(self._history_X.detach(), self._history_y.detach())

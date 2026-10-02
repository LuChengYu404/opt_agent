"""Minimal IOH backend adapter (BocodeBotorchBackend-compatible surface).

IOH minimizes; we negate values at storage time so the v3 maximization
convention holds throughout. problems are BBOB-class (both consumers);
problem_id encodes "fid:dim:instance" for checkpoint restore.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class IOHBackend:
    normalize_inputs: bool = False
    problem: any = field(default=None, init=False)
    bounds_lb: np.ndarray = field(default=None, init=False)
    bounds_ub: np.ndarray = field(default=None, init=False)
    dim: int = field(default=0, init=False)
    _history_X: Optional[torch.Tensor] = field(default=None, init=False)
    _history_y: Optional[torch.Tensor] = field(default=None, init=False)
    _fitted: bool = field(default=False, init=False)
    _model: any = field(default=None, init=False)

    def initialize(self, problem, n_init: int = 5, seed: int = 0) -> None:
        from torch.quasirandom import SobolEngine

        self.problem = problem
        self.dim = int(problem.meta_data.n_variables)
        lb = np.asarray(problem.bounds.lb, dtype=float).flatten()
        ub = np.asarray(problem.bounds.ub, dtype=float).flatten()
        if lb.size == 1:  # scalar bounds broadcast per-dim
            lb, ub = np.full(self.dim, lb[0]), np.full(self.dim, ub[0])
        self.bounds_lb, self.bounds_ub = lb, ub

        torch.manual_seed(seed)
        unit = SobolEngine(dimension=self.dim, scramble=True,
                           seed=seed).draw(n_init).to(torch.double)
        X0 = (torch.tensor(lb, dtype=torch.double)
              + unit * torch.tensor(ub - lb, dtype=torch.double))
        self._history_X = X0.detach().clone()
        self._history_y = -self._raw_eval(X0)
        self._fitted = False

    def _raw_eval(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.tensor(
                [self.problem(x.cpu().numpy()) for x in X],
                dtype=torch.double).reshape(-1, 1)

    def evaluate(self, X: torch.Tensor):
        new_y = -self._raw_eval(X)  # negate for maximization semantics
        self._history_X = torch.cat([self._history_X, X.detach()])
        self._history_y = torch.cat([self._history_y, new_y.detach()])
        self._fitted = False
        return new_y.detach()

    def fit(self) -> None:
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize
        from gpytorch.mlls import ExactMarginalLogLikelihood

        transform = (Normalize(d=self.dim, bounds=self.bounds)
                     if self.normalize_inputs else None)
        self._model = SingleTaskGP(self._history_X.clone(),
                                   self._history_y.clone(),
                                   input_transform=transform)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(
            self._model.likelihood, self._model))
        self._fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def propose(self, n: int = 1, af: str = "LogEI",
                bounds: torch.Tensor | None = None) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("Backend not fitted — call fit() first.")
        if n != 1:
            raise ValueError("IOHBackend only supports n=1 propose.")
        from opt_agent.backend.acquisitions import propose_candidate
        return propose_candidate(
            af, model=self._model,
            best_f=float(self._history_y.max().item()),  # negated-min best
            bounds=self.bounds if bounds is None else bounds)

    def predict(self, X: torch.Tensor):
        if not self._fitted:
            raise RuntimeError("not fitted")
        self._model.eval()
        with torch.no_grad():
            posterior = self._model.posterior(X)
            var = posterior.variance
            if var.ndim == 1:
                var = var.unsqueeze(-1)
            return posterior.mean.detach(), var.clamp(min=0.0).sqrt().detach()

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

    # -- checkpoint / restore (atomic greedy actions roll back through these)

    def checkpoint(self):
        from io import BytesIO

        from opt_agent.backend.checkpoint import Checkpoint

        model_bytes = None
        if self._fitted and self._model is not None:
            buf = BytesIO()
            torch.save(self._model.state_dict(), buf)
            model_bytes = buf.getvalue()
        rng_buf = BytesIO()
        torch.save(torch.random.get_rng_state(), rng_buf)
        md = self.problem.meta_data
        return Checkpoint(
            backend_id=type(self).__module__ + "." + type(self).__name__,
            problem_id=f"{md.problem_id}:{self.dim}:{md.instance}",
            problem_bounds=list(zip(self.bounds_lb.tolist(),
                                    self.bounds_ub.tolist())),
            X=self._history_X.clone(),
            objectives=self._history_y.clone(),
            constraints=torch.zeros((self._history_X.shape[0], 0),
                                    dtype=torch.double),
            fitted=self._fitted,
            model_state_dict=model_bytes,
            torch_rng_state=rng_buf.getvalue(),
            normalize_inputs=self.normalize_inputs,
        )

    def restore(self, cp) -> None:
        from io import BytesIO

        from opt_agent.backend.checkpoint import Checkpoint

        assert isinstance(cp, Checkpoint)
        fid, dim, inst = (int(v) for v in cp.problem_id.split(":"))
        problem = self._problem_matching(fid, dim, inst)
        self.problem = problem
        self.dim = dim
        self._bounds_tensor = torch.tensor(
            cp.problem_bounds, dtype=torch.double).T.contiguous()
        self.bounds_lb = self._bounds_tensor[0].numpy().astype(float)
        self.bounds_ub = self._bounds_tensor[1].numpy().astype(float)
        self._history_X = cp.X.clone()
        self._history_y = cp.objectives.clone()
        self.normalize_inputs = bool(cp.normalize_inputs)

        if cp.fitted and cp.model_state_dict is not None:
            from botorch.models import SingleTaskGP
            from botorch.models.transforms import Normalize

            model = SingleTaskGP(
                self._history_X, self._history_y,
                input_transform=(Normalize(d=dim, bounds=self._bounds_tensor)
                                 if self.normalize_inputs else None))
            model.load_state_dict(
                torch.load(BytesIO(cp.model_state_dict), weights_only=False))
            model.eval()  # normalizes train inputs; see _entropy_workspace
            self._model = model
            self._fitted = True
        else:
            self._model = None
            self._fitted = False
        torch.random.set_rng_state(
            torch.load(BytesIO(cp.torch_rng_state), weights_only=False))

    def _problem_matching(self, fid, dim, inst):
        md = getattr(self.problem, "meta_data", None)
        if md is not None and (md.problem_id, self.dim, md.instance) == (
                fid, dim, inst):
            return self.problem
        import ioh
        return ioh.get_problem(fid=fid, dimension=dim, instance=inst,
                               problem_class=ioh.ProblemClass.BBOB)

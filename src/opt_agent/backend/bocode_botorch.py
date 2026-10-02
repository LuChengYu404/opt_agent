"""BOCoDe problem + BoTorch GP backend (maximization convention).

Kept deliberately lean (2026-10-01 slimming pass): structural gates only —
no allowlist, no input re-validation, no legacy-checkpoint inference.
Single-objective / unconstrained is enforced because the
SingleTaskGP + analytic-AF stack cannot handle anything else.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from opt_agent.backend.checkpoint import Checkpoint


@dataclass
class EvaluationBatch:
    """Evaluated points. objectives/constraints follow BOCoDe's MAXIMIZE
    frame; feasible iff all constraints <= 0."""

    X: torch.Tensor                 # [n, d]
    objectives: torch.Tensor        # [n, m]
    constraints: torch.Tensor       # [n, c]
    feasible: torch.Tensor          # [n, 1] bool

    def __len__(self) -> int:
        return int(self.X.shape[0])


@dataclass
class BackendConfig:
    problem_id: str = "Branin"
    seed: int = 0
    n_init: int = 5


class UnifiedBackend(ABC):
    @property
    @abstractmethod
    def is_fitted(self) -> bool: ...

    @abstractmethod
    def initialize(self, config: BackendConfig) -> EvaluationBatch: ...
    @abstractmethod
    def evaluate(self, X: torch.Tensor) -> EvaluationBatch: ...
    @abstractmethod
    def fit(self) -> None: ...
    @abstractmethod
    def propose(self, n: int = 1, af: str = "LogEI",
                bounds: Optional[torch.Tensor] = None) -> torch.Tensor: ...
    @abstractmethod
    def get_history(self) -> EvaluationBatch: ...
    @abstractmethod
    def predict(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    @abstractmethod
    def checkpoint(self) -> "Checkpoint": ...
    @abstractmethod
    def restore(self, cp: "Checkpoint") -> None: ...


class BocodeBotorchBackend(UnifiedBackend):
    """SingleTaskGP + LMABO AF portfolio (default LogEI), Sobol init."""

    def __init__(self, *, objective_scale: float = 1.0,
                 objective_shift: float = 0.0,
                 normalize_inputs: bool = False) -> None:
        self.config: Optional[BackendConfig] = None
        self._problem = None
        self._history_X = None
        self._history_obj = None
        self._history_cons = None
        self._model = None
        self._fitted = False
        self.normalize_inputs = bool(normalize_inputs)
        # Non-finite transforms would silently corrupt the whole pipeline.
        if not (math.isfinite(objective_scale) and objective_scale > 0
                and math.isfinite(objective_shift)):
            raise ValueError("objective_scale must be finite > 0 and "
                             "objective_shift finite.")
        self._objective_scale = float(objective_scale)
        self._objective_shift = float(objective_shift)

    # -- helpers -------------------------------------------------------------

    def _apply_objective_transform(self, obj: torch.Tensor) -> torch.Tensor:
        if self._objective_scale == 1.0 and self._objective_shift == 0.0:
            return obj
        return self._objective_scale * obj + self._objective_shift

    def _make_gp(self, X: torch.Tensor, Y: torch.Tensor):
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize

        transform = (Normalize(d=X.shape[1], bounds=self.bounds)
                     if self.normalize_inputs else None)
        return SingleTaskGP(X, Y, input_transform=transform)

    def _slice_batch(self, X, obj, cons) -> EvaluationBatch:
        if cons.shape[-1] > 0:
            feasible = (cons <= 0).all(dim=-1, keepdim=True)
        else:
            feasible = torch.ones((X.shape[0], 1), dtype=torch.bool)
        return EvaluationBatch(X, obj, cons, feasible)

    # -- UnifiedBackend API --------------------------------------------------

    def initialize(self, config: BackendConfig) -> EvaluationBatch:
        import bocode  # bocode uses __getattr__ magic
        from torch.quasirandom import SobolEngine

        self.config = config
        prob = bocode.get_problem(config.problem_id)()
        obj, cons = prob.evaluate(
            torch.tensor([[(b[0] + b[1]) / 2 for b in prob.bounds]],
                         dtype=torch.double))
        if obj.shape[-1] != 1:
            raise ValueError(f"requires n_obj == 1; got {obj.shape[-1]}")
        if cons.shape[-1] != 0:
            raise ValueError(f"requires n_constr == 0; got {cons.shape[-1]}")

        self._problem = prob
        self._bounds_tensor = torch.tensor(
            [[b[0] for b in prob.bounds], [b[1] for b in prob.bounds]],
            dtype=torch.double)
        d = self._bounds_tensor.shape[1]

        torch.manual_seed(config.seed)
        unit = SobolEngine(dimension=d, scramble=True,
                           seed=config.seed).draw(config.n_init).to(torch.double)
        X0 = self._bounds_tensor[0] + unit * (self._bounds_tensor[1] - self._bounds_tensor[0])
        with torch.no_grad():
            obj0, cons0 = prob.evaluate(X0)

        self._history_X = X0.detach().clone()
        self._history_obj = self._apply_objective_transform(obj0).detach().clone()
        self._history_cons = cons0.detach().clone()
        self._fitted = False
        return self.get_history()

    def evaluate(self, X: torch.Tensor) -> EvaluationBatch:
        if self._problem is None:
            raise RuntimeError("Backend not initialized — call initialize() first.")
        with torch.no_grad():
            obj, cons = self._problem.evaluate(X)
        self._history_X = torch.cat([self._history_X, X.detach()], dim=0)
        self._history_obj = torch.cat(
            [self._history_obj, self._apply_objective_transform(obj).detach()])
        self._history_cons = torch.cat([self._history_cons, cons.detach()])
        self._fitted = False
        # Returned slice uses transformed values (consistent with history).
        return self._slice_batch(
            X.detach(), self._apply_objective_transform(obj).detach(),
            cons.detach())

    def fit(self) -> None:
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        self._model = self._make_gp(
            self._history_X.clone(), self._history_obj.clone())
        mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
        fit_gpytorch_mll(mll)
        self._fitted = True

    def propose(self, n: int = 1, af: str = "LogEI",
                bounds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Maximize the named portfolio AF (backend.acquisitions); `bounds`
        optionally restricts the search box (trust regions)."""
        if not self._fitted:
            raise RuntimeError("Backend not fitted — call fit() first.")
        if n != 1:
            raise ValueError("propose() only supports n=1.")
        from opt_agent.backend.acquisitions import propose_candidate
        return propose_candidate(
            af, model=self._model,
            best_f=self._history_obj.max().item(),  # MAXIMIZE convention
            bounds=self.bounds if bounds is None else bounds,
        )

    def get_history(self) -> EvaluationBatch:
        return self._slice_batch(
            self._history_X.clone(), self._history_obj.clone(),
            self._history_cons.clone())

    def predict(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior (mean, std), each [n, 1]. Requires is_fitted."""
        if not self._fitted:
            raise RuntimeError("Backend not fitted — call fit() first.")
        self._model.eval()
        with torch.no_grad():
            posterior = self._model.posterior(X)
            var = posterior.variance
            if var.ndim == 1:
                var = var.unsqueeze(-1)
            std = var.clamp(min=0.0).sqrt()
        return posterior.mean.detach(), std.detach()

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def bounds(self) -> Optional[torch.Tensor]:
        return getattr(self, "_bounds_tensor", None)

    # -- checkpoint / restore ------------------------------------------------
    # Counterfactual rollouts replay from an EXACT backend state: history +
    # fitted GP + torch RNG. Restore failures propagate.

    def checkpoint(self) -> "Checkpoint":
        from io import BytesIO

        from opt_agent.backend.checkpoint import Checkpoint

        model_bytes = None
        if self._fitted and self._model is not None:
            buf = BytesIO()
            torch.save(self._model.state_dict(), buf)
            model_bytes = buf.getvalue()
        rng_buf = BytesIO()
        torch.save(torch.random.get_rng_state(), rng_buf)
        return Checkpoint(
            backend_id=type(self).__module__ + "." + type(self).__name__,
            problem_id=type(self._problem).__name__,
            problem_bounds=[(float(b[0]), float(b[1]))
                            for b in self._problem.bounds],
            X=self._history_X.clone(),
            objectives=self._history_obj.clone(),
            constraints=self._history_cons.clone(),
            fitted=self._fitted,
            model_state_dict=model_bytes,
            torch_rng_state=rng_buf.getvalue(),
            normalize_inputs=self.normalize_inputs,
        )

    def restore(self, cp: "Checkpoint") -> None:
        from io import BytesIO

        from opt_agent.backend.checkpoint import Checkpoint
        from gpytorch.mlls import ExactMarginalLogLikelihood

        assert isinstance(cp, Checkpoint)
        problem = self._problem
        if problem is None or type(problem).__name__ != cp.problem_id:
            import bocode
            problem = bocode.get_problem(cp.problem_id)()

        self._problem = problem
        self._bounds_tensor = torch.tensor(
            cp.problem_bounds, dtype=torch.double).T.contiguous()
        self._history_X = cp.X.clone()
        self._history_obj = cp.objectives.clone()
        self._history_cons = cp.constraints.clone()
        if cp.normalize_inputs is not None:
            self.normalize_inputs = bool(cp.normalize_inputs)

        if cp.fitted and cp.model_state_dict is not None:
            self._model = self._make_gp(self._history_X, self._history_obj)
            self._model.load_state_dict(
                torch.load(BytesIO(cp.model_state_dict), weights_only=False))
            # eval() normalizes train inputs (see acquisitions._entropy_workspace).
            self._model.eval()
            self._fitted = True
        else:
            self._model = None
            self._fitted = False
        torch.random.set_rng_state(
            torch.load(BytesIO(cp.torch_rng_state), weights_only=False))

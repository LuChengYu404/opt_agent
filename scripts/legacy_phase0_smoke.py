"""Legacy Phase 0 smoke test — kept for historical reference.

This script probes the BOCoDe / BoTorch API empirically and runs a short
baseline BO loop on three synthetic problems. It was the first concrete
artifact in the project (2025-09-23) and recorded several API discoveries:

  - BOCoDe's `bocode.get_problem(name)` is the canonical way to load a problem.
  - `evaluate(X)` returns `(objectives, constraints)`, NOT a flat tensor.
  - BOCoDe objectives are in MAXIMIZATION frame (Brannin optimum = -0.397887).
  - BoTorch 0.18 removed `botorch.sampling.stochastic.ManualSeed`; use
    `torch.manual_seed` instead.
  - GPyTorch 1.15 moved MarginalLogLikelihood into `gpytorch.mlls`.

This script has been SUPERSEDED by:
  - `src/opt_agent/backend/bocode_botorch.py` (UnifiedBackend wrapper)
  - `tests/conftest.py` RecordingBackend stub
  - `tests/test_bocode_botorch_backend.py` (19 tests)

Run for historical reference only:
    uv run python scripts/legacy_phase0_smoke.py
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

logging.getLogger("botorch").setLevel(logging.WARNING)
logging.getLogger("gpytorch").setLevel(logging.WARNING)


@dataclass
class ProblemSpec:
    display_name: str
    bocode_name: str


DEFAULT_SPECS: list[ProblemSpec] = [
    ProblemSpec(display_name="Branin-2D",  bocode_name="Branin"),
    ProblemSpec(display_name="Hartmann-6D", bocode_name="Hartmann"),
    ProblemSpec(display_name="Ackley-10D",  bocode_name="Ackley"),
]


def get_bocode_problem(name: str):
    import bocode
    return bocode.get_problem(name)()


def bounds_to_tensor(bounds) -> torch.Tensor:
    lo = torch.tensor([b[0] for b in bounds], dtype=torch.double)
    hi = torch.tensor([b[1] for b in bounds], dtype=torch.double)
    return torch.stack([lo, hi], dim=0)


def _scalarize(obj: torch.Tensor, cons: torch.Tensor, penalty: float = 1e6) -> torch.Tensor:
    y = obj.sum(dim=-1, keepdim=True) if obj.shape[-1] > 1 else obj
    feasible = (cons <= 0).all(dim=-1, keepdim=True)
    return torch.where(feasible, y, y.new_tensor(penalty))


def run_short_bo(spec: ProblemSpec, n_init: int = 5, n_iter: int = 5, seed: int = 0) -> dict:
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from torch.quasirandom import SobolEngine

    prob = get_bocode_problem(spec.bocode_name)
    bounds_t = bounds_to_tensor(prob.bounds)
    d = bounds_t.shape[1]

    torch.manual_seed(seed)
    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    unit_x = sobol.draw(n_init).to(dtype=torch.double)
    train_x = bounds_t[0] + unit_x * (bounds_t[1] - bounds_t[0])
    with torch.no_grad():
        train_obj, train_cons = prob.evaluate(train_x)
    train_y = _scalarize(train_obj, train_cons)

    history = []
    t0 = time.time()
    for it in range(n_init + n_iter):
        best_so_far = train_y.min().item() if train_y.numel() > 0 else 0.0
        history.append(
            {
                "iter": it,
                "n_samples": int(train_x.shape[0]),
                "best_value": float(best_so_far),
            }
        )
        if train_x.shape[0] >= n_init + n_iter:
            break

        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            history[-1]["fit_error"] = f"{type(e).__name__}: {e}"
            break

        ei = LogExpectedImprovement(model=model, best_f=best_so_far)
        try:
            candidate, _ = optimize_acqf(
                acq_function=ei, bounds=bounds_t, q=1,
                num_restarts=4, raw_samples=16,
            )
        except Exception as e:
            history[-1]["acq_error"] = f"{type(e).__name__}: {e}"
            break

        with torch.no_grad():
            new_obj, new_cons = prob.evaluate(candidate)
        new_y = _scalarize(new_obj, new_cons)
        train_x = torch.cat([train_x, candidate], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)

    return {
        "history": history,
        "final_best": float(train_y.min().item()) if train_y.numel() > 0 else float("nan"),
        "elapsed_sec": round(time.time() - t0, 3),
        "final_n": int(train_x.shape[0]),
        "dim": d,
    }


def main() -> int:
    print("Legacy Phase 0 smoke test (superseded — see module docstring).")
    overall_ok = True
    for spec in DEFAULT_SPECS:
        print(f"\n--- {spec.display_name} ---")
        try:
            result = run_short_bo(spec)
            print(
                f"  final_best={result['final_best']:.6f} "
                f"in {result['elapsed_sec']}s, n={result['final_n']}, d={result['dim']}"
            )
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            overall_ok = False
    print("\nOVERALL:", "OK ✅" if overall_ok else "FAIL ❌")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

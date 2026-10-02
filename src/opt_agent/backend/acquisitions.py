"""LMABO 12-AF portfolio registry (arxiv_2602.07904, Appendix A/B.2).

Protocol per the paper: analytic AFs via multi-start L-BFGS-B; TS and PES
maximized discretely over a random candidate set. Design-time constants the
paper leaves unspecified: UCB_KAPPA=2.0, DISCRETE_CANDIDATES=512,
NUM_OPTIMAL_SAMPLES=8, optimizer budget 4/16.

MAXIMIZATION frame; best_f is the observed incumbent. All randomness flows
through the global torch RNG (checkpoint-restorable). qKG/qPES/qMES/qJES
are ~100x costlier than the analytic AFs — no default action selects them.

Entropy-AF coordinates (BoTorch 0.18): their EP system reads the model's
ALREADY-NORMALIZED train_inputs and re-transforms through posterior(), so
no caller-side input space is consistent under an input transform.
`_entropy_workspace` runs them on a transform-stripped deepcopy in the
model's internal space and maps the chosen point back. ORDER MATTERS:
eval() FIRST (normalizes train inputs), delete the transform SECOND — a
restored GP sits in train mode holding physical train inputs.
"""

from __future__ import annotations

import torch

AF_DEFAULT = "LogEI"
UCB_KAPPA = 2.0          # paper fixes kappa but states no value
DISCRETE_CANDIDATES = 512
NUM_OPTIMAL_SAMPLES = 8
NUM_RESTARTS = 4
RAW_SAMPLES = 16

ACQUISITION_NAMES: tuple[str, ...] = (
    "PI", "LogPI", "EI", "LogEI", "UCB", "PosMean", "PosSTD",
    "TS", "qKG", "qPES", "qMES", "qJES",
)
ACQUISITION_SET: frozenset[str] = frozenset(ACQUISITION_NAMES)

# Paper's exploration/exploitation grouping (Appendix A).
EXPLORATORY_AFS: frozenset[str] = frozenset(
    {"PosSTD", "UCB", "TS", "qKG", "qPES", "qMES", "qJES"})
EXPLOITATIVE_AFS: frozenset[str] = frozenset(
    {"PI", "LogPI", "EI", "LogEI", "PosMean"})


def _candidate_set(bounds: torch.Tensor, size: int) -> torch.Tensor:
    """Scrambled Sobol candidates in `bounds`; seed from the global RNG."""
    from torch.quasirandom import SobolEngine

    seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    unit = SobolEngine(dimension=bounds.shape[1], scramble=True,
                       seed=seed).draw(size).to(torch.double)
    return bounds[0] + unit * (bounds[1] - bounds[0])


def _draw_optimal_samples(model, bounds, n):
    """Thompson-style argmax draws -> (inputs [n,d], outputs [n,1]); the
    optimal-input draws qPES/qJES require in BoTorch >=0.18."""
    X = _candidate_set(bounds, DISCRETE_CANDIDATES)
    with torch.no_grad():
        flat = model.posterior(X).rsample(torch.Size([n])).reshape(n, -1)
    idx = flat.argmax(dim=-1)
    return X[idx], flat.gather(1, idx.unsqueeze(-1))


def _entropy_workspace(model, bounds):
    """(af_model, af_bounds, to_physical) — see module docstring."""
    tf = getattr(model, "input_transform", None)
    if tf is None:
        return model, bounds, lambda x: x
    import copy

    af_model = copy.deepcopy(model)
    af_model.eval()
    del af_model.input_transform
    return af_model, tf(bounds), lambda x: tf.untransform(x)


def _maximize_lbfgsb(acqf, bounds):
    from botorch.optim import optimize_acqf

    candidate, _ = optimize_acqf(
        acq_function=acqf, bounds=bounds, q=1,
        num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES)
    return candidate.detach()


def _maximize_discrete(acqf, bounds):
    X = _candidate_set(bounds, DISCRETE_CANDIDATES).unsqueeze(1)  # [C,1,d]
    with torch.no_grad():
        idx = int(acqf(X).reshape(-1).argmax().item())
    return X[idx, 0].unsqueeze(0)


def _thompson(model, bounds):
    X = _candidate_set(bounds, DISCRETE_CANDIDATES)
    with torch.no_grad():
        sample = model.posterior(X).rsample(torch.Size([1])).reshape(-1)
    return X[int(sample.argmax().item())].unsqueeze(0)


def _analytic(af, model, best_f):
    from botorch.acquisition.analytic import (
        ExpectedImprovement,
        LogExpectedImprovement,
        LogProbabilityOfImprovement,
        PosteriorMean,
        PosteriorStandardDeviation,
        ProbabilityOfImprovement,
        UpperConfidenceBound,
    )

    if af == "PI":
        return ProbabilityOfImprovement(model, best_f=best_f)
    if af == "LogPI":
        return LogProbabilityOfImprovement(model, best_f=best_f)
    if af == "EI":
        return ExpectedImprovement(model, best_f=best_f)
    if af == "LogEI":
        return LogExpectedImprovement(model, best_f=best_f)
    if af == "UCB":
        return UpperConfidenceBound(model, beta=UCB_KAPPA**2)
    if af == "PosMean":
        return PosteriorMean(model)
    return PosteriorStandardDeviation(model)  # PosSTD


def propose_candidate(af: str, *, model, best_f: float,
                      bounds: torch.Tensor) -> torch.Tensor:
    """Maximize the named portfolio AF; return one point [1, d]."""
    if af == "TS":
        return _thompson(model, bounds)
    if af in ("PI", "LogPI", "EI", "LogEI", "UCB", "PosMean", "PosSTD"):
        return _maximize_lbfgsb(_analytic(af, model, best_f), bounds)
    if af == "qKG":
        from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
        return _maximize_lbfgsb(qKnowledgeGradient(model=model), bounds)
    if af == "qPES":
        from botorch.acquisition.predictive_entropy_search import (
            qPredictiveEntropySearch,
        )
        m, b, to_phys = _entropy_workspace(model, bounds)
        oi, _ = _draw_optimal_samples(m, b, NUM_OPTIMAL_SAMPLES)
        acqf = qPredictiveEntropySearch(model=m, optimal_inputs=oi)
        return to_phys(_maximize_discrete(acqf, b))  # discrete per paper
    if af == "qMES":
        from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
        m, b, to_phys = _entropy_workspace(model, bounds)
        acqf = qMaxValueEntropy(
            model=m, candidate_set=_candidate_set(b, DISCRETE_CANDIDATES))
        return to_phys(_maximize_lbfgsb(acqf, b))
    if af == "qJES":
        from botorch.acquisition.joint_entropy_search import qJointEntropySearch
        m, b, to_phys = _entropy_workspace(model, bounds)
        oi, oo = _draw_optimal_samples(m, b, NUM_OPTIMAL_SAMPLES)
        acqf = qJointEntropySearch(
            model=m, optimal_inputs=oi, optimal_outputs=oo)
        return to_phys(_maximize_lbfgsb(acqf, b))
    raise ValueError(
        f"Unknown acquisition function {af!r}; portfolio is {ACQUISITION_NAMES}.")

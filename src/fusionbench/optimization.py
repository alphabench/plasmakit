"""Global and surrogate-assisted optimization.

Wraps differential evolution (Storn & Price 1997) for derivative-free
global minimization over named, bounded parameters, plus a one-shot
surrogate-assisted path for expensive objectives: sample a QMC design,
train a Gaussian process, optimize its lower confidence bound, and
verify the candidate with a single true-model evaluation. The surrogate
answer is only as good as the GP fit - compare ``surrogate_value``
against ``best_value``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
import scipy.optimize

from fusionbench import surrogates, uncertainty
from fusionbench.errors import FusionbenchError
from fusionbench.provenance import Provenance, build_provenance

MODEL_ID = "storn-price-1997"


@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of an optimization run.

    Attributes
    ----------
    best_parameters : mapping of str to float
        Minimizing parameter values.
    best_value : float
        Objective at the minimizer (for the surrogate path, the value
        of the *true* objective at the candidate).
    n_evaluations : int
        Objective evaluations performed (true-model calls for the
        surrogate path).
    success, message : bool, str
        Optimizer status.
    provenance : Provenance
        Bounds, settings, and (for the surrogate path) GP record.
    surrogate_value : float or None
        GP-predicted optimum (surrogate path only); compare with
        ``best_value`` to judge the emulator.
    """

    best_parameters: Mapping[str, float]
    best_value: float
    n_evaluations: int
    success: bool
    message: str
    provenance: Provenance
    surrogate_value: float | None = None


def _validate_bounds(bounds: Mapping[str, tuple[float, float]]) -> list[str]:
    if not bounds:
        raise FusionbenchError("bounds must be non-empty")
    for name, (low, high) in bounds.items():
        if low >= high:
            raise FusionbenchError(f"bounds for {name!r} require low < high, got ({low}, {high})")
    return list(bounds)


def optimize(
    objective: Callable[..., float],
    bounds: Mapping[str, tuple[float, float]],
    *,
    constraints: Sequence[Callable[..., float]] | None = None,
    method: str = "differential-evolution",
    seed: int = 0,
    maxiter: int = 200,
    tol: float = 1e-8,
) -> OptimizationResult:
    """Minimize an objective over bounded named parameters.

    Parameters
    ----------
    objective : callable
        Called as ``objective(**params)``; returns the value to minimize.
    bounds : mapping of str to (low, high)
        Box bounds per parameter.
    constraints : sequence of callables, optional
        Each called as ``g(**params)``; feasible when ``g(**params) <= 0``.
    method : str
        Only ``"differential-evolution"`` is supported.
    seed : int
        Optimizer seed (deterministic).
    maxiter, tol
        Passed to :func:`scipy.optimize.differential_evolution`.
    """
    if method != "differential-evolution":
        raise FusionbenchError(f"unknown method {method!r}; use 'differential-evolution'")
    names = _validate_bounds(bounds)

    def wrapped(x: npt.NDArray[np.float64]) -> float:
        return float(objective(**dict(zip(names, x.tolist(), strict=True))))

    scipy_constraints = tuple(
        scipy.optimize.NonlinearConstraint(
            lambda x, g=g: float(g(**dict(zip(names, x.tolist(), strict=True)))),
            -np.inf,
            0.0,
        )
        for g in (constraints or ())
    )
    result = scipy.optimize.differential_evolution(
        wrapped,
        [bounds[name] for name in names],
        seed=seed,
        maxiter=maxiter,
        tol=tol,
        init="sobol",
        polish=True,
        constraints=scipy_constraints,
    )
    provenance = build_provenance(
        models=[MODEL_ID],
        inputs={
            "bounds": {name: list(bounds[name]) for name in names},
            "seed": seed,
            "maxiter": maxiter,
            "tol": tol,
            "n_constraints": len(scipy_constraints),
        },
    )
    return OptimizationResult(
        best_parameters=MappingProxyType(
            dict(zip(names, np.asarray(result.x, dtype=np.float64).tolist(), strict=True))
        ),
        best_value=float(result.fun),
        n_evaluations=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
        provenance=provenance,
    )


def optimize_surrogate(
    fn: Callable[..., float],
    bounds: Mapping[str, tuple[float, float]],
    *,
    n_train: int = 32,
    kappa: float = 0.0,
    seed: int = 0,
    maxiter: int = 200,
) -> OptimizationResult:
    """One-shot surrogate-assisted minimization of an expensive objective.

    Trains a Gaussian process on a QMC design of ``n_train`` true-model
    evaluations, minimizes the GP lower confidence bound
    ``mean - kappa * std`` with differential evolution, then evaluates
    the true objective once at the candidate. ``n_evaluations`` counts
    true-model calls (``n_train + 1``). This is deliberately not an
    iterative Bayesian-optimization loop.
    """
    names = _validate_bounds(bounds)
    surrogate = surrogates.Surrogate.from_function(fn, dict(bounds), n_train=n_train, seed=seed)

    def lcb(**params: float) -> float:
        mean, std = surrogate.predict(**params)
        return mean - kappa * std

    inner = optimize(lcb, bounds, seed=seed, maxiter=maxiter)
    candidate = dict(inner.best_parameters)
    true_value = float(fn(**candidate))
    provenance = build_provenance(
        models=[MODEL_ID, surrogates.MODEL_ID, uncertainty.MODEL_ID],
        inputs={
            "bounds": {name: list(bounds[name]) for name in names},
            "n_train": n_train,
            "kappa": kappa,
            "seed": seed,
            "maxiter": maxiter,
            "gp": surrogate.gp.to_dict(),
        },
    )
    return OptimizationResult(
        best_parameters=MappingProxyType(candidate),
        best_value=true_value,
        n_evaluations=surrogate.gp.to_dict()["n_train"] + 1,
        success=inner.success,
        message=inner.message,
        provenance=provenance,
        surrogate_value=float(inner.best_value),
    )

"""Monte Carlo uncertainty propagation and variance-based sensitivity.

Model functions are called as ``fn(**params)`` with keyword arguments
named after the entries of the parameter mapping; entries may be
:class:`~plasmakit.distributions.Distribution` objects (sampled) or
plain floats (held fixed). Sampling uses scrambled Sobol quasi-Monte
Carlo sequences by default, mapped through each distribution's inverse
CDF; sample counts are rounded up to a power of two.

Sensitivity indices use the Saltelli radial design with Jansen
estimators (Saltelli et al. 2010, Comput. Phys. Commun. 181, 259).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.stats.qmc

from plasmakit.constants import ArrayLike, as_float64, scalar_like
from plasmakit.distributions import Distribution
from plasmakit.errors import PlasmakitError
from plasmakit.neutronics import TallyValue
from plasmakit.provenance import Provenance, build_provenance

MODEL_ID = "sobol-qmc"
SALTELLI_MODEL_ID = "saltelli-2010"

Parameters: TypeAlias = Mapping[str, "Distribution | float"]
"""Named model inputs: distributions are sampled, floats are held fixed."""


def _split_parameters(
    parameters: Parameters,
) -> tuple[list[str], list[Distribution], dict[str, float]]:
    free_names: list[str] = []
    free_dists: list[Distribution] = []
    fixed: dict[str, float] = {}
    for name, value in parameters.items():
        if isinstance(value, Distribution):
            free_names.append(name)
            free_dists.append(value)
        else:
            fixed[name] = float(value)
    if not free_names:
        raise PlasmakitError("at least one parameter must be a Distribution")
    return free_names, free_dists, fixed


def _unit_samples(
    n_samples: int, dimension: int, seed: int, method: str
) -> npt.NDArray[np.float64]:
    """Quantile samples in (0, 1)^d; Sobol counts round up to a power of two."""
    if method == "sobol":
        m = max(1, int(np.ceil(np.log2(n_samples))))
        sampler = scipy.stats.qmc.Sobol(d=dimension, scramble=True, seed=seed)
        return as_float64(sampler.random_base2(m))
    if method == "random":
        rng = np.random.default_rng(seed)
        return as_float64(rng.uniform(size=(n_samples, dimension)))
    raise PlasmakitError(f"unknown sampling method {method!r}; use 'sobol' or 'random'")


def _map_quantiles(
    unit: npt.NDArray[np.float64], dists: list[Distribution]
) -> npt.NDArray[np.float64]:
    # clip away exact 0/1 so unbounded distributions never produce +/- inf
    clipped = np.clip(unit, 1e-12, 1.0 - 1e-12)
    columns = [as_float64(dist.ppf(clipped[:, j])) for j, dist in enumerate(dists)]
    return np.column_stack(columns)


def _evaluate(
    fn: Callable[..., float],
    values: npt.NDArray[np.float64],
    free_names: list[str],
    fixed: dict[str, float],
    vectorized: bool,
) -> npt.NDArray[np.float64]:
    n = values.shape[0]
    if vectorized:
        kwargs: dict[str, Any] = {name: values[:, j] for j, name in enumerate(free_names)} | fixed
        out = as_float64(fn(**kwargs))
        if out.shape != (n,):
            raise PlasmakitError(f"vectorized fn must return shape ({n},), got {out.shape}")
        return out
    results = np.empty(n)
    for i in range(n):
        kwargs = {name: float(values[i, j]) for j, name in enumerate(free_names)} | fixed
        results[i] = float(fn(**kwargs))
    return results


def _sampling_provenance(
    parameters: Parameters,
    n_actual: int,
    seed: int,
    method: str,
    models: list[str],
    extra: dict[str, Any] | None = None,
) -> Provenance:
    specs = {
        name: (value.to_dict() if isinstance(value, Distribution) else float(value))
        for name, value in parameters.items()
    }
    inputs: dict[str, Any] = {
        "parameters": specs,
        "n_samples": n_actual,
        "seed": seed,
        "method": method,
        "sampler": "scipy.stats.qmc.Sobol(scramble=True)" if method == "sobol" else "numpy",
    }
    if extra:
        inputs.update(extra)
    return build_provenance(models=models, inputs=inputs)


@dataclass(frozen=True)
class UncertainResult:
    """Samples of a propagated quantity with summary statistics.

    Attributes
    ----------
    samples : ndarray
        Model outputs, one per input sample.
    provenance : Provenance
        Distribution specs, sample count, seed, and sampler.
    extra_variance : float
        Additional variance folded into :attr:`std` (used by
        :func:`propagate_transport` for Monte Carlo tally noise); zero
        for deterministic model chains.
    """

    samples: npt.NDArray[np.float64]
    provenance: Provenance
    extra_variance: float = 0.0

    @property
    def mean(self) -> float:
        """Sample mean."""
        return float(np.mean(self.samples))

    @property
    def std(self) -> float:
        """Total standard deviation: parametric spread plus extra variance."""
        return float(np.sqrt(np.var(self.samples, ddof=1) + self.extra_variance))

    def percentile(self, q: ArrayLike) -> ArrayLike:
        """Percentile(s) of the samples (excludes ``extra_variance``)."""
        return scalar_like(as_float64(np.percentile(self.samples, q)), q)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary statistics."""
        return {
            "mean": self.mean,
            "std": self.std,
            "p5": float(np.percentile(self.samples, 5)),
            "p50": float(np.percentile(self.samples, 50)),
            "p95": float(np.percentile(self.samples, 95)),
            "n_samples": int(self.samples.size),
            "extra_variance": self.extra_variance,
        }


def propagate(
    fn: Callable[..., float],
    parameters: Parameters,
    *,
    n_samples: int = 1000,
    seed: int = 0,
    method: str = "sobol",
    vectorized: bool = False,
) -> UncertainResult:
    """Propagate input distributions through a model function.

    Parameters
    ----------
    fn : callable
        Model called as ``fn(**params)`` returning a float (or, with
        ``vectorized=True``, arrays in and one ``(n,)`` array out).
    parameters : mapping
        Distribution-valued inputs are sampled; float values are fixed.
    n_samples : int
        Requested samples; rounded up to a power of two for Sobol.
    seed : int
        Sampler seed (results are deterministic for a given seed).
    method : {"sobol", "random"}
        Scrambled Sobol QMC (default) or pseudo-random sampling.
    vectorized : bool
        Whether ``fn`` accepts arrays for the sampled parameters.
    """
    free_names, free_dists, fixed = _split_parameters(parameters)
    unit = _unit_samples(n_samples, len(free_names), seed, method)
    values = _map_quantiles(unit, free_dists)
    outputs = _evaluate(fn, values, free_names, fixed, vectorized)
    provenance = _sampling_provenance(parameters, values.shape[0], seed, method, [MODEL_ID])
    return UncertainResult(samples=outputs, provenance=provenance)


@dataclass(frozen=True)
class SobolIndices:
    """First-order and total Sobol sensitivity indices per parameter."""

    parameters: tuple[str, ...]
    first_order: Mapping[str, float]
    total_order: Mapping[str, float]
    first_order_std: Mapping[str, float]
    total_order_std: Mapping[str, float]
    n_samples: int
    provenance: Provenance

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary."""
        return {
            "parameters": list(self.parameters),
            "first_order": dict(self.first_order),
            "total_order": dict(self.total_order),
            "first_order_std": dict(self.first_order_std),
            "total_order_std": dict(self.total_order_std),
            "n_samples": self.n_samples,
        }


def sobol_indices(
    fn: Callable[..., float],
    parameters: Parameters,
    *,
    n_samples: int = 1024,
    seed: int = 0,
    n_bootstrap: int = 100,
    vectorized: bool = False,
) -> SobolIndices:
    """Variance-based first-order and total sensitivity indices.

    Saltelli radial design with Jansen estimators: one QMC draw supplies
    matrices A and B; each AB_i replaces column i of A with B's. The
    model is evaluated ``(d + 2) * n`` times. Standard errors come from
    bootstrap resampling of the sample rows.

    Parameters are as in :func:`propagate`; only Distribution-valued
    entries receive indices.
    """
    free_names, free_dists, fixed = _split_parameters(parameters)
    d = len(free_names)
    unit = _unit_samples(n_samples, 2 * d, seed, "sobol")
    n = unit.shape[0]
    a = _map_quantiles(unit[:, :d], free_dists)
    b = _map_quantiles(unit[:, d:], free_dists)

    f_a = _evaluate(fn, a, free_names, fixed, vectorized)
    f_b = _evaluate(fn, b, free_names, fixed, vectorized)
    f_ab = np.empty((d, n))
    for i in range(d):
        ab_i = a.copy()
        ab_i[:, i] = b[:, i]
        f_ab[i] = _evaluate(fn, ab_i, free_names, fixed, vectorized)

    def indices(rows: npt.NDArray[np.intp]) -> tuple[npt.NDArray[np.float64], ...]:
        fa, fb, fab = f_a[rows], f_b[rows], f_ab[:, rows]
        variance = float(np.var(np.concatenate([fa, fb]), ddof=1))
        if variance == 0.0:
            raise PlasmakitError("model output has zero variance; indices undefined")
        first = (variance - 0.5 * np.mean((fb[None, :] - fab) ** 2, axis=1)) / variance
        total = 0.5 * np.mean((fa[None, :] - fab) ** 2, axis=1) / variance
        return as_float64(first), as_float64(total)

    all_rows = np.arange(n)
    first, total = indices(all_rows)

    rng = np.random.default_rng(seed + 1)
    boot_first = np.empty((n_bootstrap, d))
    boot_total = np.empty((n_bootstrap, d))
    for k in range(n_bootstrap):
        rows = np.asarray(rng.integers(0, n, size=n), dtype=np.intp)
        boot_first[k], boot_total[k] = indices(rows)

    provenance = _sampling_provenance(
        parameters,
        n,
        seed,
        "sobol",
        [MODEL_ID, SALTELLI_MODEL_ID],
        extra={"n_bootstrap": n_bootstrap, "estimator": "jansen"},
    )
    return SobolIndices(
        parameters=tuple(free_names),
        first_order=MappingProxyType(dict(zip(free_names, first.tolist(), strict=True))),
        total_order=MappingProxyType(dict(zip(free_names, total.tolist(), strict=True))),
        first_order_std=MappingProxyType(
            dict(zip(free_names, np.std(boot_first, axis=0, ddof=1).tolist(), strict=True))
        ),
        total_order_std=MappingProxyType(
            dict(zip(free_names, np.std(boot_total, axis=0, ddof=1).tolist(), strict=True))
        ),
        n_samples=n,
        provenance=provenance,
    )


def propagate_transport(
    fn: Callable[..., TallyValue],
    parameters: Parameters,
    *,
    n_samples: int = 8,
    seed: int = 0,
    method: str = "sobol",
) -> UncertainResult:
    """Propagate uncertainty through a Monte Carlo (tally-valued) model.

    Like :func:`propagate` but for functions returning a
    :class:`~plasmakit.neutronics.TallyValue` (e.g. a blanket TBR from
    ``run_neutronics``). The per-run tally variance is folded into the
    result: ``std^2 = Var(sample values) + mean(tally std^2)``,
    separating parametric uncertainty from transport statistics.

    Sample counts should stay small (each sample is a transport run).
    """
    free_names, free_dists, fixed = _split_parameters(parameters)
    unit = _unit_samples(n_samples, len(free_names), seed, method)
    values = _map_quantiles(unit, free_dists)
    n = values.shape[0]
    outputs = np.empty(n)
    tally_variances = np.empty(n)
    for i in range(n):
        kwargs = {name: float(values[i, j]) for j, name in enumerate(free_names)} | fixed
        result = fn(**kwargs)
        if not isinstance(result, TallyValue):
            raise PlasmakitError(f"fn must return a TallyValue, got {type(result).__name__}")
        outputs[i] = result.value
        tally_variances[i] = result.std_dev**2
    provenance = _sampling_provenance(
        parameters,
        n,
        seed,
        method,
        [MODEL_ID],
        extra={"tally_variance_mean": float(np.mean(tally_variances))},
    )
    return UncertainResult(
        samples=outputs,
        provenance=provenance,
        extra_variance=float(np.mean(tally_variances)),
    )

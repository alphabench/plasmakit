"""Bayesian parameter estimation via Markov-chain Monte Carlo.

Random-walk Metropolis-Hastings (Metropolis et al. 1953; Hastings 1970)
with a Gaussian likelihood: given observations ``y`` of a model
``fn(**theta)`` with known observation noise, samples the posterior
``p(theta | y)``, proportional to ``p(y | theta) p(theta)``, under the
supplied priors.

Diagnostics are deliberately minimal: :attr:`Posterior.acceptance_rate`
is exposed (a healthy random walk targets roughly 0.2-0.5; tune
``proposal_scale``), and chains in :attr:`Posterior.samples` should be
inspected for mixing. Single-chain convergence statistics such as R-hat
are not computed - they are unreliable without multiple chains.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.optimize

from fusionbench.constants import ArrayLike, as_float64, scalar_like
from fusionbench.distributions import Distribution
from fusionbench.errors import FusionbenchError
from fusionbench.provenance import Provenance, build_provenance

MODEL_ID = "metropolis-hastings"


@dataclass(frozen=True)
class Posterior:
    """Posterior samples from a Metropolis-Hastings fit.

    Attributes
    ----------
    parameters : tuple of str
        Fitted parameter names.
    samples : mapping of str to ndarray
        Post-burn-in, thinned chain per parameter.
    map_estimate : mapping of str to float
        Maximum a posteriori point (from a deterministic optimization).
    acceptance_rate : float
        Fraction of accepted proposals after burn-in.
    n_samples : int
        Retained samples per parameter.
    provenance : Provenance
        Priors, observations, chain settings, and seed.
    """

    parameters: tuple[str, ...]
    samples: Mapping[str, npt.NDArray[np.float64]]
    map_estimate: Mapping[str, float]
    acceptance_rate: float
    n_samples: int
    provenance: Provenance

    def mean(self, name: str) -> float:
        """Posterior mean of one parameter."""
        return float(np.mean(self.samples[name]))

    def std(self, name: str) -> float:
        """Posterior standard deviation of one parameter."""
        return float(np.std(self.samples[name], ddof=1))

    def percentile(self, name: str, q: ArrayLike) -> ArrayLike:
        """Posterior percentile(s) of one parameter."""
        return scalar_like(as_float64(np.percentile(self.samples[name], q)), q)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe posterior summary."""
        return {
            name: {
                "mean": self.mean(name),
                "std": self.std(name),
                "p5": float(np.percentile(self.samples[name], 5)),
                "p95": float(np.percentile(self.samples[name], 95)),
                "map": self.map_estimate[name],
            }
            for name in self.parameters
        } | {"acceptance_rate": self.acceptance_rate, "n_samples": self.n_samples}


def _support_bounds(dist: Distribution) -> tuple[float, float]:
    p = dist.parameters
    if dist.kind == "uniform":
        return (p["low"], p["high"])
    if dist.kind == "triangular":
        return (p["low"], p["high"])
    if dist.kind == "lognormal":
        return (1e-300, np.inf)
    return (-np.inf, np.inf)


def fit(
    fn: Callable[..., float | npt.NDArray[np.float64]],
    priors: Mapping[str, Distribution],
    observed: float | npt.NDArray[np.float64],
    noise_std: float,
    *,
    n_samples: int = 5000,
    burn_in: int = 1000,
    thin: int = 1,
    proposal_scale: float = 0.5,
    seed: int = 0,
) -> Posterior:
    """Sample the posterior of model parameters given noisy observations.

    Parameters
    ----------
    fn : callable
        Model called as ``fn(**theta)`` returning the predicted
        observable (scalar or array matching ``observed``).
    priors : mapping of str to Distribution
        Prior distribution per fitted parameter.
    observed : float or ndarray
        Observed value(s) of the model output.
    noise_std : float
        Gaussian observation-noise standard deviation.
    n_samples : int
        Retained posterior samples (after burn-in and thinning).
    burn_in : int
        Discarded initial chain steps.
    thin : int
        Keep every ``thin``-th step after burn-in.
    proposal_scale : float
        Random-walk step size as a fraction of each prior's std.
    seed : int
        Chain seed (deterministic for a given seed).
    """
    if not priors:
        raise FusionbenchError("priors must be non-empty")
    if noise_std <= 0.0:
        raise FusionbenchError("noise_std must be positive")
    if n_samples <= 0 or burn_in < 0 or thin <= 0:
        raise FusionbenchError("n_samples and thin must be positive; burn_in non-negative")

    names = list(priors)
    dists = [priors[name] for name in names]
    y = as_float64(observed)
    prior_stds = np.array([dist.std for dist in dists])

    def log_posterior(theta: npt.NDArray[np.float64]) -> float:
        log_prior = 0.0
        for value, dist in zip(theta, dists, strict=True):
            lp = float(dist.logpdf(float(value)))
            if not np.isfinite(lp):
                return -np.inf
            log_prior += lp
        prediction = as_float64(fn(**dict(zip(names, theta.tolist(), strict=True))))
        residual = (y - prediction) / noise_std
        return log_prior - 0.5 * float(np.sum(residual**2))

    rng = np.random.default_rng(seed)
    theta = np.array([dist.mean for dist in dists])
    current = log_posterior(theta)
    total_steps = burn_in + n_samples * thin
    chain = np.empty((n_samples, len(names)))
    accepted_post_burn_in = 0
    kept = 0
    for step in range(total_steps):
        proposal = theta + rng.normal(0.0, proposal_scale * prior_stds)
        candidate = log_posterior(proposal)
        accept = np.log(rng.uniform()) < candidate - current
        if accept:
            theta, current = proposal, candidate
        if step >= burn_in:
            if accept:
                accepted_post_burn_in += 1
            if (step - burn_in) % thin == 0 and kept < n_samples:
                chain[kept] = theta
                kept += 1
    acceptance_rate = accepted_post_burn_in / max(1, total_steps - burn_in)

    def neg_log_posterior(theta: npt.NDArray[np.float64]) -> float:
        value = log_posterior(as_float64(theta))
        return 1.0e25 if not np.isfinite(value) else -value

    bounds = [_support_bounds(dist) for dist in dists]
    result = scipy.optimize.minimize(
        neg_log_posterior,
        np.array([dist.mean for dist in dists]),
        method="L-BFGS-B",
        bounds=bounds,
    )
    map_estimate = dict(zip(names, as_float64(result.x).tolist(), strict=True))

    provenance = build_provenance(
        models=[MODEL_ID],
        inputs={
            "priors": {name: priors[name].to_dict() for name in names},
            "observed": y.tolist(),
            "noise_std": noise_std,
            "n_samples": n_samples,
            "burn_in": burn_in,
            "thin": thin,
            "proposal_scale": proposal_scale,
            "seed": seed,
            "acceptance_rate": acceptance_rate,
        },
    )
    return Posterior(
        parameters=tuple(names),
        samples=MappingProxyType({name: chain[:, j].copy() for j, name in enumerate(names)}),
        map_estimate=MappingProxyType(map_estimate),
        acceptance_rate=acceptance_rate,
        n_samples=n_samples,
        provenance=provenance,
    )

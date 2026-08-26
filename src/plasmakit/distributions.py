"""Input probability distributions for uncertainty quantification.

A :class:`Distribution` couples a JSON-serializable specification (kind
plus named parameters, recorded verbatim in provenance) with the
corresponding ``scipy.stats`` machinery for sampling, quantiles, and
densities.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.stats

from plasmakit.constants import ArrayLike, as_float64, scalar_like
from plasmakit.errors import PlasmakitError

_KINDS = ("normal", "lognormal", "uniform", "triangular")


@dataclass(frozen=True)
class Distribution:
    """A univariate input distribution with a JSON-serializable spec.

    Build instances via the factory classmethods (:meth:`normal`,
    :meth:`lognormal`, :meth:`uniform`, :meth:`triangular`); the spec
    (``kind`` + ``parameters``) is what provenance records.
    """

    kind: str
    parameters: Mapping[str, float]

    def __post_init__(self) -> None:
        """Validate the spec and freeze the parameter mapping."""
        if self.kind not in _KINDS:
            raise PlasmakitError(f"unknown distribution kind {self.kind!r}; known: {_KINDS}")
        p = dict(self.parameters)
        if self.kind in ("normal", "lognormal"):
            if set(p) != {"mean", "std"}:
                raise PlasmakitError(f"{self.kind} needs parameters mean and std, got {set(p)}")
            if p["std"] <= 0.0:
                raise PlasmakitError("std must be positive")
            if self.kind == "lognormal" and p["mean"] <= 0.0:
                raise PlasmakitError("lognormal mean must be positive")
        elif self.kind == "uniform":
            if set(p) != {"low", "high"}:
                raise PlasmakitError(f"uniform needs parameters low and high, got {set(p)}")
            if p["low"] >= p["high"]:
                raise PlasmakitError("uniform requires low < high")
        else:  # triangular
            if set(p) != {"low", "mode", "high"}:
                raise PlasmakitError(f"triangular needs parameters low, mode, high, got {set(p)}")
            if not (p["low"] <= p["mode"] <= p["high"]) or p["low"] >= p["high"]:
                raise PlasmakitError("triangular requires low <= mode <= high and low < high")
        object.__setattr__(self, "parameters", MappingProxyType(p))

    @classmethod
    def normal(cls, mean: float, std: float) -> Distribution:
        """Gaussian distribution with the given mean and standard deviation."""
        return cls(kind="normal", parameters={"mean": mean, "std": std})

    @classmethod
    def lognormal(cls, mean: float, std: float) -> Distribution:
        """Lognormal distribution parameterized by its actual mean and std.

        The (mu, sigma) of the underlying normal are derived internally:
        ``sigma^2 = ln(1 + std^2/mean^2)``, ``mu = ln(mean) - sigma^2/2``.
        """
        return cls(kind="lognormal", parameters={"mean": mean, "std": std})

    @classmethod
    def uniform(cls, low: float, high: float) -> Distribution:
        """Uniform distribution on [low, high]."""
        return cls(kind="uniform", parameters={"low": low, "high": high})

    @classmethod
    def triangular(cls, low: float, mode: float, high: float) -> Distribution:
        """Triangular distribution on [low, high] with the given mode."""
        return cls(kind="triangular", parameters={"low": low, "mode": mode, "high": high})

    def _frozen(self) -> Any:
        p = self.parameters
        if self.kind == "normal":
            return scipy.stats.norm(loc=p["mean"], scale=p["std"])
        if self.kind == "lognormal":
            sigma2 = np.log(1.0 + (p["std"] / p["mean"]) ** 2)
            mu = np.log(p["mean"]) - sigma2 / 2.0
            return scipy.stats.lognorm(s=np.sqrt(sigma2), scale=np.exp(mu))
        if self.kind == "uniform":
            return scipy.stats.uniform(loc=p["low"], scale=p["high"] - p["low"])
        span = p["high"] - p["low"]
        return scipy.stats.triang(c=(p["mode"] - p["low"]) / span, loc=p["low"], scale=span)

    def sample(self, n: int, rng: np.random.Generator) -> npt.NDArray[np.float64]:
        """Draw ``n`` samples using the given random generator."""
        return as_float64(self._frozen().rvs(size=n, random_state=rng))

    def ppf(self, q: ArrayLike) -> ArrayLike:
        """Percent-point function (inverse CDF) at quantiles ``q`` in (0, 1)."""
        return scalar_like(as_float64(self._frozen().ppf(q)), q)

    def logpdf(self, x: ArrayLike) -> ArrayLike:
        """Log probability density at ``x`` (``-inf`` outside the support)."""
        return scalar_like(as_float64(self._frozen().logpdf(x)), x)

    @property
    def mean(self) -> float:
        """Distribution mean."""
        return float(self._frozen().mean())

    @property
    def std(self) -> float:
        """Distribution standard deviation."""
        return float(self._frozen().std())

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe spec for provenance records."""
        return {"kind": self.kind, "parameters": dict(self.parameters)}

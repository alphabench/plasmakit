"""Gaussian-process surrogate models.

A small RBF-kernel Gaussian-process regressor (Rasmussen & Williams,
"Gaussian Processes for Machine Learning", MIT Press 2006, Algorithm
2.1) with per-dimension (ARD) length scales, plus a :class:`Surrogate`
convenience that emulates an expensive model function over named
parameters and is a drop-in replacement for it in
:func:`~plasmakit.uncertainty.propagate` and
:func:`~plasmakit.optimization.optimize`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.optimize

from plasmakit import uncertainty
from plasmakit.constants import as_float64
from plasmakit.distributions import Distribution
from plasmakit.errors import PlasmakitError
from plasmakit.provenance import Provenance, build_provenance

MODEL_ID = "gp-rbf"

_LOG_BOUNDS = (np.log(1e-3), np.log(1e3))


def _kernel(
    xa: npt.NDArray[np.float64],
    xb: npt.NDArray[np.float64],
    length_scales: npt.NDArray[np.float64],
    signal_std: float,
) -> npt.NDArray[np.float64]:
    diff = xa[:, None, :] - xb[None, :, :]
    sq = np.sum((diff / length_scales) ** 2, axis=-1)
    return as_float64(signal_std**2 * np.exp(-0.5 * sq))


@dataclass(frozen=True)
class GaussianProcess:
    """An RBF-kernel Gaussian-process regressor.

    Inputs and outputs are standardized internally; hyperparameters
    (``length_scales``, ``signal_std``) live in standardized space.
    Build instances with :meth:`train`.
    """

    x_train: npt.NDArray[np.float64]
    y_mean: float
    y_std: float
    length_scales: npt.NDArray[np.float64]
    signal_std: float
    noise: float
    chol: npt.NDArray[np.float64]
    alpha: npt.NDArray[np.float64]
    x_shift: npt.NDArray[np.float64]
    x_scale: npt.NDArray[np.float64]

    @classmethod
    def train(
        cls,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        *,
        length_scale: float | Sequence[float] | None = None,
        noise: float = 1e-10,
        n_restarts: int = 3,
        seed: int = 0,
    ) -> GaussianProcess:
        """Fit a GP to training data.

        Parameters
        ----------
        x : ndarray
            Training inputs, shape ``(n, d)`` (or ``(n,)`` for 1-D).
        y : ndarray
            Training outputs, shape ``(n,)``.
        length_scale : float or sequence, optional
            Fixed length scale(s) in standardized-input units; ``None``
            optimizes them (with the signal amplitude) by maximizing the
            log marginal likelihood with L-BFGS-B restarts.
        noise : float
            Observation-noise variance added to the kernel diagonal.
        n_restarts : int
            Random restarts for the hyperparameter optimization.
        seed : int
            Seed for the restart draws.
        """
        x_arr = as_float64(x)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        y_arr = as_float64(y)
        n, d = x_arr.shape
        if y_arr.shape != (n,):
            raise PlasmakitError(f"y must have shape ({n},), got {y_arr.shape}")
        if n < 2:
            raise PlasmakitError("need at least 2 training points")
        if noise < 0.0:
            raise PlasmakitError("noise must be non-negative")
        if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
            raise PlasmakitError("training data must be finite")

        x_shift = np.mean(x_arr, axis=0)
        x_scale = np.std(x_arr, axis=0)
        x_scale = np.where(x_scale > 0.0, x_scale, 1.0)
        xs = (x_arr - x_shift) / x_scale
        y_mean = float(np.mean(y_arr))
        y_scale = float(np.std(y_arr))
        y_scale = y_scale if y_scale > 0.0 else 1.0
        ys = (y_arr - y_mean) / y_scale

        def neg_lml(theta: npt.NDArray[np.float64]) -> float:
            scales = np.exp(theta[:d])
            amplitude = float(np.exp(theta[d]))
            k = _kernel(xs, xs, scales, amplitude) + (noise + 1e-12) * np.eye(n)
            try:
                lower = scipy.linalg.cholesky(k, lower=True)
            except scipy.linalg.LinAlgError:
                return 1.0e25  # finite penalty keeps L-BFGS-B finite differences well-defined
            alpha = scipy.linalg.cho_solve((lower, True), ys)
            return float(
                0.5 * ys @ alpha + np.sum(np.log(np.diag(lower))) + 0.5 * n * np.log(2 * np.pi)
            )

        if length_scale is not None:
            scales = as_float64(np.broadcast_to(np.asarray(length_scale, dtype=np.float64), (d,)))
            amplitude = 1.0
        else:
            rng = np.random.default_rng(seed)
            best_theta = np.zeros(d + 1)
            best_value = neg_lml(best_theta)
            starts = [np.zeros(d + 1)] + [
                rng.uniform(_LOG_BOUNDS[0], _LOG_BOUNDS[1], size=d + 1) for _ in range(n_restarts)
            ]
            for start in starts:
                result = scipy.optimize.minimize(
                    neg_lml,
                    start,
                    method="L-BFGS-B",
                    bounds=[_LOG_BOUNDS] * (d + 1),
                )
                if float(result.fun) < best_value:
                    best_value = float(result.fun)
                    best_theta = as_float64(result.x)
            scales = as_float64(np.exp(best_theta[:d]))
            amplitude = float(np.exp(best_theta[d]))

        k = _kernel(xs, xs, scales, amplitude) + (noise + 1e-12) * np.eye(n)
        lower = scipy.linalg.cholesky(k, lower=True)
        alpha = as_float64(scipy.linalg.cho_solve((lower, True), ys))
        return cls(
            x_train=xs,
            y_mean=y_mean,
            y_std=y_scale,
            length_scales=scales,
            signal_std=amplitude,
            noise=noise,
            chol=as_float64(lower),
            alpha=alpha,
            x_shift=as_float64(x_shift),
            x_scale=as_float64(x_scale),
        )

    def predict(
        self, x: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Predictive mean and standard deviation at query points.

        Parameters
        ----------
        x : ndarray
            Query inputs, shape ``(m, d)`` (or ``(m,)`` for 1-D GPs).

        Returns
        -------
        tuple of ndarray
            ``(mean, std)``, each shape ``(m,)``, in original units.
        """
        x_arr = as_float64(x)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        xs = (x_arr - self.x_shift) / self.x_scale
        k_star = _kernel(self.x_train, xs, self.length_scales, self.signal_std)
        mean = k_star.T @ self.alpha
        v = scipy.linalg.solve_triangular(self.chol, k_star, lower=True)
        variance = np.clip(self.signal_std**2 - np.sum(v**2, axis=0), 0.0, None)
        return (
            as_float64(mean * self.y_std + self.y_mean),
            as_float64(np.sqrt(variance) * self.y_std),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe hyperparameter record for provenance."""
        return {
            "kernel": "rbf-ard",
            "length_scales": self.length_scales.tolist(),
            "signal_std": self.signal_std,
            "noise": self.noise,
            "n_train": int(self.x_train.shape[0]),
        }


@dataclass(frozen=True)
class Surrogate:
    """A GP emulator over named parameters, callable like the true model.

    ``surrogate(**params)`` returns the GP predictive mean, so a trained
    surrogate is a drop-in substitute for the emulated function in
    :func:`~plasmakit.uncertainty.propagate` or
    :func:`~plasmakit.optimization.optimize`.
    """

    parameters: tuple[str, ...]
    gp: GaussianProcess
    provenance: Provenance

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., float],
        parameters: Mapping[str, Distribution | tuple[float, float]],
        *,
        n_train: int = 64,
        noise: float = 1e-10,
        seed: int = 0,
        vectorized: bool = False,
    ) -> Surrogate:
        """Train a surrogate by sampling ``fn`` on a QMC design.

        Parameters
        ----------
        fn : callable
            Expensive model, called as ``fn(**params)``.
        parameters : mapping
            Per-parameter :class:`Distribution` or ``(low, high)``
            bounds (treated as uniform for the design).
        n_train : int
            Design size (rounded up to a power of two).
        noise : float
            GP observation-noise variance.
        seed : int
            Design and hyperparameter-optimization seed.
        vectorized : bool
            Whether ``fn`` accepts arrays (as in ``propagate``).
        """
        dists = {
            name: (value if isinstance(value, Distribution) else Distribution.uniform(*value))
            for name, value in parameters.items()
        }
        names, free_dists, _ = uncertainty._split_parameters(dict(dists))
        unit = uncertainty._unit_samples(n_train, len(names), seed, "sobol")
        x = uncertainty._map_quantiles(unit, free_dists)
        y = uncertainty._evaluate(fn, x, names, {}, vectorized)
        gp = GaussianProcess.train(x, y, noise=noise, seed=seed)
        provenance = build_provenance(
            models=[MODEL_ID, uncertainty.MODEL_ID],
            inputs={
                "parameters": {n: dists[n].to_dict() for n in names},
                "n_train": int(y.size),
                "seed": seed,
                "gp": gp.to_dict(),
                "design": "scipy.stats.qmc.Sobol(scramble=True)",
            },
        )
        return cls(parameters=tuple(names), gp=gp, provenance=provenance)

    def __call__(self, **params: float) -> Any:
        """GP predictive mean at one point (or arrays of points)."""
        scalar_input = all(np.ndim(params[name]) == 0 for name in self.parameters)
        columns = [np.atleast_1d(as_float64(params[name])) for name in self.parameters]
        mean, _ = self.gp.predict(np.column_stack(columns))
        return float(mean[0]) if scalar_input else mean

    def predict(self, **params: float) -> tuple[float, float]:
        """GP predictive mean and standard deviation at one point."""
        x = np.array([[float(params[name]) for name in self.parameters]])
        mean, std = self.gp.predict(x)
        return float(mean[0]), float(std[0])

    def propagate(
        self,
        parameters: uncertainty.Parameters,
        *,
        n_samples: int = 10_000,
        seed: int = 0,
        method: str = "sobol",
    ) -> uncertainty.UncertainResult:
        """Run cheap uncertainty propagation on the surrogate itself."""
        return uncertainty.propagate(
            self.__call__,
            parameters,
            n_samples=n_samples,
            seed=seed,
            method=method,
            vectorized=True,
        )

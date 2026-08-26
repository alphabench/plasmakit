"""Radial plasma profiles on normalized minor radius."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from fusionbench.constants import ArrayLike, as_float64, scalar_like
from fusionbench.errors import FusionbenchError
from fusionbench.plasma import PlasmaState


@dataclass(frozen=True)
class RadialProfile:
    """A 1-D profile on normalized minor radius ``rho`` in [0, 1].

    Values are stored on a grid and evaluated by linear interpolation.
    All values must be strictly positive: downstream plasma states reject
    non-positive temperatures and densities, so profiles that vanish at the
    edge should use a small positive pedestal instead of exactly zero.

    Parameters
    ----------
    rho : ndarray
        Strictly increasing grid spanning [0, 1] (``rho[0] == 0``,
        ``rho[-1] == 1``).
    values : ndarray
        Profile values on ``rho``; same shape, all > 0.
    """

    rho: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate and freeze the grid and values."""
        rho = as_float64(self.rho)
        values = as_float64(self.values)
        if rho.ndim != 1 or rho.size < 2:
            raise FusionbenchError("rho must be a 1-D grid with at least 2 points")
        if values.shape != rho.shape:
            raise FusionbenchError(
                f"values shape {values.shape} does not match rho shape {rho.shape}"
            )
        if np.any(np.diff(rho) <= 0.0):
            raise FusionbenchError("rho must be strictly increasing")
        if rho[0] != 0.0 or rho[-1] != 1.0:
            raise FusionbenchError("rho must span [0, 1] exactly")
        if np.any(values <= 0.0):
            raise FusionbenchError(
                "profile values must be strictly positive; use a small edge pedestal "
                "instead of zero"
            )
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "values", values)

    def __call__(self, rho: ArrayLike) -> ArrayLike:
        """Evaluate the profile at ``rho`` in [0, 1] by linear interpolation."""
        r = as_float64(rho)
        if np.any((r < 0.0) | (r > 1.0)):
            raise FusionbenchError("rho must lie in [0, 1]")
        return scalar_like(np.interp(r, self.rho, self.values), rho)

    @classmethod
    def from_callable(
        cls,
        f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        n_points: int = 129,
    ) -> RadialProfile:
        """Sample a callable ``f(rho)`` onto a uniform grid of ``n_points``."""
        rho = np.linspace(0.0, 1.0, n_points)
        return cls(rho=rho, values=as_float64(f(rho)))

    @classmethod
    def parabolic(
        cls, center: float, edge: float, exponent: float = 1.0, n_points: int = 129
    ) -> RadialProfile:
        """Build ``edge + (center - edge) * (1 - rho^2)^exponent``.

        Parameters
        ----------
        center : float
            On-axis value (rho = 0).
        edge : float
            Edge value (rho = 1); must be positive.
        exponent : float
            Peaking exponent; 1.0 gives a parabola. Use equal center and
            edge for a flat profile.
        n_points : int
            Grid resolution.
        """
        return cls.from_callable(
            lambda rho: edge + (center - edge) * (1.0 - rho**2) ** exponent, n_points
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation."""
        return {"rho": self.rho.tolist(), "values": self.values.tolist()}


@dataclass(frozen=True)
class PlasmaProfiles:
    """Profile-resolved plasma description.

    Complements (does not replace) the 0-D
    :class:`~fusionbench.plasma.PlasmaState`: evaluating the profiles at a
    set of radii yields an array-valued plasma state that all Phase-1
    physics functions accept unchanged.

    Parameters
    ----------
    ion_temperature : RadialProfile
        Ion temperature profile, keV.
    ion_density : RadialProfile
        Total fuel-ion density profile, m^-3.
    fuel : mapping of str to float
        Fuel-ion fractions (as in :class:`~fusionbench.plasma.PlasmaState`).
    """

    ion_temperature: RadialProfile
    ion_density: RadialProfile
    fuel: Mapping[str, float] = field(default_factory=lambda: {"D": 0.5, "T": 0.5})

    def __post_init__(self) -> None:
        """Validate eagerly by constructing a plasma state on axis."""
        state = self.state_at(0.0)
        object.__setattr__(self, "fuel", MappingProxyType(dict(state.fuel)))

    def state_at(self, rho: ArrayLike) -> PlasmaState:
        """Evaluate the profiles at ``rho`` and return a plasma state."""
        return PlasmaState(
            ion_temperature=self.ion_temperature(rho),
            ion_density=self.ion_density(rho),
            fuel=dict(self.fuel),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation."""
        return {
            "ion_temperature": self.ion_temperature.to_dict(),
            "ion_density": self.ion_density.to_dict(),
            "fuel": dict(self.fuel),
        }

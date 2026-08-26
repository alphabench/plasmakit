"""Plasma state description."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from plasmakit.constants import ArrayLike, as_float64, scalar_like
from plasmakit.errors import PlasmakitError

_FUEL_SPECIES = ("D", "T", "3He")


@dataclass(frozen=True)
class PlasmaState:
    """An immutable 0-D (point-wise vectorizable) plasma state.

    Parameters
    ----------
    ion_temperature : float or ndarray
        Ion temperature in keV.
    ion_density : float or ndarray
        Total fuel-ion number density in m^-3.
    fuel : mapping of str to float
        Fuel-ion number fractions by species (``"D"``, ``"T"``, ``"3He"``),
        summing to 1. Example: ``{"D": 0.5, "T": 0.5}``.

    Notes
    -----
    ``ion_temperature`` and ``ion_density`` may be broadcast-compatible
    arrays, in which case all derived quantities are computed point-wise.
    """

    ion_temperature: ArrayLike
    ion_density: ArrayLike
    fuel: Mapping[str, float] = field(default_factory=lambda: {"D": 0.5, "T": 0.5})

    def __post_init__(self) -> None:
        """Validate the state and freeze the fuel mapping."""
        t = as_float64(self.ion_temperature)
        n = as_float64(self.ion_density)
        if np.any(t <= 0.0):
            raise PlasmakitError("ion_temperature must be positive (keV)")
        if np.any(n <= 0.0):
            raise PlasmakitError("ion_density must be positive (m^-3)")
        try:
            np.broadcast_shapes(t.shape, n.shape)
        except ValueError:
            raise PlasmakitError(
                f"ion_temperature shape {t.shape} and ion_density shape {n.shape} "
                "are not broadcast-compatible"
            ) from None
        unknown = set(self.fuel) - set(_FUEL_SPECIES)
        if unknown:
            raise PlasmakitError(
                f"unknown fuel species {sorted(unknown)}; supported: {list(_FUEL_SPECIES)}"
            )
        fractions = list(self.fuel.values())
        if any(f < 0.0 or f > 1.0 for f in fractions):
            raise PlasmakitError("fuel fractions must be in [0, 1]")
        total = sum(fractions)
        if abs(total - 1.0) > 1e-8:
            raise PlasmakitError(f"fuel fractions must sum to 1, got {total}")
        object.__setattr__(self, "fuel", MappingProxyType(dict(self.fuel)))

    def density(self, species: str) -> ArrayLike:
        """Return the number density (m^-3) of one fuel species; 0 if absent."""
        fraction = self.fuel.get(species, 0.0)
        return scalar_like(fraction * as_float64(self.ion_density), self.ion_density)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dictionary representation (arrays become nested lists)."""
        return {
            "ion_temperature": np.asarray(self.ion_temperature).tolist(),
            "ion_density": np.asarray(self.ion_density).tolist(),
            "fuel": dict(self.fuel),
        }

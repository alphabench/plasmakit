"""Physical constants and array-handling conventions.

Units used throughout the package:

- temperatures and particle energies: keV
- rest-mass energies (``m c^2``): keV
- number densities: m^-3
- cross sections: m^2
- reactivities: m^3/s
- power densities: W/m^3

Rest-mass energies are CODATA 2018 recommended values.
"""

from __future__ import annotations

from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt

ArrayLike: TypeAlias = float | npt.NDArray[np.float64]
"""Scalar or NumPy array input accepted by all public functions."""

KEV_TO_JOULE: Final = 1.602176634e-16
"""Exact conversion factor, J/keV (SI definition of the electronvolt)."""

AVOGADRO: Final = 6.02214076e23
"""Avogadro constant, 1/mol (exact, SI 2019)."""

MILLIBARN_TO_M2: Final = 1.0e-31

NEUTRON_MASS_KEV: Final = 939_565.42052
PROTON_MASS_KEV: Final = 938_272.08816
DEUTERON_MASS_KEV: Final = 1_875_612.94257
TRITON_MASS_KEV: Final = 2_808_921.13668
HELION_MASS_KEV: Final = 2_808_391.60743
ALPHA_MASS_KEV: Final = 3_727_379.4066

SPECIES_MASS_KEV: Final[dict[str, float]] = {
    "n": NEUTRON_MASS_KEV,
    "p": PROTON_MASS_KEV,
    "D": DEUTERON_MASS_KEV,
    "T": TRITON_MASS_KEV,
    "3He": HELION_MASS_KEV,
    "4He": ALPHA_MASS_KEV,
}
"""Rest-mass energies (keV) of every particle species the package knows about."""


def as_float64(x: ArrayLike) -> npt.NDArray[np.float64]:
    """Coerce a scalar or array input to a float64 ndarray."""
    return np.asarray(x, dtype=np.float64)


def scalar_like(result: npt.NDArray[np.float64], *inputs: ArrayLike) -> ArrayLike:
    """Return ``result`` as a Python float when every input was scalar."""
    if all(np.ndim(x) == 0 for x in inputs):
        return float(result)
    return result

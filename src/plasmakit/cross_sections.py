"""Fusion cross sections.

Implements the Bosch-Hale (1992) S-function parameterization
(Nucl. Fusion 32, 611, Eq. 8 and Table IV):

``S(E) = (A1 + E(A2 + E(A3 + E(A4 + E*A5)))) / (1 + E(B1 + E(B2 + E(B3 + E*B4))))``

``sigma(E) = S(E) / (E * exp(B_G / sqrt(E)))``

with sigma in millibarn and E the center-of-mass energy in keV; results are
converted to m^2.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from plasmakit import bosch_hale
from plasmakit.bosch_hale import CrossSectionCoefficients
from plasmakit.constants import MILLIBARN_TO_M2, ArrayLike, as_float64, scalar_like
from plasmakit.errors import warn_outside_range
from plasmakit.reactions import Reaction, reaction


def _sigma_millibarn(
    coeff: CrossSectionCoefficients, e: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    a1, a2, a3, a4, a5 = coeff.a
    b1, b2, b3, b4 = coeff.b
    s = (a1 + e * (a2 + e * (a3 + e * (a4 + e * a5)))) / (
        1.0 + e * (b1 + e * (b2 + e * (b3 + e * b4)))
    )
    return s / (e * np.exp(coeff.b_g / np.sqrt(e)))


def cross_section(fusion_reaction: str | Reaction, energy: ArrayLike) -> ArrayLike:
    """Fusion cross section ``sigma(E)``.

    Parameters
    ----------
    fusion_reaction : str or Reaction
        Reaction identifier or :class:`~plasmakit.reactions.Reaction`.
    energy : float or ndarray
        Center-of-mass energy in keV. Scalar or array (vectorized).

    Returns
    -------
    float or ndarray
        Cross section in m^2, matching the shape of ``energy``.

    Warns
    -----
    ValidityRangeWarning
        For energies outside the fitted range of the parameterization.
    """
    rxn = reaction(fusion_reaction)
    segments = bosch_hale.CROSS_SECTION[rxn.id]
    e = as_float64(energy)
    full_range = (segments[0].e_range[0], segments[-1].e_range[1])
    warn_outside_range(e, full_range, bosch_hale.MODEL_ID, "center-of-mass energy")

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sigma = _sigma_millibarn(segments[0], e)
        for segment in segments[1:]:
            sigma = np.where(e >= segment.e_range[0], _sigma_millibarn(segment, e), sigma)
    sigma = np.where(e > 0.0, sigma, 0.0) * MILLIBARN_TO_M2
    return scalar_like(sigma, energy)

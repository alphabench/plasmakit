"""Maxwellian-averaged fusion reactivities.

Implements the Bosch-Hale (1992) parameterization (Nucl. Fusion 32, 611,
Eqs. 12-14), accurate to within 0.25% RMS over each reaction's fitted
temperature range.
"""

from __future__ import annotations

import numpy as np

from plasmakit import bosch_hale
from plasmakit.constants import ArrayLike, as_float64, scalar_like
from plasmakit.errors import warn_outside_range
from plasmakit.reactions import Reaction, reaction

_CM3_TO_M3 = 1.0e-6


def maxwellian_reactivity(fusion_reaction: str | Reaction, ion_temperature: ArrayLike) -> ArrayLike:
    """Maxwellian-averaged reactivity ``<sigma*v>``.

    Parameters
    ----------
    fusion_reaction : str or Reaction
        Reaction identifier (``"DT"``, ``"DDn"``, ``"DDp"``, ``"DHe3"``)
        or a :class:`~plasmakit.reactions.Reaction`.
    ion_temperature : float or ndarray
        Ion temperature in keV. Scalar or array (vectorized).

    Returns
    -------
    float or ndarray
        Reactivity in m^3/s, matching the shape of ``ion_temperature``.

    Warns
    -----
    ValidityRangeWarning
        For temperatures outside the Bosch-Hale fit range of the reaction.
    """
    rxn = reaction(fusion_reaction)
    coeff = bosch_hale.REACTIVITY[rxn.id]
    t = as_float64(ion_temperature)
    warn_outside_range(t, coeff.t_range, bosch_hale.MODEL_ID, "ion temperature")

    c1, c2, c3, c4, c5, c6, c7 = coeff.c
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = t / (1.0 - (t * (c2 + t * (c4 + t * c6))) / (1.0 + t * (c3 + t * (c5 + t * c7))))
        xi = (coeff.b_g**2 / (4.0 * theta)) ** (1.0 / 3.0)
        sigma_v = c1 * theta * np.sqrt(xi / (coeff.mrc2 * t**3)) * np.exp(-3.0 * xi)
    sigma_v = np.where(t > 0.0, sigma_v, 0.0) * _CM3_TO_M3
    return scalar_like(sigma_v, ion_temperature)

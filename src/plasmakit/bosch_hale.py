"""Bosch-Hale (1992) parameterization coefficients.

Reference: H.-S. Bosch and G.M. Hale, "Improved formulas for fusion
cross-sections and thermal reactivities", Nuclear Fusion 32 (1992) 611.

Cross-section coefficients are from Table IV (sigma in millibarn, E is the
center-of-mass energy in keV). Reactivity coefficients are from Table VII
(<sigma*v> in cm^3/s, T in keV). Unit conversion to SI happens in the
consuming modules, not here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

MODEL_ID = "bosch-hale-1992"


@dataclass(frozen=True, slots=True)
class ReactivityCoefficients:
    """Table VII coefficients for the Maxwellian reactivity fit of one reaction."""

    b_g: float
    """Gamow constant, keV^(1/2)."""
    mrc2: float
    """Reduced mass energy m_r c^2, keV."""
    c: tuple[float, float, float, float, float, float, float]
    """Fit coefficients C1..C7."""
    t_range: tuple[float, float]
    """Validity range of the fit in ion temperature, keV."""


@dataclass(frozen=True, slots=True)
class CrossSectionCoefficients:
    """Table IV coefficients for one energy segment of the S-function fit."""

    b_g: float
    """Gamow constant, keV^(1/2)."""
    a: tuple[float, float, float, float, float]
    """Numerator polynomial coefficients A1..A5."""
    b: tuple[float, float, float, float]
    """Denominator polynomial coefficients B1..B4."""
    e_range: tuple[float, float]
    """Validity range of the segment in center-of-mass energy, keV."""


REACTIVITY: Mapping[str, ReactivityCoefficients] = MappingProxyType(
    {
        "DT": ReactivityCoefficients(
            b_g=34.3827,
            mrc2=1_124_656.0,
            c=(1.17302e-9, 1.51361e-2, 7.51886e-2, 4.60643e-3, 1.35000e-2, -1.06750e-4, 1.36600e-5),
            t_range=(0.2, 100.0),
        ),
        "DDn": ReactivityCoefficients(
            b_g=31.3970,
            mrc2=937_814.0,
            c=(5.43360e-12, 5.85778e-3, 7.68222e-3, 0.0, -2.96400e-6, 0.0, 0.0),
            t_range=(0.2, 100.0),
        ),
        "DDp": ReactivityCoefficients(
            b_g=31.3970,
            mrc2=937_814.0,
            c=(5.65718e-12, 3.41267e-3, 1.99167e-3, 0.0, 1.05060e-5, 0.0, 0.0),
            t_range=(0.2, 100.0),
        ),
        "DHe3": ReactivityCoefficients(
            b_g=68.7508,
            mrc2=1_124_572.0,
            c=(5.51036e-10, 6.41918e-3, -2.02896e-3, -1.91080e-5, 1.35776e-4, 0.0, 0.0),
            t_range=(0.5, 190.0),
        ),
    }
)


CROSS_SECTION: Mapping[str, tuple[CrossSectionCoefficients, ...]] = MappingProxyType(
    {
        "DT": (
            CrossSectionCoefficients(
                b_g=34.3827,
                a=(6.927e4, 7.454e8, 2.050e6, 5.2002e4, 0.0),
                b=(6.38e1, -9.95e-1, 6.981e-5, 1.728e-4),
                e_range=(0.5, 550.0),
            ),
            CrossSectionCoefficients(
                b_g=34.3827,
                a=(-1.4714e6, 0.0, 0.0, 0.0, 0.0),
                b=(-8.4127e-3, 4.7983e-6, -1.0748e-9, 8.5184e-14),
                e_range=(550.0, 4700.0),
            ),
        ),
        "DDn": (
            CrossSectionCoefficients(
                b_g=31.3970,
                a=(5.3701e4, 3.3027e2, -1.2706e-1, 2.9327e-5, -2.5151e-9),
                b=(0.0, 0.0, 0.0, 0.0),
                e_range=(0.5, 4900.0),
            ),
        ),
        "DDp": (
            CrossSectionCoefficients(
                b_g=31.3970,
                a=(5.5576e4, 2.1054e2, -3.2638e-2, 1.4987e-6, 1.8181e-10),
                b=(0.0, 0.0, 0.0, 0.0),
                e_range=(0.5, 5000.0),
            ),
        ),
        "DHe3": (
            CrossSectionCoefficients(
                b_g=68.7508,
                a=(5.7501e6, 2.5226e3, 4.5566e1, 0.0, 0.0),
                b=(-3.1995e-3, -8.5530e-6, 5.9014e-8, 0.0),
                e_range=(0.3, 900.0),
            ),
        ),
    }
)

"""Parameterized axisymmetric tokamak flux-surface geometry.

Implements a Miller-like local parameterization (R.L. Miller, M.S. Chu,
J.M. Greene, Y.R. Lin-Liu and R.E. Waltz, Phys. Plasmas 5 (1998) 973)
with linear-in-rho triangularity and a parabolic Shafranov shift
``Delta(rho) = Delta0 * (1 - rho^2)``:

``R(rho, theta) = R0 + Delta(rho) + a*rho*cos(theta + arcsin(delta*rho)*sin(theta))``
``Z(rho, theta) = kappa * a * rho * sin(theta)``

This is a prescribed geometry, not a Grad-Shafranov equilibrium solution;
``rho`` labels nested surfaces on [0, 1] with rho = 1 the last closed flux
surface. Lengths are in metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fusionbench.constants import ArrayLike, as_float64, scalar_like
from fusionbench.errors import FusionbenchError

MODEL_ID = "miller-1998"


@dataclass(frozen=True)
class TokamakGeometry:
    """Nested Miller-like flux surfaces of an axisymmetric tokamak.

    Parameters
    ----------
    major_radius : float
        Geometric major radius R0 of the last closed flux surface, m.
    minor_radius : float
        Minor radius a, m.
    elongation : float
        Vertical elongation kappa.
    triangularity : float
        Triangularity delta of the last closed flux surface, |delta| < 1.
    shafranov_shift : float
        On-axis outward shift Delta0, m (the shift decays as 1 - rho^2).
    """

    major_radius: float
    minor_radius: float
    elongation: float = 1.0
    triangularity: float = 0.0
    shafranov_shift: float = 0.0

    def __post_init__(self) -> None:
        """Validate the shape parameters."""
        if self.major_radius <= 0.0:
            raise FusionbenchError("major_radius must be positive (m)")
        if self.minor_radius <= 0.0:
            raise FusionbenchError("minor_radius must be positive (m)")
        if self.elongation <= 0.0:
            raise FusionbenchError("elongation must be positive")
        if abs(self.triangularity) >= 1.0:
            raise FusionbenchError("triangularity must satisfy |delta| < 1")
        if self.major_radius + self.shafranov_shift - self.minor_radius <= 0.0:
            raise FusionbenchError("innermost surfaces must stay at R > 0")

    def flux_surface(
        self, rho: ArrayLike, theta: ArrayLike
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """(R, Z) coordinates (m) of the surface ``rho`` at poloidal angle ``theta``."""
        r, th = np.broadcast_arrays(as_float64(rho), as_float64(theta))
        x = np.arcsin(self.triangularity * r)
        big_r = (
            self.major_radius
            + self.shafranov_shift * (1.0 - r**2)
            + self.minor_radius * r * np.cos(th + x * np.sin(th))
        )
        z = self.elongation * self.minor_radius * r * np.sin(th)
        return big_r, z

    def jacobian(self, rho: ArrayLike, theta: ArrayLike) -> npt.NDArray[np.float64]:
        """Poloidal-plane Jacobian |dR/drho * dZ/dth - dR/dth * dZ/drho| (m^2/rad).

        Closed-form partial derivatives of the Miller parameterization; the
        circular limit gives ``kappa * a^2 * rho`` exactly.
        """
        r, th = np.broadcast_arrays(as_float64(rho), as_float64(theta))
        a = self.minor_radius
        delta = self.triangularity
        x = np.arcsin(delta * r)
        phi = th + x * np.sin(th)
        dx_drho = delta / np.sqrt(1.0 - (delta * r) ** 2)
        dr_dth = -a * r * np.sin(phi) * (1.0 + x * np.cos(th))
        dr_drho = (
            -2.0 * self.shafranov_shift * r
            + a * np.cos(phi)
            - a * r * np.sin(phi) * dx_drho * np.sin(th)
        )
        dz_dth = self.elongation * a * r * np.cos(th)
        dz_drho = self.elongation * a * np.sin(th)
        return np.abs(dr_drho * dz_dth - dr_dth * dz_drho)

    def dvolume_drho(self, rho: ArrayLike, n_theta: int = 256) -> ArrayLike:
        """Differential volume dV/drho (m^3) of the surface at ``rho``.

        ``dV/drho = 2*pi * integral_0^{2pi} R(rho, th) J(rho, th) dth``,
        integrated by the trapezoid rule on a uniform theta grid (the
        integrand is periodic, so convergence is spectral).
        """
        r = as_float64(rho)
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
        big_r, _ = self.flux_surface(r[..., None], theta)
        jac = self.jacobian(r[..., None], theta)
        integral = np.trapezoid(big_r * jac, theta, axis=-1)
        return scalar_like(2.0 * np.pi * integral, rho)

    def volume(self, rho: float = 1.0, *, n_rho: int = 257, n_theta: int = 256) -> float:
        """Plasma volume (m^3) enclosed by the surface ``rho``."""
        if not 0.0 <= rho <= 1.0:
            raise FusionbenchError("rho must lie in [0, 1]")
        grid = np.linspace(0.0, rho, n_rho)
        dv = as_float64(self.dvolume_drho(grid, n_theta=n_theta))
        return float(np.trapezoid(dv, grid))

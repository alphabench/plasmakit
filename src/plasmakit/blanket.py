"""Parameterized concentric-torus breeding blanket.

Layers are circular-cross-section torus shells centered on the plasma
major radius, ordered inward-out from the first wall. Miller shaping
(elongation, triangularity, Shafranov shift) cannot be represented by
circular tori; :meth:`Blanket.from_geometry` conservatively takes the
first-wall minor radius as the maximum poloidal-plane distance of the
last closed flux surface from ``(R0, 0)``, so shaped plasmas are fully
enclosed. This circular-shell approximation is recorded in provenance.
"""

from __future__ import annotations

import itertools
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from plasmakit.errors import PlasmakitError
from plasmakit.geometry import TokamakGeometry
from plasmakit.materials import Material

if TYPE_CHECKING:
    from plasmakit.neutronics import BlanketResult, SourceInput


def torus_shell_volume(major_radius: float, r_inner: float, r_outer: float) -> float:
    """Volume (m^3) of a circular torus shell: ``2 pi^2 R0 (r_out^2 - r_in^2)``."""
    if not 0.0 <= r_inner < r_outer:
        raise PlasmakitError("need 0 <= r_inner < r_outer")
    if r_outer >= major_radius:
        raise PlasmakitError("r_outer must be smaller than the major radius")
    return 2.0 * np.pi**2 * major_radius * (r_outer**2 - r_inner**2)


@dataclass(frozen=True)
class Layer:
    """One blanket layer.

    Parameters
    ----------
    name : str
        Unique layer name (used to key per-layer results).
    material : Material
        Homogeneous layer material.
    thickness : float
        Radial thickness, m.
    """

    name: str
    material: Material
    thickness: float

    def __post_init__(self) -> None:
        """Validate the layer."""
        if not self.name:
            raise PlasmakitError("layer name must be non-empty")
        if self.thickness <= 0.0:
            raise PlasmakitError("layer thickness must be positive (m)")


@dataclass(frozen=True)
class Blanket:
    """Concentric circular torus shells outward from the first wall.

    Parameters
    ----------
    layers : tuple of Layer
        Layers in inward-out order (first entry faces the plasma).
    major_radius : float
        Torus major radius R0, m.
    first_wall_radius : float
        Minor radius of the plasma-facing surface, m.
    """

    layers: tuple[Layer, ...]
    major_radius: float
    first_wall_radius: float

    def __post_init__(self) -> None:
        """Validate the radial build."""
        if not self.layers:
            raise PlasmakitError("blanket needs at least one layer")
        object.__setattr__(self, "layers", tuple(self.layers))
        names = [layer.name for layer in self.layers]
        if len(set(names)) != len(names):
            raise PlasmakitError(f"layer names must be unique, got {names}")
        if self.major_radius <= 0.0:
            raise PlasmakitError("major_radius must be positive (m)")
        if self.first_wall_radius <= 0.0:
            raise PlasmakitError("first_wall_radius must be positive (m)")
        outer = self.first_wall_radius + sum(layer.thickness for layer in self.layers)
        if outer >= self.major_radius:
            raise PlasmakitError(
                f"radial build extends to {outer:.3f} m, which reaches the "
                f"major radius {self.major_radius} m (torus would self-intersect)"
            )

    @classmethod
    def from_geometry(
        cls, layers: Sequence[Layer], geometry: TokamakGeometry, *, gap: float = 0.0
    ) -> Blanket:
        """Build a blanket enclosing the last closed flux surface of ``geometry``.

        The first-wall minor radius is the maximum poloidal-plane distance
        of the LCFS from ``(R0, 0)`` plus ``gap``. A shaped plasma
        (elongation != 1 or triangularity != 0) is approximated by
        circular shells; a ``UserWarning`` notes this.
        """
        theta = np.linspace(0.0, 2.0 * np.pi, 513)
        r, z = geometry.flux_surface(1.0, theta)
        distance = float(np.max(np.hypot(r - geometry.major_radius, z)))
        if geometry.elongation != 1.0 or geometry.triangularity != 0.0:
            warnings.warn(
                "shaped plasma approximated by circular blanket shells enclosing the LCFS",
                UserWarning,
                stacklevel=2,
            )
        return cls(
            layers=tuple(layers),
            major_radius=geometry.major_radius,
            first_wall_radius=distance + gap,
        )

    def boundaries(self) -> tuple[float, ...]:
        """Minor radii (m) of the ``n_layers + 1`` torus surfaces, inward-out."""
        radii = [self.first_wall_radius]
        for layer in self.layers:
            radii.append(radii[-1] + layer.thickness)
        return tuple(radii)

    def layer_volumes(self) -> tuple[float, ...]:
        """Analytic torus-shell volume (m^3) of each layer."""
        return tuple(
            torus_shell_volume(self.major_radius, inner, outer)
            for inner, outer in itertools.pairwise(self.boundaries())
        )

    def first_wall_area(self) -> float:
        """Torus surface area (m^2) of the first wall: ``4 pi^2 R0 r_fw``."""
        return 4.0 * np.pi**2 * self.major_radius * self.first_wall_radius

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe description for provenance records."""
        return {
            "major_radius": self.major_radius,
            "first_wall_radius": self.first_wall_radius,
            "layers": [
                {
                    "name": layer.name,
                    "thickness": layer.thickness,
                    "material": layer.material.to_dict(),
                }
                for layer in self.layers
            ],
        }

    def run_neutronics(
        self,
        source: SourceInput,
        *,
        particles: int = 100_000,
        batches: int = 10,
        seed: int = 1,
        source_rate: float | None = None,
        max_sources: int | None = 1000,
        displacement_energy_ev: float = 40.0,
        cwd: str | os.PathLike[str] | None = None,
    ) -> BlanketResult:
        """Run an OpenMC transport calculation for this blanket (requires openmc).

        See :func:`plasmakit.neutronics.run_neutronics` for parameters.
        """
        from plasmakit import neutronics

        return neutronics.run_neutronics(
            self,
            source,
            particles=particles,
            batches=batches,
            seed=seed,
            source_rate=source_rate,
            max_sources=max_sources,
            displacement_energy_ev=displacement_energy_ev,
            cwd=cwd,
        )

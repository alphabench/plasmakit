"""Spatially resolved neutron sources.

Turns plasma profiles on flux surfaces (or 2-D R-Z fields) into
cell-resolved neutron emissivity, power density, and exportable source
terms. All physics is delegated to the 0-D modules: each cell is an entry
of one array-valued :class:`~fusionbench.plasma.PlasmaState`.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from fusionbench import bosch_hale, spectra, vtk_io
from fusionbench.constants import as_float64
from fusionbench.errors import FusionbenchError
from fusionbench.geometry import MODEL_ID as GEOMETRY_MODEL_ID
from fusionbench.geometry import TokamakGeometry
from fusionbench.plasma import PlasmaState
from fusionbench.profiles import PlasmaProfiles
from fusionbench.provenance import Provenance, build_provenance
from fusionbench.rates import applicable_reactions, power_partition, reaction_rate_density


def _neutronic_reactions(fuel: Mapping[str, float]) -> tuple[str, ...]:
    ids = tuple(r.id for r in applicable_reactions(fuel) if r.neutronic)
    if not ids:
        raise FusionbenchError(f"fuel {dict(fuel)} produces no neutrons from registered reactions")
    return ids


@dataclass(frozen=True)
class SourceTerms:
    """Flattened per-cell, per-reaction ring source terms.

    Each entry describes one axisymmetric ring: position (R, Z), absolute
    neutron emission rate, and the local Brysk Gaussian energy spectrum.
    This is the exporter-independent representation used by
    :meth:`SpatialNeutronSource.to_openmc`.

    Attributes
    ----------
    r, z : ndarray
        Ring positions, m.
    strength : ndarray
        Neutron emission rate per ring, 1/s.
    energy_mean, energy_std : ndarray
        Brysk spectrum mean and standard deviation at the local ion
        temperature, keV.
    reaction_id : tuple of str
        Producing reaction of each ring.
    """

    r: npt.NDArray[np.float64]
    z: npt.NDArray[np.float64]
    strength: npt.NDArray[np.float64]
    energy_mean: npt.NDArray[np.float64]
    energy_std: npt.NDArray[np.float64]
    reaction_id: tuple[str, ...]


@dataclass(frozen=True)
class SpatialNeutronSource:
    """Cell-resolved neutron source on a structured axisymmetric grid.

    Built via :meth:`from_profiles` (flux-surface pathway) or
    :meth:`from_rz` (R-Z mesh pathway). All fields are cell-centered with
    shape ``(n1, n2)`` following ``dims``; ``r_corners``/``z_corners``
    hold the mesh nodes with shape ``(n1+1, n2+1)``.

    Units: positions m, volumes m^3, emissivity m^-3 s^-1 (neutrons),
    power density W/m^3, temperature keV, density m^-3.
    """

    r: npt.NDArray[np.float64]
    z: npt.NDArray[np.float64]
    r_corners: npt.NDArray[np.float64]
    z_corners: npt.NDArray[np.float64]
    volume: npt.NDArray[np.float64]
    emissivity: npt.NDArray[np.float64]
    emissivity_by_reaction: Mapping[str, npt.NDArray[np.float64]]
    power_density: npt.NDArray[np.float64]
    ion_temperature: npt.NDArray[np.float64]
    ion_density: npt.NDArray[np.float64]
    fuel: Mapping[str, float]
    dims: tuple[str, str]
    coords: Mapping[str, npt.NDArray[np.float64]]
    provenance: Provenance

    @property
    def total_rate(self) -> float:
        """Total neutron emission rate, 1/s."""
        return float(np.sum(self.emissivity * self.volume))

    @property
    def total_fusion_power(self) -> float:
        """Total fusion power, W."""
        return float(np.sum(self.power_density * self.volume))

    @classmethod
    def from_profiles(
        cls,
        profiles: PlasmaProfiles,
        geometry: TokamakGeometry,
        *,
        n_rho: int = 64,
        n_theta: int = 128,
    ) -> SpatialNeutronSource:
        """Build the source from radial profiles on flux surfaces.

        Parameters
        ----------
        profiles : PlasmaProfiles
            Ion temperature and density profiles with fuel composition.
        geometry : TokamakGeometry
            Flux-surface geometry the profiles live on.
        n_rho, n_theta : int
            Number of radial and poloidal cells (midpoint rule; volumes
            converge to the analytic value at second order).
        """
        rho_edges = np.linspace(0.0, 1.0, n_rho + 1)
        theta_edges = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
        rho_c = 0.5 * (rho_edges[:-1] + rho_edges[1:])
        theta_c = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        d_rho = rho_edges[1] - rho_edges[0]
        d_theta = theta_edges[1] - theta_edges[0]

        state = profiles.state_at(rho_c)
        shape = (n_rho, n_theta)

        def field(radial: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.broadcast_to(radial[:, None], shape).copy()

        by_reaction = {
            rid: field(as_float64(reaction_rate_density(state, rid)))
            for rid in _neutronic_reactions(profiles.fuel)
        }
        power = field(as_float64(power_partition(state).total))

        r_cells, z_cells = geometry.flux_surface(rho_c[:, None], theta_c[None, :])
        r_corners, z_corners = geometry.flux_surface(rho_edges[:, None], theta_edges[None, :])
        jac = geometry.jacobian(rho_c[:, None], theta_c[None, :])
        volume = 2.0 * np.pi * r_cells * jac * d_rho * d_theta

        provenance = build_provenance(
            models=[bosch_hale.MODEL_ID, spectra.MODEL_ID, GEOMETRY_MODEL_ID],
            inputs={
                "profiles": profiles.to_dict(),
                "geometry": asdict(geometry),
                "n_rho": n_rho,
                "n_theta": n_theta,
            },
        )
        return cls(
            r=r_cells,
            z=z_cells,
            r_corners=r_corners,
            z_corners=z_corners,
            volume=volume,
            emissivity=sum(by_reaction.values(), np.zeros(shape)),
            emissivity_by_reaction=MappingProxyType(by_reaction),
            power_density=power,
            ion_temperature=field(as_float64(state.ion_temperature)),
            ion_density=field(as_float64(state.ion_density)),
            fuel=MappingProxyType(dict(profiles.fuel)),
            dims=("rho", "theta"),
            coords=MappingProxyType({"rho": rho_c, "theta": theta_c}),
            provenance=provenance,
        )

    @classmethod
    def from_rz(
        cls,
        r_edges: npt.NDArray[np.float64],
        z_edges: npt.NDArray[np.float64],
        ion_temperature: npt.NDArray[np.float64],
        ion_density: npt.NDArray[np.float64],
        fuel: Mapping[str, float] | None = None,
    ) -> SpatialNeutronSource:
        """Build the source from 2-D fields on a rectilinear R-Z mesh.

        Parameters
        ----------
        r_edges, z_edges : ndarray
            Strictly increasing cell-edge coordinates, m (``r_edges >= 0``).
        ion_temperature, ion_density : ndarray
            Cell-centered fields with shape
            ``(len(r_edges) - 1, len(z_edges) - 1)``, keV and m^-3.
            Values must be positive everywhere: represent vacuum cells with
            floor values (e.g. 0.2 keV — the bottom of the Bosch-Hale
            D-T fit range, avoiding validity warnings — and 1 m^-3, whose
            emissivity is ~40 orders of magnitude below the core).
        fuel : mapping, optional
            Fuel-ion fractions; defaults to 50/50 D-T.
        """
        r_e = as_float64(r_edges)
        z_e = as_float64(z_edges)
        if r_e.ndim != 1 or z_e.ndim != 1 or r_e.size < 2 or z_e.size < 2:
            raise FusionbenchError("r_edges and z_edges must be 1-D with at least 2 points")
        if np.any(np.diff(r_e) <= 0.0) or np.any(np.diff(z_e) <= 0.0):
            raise FusionbenchError("r_edges and z_edges must be strictly increasing")
        if r_e[0] < 0.0:
            raise FusionbenchError("r_edges must be non-negative (m)")
        shape = (r_e.size - 1, z_e.size - 1)
        temperature = as_float64(ion_temperature)
        density = as_float64(ion_density)
        if temperature.shape != shape or density.shape != shape:
            raise FusionbenchError(
                f"fields must have cell shape {shape}, got {temperature.shape} and {density.shape}"
            )

        state = PlasmaState(
            ion_temperature=temperature,
            ion_density=density,
            fuel=dict(fuel) if fuel is not None else {"D": 0.5, "T": 0.5},
        )
        by_reaction = {
            rid: as_float64(reaction_rate_density(state, rid))
            for rid in _neutronic_reactions(state.fuel)
        }
        power = as_float64(power_partition(state).total)

        r_c = 0.5 * (r_e[:-1] + r_e[1:])
        z_c = 0.5 * (z_e[:-1] + z_e[1:])
        volume = (np.pi * (r_e[1:] ** 2 - r_e[:-1] ** 2))[:, None] * np.diff(z_e)[None, :]
        r_corners, z_corners = np.meshgrid(r_e, z_e, indexing="ij")

        provenance = build_provenance(
            models=[bosch_hale.MODEL_ID, spectra.MODEL_ID],
            inputs={
                "r_edges": r_e.tolist(),
                "z_edges": z_e.tolist(),
                "ion_temperature": temperature.tolist(),
                "ion_density": density.tolist(),
                "fuel": dict(state.fuel),
            },
        )
        return cls(
            r=np.broadcast_to(r_c[:, None], shape).copy(),
            z=np.broadcast_to(z_c[None, :], shape).copy(),
            r_corners=r_corners,
            z_corners=z_corners,
            volume=volume,
            emissivity=sum(by_reaction.values(), np.zeros(shape)),
            emissivity_by_reaction=MappingProxyType(by_reaction),
            power_density=power,
            ion_temperature=temperature,
            ion_density=density,
            fuel=state.fuel,
            dims=("r", "z"),
            coords=MappingProxyType({"r": r_c, "z": z_c}),
            provenance=provenance,
        )

    def source_terms(
        self, *, max_sources: int | None = None, min_strength_fraction: float = 0.0
    ) -> SourceTerms:
        """Flatten the source into per-cell, per-reaction ring terms.

        Parameters
        ----------
        max_sources : int, optional
            Keep only the strongest ``max_sources`` rings; a warning
            reports the discarded emission fraction.
        min_strength_fraction : float
            Drop rings weaker than this fraction of the total rate.
        """
        r_parts, z_parts, s_parts, mean_parts, std_parts, ids = [], [], [], [], [], []
        for rid, rate in self.emissivity_by_reaction.items():
            strength = (rate * self.volume).ravel()
            temperature = self.ion_temperature.ravel()
            r_parts.append(self.r.ravel())
            z_parts.append(self.z.ravel())
            s_parts.append(strength)
            mean_parts.append(as_float64(spectra.neutron_mean_energy(rid, temperature)))
            std_parts.append(as_float64(spectra.neutron_std(rid, temperature)))
            ids.extend([rid] * strength.size)

        r = np.concatenate(r_parts)
        z = np.concatenate(z_parts)
        strength = np.concatenate(s_parts)
        mean = np.concatenate(mean_parts)
        std = np.concatenate(std_parts)
        reaction_id = np.array(ids)

        total = float(np.sum(strength))
        keep = strength >= min_strength_fraction * total
        if max_sources is not None and int(np.sum(keep)) > max_sources:
            threshold = np.sort(strength[keep])[-max_sources]
            keep &= strength >= threshold
        kept = float(np.sum(strength[keep]))
        if kept < total * (1.0 - 1e-12):
            warnings.warn(
                f"truncated source terms discard {1.0 - kept / total:.2e} "
                "of the total neutron rate",
                UserWarning,
                stacklevel=2,
            )
        return SourceTerms(
            r=r[keep],
            z=z[keep],
            strength=strength[keep],
            energy_mean=mean[keep],
            energy_std=std[keep],
            reaction_id=tuple(reaction_id[keep]),
        )

    def to_vtk(self, path: str | os.PathLike[str]) -> None:
        """Write the source as a legacy ASCII VTK structured grid.

        The poloidal (R, Z) mesh becomes the grid plane; emissivity, power
        density, plasma fields, and cell volumes are written as cell data.
        Readable directly by ParaView/VisIt; no VTK dependency required.
        """
        cell_data = {
            "emissivity": self.emissivity,
            "power_density": self.power_density,
            "ion_temperature": self.ion_temperature,
            "ion_density": self.ion_density,
            "volume": self.volume,
        }
        vtk_io.write_structured_grid(path, self.r_corners, self.z_corners, cell_data)

    def to_xarray(self) -> Any:
        """Export the fields as an ``xarray.Dataset``.

        Requires the optional ``xarray`` dependency
        (``pip install fusionbench[xarray]``). Coordinates follow
        ``self.dims`` with 2-D ``R``/``Z`` coordinates attached; dataset
        attributes carry the integrated totals and the provenance record.
        """
        import xarray as xr

        coords: dict[str, Any] = {name: (name, values) for name, values in self.coords.items()}
        coords["R"] = (self.dims, self.r)
        coords["Z"] = (self.dims, self.z)
        data_vars = {
            "emissivity": (self.dims, self.emissivity, {"units": "m^-3 s^-1"}),
            "power_density": (self.dims, self.power_density, {"units": "W m^-3"}),
            "ion_temperature": (self.dims, self.ion_temperature, {"units": "keV"}),
            "ion_density": (self.dims, self.ion_density, {"units": "m^-3"}),
            "volume": (self.dims, self.volume, {"units": "m^3"}),
        }
        for rid, rate in self.emissivity_by_reaction.items():
            data_vars[f"emissivity_{rid}"] = (self.dims, rate, {"units": "m^-3 s^-1"})
        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "total_rate": self.total_rate,
                "total_fusion_power": self.total_fusion_power,
                "provenance": self.provenance.to_json(indent=None),
            },
        )

    def to_openmc(self, *, max_sources: int | None = 1000) -> list[Any]:
        """Export as a list of weighted ``openmc.IndependentSource`` rings.

        Requires the optional ``openmc`` dependency. One ring source is
        created per (cell, reaction): axisymmetric in phi, discrete in
        (r, z), isotropic, with a Gaussian energy spectrum at the local
        ion temperature. OpenMC uses cm and eV; conversion is handled
        here.

        Source strengths are normalized weights summing to 1 — multiply
        tallies by :attr:`total_rate` (neutrons/s) for absolute results.

        Parameters
        ----------
        max_sources : int, optional
            Cap on the number of rings (strongest kept); ``None`` exports
            every cell.
        """
        import openmc

        terms = self.source_terms(max_sources=max_sources)
        total = float(np.sum(terms.strength))
        sources = []
        for i in range(terms.strength.size):
            space = openmc.stats.CylindricalIndependent(
                r=openmc.stats.Discrete([terms.r[i] * 100.0], [1.0]),
                phi=openmc.stats.Uniform(0.0, 2.0 * np.pi),
                z=openmc.stats.Discrete([terms.z[i] * 100.0], [1.0]),
            )
            energy = openmc.stats.Normal(terms.energy_mean[i] * 1e3, terms.energy_std[i] * 1e3)
            sources.append(
                openmc.IndependentSource(
                    space=space,
                    angle=openmc.stats.Isotropic(),
                    energy=energy,
                    strength=float(terms.strength[i]) / total,
                )
            )
        return sources

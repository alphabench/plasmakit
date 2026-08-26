"""Blanket neutronics via OpenMC transport.

Couples a :class:`~plasmakit.blanket.Blanket` and a neutron source into
an OpenMC fixed-source calculation and post-processes the tallies into a
:class:`BlanketResult`: tritium breeding ratio, neutron wall load,
per-layer energy deposition and tritium production, and NRT displacement
damage.

Definitions
-----------
- Neutron wall load: energy carried by neutrons crossing from the plasma
  chamber into the first wall, divided by the first-wall torus area
  (MW/m^2). Includes backscattered re-entries; dominated by the
  uncollided first crossing.
- Heating uses OpenMC's local energy-deposition score (photon transport
  is not run; secondary-photon energy is deposited locally).
- DPA follows the NRT model (Norgett-Robinson-Torrens 1975):
  ``dpa/s = 0.8 * E_damage_rate / (2 * E_d) / N_atoms`` with the ASTM
  E693 displacement energy ``E_d = 40 eV`` for Fe/W by default.

All tallies are per source particle; results are normalized by the
absolute source rate (neutrons/s).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from plasmakit import bosch_hale, spectra
from plasmakit.blanket import Blanket
from plasmakit.constants import as_float64
from plasmakit.errors import PlasmakitError
from plasmakit.materials import MODEL_ID as MATERIALS_MODEL_ID
from plasmakit.plasma import PlasmaState
from plasmakit.provenance import Provenance, build_provenance
from plasmakit.rates import applicable_reactions, reaction_rate_density
from plasmakit.sources import NeutronSource
from plasmakit.spatial import SourceTerms, SpatialNeutronSource

MODEL_ID = "nrt-1975"

SECONDS_PER_FULL_POWER_YEAR: Final = 3.1536e7
EV_TO_JOULE: Final = 1.602176634e-19

WALL_LOAD_ENERGY_EDGES_EV: Final[npt.NDArray[np.float64]] = np.logspace(3.0, np.log10(2.0e7), 101)
"""Energy-bin edges (eV) for the first-wall current tally."""

SourceInput = SpatialNeutronSource | SourceTerms | NeutronSource | PlasmaState
"""Accepted neutron-source inputs for the transport coupling."""


@dataclass(frozen=True)
class TallyValue:
    """A Monte Carlo estimate: value with 1-sigma standard deviation."""

    value: float
    std_dev: float


def nrt_dpa_rate(
    damage_energy_ev_per_s: float, n_atoms: float, *, displacement_energy_ev: float = 40.0
) -> float:
    """NRT displacement rate in dpa/s.

    ``dpa/s = 0.8 * E_dam / (2 * E_d) / N_atoms``.

    Parameters
    ----------
    damage_energy_ev_per_s : float
        Total damage-energy production rate in the region, eV/s.
    n_atoms : float
        Number of atoms in the region.
    displacement_energy_ev : float
        Effective displacement threshold E_d; 40 eV is the ASTM E693
        convention for iron (also standard for tungsten).
    """
    if n_atoms <= 0.0:
        raise PlasmakitError("n_atoms must be positive")
    if displacement_energy_ev <= 0.0:
        raise PlasmakitError("displacement_energy_ev must be positive")
    return 0.8 * damage_energy_ev_per_s / (2.0 * displacement_energy_ev) / n_atoms


def wall_load_mw_per_m2(
    energy_ev: npt.NDArray[np.float64],
    current_per_sp: npt.NDArray[np.float64],
    current_std: npt.NDArray[np.float64],
    total_rate: float,
    area_m2: float,
) -> TallyValue:
    """Neutron wall loading (MW/m^2) from an energy-binned surface current.

    Parameters
    ----------
    energy_ev : ndarray
        Bin representative energies, eV.
    current_per_sp, current_std : ndarray
        Partial current per bin and its std, neutrons per source particle.
    total_rate : float
        Absolute source rate, neutrons/s.
    area_m2 : float
        Wall surface area, m^2.
    """
    if area_m2 <= 0.0:
        raise PlasmakitError("area_m2 must be positive")
    scale = total_rate * EV_TO_JOULE / area_m2 / 1.0e6
    value = float(np.sum(energy_ev * current_per_sp)) * scale
    std = float(np.sqrt(np.sum((energy_ev * current_std) ** 2))) * scale
    return TallyValue(value=value, std_dev=std)


@dataclass(frozen=True)
class BlanketResult:
    """Normalized blanket neutronics results with Monte Carlo uncertainties.

    Attributes
    ----------
    tbr : TallyValue
        Tritium breeding ratio, tritons per source neutron (all layers).
    neutron_wall_load : TallyValue
        Area-averaged first-wall neutron loading, MW/m^2.
    energy_deposition : mapping of str to TallyValue
        Deposited power per layer, W.
    tritium_production : mapping of str to TallyValue
        Tritium production rate per layer, atoms/s.
    dpa : TallyValue
        First-wall-layer displacement rate, dpa/s (NRT).
    dpa_per_fpy : TallyValue
        Displacement dose per full-power year, dpa.
    total_rate : float
        Source rate (n/s) used for normalization.
    particles, batches : int
        Monte Carlo run parameters.
    provenance : Provenance
        Full reproducibility record (chains the source provenance).
    """

    tbr: TallyValue
    neutron_wall_load: TallyValue
    energy_deposition: Mapping[str, TallyValue]
    tritium_production: Mapping[str, TallyValue]
    dpa: TallyValue
    dpa_per_fpy: TallyValue
    total_rate: float
    particles: int
    batches: int
    provenance: Provenance

    @classmethod
    def from_tallies(
        cls,
        *,
        blanket: Blanket,
        total_rate: float,
        particles: int,
        batches: int,
        seed: int,
        h3_per_layer: Sequence[tuple[float, float]],
        heating_ev_per_layer: Sequence[tuple[float, float]],
        damage_energy_ev: tuple[float, float],
        wall_current: tuple[
            npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
        ],
        displacement_energy_ev: float = 40.0,
        source_provenance: Provenance | None = None,
        openmc_version: str | None = None,
        cross_sections: str | None = None,
    ) -> BlanketResult:
        """Assemble a result from per-source-particle tally numbers.

        Pure arithmetic (no OpenMC required): layer sequences follow
        ``blanket.layers`` order; ``damage_energy_ev`` refers to the
        first (plasma-facing) layer; ``wall_current`` is
        ``(energies_ev, mean, std)`` per energy bin. Standard deviations
        combine in quadrature; tally correlations are ignored.
        """
        names = [layer.name for layer in blanket.layers]
        if len(h3_per_layer) != len(names) or len(heating_ev_per_layer) != len(names):
            raise PlasmakitError("per-layer tallies must match the number of layers")

        tbr_value = sum(mean for mean, _ in h3_per_layer)
        tbr_std = math.sqrt(sum(std**2 for _, std in h3_per_layer))

        tritium = {
            name: TallyValue(value=mean * total_rate, std_dev=std * total_rate)
            for name, (mean, std) in zip(names, h3_per_layer, strict=True)
        }
        heating = {
            name: TallyValue(
                value=mean * total_rate * EV_TO_JOULE, std_dev=std * total_rate * EV_TO_JOULE
            )
            for name, (mean, std) in zip(names, heating_ev_per_layer, strict=True)
        }

        first_wall = blanket.layers[0]
        n_atoms = first_wall.material.atom_density * blanket.layer_volumes()[0]
        dpa_rate = nrt_dpa_rate(
            damage_energy_ev[0] * total_rate,
            n_atoms,
            displacement_energy_ev=displacement_energy_ev,
        )
        dpa_std = nrt_dpa_rate(
            damage_energy_ev[1] * total_rate,
            n_atoms,
            displacement_energy_ev=displacement_energy_ev,
        )

        energies, current_mean, current_std = wall_current
        wall_load = wall_load_mw_per_m2(
            as_float64(energies),
            as_float64(current_mean),
            as_float64(current_std),
            total_rate,
            blanket.first_wall_area(),
        )

        models = [MODEL_ID, MATERIALS_MODEL_ID]
        if openmc_version is not None:
            models.insert(0, "openmc-2015")
        inputs: dict[str, Any] = {
            "blanket": blanket.to_dict(),
            "total_rate": total_rate,
            "particles": particles,
            "batches": batches,
            "seed": seed,
            "displacement_energy_ev": displacement_energy_ev,
            "openmc_version": openmc_version,
            "cross_sections": cross_sections,
        }
        if source_provenance is not None:
            inputs["source"] = json.loads(source_provenance.to_json())
        return cls(
            tbr=TallyValue(value=tbr_value, std_dev=tbr_std),
            neutron_wall_load=wall_load,
            energy_deposition=MappingProxyType(heating),
            tritium_production=MappingProxyType(tritium),
            dpa=TallyValue(value=dpa_rate, std_dev=dpa_std),
            dpa_per_fpy=TallyValue(
                value=dpa_rate * SECONDS_PER_FULL_POWER_YEAR,
                std_dev=dpa_std * SECONDS_PER_FULL_POWER_YEAR,
            ),
            total_rate=total_rate,
            particles=particles,
            batches=batches,
            provenance=build_provenance(models=models, inputs=inputs),
        )


def resolve_source_terms(
    source: SourceInput,
    blanket: Blanket,
    *,
    source_rate: float | None = None,
    max_sources: int | None = 1000,
) -> tuple[SourceTerms, float, Provenance | None]:
    """Reduce any accepted source input to ring terms plus an absolute rate.

    Parameters
    ----------
    source : SpatialNeutronSource, SourceTerms, NeutronSource, or PlasmaState
        The neutron source. 0-D inputs (``NeutronSource``/``PlasmaState``)
        carry only a rate *density*, so ``source_rate`` is then required
        and the source is placed as rings on the magnetic axis.
    blanket : Blanket
        Used to validate that every ring lies inside the first wall.
    source_rate : float, optional
        Absolute neutron rate (n/s); overrides the source's own total.
    max_sources : int, optional
        Ring-count cap for spatial sources.
    """
    provenance: Provenance | None = None
    if isinstance(source, SpatialNeutronSource):
        terms = source.source_terms(max_sources=max_sources)
        rate = source_rate if source_rate is not None else source.total_rate
        provenance = source.provenance
    elif isinstance(source, SourceTerms):
        terms = source
        rate = source_rate if source_rate is not None else float(np.sum(source.strength))
    else:
        state = source.plasma if isinstance(source, NeutronSource) else source
        if source_rate is None:
            raise PlasmakitError(
                "a 0-D plasma has only a neutron rate density; pass source_rate "
                "(neutrons/s) or provide a SpatialNeutronSource"
            )
        if np.ndim(state.ion_temperature) != 0:
            raise PlasmakitError("0-D source input requires a scalar ion_temperature")
        temperature = float(state.ion_temperature)
        neutronic = [r for r in applicable_reactions(state.fuel) if r.neutronic]
        if not neutronic:
            raise PlasmakitError(
                f"fuel {dict(state.fuel)} produces no neutrons from registered reactions"
            )
        weights = np.asarray([float(reaction_rate_density(state, r)) for r in neutronic])
        weights = weights / np.sum(weights)
        terms = SourceTerms(
            r=np.full(len(neutronic), blanket.major_radius),
            z=np.zeros(len(neutronic)),
            strength=weights * source_rate,
            energy_mean=np.asarray(
                [float(spectra.neutron_mean_energy(r, temperature)) for r in neutronic]
            ),
            energy_std=np.asarray([float(spectra.neutron_std(r, temperature)) for r in neutronic]),
            reaction_id=tuple(r.id for r in neutronic),
        )
        rate = source_rate
        provenance = build_provenance(
            models=[bosch_hale.MODEL_ID, spectra.MODEL_ID],
            inputs={"plasma": state.to_dict(), "source_rate": source_rate},
        )

    distance = np.hypot(terms.r - blanket.major_radius, terms.z)
    if np.any(distance >= blanket.first_wall_radius):
        raise PlasmakitError(
            "source rings must lie inside the first wall "
            f"(max distance {float(np.max(distance)):.3f} m >= "
            f"first_wall_radius {blanket.first_wall_radius} m)"
        )
    return terms, rate, provenance


def _terms_to_openmc(terms: SourceTerms) -> list[Any]:
    import openmc

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


def build_model(
    blanket: Blanket,
    source: SourceInput,
    *,
    particles: int = 100_000,
    batches: int = 10,
    seed: int = 1,
    source_rate: float | None = None,
    max_sources: int | None = 1000,
) -> Any:
    """Build the ``openmc.Model`` for a blanket calculation (requires openmc).

    Geometry: one circular ``openmc.ZTorus`` per layer boundary (cm), the
    outermost a vacuum boundary; a void plasma chamber cell plus one cell
    per layer. Tallies: per-layer H3 production and heating, first-wall
    damage energy, and an energy-binned plasma-to-wall partial current.
    """
    import openmc

    cm = 100.0
    r0 = blanket.major_radius * cm
    tori = [
        openmc.ZTorus(x0=0.0, y0=0.0, z0=0.0, a=r0, b=r * cm, c=r * cm)
        for r in blanket.boundaries()
    ]
    tori[-1].boundary_type = "vacuum"

    openmc_materials = [layer.material.to_openmc() for layer in blanket.layers]
    plasma_cell = openmc.Cell(name="plasma", fill=None, region=-tori[0])
    layer_cells = [
        openmc.Cell(name=layer.name, fill=material, region=+inner & -outer)
        for layer, material, inner, outer in zip(
            blanket.layers, openmc_materials, tori[:-1], tori[1:], strict=True
        )
    ]
    geometry = openmc.Geometry([plasma_cell, *layer_cells])

    terms, _, _ = resolve_source_terms(
        source, blanket, source_rate=source_rate, max_sources=max_sources
    )

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.particles = particles
    settings.batches = batches
    settings.seed = seed
    settings.source = _terms_to_openmc(terms)

    layer_filter = openmc.CellFilter(layer_cells)
    tritium_tally = openmc.Tally(name="tritium")
    tritium_tally.filters = [layer_filter]
    tritium_tally.scores = ["H3-production"]

    heating_tally = openmc.Tally(name="heating")
    heating_tally.filters = [layer_filter]
    heating_tally.scores = ["heating"]

    damage_tally = openmc.Tally(name="damage")
    damage_tally.filters = [openmc.CellFilter([layer_cells[0]])]
    damage_tally.scores = ["damage-energy"]

    wall_tally = openmc.Tally(name="wall_current")
    wall_tally.filters = [
        openmc.CellFromFilter(plasma_cell),
        openmc.SurfaceFilter(tori[0]),
        openmc.EnergyFilter(WALL_LOAD_ENERGY_EDGES_EV.tolist()),
    ]
    wall_tally.scores = ["current"]

    tallies = openmc.Tallies([tritium_tally, heating_tally, damage_tally, wall_tally])
    return openmc.Model(geometry=geometry, settings=settings, tallies=tallies)


def _cross_sections_id() -> str | None:
    import openmc

    try:
        path = openmc.config.get("cross_sections")
    except Exception:
        path = None
    return str(path) if path else os.environ.get("OPENMC_CROSS_SECTIONS")


def run_neutronics(
    blanket: Blanket,
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
    """Run OpenMC transport for a blanket and post-process the tallies.

    Parameters
    ----------
    blanket : Blanket
        The layered blanket.
    source : SpatialNeutronSource, SourceTerms, NeutronSource, or PlasmaState
        Neutron source; 0-D inputs require ``source_rate``.
    particles, batches, seed : int
        Monte Carlo run parameters.
    source_rate : float, optional
        Absolute neutron rate (n/s) for normalization; defaults to the
        source's own total rate where it has one.
    max_sources : int, optional
        Ring-count cap for spatial sources.
    displacement_energy_ev : float
        NRT displacement threshold for the first-wall DPA.
    cwd : path-like, optional
        Working directory for OpenMC files; a temporary directory by
        default.
    """
    import openmc

    terms, total_rate, source_provenance = resolve_source_terms(
        source, blanket, source_rate=source_rate, max_sources=max_sources
    )
    model = build_model(
        blanket,
        terms,
        particles=particles,
        batches=batches,
        seed=seed,
        source_rate=total_rate,
        max_sources=None,
    )

    def extract(sp_path: str | os.PathLike[str]) -> BlanketResult:
        with openmc.StatePoint(sp_path) as statepoint:
            tally = statepoint.get_tally(name="tritium")
            h3 = list(zip(tally.mean.ravel().tolist(), tally.std_dev.ravel().tolist(), strict=True))
            tally = statepoint.get_tally(name="heating")
            heat = list(
                zip(tally.mean.ravel().tolist(), tally.std_dev.ravel().tolist(), strict=True)
            )
            tally = statepoint.get_tally(name="damage")
            damage = (float(tally.mean.ravel()[0]), float(tally.std_dev.ravel()[0]))
            tally = statepoint.get_tally(name="wall_current")
            midpoints = np.sqrt(WALL_LOAD_ENERGY_EDGES_EV[:-1] * WALL_LOAD_ENERGY_EDGES_EV[1:])
            wall = (midpoints, tally.mean.ravel(), tally.std_dev.ravel())
        return BlanketResult.from_tallies(
            blanket=blanket,
            total_rate=total_rate,
            particles=particles,
            batches=batches,
            seed=seed,
            h3_per_layer=h3,
            heating_ev_per_layer=heat,
            damage_energy_ev=damage,
            wall_current=wall,
            displacement_energy_ev=displacement_energy_ev,
            source_provenance=source_provenance,
            openmc_version=str(openmc.__version__),
            cross_sections=_cross_sections_id(),
        )

    if cwd is not None:
        return extract(model.run(cwd=cwd))
    with tempfile.TemporaryDirectory() as tmp:
        return extract(model.run(cwd=tmp))

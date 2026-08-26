"""Fusion nuclear physics engine.

Connects plasma conditions to fusion reaction rates, neutron spectra, and
power densities, with every calculation validated against published
references (run :func:`validate`).

Quickstart
----------
>>> import plasmakit as fb
>>> plasma = fb.PlasmaState(ion_temperature=15.0, ion_density=1.0e20,
...                         fuel={"D": 0.5, "T": 0.5})
>>> source = fb.NeutronSource(plasma)
>>> rate = source.rate_density()          # neutrons / m^3 / s
>>> energy = source.mean_energy()         # keV, ~14070 for D-T

Spatially resolved sources from radial profiles on flux surfaces:

>>> profiles = fb.PlasmaProfiles(
...     ion_temperature=fb.RadialProfile.parabolic(20.0, 1.0),
...     ion_density=fb.RadialProfile.parabolic(1.0e20, 1.0e18),
... )
>>> geometry = fb.TokamakGeometry(major_radius=6.0, minor_radius=2.0,
...                               elongation=1.7, triangularity=0.33)
>>> spatial = fb.SpatialNeutronSource.from_profiles(profiles, geometry)
>>> total = spatial.total_rate            # neutrons / s

Units: temperatures and energies in keV, densities in m^-3, cross sections
in m^2, reactivities in m^3/s, power densities in W/m^3, lengths in m.
"""

from plasmakit.benchmarks import BenchmarkReport, validate
from plasmakit.blanket import Blanket, Layer
from plasmakit.cross_sections import cross_section
from plasmakit.distributions import Distribution
from plasmakit.estimation import Posterior, fit
from plasmakit.geometry import TokamakGeometry
from plasmakit.materials import MATERIALS, Material
from plasmakit.neutronics import BlanketResult, TallyValue
from plasmakit.optimization import OptimizationResult, optimize, optimize_surrogate
from plasmakit.plasma import PlasmaState
from plasmakit.profiles import PlasmaProfiles, RadialProfile
from plasmakit.provenance import Provenance
from plasmakit.rates import (
    PowerPartition,
    fusion_power_density,
    power_partition,
    reaction_rate_density,
)
from plasmakit.reactions import REACTIONS, Reaction
from plasmakit.reactivity import maxwellian_reactivity
from plasmakit.sources import NeutronSource
from plasmakit.spatial import SourceTerms, SpatialNeutronSource
from plasmakit.spectra import (
    NeutronSpectrum,
    neutron_mean_energy,
    neutron_spectrum,
    neutron_std,
)
from plasmakit.surrogates import GaussianProcess, Surrogate
from plasmakit.tritium import CycleHistory, TritiumCycle
from plasmakit.uncertainty import (
    SobolIndices,
    UncertainResult,
    propagate,
    propagate_transport,
    sobol_indices,
)

__version__ = "0.1.0"

__all__ = [
    "MATERIALS",
    "REACTIONS",
    "BenchmarkReport",
    "Blanket",
    "BlanketResult",
    "CycleHistory",
    "Distribution",
    "GaussianProcess",
    "Layer",
    "Material",
    "NeutronSource",
    "NeutronSpectrum",
    "OptimizationResult",
    "PlasmaProfiles",
    "PlasmaState",
    "Posterior",
    "PowerPartition",
    "Provenance",
    "RadialProfile",
    "Reaction",
    "SobolIndices",
    "SourceTerms",
    "SpatialNeutronSource",
    "Surrogate",
    "TallyValue",
    "TokamakGeometry",
    "TritiumCycle",
    "UncertainResult",
    "__version__",
    "cross_section",
    "fit",
    "fusion_power_density",
    "maxwellian_reactivity",
    "neutron_mean_energy",
    "neutron_spectrum",
    "neutron_std",
    "optimize",
    "optimize_surrogate",
    "power_partition",
    "propagate",
    "propagate_transport",
    "reaction_rate_density",
    "sobol_indices",
    "validate",
]

"""Fusion nuclear physics engine.

Connects plasma conditions to fusion reaction rates, neutron spectra, and
power densities, with every calculation validated against published
references (run :func:`validate`).

Quickstart
----------
>>> import fusionbench as fb
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

from fusionbench.benchmarks import BenchmarkReport, validate
from fusionbench.blanket import Blanket, Layer
from fusionbench.cross_sections import cross_section
from fusionbench.distributions import Distribution
from fusionbench.estimation import Posterior, fit
from fusionbench.geometry import TokamakGeometry
from fusionbench.materials import MATERIALS, Material
from fusionbench.neutronics import BlanketResult, TallyValue
from fusionbench.optimization import OptimizationResult, optimize, optimize_surrogate
from fusionbench.plasma import PlasmaState
from fusionbench.profiles import PlasmaProfiles, RadialProfile
from fusionbench.provenance import Provenance
from fusionbench.rates import (
    PowerPartition,
    fusion_power_density,
    power_partition,
    reaction_rate_density,
)
from fusionbench.reactions import REACTIONS, Reaction
from fusionbench.reactivity import maxwellian_reactivity
from fusionbench.sources import NeutronSource
from fusionbench.spatial import SourceTerms, SpatialNeutronSource
from fusionbench.spectra import (
    NeutronSpectrum,
    neutron_mean_energy,
    neutron_spectrum,
    neutron_std,
)
from fusionbench.surrogates import GaussianProcess, Surrogate
from fusionbench.uncertainty import (
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

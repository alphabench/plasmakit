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

Units: temperatures and energies in keV, densities in m^-3, cross sections
in m^2, reactivities in m^3/s, power densities in W/m^3.
"""

from fusionbench.benchmarks import BenchmarkReport, validate
from fusionbench.cross_sections import cross_section
from fusionbench.plasma import PlasmaState
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
from fusionbench.spectra import NeutronSpectrum, neutron_mean_energy, neutron_spectrum

__version__ = "0.1.0"

__all__ = [
    "REACTIONS",
    "BenchmarkReport",
    "NeutronSource",
    "NeutronSpectrum",
    "PlasmaState",
    "PowerPartition",
    "Provenance",
    "Reaction",
    "__version__",
    "cross_section",
    "fusion_power_density",
    "maxwellian_reactivity",
    "neutron_mean_energy",
    "neutron_spectrum",
    "power_partition",
    "reaction_rate_density",
    "validate",
]

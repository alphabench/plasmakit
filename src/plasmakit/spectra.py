"""Thermally broadened neutron energy spectra.

Implements the Gaussian model of H. Brysk, "Fusion neutron energies and
spectra", Plasma Physics 15 (1973) 611, for a Maxwellian plasma at ion
temperature T (keV):

- mean energy: ``<E_n> = E_0 + (3T/2) * m_n / (m_n + m_heavy)``
- variance: ``sigma_E^2 = 2 * m_n / (m_n + m_heavy) * E_0 * T``

where ``E_0`` is the cold-plasma neutron birth energy and ``m_heavy`` the
mass of the companion product. This yields the familiar widths
``FWHM_DT ~= 177 sqrt(T) keV`` and ``FWHM_DD ~= 82.5 sqrt(T) keV``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from plasmakit.constants import SPECIES_MASS_KEV, ArrayLike, as_float64, scalar_like
from plasmakit.errors import PlasmakitError
from plasmakit.reactions import Reaction, reaction

MODEL_ID = "brysk-1973"

_FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))


def _neutron_mass_fraction(rxn: Reaction) -> float:
    """m_n / (m_n + m_heavy) for the neutron's companion product."""
    heavy = rxn.products[1] if rxn.products[0] == "n" else rxn.products[0]
    m_n = SPECIES_MASS_KEV["n"]
    return m_n / (m_n + SPECIES_MASS_KEV[heavy])


def _require_neutronic(rxn: Reaction) -> None:
    if not rxn.neutronic:
        raise PlasmakitError(f"{rxn.label} is aneutronic; no neutron spectrum exists")


@dataclass(frozen=True)
class NeutronSpectrum:
    """Gaussian (Brysk) neutron energy spectrum at one ion temperature.

    Attributes
    ----------
    reaction : Reaction
        The neutron-producing fusion reaction.
    ion_temperature : float
        Ion temperature in keV.
    mean_energy : float
        Mean neutron energy in keV (thermally shifted).
    std : float
        Standard deviation of the spectrum in keV.
    """

    reaction: Reaction
    ion_temperature: float
    mean_energy: float
    std: float

    @property
    def fwhm(self) -> float:
        """Full width at half maximum in keV."""
        return _FWHM_PER_SIGMA * self.std

    def pdf(self, energy: ArrayLike) -> ArrayLike:
        """Evaluate the normalized probability density (1/keV) at the given energies (keV)."""
        e = as_float64(energy)
        norm = 1.0 / (self.std * math.sqrt(2.0 * math.pi))
        density = norm * np.exp(-0.5 * ((e - self.mean_energy) / self.std) ** 2)
        return scalar_like(density, energy)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> ArrayLike:
        """Draw ``n`` neutron energies (keV) from the spectrum."""
        generator = rng if rng is not None else np.random.default_rng()
        return generator.normal(self.mean_energy, self.std, size=n)


def neutron_mean_energy(fusion_reaction: str | Reaction, ion_temperature: ArrayLike) -> ArrayLike:
    """Mean neutron energy (keV) including the Brysk thermal shift.

    Parameters
    ----------
    fusion_reaction : str or Reaction
        A neutron-producing reaction (``"DT"`` or ``"DDn"``).
    ion_temperature : float or ndarray
        Ion temperature in keV.
    """
    rxn = reaction(fusion_reaction)
    _require_neutronic(rxn)
    t = as_float64(ion_temperature)
    assert rxn.neutron_energy is not None
    mean = rxn.neutron_energy + 1.5 * t * _neutron_mass_fraction(rxn)
    return scalar_like(mean, ion_temperature)


def neutron_std(fusion_reaction: str | Reaction, ion_temperature: ArrayLike) -> ArrayLike:
    """Compute the standard deviation (keV) of the Brysk neutron spectrum.

    Parameters
    ----------
    fusion_reaction : str or Reaction
        A neutron-producing reaction (``"DT"`` or ``"DDn"``).
    ion_temperature : float or ndarray
        Ion temperature in keV.
    """
    rxn = reaction(fusion_reaction)
    _require_neutronic(rxn)
    t = as_float64(ion_temperature)
    assert rxn.neutron_energy is not None
    std = np.sqrt(2.0 * _neutron_mass_fraction(rxn) * rxn.neutron_energy * t)
    return scalar_like(std, ion_temperature)


def neutron_spectrum(fusion_reaction: str | Reaction, ion_temperature: float) -> NeutronSpectrum:
    """Build the Brysk Gaussian spectrum for a reaction at one temperature.

    Parameters
    ----------
    fusion_reaction : str or Reaction
        A neutron-producing reaction (``"DT"`` or ``"DDn"``).
    ion_temperature : float
        Ion temperature in keV (scalar; one spectrum per temperature).
    """
    rxn = reaction(fusion_reaction)
    _require_neutronic(rxn)
    t = float(ion_temperature)
    if t <= 0.0:
        raise PlasmakitError("ion_temperature must be positive (keV)")
    assert rxn.neutron_energy is not None
    return NeutronSpectrum(
        reaction=rxn,
        ion_temperature=t,
        mean_energy=float(neutron_mean_energy(rxn, t)),
        std=float(neutron_std(rxn, t)),
    )

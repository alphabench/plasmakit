"""Neutron source derived from a plasma state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from fusionbench import bosch_hale, spectra
from fusionbench.constants import ArrayLike, as_float64, scalar_like
from fusionbench.errors import FusionbenchError
from fusionbench.plasma import PlasmaState
from fusionbench.provenance import Provenance, build_provenance
from fusionbench.rates import PowerPartition, applicable_reactions, power_partition
from fusionbench.rates import reaction_rate_density as _rate
from fusionbench.reactions import Reaction, reaction
from fusionbench.reactivity import maxwellian_reactivity
from fusionbench.spectra import NeutronSpectrum


@dataclass(frozen=True)
class NeutronSource:
    """Neutron source of a fusing plasma (0-D).

    A thin facade over the functional modules: reactivities, rate densities,
    spectra, and power partition all evaluated for ``plasma``.

    Parameters
    ----------
    plasma : PlasmaState
        The plasma producing the neutrons.
    """

    plasma: PlasmaState

    def _neutronic_reactions(self) -> tuple[Reaction, ...]:
        rxns = tuple(r for r in applicable_reactions(self.plasma.fuel) if r.neutronic)
        if not rxns:
            raise FusionbenchError(
                f"fuel {dict(self.plasma.fuel)} produces no neutrons from registered reactions"
            )
        return rxns

    def _primary_reaction(self) -> Reaction:
        """Return the dominant neutron-producing reaction for this fuel (DT over DDn)."""
        rxns = self._neutronic_reactions()
        by_id = {r.id: r for r in rxns}
        return by_id.get("DT") or by_id.get("DDn") or rxns[0]

    def reactivity(self, fusion_reaction: str | Reaction | None = None) -> ArrayLike:
        """Maxwellian reactivity ``<sigma*v>`` (m^3/s) at the plasma temperature.

        Parameters
        ----------
        fusion_reaction : str or Reaction, optional
            Defaults to the fuel's primary neutron-producing reaction.
        """
        rxn = self._primary_reaction() if fusion_reaction is None else reaction(fusion_reaction)
        return maxwellian_reactivity(rxn, self.plasma.ion_temperature)

    def rate_density(self, fusion_reaction: str | Reaction | None = None) -> ArrayLike:
        """Neutron production rate density (m^-3 s^-1).

        Parameters
        ----------
        fusion_reaction : str or Reaction, optional
            A single reaction, or ``None`` to sum every neutron-producing
            reaction available to the fuel.
        """
        if fusion_reaction is not None:
            return _rate(self.plasma, fusion_reaction)
        total = np.zeros(())
        for rxn in self._neutronic_reactions():
            total = total + as_float64(_rate(self.plasma, rxn))
        return scalar_like(total, self.plasma.ion_temperature, self.plasma.ion_density)

    def mean_energy(self, fusion_reaction: str | Reaction | None = None) -> ArrayLike:
        """Mean neutron energy (keV) including the thermal shift."""
        rxn = self._primary_reaction() if fusion_reaction is None else reaction(fusion_reaction)
        return spectra.neutron_mean_energy(rxn, self.plasma.ion_temperature)

    def spectrum(self, fusion_reaction: str | Reaction | None = None) -> NeutronSpectrum:
        """Thermally broadened neutron spectrum (scalar-temperature plasmas)."""
        rxn = self._primary_reaction() if fusion_reaction is None else reaction(fusion_reaction)
        if np.ndim(self.plasma.ion_temperature) != 0:
            raise FusionbenchError(
                "spectrum() requires a scalar ion_temperature; build spectra per point"
            )
        return spectra.neutron_spectrum(rxn, float(self.plasma.ion_temperature))

    def power_density(self) -> PowerPartition:
        """Fusion power density partition (W/m^3) over all applicable reactions."""
        return power_partition(self.plasma)

    @property
    def provenance(self) -> Provenance:
        """Reproducibility record for results derived from this source."""
        return build_provenance(
            models=[bosch_hale.MODEL_ID, spectra.MODEL_ID],
            inputs={"plasma": self.plasma.to_dict()},
        )

    def summary(self) -> dict[str, Any]:
        """JSON-safe summary: rates, mean energies, and power densities."""
        power = self.power_density()
        return {
            "neutron_rate_density": np.asarray(self.rate_density()).tolist(),
            "mean_energy": {
                r.id: np.asarray(self.mean_energy(r)).tolist() for r in self._neutronic_reactions()
            },
            "power_density": {
                "neutron": np.asarray(power.neutron).tolist(),
                "charged": np.asarray(power.charged).tolist(),
                "total": np.asarray(power.total).tolist(),
            },
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize :meth:`summary` plus provenance to JSON."""
        record = {"summary": self.summary(), "provenance": json.loads(self.provenance.to_json())}
        return json.dumps(record, indent=indent)

"""Reaction rate and fusion power densities.

The volumetric rate for reactants i, j is ``R = n_i n_j <sigma*v> / (1 + delta_ij)``,
where the Kronecker delta halves the rate for identical reactants (D-D).

Secondary reactions of fusion products (e.g. burnup of D-D tritons) are not
modelled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from fusionbench.constants import KEV_TO_JOULE, ArrayLike, as_float64, scalar_like
from fusionbench.errors import FusionbenchError
from fusionbench.plasma import PlasmaState
from fusionbench.reactions import REACTIONS, Reaction, reaction
from fusionbench.reactivity import maxwellian_reactivity


def applicable_reactions(fuel: Mapping[str, float]) -> tuple[Reaction, ...]:
    """Reactions whose reactants are both present in the given fuel."""
    present = {s for s, fraction in fuel.items() if fraction > 0.0}
    return tuple(r for r in REACTIONS.values() if set(r.reactants) <= present)


def reaction_rate_density(state: PlasmaState, fusion_reaction: str | Reaction) -> ArrayLike:
    """Volumetric reaction rate in m^-3 s^-1.

    Parameters
    ----------
    state : PlasmaState
        Plasma state supplying temperature, density, and fuel composition.
    fusion_reaction : str or Reaction
        Reaction identifier or :class:`~fusionbench.reactions.Reaction`.
    """
    rxn = reaction(fusion_reaction)
    n1 = as_float64(state.density(rxn.reactants[0]))
    n2 = as_float64(state.density(rxn.reactants[1]))
    sigma_v = as_float64(maxwellian_reactivity(rxn, state.ion_temperature))
    delta = 1.0 if rxn.identical_reactants else 0.0
    rate = n1 * n2 * sigma_v / (1.0 + delta)
    return scalar_like(rate, state.ion_temperature, state.ion_density)


@dataclass(frozen=True)
class PowerPartition:
    """Fusion power density split between neutrons and charged products (W/m^3)."""

    neutron: ArrayLike
    charged: ArrayLike

    @property
    def total(self) -> ArrayLike:
        """Total fusion power density, W/m^3."""
        return self.neutron + self.charged


def power_partition(
    state: PlasmaState, reactions: Sequence[str | Reaction] | None = None
) -> PowerPartition:
    """Neutron/charged split of the fusion power density.

    Parameters
    ----------
    state : PlasmaState
        Plasma state.
    reactions : sequence of str or Reaction, optional
        Reactions to include. Defaults to every registered reaction whose
        reactants are present in ``state.fuel``.

    Returns
    -------
    PowerPartition
        Birth-energy partition from two-body kinematics; slowing-down and
        transport are not modelled.
    """
    selected = (
        applicable_reactions(state.fuel)
        if reactions is None
        else tuple(reaction(r) for r in reactions)
    )
    if not selected:
        raise FusionbenchError(f"no applicable reactions for fuel {dict(state.fuel)}")
    neutron = np.zeros(())
    charged = np.zeros(())
    for rxn in selected:
        rate = as_float64(reaction_rate_density(state, rxn))
        neutron = neutron + rate * (rxn.neutron_energy or 0.0) * KEV_TO_JOULE
        charged = charged + rate * rxn.charged_energy * KEV_TO_JOULE
    inputs = (state.ion_temperature, state.ion_density)
    return PowerPartition(
        neutron=scalar_like(neutron, *inputs), charged=scalar_like(charged, *inputs)
    )


def fusion_power_density(
    state: PlasmaState, reactions: Sequence[str | Reaction] | None = None
) -> ArrayLike:
    """Total fusion power density in W/m^3 (see :func:`power_partition`)."""
    return power_partition(state, reactions).total

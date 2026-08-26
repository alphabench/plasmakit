"""Fusion reaction metadata.

Q values and two-body product energies are derived at import time from the
nuclide rest-mass energies in :mod:`fusionbench.constants`, so no product
energy is hand-copied. Product energies use non-relativistic two-body
kinematics for reactants at rest:

``E_product = Q * m_other / (m_product + m_other)``
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from fusionbench.constants import SPECIES_MASS_KEV
from fusionbench.errors import UnknownReactionError


@dataclass(frozen=True, slots=True)
class Reaction:
    """A fusion reaction and its energetics.

    Attributes
    ----------
    id : str
        Registry identifier, e.g. ``"DT"``.
    label : str
        Conventional nuclear notation, e.g. ``"T(d,n)4He"``.
    reactants : tuple of str
        Reactant species symbols.
    products : tuple of str
        Product species symbols.
    q_value : float
        Reaction Q value in keV.
    neutron_energy : float or None
        Birth energy (keV) of the emitted neutron for reactants at rest;
        ``None`` for aneutronic reactions.
    charged_energy : float
        Total kinetic energy (keV) carried by charged products.
    """

    id: str
    label: str
    reactants: tuple[str, str]
    products: tuple[str, str]
    q_value: float
    neutron_energy: float | None
    charged_energy: float

    @property
    def identical_reactants(self) -> bool:
        """Whether the two reactants are the same species (e.g. D-D)."""
        return self.reactants[0] == self.reactants[1]

    @property
    def neutronic(self) -> bool:
        """Whether the reaction emits a neutron."""
        return self.neutron_energy is not None

    def product_energy(self, species: str) -> float:
        """Birth kinetic energy (keV) of one product for reactants at rest.

        Parameters
        ----------
        species : str
            One of the two product species symbols.
        """
        if species not in self.products:
            raise ValueError(f"{species!r} is not a product of {self.label}")
        other = self.products[1] if species == self.products[0] else self.products[0]
        m_product = SPECIES_MASS_KEV[species]
        m_other = SPECIES_MASS_KEV[other]
        return self.q_value * m_other / (m_product + m_other)


def _make(id: str, label: str, reactants: tuple[str, str], products: tuple[str, str]) -> Reaction:
    q_value = sum(SPECIES_MASS_KEV[s] for s in reactants) - sum(
        SPECIES_MASS_KEV[s] for s in products
    )
    neutron_energy: float | None = None
    if "n" in products:
        other = products[1] if products[0] == "n" else products[0]
        m_n = SPECIES_MASS_KEV["n"]
        neutron_energy = q_value * SPECIES_MASS_KEV[other] / (m_n + SPECIES_MASS_KEV[other])
    charged_energy = q_value - (neutron_energy or 0.0)
    return Reaction(
        id=id,
        label=label,
        reactants=reactants,
        products=products,
        q_value=q_value,
        neutron_energy=neutron_energy,
        charged_energy=charged_energy,
    )


REACTIONS: Mapping[str, Reaction] = MappingProxyType(
    {
        "DT": _make("DT", "T(d,n)4He", ("D", "T"), ("n", "4He")),
        "DDn": _make("DDn", "D(d,n)3He", ("D", "D"), ("n", "3He")),
        "DDp": _make("DDp", "D(d,p)T", ("D", "D"), ("p", "T")),
        "DHe3": _make("DHe3", "3He(d,p)4He", ("D", "3He"), ("p", "4He")),
    }
)
"""Immutable registry of supported fusion reactions, keyed by identifier."""


def reaction(id: str | Reaction) -> Reaction:
    """Resolve a reaction identifier or pass a :class:`Reaction` through.

    Raises
    ------
    UnknownReactionError
        If ``id`` is not a registered reaction identifier.
    """
    if isinstance(id, Reaction):
        return id
    try:
        return REACTIONS[id]
    except KeyError:
        known = ", ".join(REACTIONS)
        raise UnknownReactionError(f"unknown reaction {id!r}; known reactions: {known}") from None

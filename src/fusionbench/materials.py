"""Fusion-relevant materials with literature compositions and densities.

A :class:`Material` is a homogeneous mixture described by mass density and
element/nuclide fractions. The registry provides standard blanket
materials (armor, structure, multiplier, breeders, coolants) with cited
densities and compositions. Lithium-bearing breeders take a Li-6
enrichment parameter, expressed as the atom fraction of Li-6 in lithium
(the convention in blanket-design literature).

Atomic-weight data: CIAAW 2021 standard atomic weights (nuclide masses
from AME2020 where individual isotopes are used).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal

from fusionbench.constants import AVOGADRO
from fusionbench.errors import FusionbenchError

MODEL_ID = "ciaaw-2021"

ATOMIC_MASS_U: Final[Mapping[str, float]] = MappingProxyType(
    {
        "H": 1.008,
        "He": 4.0026,
        "H3": 3.0160492,
        "Li": 6.94,
        "Li6": 6.0151228,
        "Li7": 7.0160034,
        "Be": 9.0121831,
        "C": 12.011,
        "O": 15.999,
        "Si": 28.085,
        "V": 50.9415,
        "Cr": 51.9961,
        "Mn": 54.938043,
        "Fe": 55.845,
        "Ta": 180.94788,
        "W": 183.84,
        "Pb": 207.2,
    }
)
"""Standard atomic weights (g/mol); nuclides (e.g. ``"Li6"``) use isotopic masses."""

LI6_NATURAL_ABUNDANCE: Final = 0.0759
"""Natural Li-6 atom fraction in lithium (IUPAC/CIAAW)."""


@dataclass(frozen=True)
class Material:
    """A homogeneous material: density plus element/nuclide composition.

    Parameters
    ----------
    name : str
        Material name.
    density : float
        Mass density, g/cm^3.
    composition : mapping of str to float
        Species (element symbol like ``"Fe"`` or nuclide like ``"Li6"``)
        to fraction. Fractions need not be normalized.
    percent_type : {"ao", "wo"}
        Whether fractions are atom ("ao") or weight ("wo") fractions.
    reference : str
        Citation for the density and composition.
    """

    name: str
    density: float
    composition: Mapping[str, float]
    percent_type: Literal["ao", "wo"] = "ao"
    reference: str = ""

    def __post_init__(self) -> None:
        """Validate and freeze the composition."""
        if not self.name:
            raise FusionbenchError("material name must be non-empty")
        if self.density <= 0.0:
            raise FusionbenchError("density must be positive (g/cm^3)")
        if not self.composition:
            raise FusionbenchError("composition must be non-empty")
        for species, fraction in self.composition.items():
            if species not in ATOMIC_MASS_U:
                known = ", ".join(ATOMIC_MASS_U)
                raise FusionbenchError(f"unknown species {species!r}; known: {known}")
            if fraction <= 0.0:
                raise FusionbenchError(f"fraction of {species!r} must be positive")
        if self.percent_type not in ("ao", "wo"):
            raise FusionbenchError("percent_type must be 'ao' or 'wo'")
        object.__setattr__(self, "composition", MappingProxyType(dict(self.composition)))

    def atom_fractions(self) -> dict[str, float]:
        """Return normalized atom fractions, converting from weight fractions if needed."""
        if self.percent_type == "ao":
            total = sum(self.composition.values())
            return {s: f / total for s, f in self.composition.items()}
        moles = {s: f / ATOMIC_MASS_U[s] for s, f in self.composition.items()}
        total = sum(moles.values())
        return {s: m / total for s, m in moles.items()}

    @property
    def mean_atomic_mass(self) -> float:
        """Mean molar mass per atom, g/mol."""
        return sum(x * ATOMIC_MASS_U[s] for s, x in self.atom_fractions().items())

    @property
    def atom_density(self) -> float:
        """Atom number density, atoms/m^3."""
        return self.density * 1.0e6 * AVOGADRO / self.mean_atomic_mass

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe description for provenance records."""
        return {
            "name": self.name,
            "density": self.density,
            "composition": dict(self.composition),
            "percent_type": self.percent_type,
            "reference": self.reference,
        }

    def to_openmc(self) -> Any:
        """Build the corresponding ``openmc.Material`` (requires openmc)."""
        import openmc

        material = openmc.Material(name=self.name)
        material.set_density("g/cm3", self.density)
        for species, fraction in self.composition.items():
            if any(ch.isdigit() for ch in species):
                material.add_nuclide(species, fraction, self.percent_type)
            else:
                material.add_element(species, fraction, self.percent_type)
        return material


def enriched_lithium(fraction: float, li6_enrichment: float) -> dict[str, float]:
    """Split a lithium atom fraction into Li6/Li7 by Li-6 enrichment.

    Parameters
    ----------
    fraction : float
        Total lithium atom fraction in the material.
    li6_enrichment : float
        Atom fraction of Li-6 in the lithium, in [0, 1].
    """
    if not 0.0 <= li6_enrichment <= 1.0:
        raise FusionbenchError("li6_enrichment must lie in [0, 1]")
    result = {}
    if li6_enrichment > 0.0:
        result["Li6"] = fraction * li6_enrichment
    if li6_enrichment < 1.0:
        result["Li7"] = fraction * (1.0 - li6_enrichment)
    return result


_CRC = "CRC Handbook of Chemistry and Physics, 97th ed."


def tungsten() -> Material:
    """Pure tungsten (plasma-facing armor)."""
    return Material(name="tungsten", density=19.30, composition={"W": 1.0}, reference=_CRC)


def beryllium() -> Material:
    """Pure beryllium (neutron multiplier)."""
    return Material(name="beryllium", density=1.848, composition={"Be": 1.0}, reference=_CRC)


def eurofer97() -> Material:
    """EUROFER97 reduced-activation ferritic-martensitic steel."""
    return Material(
        name="eurofer97",
        density=7.798,
        composition={
            "Fe": 0.8907,
            "Cr": 0.090,
            "W": 0.011,
            "Mn": 0.004,
            "V": 0.002,
            "Ta": 0.0012,
            "C": 0.0011,
        },
        percent_type="wo",
        reference=(
            "K. Mergia and N. Boukos, J. Nucl. Mater. 373 (2008) 1; "
            "R. Lindau et al., Fusion Eng. Des. 75-79 (2005) 989"
        ),
    )


def li4sio4(li6_enrichment: float = LI6_NATURAL_ABUNDANCE) -> Material:
    """Lithium orthosilicate ceramic breeder (Li4SiO4).

    Parameters
    ----------
    li6_enrichment : float
        Atom fraction of Li-6 in the lithium; defaults to natural.
    """
    composition = {**enriched_lithium(4.0 / 9.0, li6_enrichment), "Si": 1.0 / 9.0, "O": 4.0 / 9.0}
    return Material(
        name="li4sio4",
        density=2.40,
        composition=composition,
        reference="R. Knitter et al., J. Nucl. Mater. 442 (2013) S420 (theoretical density)",
    )


def pbli(li6_enrichment: float = 0.90) -> Material:
    """Lead-lithium eutectic breeder (Pb-17Li) at 573 K.

    Parameters
    ----------
    li6_enrichment : float
        Atom fraction of Li-6 in the lithium; 90% is the common design value.
    """
    composition = {"Pb": 0.83, **enriched_lithium(0.17, li6_enrichment)}
    return Material(
        name="pbli",
        density=9.84,
        composition=composition,
        reference=(
            "E. Mas de les Valls et al., J. Nucl. Mater. 376 (2008) 353 "
            "(rho = 10520.35 - 1.19051 T kg/m^3, T = 573 K); eutectic Pb-17Li"
        ),
    )


def water() -> Material:
    """Light water coolant at 20 C."""
    return Material(
        name="water",
        density=0.998,
        composition={"H": 2.0 / 3.0, "O": 1.0 / 3.0},
        reference=_CRC,
    )


def helium(density: float = 0.0057) -> Material:
    """Helium coolant; default density for 8 MPa at 673 K (HCPB-like state).

    Parameters
    ----------
    density : float
        Gas density in g/cm^3 at the operating state of interest.
    """
    return Material(
        name="helium",
        density=density,
        composition={"He": 1.0},
        reference="Ideal-gas estimate at 8 MPa, 673 K",
    )


MATERIALS: Final[Mapping[str, Callable[[], Material]]] = MappingProxyType(
    {
        "tungsten": tungsten,
        "beryllium": beryllium,
        "eurofer97": eurofer97,
        "li4sio4": li4sio4,
        "pbli": pbli,
        "water": water,
        "helium": helium,
    }
)
"""Registry of standard fusion materials, keyed by name (call to instantiate)."""

import json

import pytest

from plasmakit.errors import PlasmakitError
from plasmakit.materials import (
    LI6_NATURAL_ABUNDANCE,
    MATERIALS,
    Material,
    enriched_lithium,
    li4sio4,
    pbli,
    tungsten,
    water,
)


def test_registry_constructs():
    for name, factory in MATERIALS.items():
        material = factory()
        assert material.name == name
        assert material.density > 0.0
        assert material.reference
        assert abs(sum(material.atom_fractions().values()) - 1.0) < 1e-12


def test_weight_to_atom_conversion_water():
    # water given as weight fractions must recover the 2:1 atom ratio
    by_weight = Material(
        name="water-wo",
        density=0.998,
        composition={"H": 0.1119, "O": 0.8881},
        percent_type="wo",
    )
    fractions = by_weight.atom_fractions()
    assert fractions["H"] == pytest.approx(2.0 / 3.0, rel=1e-3)
    assert fractions["O"] == pytest.approx(1.0 / 3.0, rel=1e-3)
    reference = water().atom_fractions()
    assert fractions["H"] == pytest.approx(reference["H"], rel=1e-3)


def test_enriched_lithium():
    natural = enriched_lithium(1.0, LI6_NATURAL_ABUNDANCE)
    assert natural["Li6"] == pytest.approx(0.0759)
    assert natural["Li7"] == pytest.approx(0.9241)
    assert enriched_lithium(1.0, 1.0) == {"Li6": 1.0}
    assert enriched_lithium(1.0, 0.0) == {"Li7": 1.0}
    with pytest.raises(PlasmakitError):
        enriched_lithium(1.0, 1.5)
    with pytest.raises(PlasmakitError):
        enriched_lithium(1.0, -0.1)


def test_li4sio4_enrichment():
    material = li4sio4(li6_enrichment=0.60)
    fractions = material.atom_fractions()
    assert fractions["Li6"] == pytest.approx(4.0 / 9.0 * 0.60, rel=1e-12)
    assert fractions["Li6"] + fractions["Li7"] == pytest.approx(4.0 / 9.0, rel=1e-12)
    assert fractions["Si"] == pytest.approx(1.0 / 9.0, rel=1e-12)


def test_pbli_composition():
    fractions = pbli().atom_fractions()
    assert fractions["Pb"] == pytest.approx(0.83, rel=1e-12)
    assert fractions["Li6"] + fractions["Li7"] == pytest.approx(0.17, rel=1e-12)
    assert fractions["Li6"] == pytest.approx(0.17 * 0.90, rel=1e-12)


def test_tungsten_atom_density():
    material = tungsten()
    assert material.mean_atomic_mass == pytest.approx(183.84)
    # rho N_A / A = 19.30 * 1e6 * 6.022e23 / 183.84 g/mol
    assert material.atom_density == pytest.approx(6.322e28, rel=1e-3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "density": 1.0, "composition": {"Fe": 1.0}},
        {"name": "x", "density": -1.0, "composition": {"Fe": 1.0}},
        {"name": "x", "density": 1.0, "composition": {}},
        {"name": "x", "density": 1.0, "composition": {"Xx": 1.0}},
        {"name": "x", "density": 1.0, "composition": {"Fe": -0.5}},
        {"name": "x", "density": 1.0, "composition": {"Fe": 1.0}, "percent_type": "vol"},
    ],
)
def test_invalid_materials_rejected(kwargs):
    with pytest.raises(PlasmakitError):
        Material(**kwargs)


def test_to_dict_json_safe():
    json.dumps(li4sio4().to_dict())


def test_to_openmc():
    openmc = pytest.importorskip("openmc")
    material = li4sio4(li6_enrichment=0.60).to_openmc()
    assert isinstance(material, openmc.Material)
    nuclides = {n.name for n in material.nuclides}
    assert "Li6" in nuclides and "Li7" in nuclides
    assert material.get_mass_density() == pytest.approx(2.40, rel=1e-6)

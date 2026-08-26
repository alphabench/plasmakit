import pytest

from plasmakit.errors import UnknownReactionError
from plasmakit.reactions import REACTIONS, reaction


def test_registry_contents():
    assert set(REACTIONS) == {"DT", "DDn", "DDp", "DHe3"}


@pytest.mark.parametrize(
    ("rid", "q_mev"),
    [("DT", 17.589), ("DDn", 3.269), ("DDp", 4.033), ("DHe3", 18.353)],
)
def test_q_values(rid, q_mev):
    assert REACTIONS[rid].q_value == pytest.approx(q_mev * 1e3, rel=1e-3)


def test_neutron_energies_from_kinematics():
    assert REACTIONS["DT"].neutron_energy == pytest.approx(14_050.0, rel=1e-3)
    assert REACTIONS["DDn"].neutron_energy == pytest.approx(2_450.0, rel=1e-3)
    assert REACTIONS["DDp"].neutron_energy is None
    assert REACTIONS["DHe3"].neutron_energy is None


def test_product_energies():
    # D-D proton branch: p carries ~3.02 MeV, T ~1.01 MeV
    assert REACTIONS["DDp"].product_energy("p") == pytest.approx(3_022.0, rel=1e-3)
    assert REACTIONS["DDp"].product_energy("T") == pytest.approx(1_011.0, rel=1e-2)
    # D-3He proton: ~14.68 MeV
    assert REACTIONS["DHe3"].product_energy("p") == pytest.approx(14_680.0, rel=2e-3)
    with pytest.raises(ValueError):
        REACTIONS["DT"].product_energy("p")


def test_energy_conservation():
    for rxn in REACTIONS.values():
        total = sum(rxn.product_energy(s) for s in rxn.products)
        assert total == pytest.approx(rxn.q_value, rel=1e-12)
        assert rxn.charged_energy == pytest.approx(
            rxn.q_value - (rxn.neutron_energy or 0.0), rel=1e-12
        )


def test_identical_reactants_flag():
    assert REACTIONS["DDn"].identical_reactants
    assert not REACTIONS["DT"].identical_reactants


def test_reaction_lookup():
    assert reaction("DT") is REACTIONS["DT"]
    assert reaction(REACTIONS["DT"]) is REACTIONS["DT"]
    with pytest.raises(UnknownReactionError):
        reaction("TT")

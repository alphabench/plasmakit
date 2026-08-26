import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from plasmakit import (
    PlasmaState,
    fusion_power_density,
    maxwellian_reactivity,
    power_partition,
    reaction_rate_density,
)
from plasmakit.errors import PlasmakitError
from plasmakit.rates import applicable_reactions


def test_dt_rate_is_quarter_n_squared(dt_plasma):
    sv = maxwellian_reactivity("DT", 10.0)
    expected = (1.0e20**2 / 4.0) * sv
    assert reaction_rate_density(dt_plasma, "DT") == pytest.approx(expected, rel=1e-12)


def test_pure_d_rate_is_half_n_squared(dd_plasma):
    sv = maxwellian_reactivity("DDn", 10.0)
    expected = (1.0e20**2 / 2.0) * sv
    assert reaction_rate_density(dd_plasma, "DDn") == pytest.approx(expected, rel=1e-12)


def test_absent_species_rate_is_zero(dd_plasma):
    assert reaction_rate_density(dd_plasma, "DT") == 0.0


def test_applicable_reactions():
    assert {r.id for r in applicable_reactions({"D": 0.5, "T": 0.5})} == {"DT", "DDn", "DDp"}
    assert {r.id for r in applicable_reactions({"D": 1.0})} == {"DDn", "DDp"}
    assert {r.id for r in applicable_reactions({"D": 0.5, "3He": 0.5})} == {"DHe3", "DDn", "DDp"}


def test_no_applicable_reactions_raises():
    state = PlasmaState(ion_temperature=10.0, ion_density=1e20, fuel={"T": 1.0})
    with pytest.raises(PlasmakitError):
        power_partition(state)


def test_dt_neutron_power_fraction(dt_plasma):
    partition = power_partition(dt_plasma, reactions=["DT"])
    fraction = partition.neutron / partition.total
    assert fraction == pytest.approx(0.799, rel=1e-3)


def test_dd_branch_ratio_near_unity(dd_plasma):
    for t in (5.0, 10.0, 20.0, 50.0):
        ratio = maxwellian_reactivity("DDp", t) / maxwellian_reactivity("DDn", t)
        assert 0.8 <= ratio <= 1.2


def test_array_broadcast(dt_plasma):
    t = np.array([5.0, 10.0, 20.0])
    state = PlasmaState(ion_temperature=t, ion_density=1e20, fuel={"D": 0.5, "T": 0.5})
    rate = reaction_rate_density(state, "DT")
    assert rate.shape == (3,)
    power = power_partition(state)
    assert power.total.shape == (3,)


@given(
    st.floats(min_value=1.0, max_value=50.0),
    st.floats(min_value=1e18, max_value=1e21),
    st.floats(min_value=0.1, max_value=0.9),
)
def test_partition_sums_to_total(temperature, density, d_fraction):
    state = PlasmaState(
        ion_temperature=temperature,
        ion_density=density,
        fuel={"D": d_fraction, "T": 1.0 - d_fraction},
    )
    partition = power_partition(state)
    assert partition.total == pytest.approx(fusion_power_density(state), rel=1e-12)
    assert partition.neutron >= 0.0
    assert partition.charged > 0.0

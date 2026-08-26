import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from fusionbench.errors import FusionbenchError
from fusionbench.plasma import PlasmaState
from fusionbench.profiles import PlasmaProfiles, RadialProfile


def test_parabolic_endpoints():
    profile = RadialProfile.parabolic(center=20.0, edge=1.0)
    assert profile(0.0) == pytest.approx(20.0)
    assert profile(1.0) == pytest.approx(1.0)


def test_parabolic_exponent_peaking():
    broad = RadialProfile.parabolic(10.0, 1.0, exponent=0.5)
    peaked = RadialProfile.parabolic(10.0, 1.0, exponent=3.0)
    assert peaked(0.5) < broad(0.5)


def test_from_callable_exact_at_nodes():
    def f(rho):
        return 1.0 + rho**3

    profile = RadialProfile.from_callable(f, n_points=65)
    assert np.array_equal(np.asarray(profile(profile.rho)), profile.values)
    # linear interpolation between nodes stays close to the cubic
    assert profile(0.3) == pytest.approx(f(np.array(0.3)), rel=1e-3)


@pytest.mark.parametrize(
    ("rho", "values"),
    [
        (np.array([0.0, 0.5, 0.5, 1.0]), np.ones(4)),  # not strictly increasing
        (np.array([0.1, 0.5, 1.0]), np.ones(3)),  # does not start at 0
        (np.array([0.0, 0.5, 0.9]), np.ones(3)),  # does not end at 1
        (np.array([0.0, 1.0]), np.array([1.0, 0.0])),  # non-positive value
        (np.array([0.0, 0.5, 1.0]), np.ones(2)),  # shape mismatch
        (np.array([0.0]), np.array([1.0])),  # too few points
    ],
)
def test_invalid_profiles_rejected(rho, values):
    with pytest.raises(FusionbenchError):
        RadialProfile(rho=rho, values=values)


def test_call_outside_unit_interval_raises():
    profile = RadialProfile.parabolic(10.0, 1.0)
    with pytest.raises(FusionbenchError):
        profile(1.5)
    with pytest.raises(FusionbenchError):
        profile(np.array([0.5, -0.1]))


def test_scalar_array_contract():
    profile = RadialProfile.parabolic(10.0, 1.0)
    assert isinstance(profile(0.5), float)
    out = profile(np.array([0.0, 0.5, 1.0]))
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)


def test_state_at_matches_profiles():
    profiles = PlasmaProfiles(
        ion_temperature=RadialProfile.parabolic(20.0, 1.0),
        ion_density=RadialProfile.parabolic(1e20, 1e18),
    )
    rho = np.array([0.0, 0.3, 1.0])
    state = profiles.state_at(rho)
    assert isinstance(state, PlasmaState)
    assert np.array_equal(
        np.asarray(state.ion_temperature), np.asarray(profiles.ion_temperature(rho))
    )
    assert np.array_equal(np.asarray(state.ion_density), np.asarray(profiles.ion_density(rho)))
    assert dict(state.fuel) == {"D": 0.5, "T": 0.5}


def test_bad_fuel_rejected_eagerly():
    with pytest.raises(FusionbenchError):
        PlasmaProfiles(
            ion_temperature=RadialProfile.parabolic(20.0, 1.0),
            ion_density=RadialProfile.parabolic(1e20, 1e18),
            fuel={"D": 0.5, "T": 0.6},
        )


def test_to_dict_json_safe():
    import json

    profiles = PlasmaProfiles(
        ion_temperature=RadialProfile.parabolic(20.0, 1.0, n_points=5),
        ion_density=RadialProfile.parabolic(1e20, 1e18, n_points=5),
    )
    json.dumps(profiles.to_dict())


@given(
    st.floats(min_value=0.1, max_value=100.0),
    st.floats(min_value=0.1, max_value=100.0),
    st.floats(min_value=0.2, max_value=5.0),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_parabolic_bounded_by_center_and_edge(center, edge, exponent, rho):
    profile = RadialProfile.parabolic(center, edge, exponent=exponent)
    value = profile(rho)
    low, high = min(center, edge), max(center, edge)
    assert low * (1.0 - 1e-9) <= value <= high * (1.0 + 1e-9)

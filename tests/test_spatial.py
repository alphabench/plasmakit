import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from plasmakit import NeutronSource, PlasmaState, fusion_power_density
from plasmakit.errors import PlasmakitError
from plasmakit.geometry import TokamakGeometry
from plasmakit.profiles import PlasmaProfiles, RadialProfile
from plasmakit.spatial import SpatialNeutronSource


@pytest.fixture
def circular_geometry() -> TokamakGeometry:
    return TokamakGeometry(major_radius=6.0, minor_radius=2.0)


@pytest.fixture
def shaped_geometry() -> TokamakGeometry:
    return TokamakGeometry(6.0, 2.0, elongation=1.7, triangularity=0.33, shafranov_shift=0.3)


@pytest.fixture
def flat_profiles() -> PlasmaProfiles:
    return PlasmaProfiles(
        ion_temperature=RadialProfile.parabolic(10.0, 10.0),
        ion_density=RadialProfile.parabolic(1.0e20, 1.0e20),
    )


def test_flat_profile_reproduces_0d(circular_geometry, flat_profiles):
    source = SpatialNeutronSource.from_profiles(flat_profiles, circular_geometry)
    state = PlasmaState(ion_temperature=10.0, ion_density=1.0e20, fuel={"D": 0.5, "T": 0.5})
    rate_0d = NeutronSource(state).rate_density()
    assert np.all(source.emissivity == rate_0d)
    volume = circular_geometry.volume()
    assert source.total_rate == pytest.approx(rate_0d * volume, rel=1e-4)
    assert source.total_fusion_power == pytest.approx(
        fusion_power_density(state) * volume, rel=1e-4
    )


def test_cell_volumes_sum_to_plasma_volume(shaped_geometry, flat_profiles):
    source = SpatialNeutronSource.from_profiles(flat_profiles, shaped_geometry)
    assert np.sum(source.volume) == pytest.approx(shaped_geometry.volume(), rel=1e-4)


def test_emissivity_is_sum_of_reactions(circular_geometry, flat_profiles):
    source = SpatialNeutronSource.from_profiles(flat_profiles, circular_geometry)
    assert set(source.emissivity_by_reaction) == {"DT", "DDn"}
    stacked = sum(source.emissivity_by_reaction.values())
    assert np.array_equal(source.emissivity, stacked)


def test_peaked_profile_peaks_on_axis(circular_geometry):
    profiles = PlasmaProfiles(
        ion_temperature=RadialProfile.parabolic(20.0, 1.0),
        ion_density=RadialProfile.parabolic(1.0e20, 1.0e18),
    )
    source = SpatialNeutronSource.from_profiles(profiles, circular_geometry)
    assert np.argmax(source.emissivity[:, 0]) == 0
    assert source.emissivity[0, 0] > 1e3 * source.emissivity[-1, 0]


def test_from_rz_flat_box():
    r_edges = np.linspace(4.0, 8.0, 5)
    z_edges = np.linspace(-2.0, 2.0, 5)
    temperature = np.full((4, 4), 10.0)
    density = np.full((4, 4), 1.0e20)
    source = SpatialNeutronSource.from_rz(r_edges, z_edges, temperature, density)
    expected = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)[:, None] * 1.0
    assert np.allclose(source.volume, expected)
    state = PlasmaState(ion_temperature=10.0, ion_density=1.0e20, fuel={"D": 0.5, "T": 0.5})
    rate_0d = NeutronSource(state).rate_density()
    assert np.all(source.emissivity == rate_0d)
    total_volume = np.pi * (8.0**2 - 4.0**2) * 4.0
    assert source.total_rate == pytest.approx(rate_0d * total_volume, rel=1e-12)


def test_cross_pathway_consistency(circular_geometry, flat_profiles):
    torus = SpatialNeutronSource.from_profiles(flat_profiles, circular_geometry, n_rho=128)
    r_edges = np.linspace(3.99, 8.01, 201)
    z_edges = np.linspace(-2.01, 2.01, 201)
    r_c = 0.5 * (r_edges[:-1] + r_edges[1:])[:, None]
    z_c = 0.5 * (z_edges[:-1] + z_edges[1:])[None, :]
    inside = (r_c - 6.0) ** 2 + z_c**2 <= 4.0
    temperature = np.where(inside, 10.0, 0.2)
    density = np.where(inside, 1.0e20, 1.0)
    box = SpatialNeutronSource.from_rz(r_edges, z_edges, temperature, density)
    assert box.total_rate == pytest.approx(torus.total_rate, rel=2e-2)


@pytest.mark.parametrize(
    ("r_edges", "z_edges"),
    [
        (np.array([1.0, 0.5]), np.array([0.0, 1.0])),  # decreasing r
        (np.array([-1.0, 1.0]), np.array([0.0, 1.0])),  # negative r
        (np.array([1.0]), np.array([0.0, 1.0])),  # too few edges
    ],
)
def test_from_rz_invalid_edges(r_edges, z_edges):
    n_r, n_z = max(r_edges.size - 1, 1), max(z_edges.size - 1, 1)
    with pytest.raises(PlasmakitError):
        SpatialNeutronSource.from_rz(
            r_edges, z_edges, np.full((n_r, n_z), 10.0), np.full((n_r, n_z), 1e20)
        )


def test_from_rz_shape_mismatch():
    with pytest.raises(PlasmakitError):
        SpatialNeutronSource.from_rz(
            np.array([4.0, 5.0, 6.0]),
            np.array([0.0, 1.0]),
            np.full((3, 3), 10.0),
            np.full((2, 1), 1e20),
        )


def test_no_neutronic_fuel_raises(circular_geometry):
    profiles = PlasmaProfiles(
        ion_temperature=RadialProfile.parabolic(10.0, 1.0),
        ion_density=RadialProfile.parabolic(1e20, 1e18),
        fuel={"T": 1.0},
    )
    with pytest.raises(PlasmakitError):
        SpatialNeutronSource.from_profiles(profiles, circular_geometry)


def test_provenance(circular_geometry, flat_profiles):
    source = SpatialNeutronSource.from_profiles(flat_profiles, circular_geometry)
    assert "miller-1998" in source.provenance.models
    json.dumps(json.loads(source.provenance.to_json()))
    rz = SpatialNeutronSource.from_rz(
        np.array([4.0, 8.0]), np.array([-1.0, 1.0]), np.full((1, 1), 10.0), np.full((1, 1), 1e20)
    )
    assert "miller-1998" not in rz.provenance.models


@settings(max_examples=15, deadline=None)
@given(st.floats(min_value=0.1, max_value=10.0))
def test_density_scaling_is_quadratic(scale):
    geometry = TokamakGeometry(6.0, 2.0)
    base = PlasmaProfiles(
        ion_temperature=RadialProfile.parabolic(10.0, 1.0, n_points=17),
        ion_density=RadialProfile.parabolic(1.0e20, 1.0e18, n_points=17),
    )
    scaled = PlasmaProfiles(
        ion_temperature=base.ion_temperature,
        ion_density=RadialProfile(rho=base.ion_density.rho, values=base.ion_density.values * scale),
    )
    rate_base = SpatialNeutronSource.from_profiles(base, geometry, n_rho=8, n_theta=8).total_rate
    rate_scaled = SpatialNeutronSource.from_profiles(
        scaled, geometry, n_rho=8, n_theta=8
    ).total_rate
    assert rate_scaled == pytest.approx(rate_base * scale**2, rel=1e-9)

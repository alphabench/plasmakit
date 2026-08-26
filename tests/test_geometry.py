import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fusionbench.errors import FusionbenchError
from fusionbench.geometry import TokamakGeometry


def test_circular_volume_analytic():
    geometry = TokamakGeometry(major_radius=6.0, minor_radius=2.0)
    analytic = 2.0 * np.pi**2 * 6.0 * 2.0**2
    assert geometry.volume(n_rho=513, n_theta=512) == pytest.approx(analytic, rel=1e-6)


def test_elongated_volume_scales_with_kappa():
    geometry = TokamakGeometry(major_radius=6.0, minor_radius=2.0, elongation=1.7)
    analytic = 2.0 * np.pi**2 * 6.0 * 2.0**2 * 1.7
    assert geometry.volume(n_rho=513, n_theta=512) == pytest.approx(analytic, rel=1e-6)


@settings(max_examples=30, deadline=None)
@given(
    st.floats(min_value=2.0, max_value=10.0),
    st.floats(min_value=0.1, max_value=0.45),
    st.floats(min_value=0.5, max_value=2.5),
)
def test_volume_matches_analytic_ellipse(major_radius, aspect, elongation):
    minor_radius = aspect * major_radius
    geometry = TokamakGeometry(major_radius, minor_radius, elongation=elongation)
    analytic = 2.0 * np.pi**2 * major_radius * minor_radius**2 * elongation
    assert geometry.volume() == pytest.approx(analytic, rel=1e-4)


def test_triangularity_definition():
    geometry = TokamakGeometry(6.0, 2.0, elongation=1.7, triangularity=0.33)
    r, z = geometry.flux_surface(1.0, np.pi / 2.0)
    assert r == pytest.approx(6.0 - 2.0 * 0.33, abs=1e-12)
    assert z == pytest.approx(1.7 * 2.0, abs=1e-12)


def test_shafranov_shift():
    geometry = TokamakGeometry(6.0, 2.0, shafranov_shift=0.3)
    r_axis, z_axis = geometry.flux_surface(0.0, 1.234)
    assert r_axis == pytest.approx(6.3)
    assert z_axis == 0.0
    r_lcfs, _ = geometry.flux_surface(1.0, 0.0)
    assert r_lcfs == pytest.approx(8.0)


def test_circular_jacobian_analytic():
    geometry = TokamakGeometry(6.0, 2.0, elongation=1.7)
    rho, theta = 0.6, 1.1
    assert geometry.jacobian(rho, theta) == pytest.approx(1.7 * 4.0 * rho, rel=1e-12)


def test_jacobian_matches_finite_differences():
    geometry = TokamakGeometry(6.0, 2.0, elongation=1.7, triangularity=0.33, shafranov_shift=0.3)
    eps = 1e-7
    rng = np.random.default_rng(7)
    for rho in rng.uniform(0.05, 0.95, 5):
        for theta in rng.uniform(0.0, 2.0 * np.pi, 5):
            r_p, z_p = geometry.flux_surface(rho + eps, theta)
            r_m, z_m = geometry.flux_surface(rho - eps, theta)
            dr_drho, dz_drho = (r_p - r_m) / (2 * eps), (z_p - z_m) / (2 * eps)
            r_p, z_p = geometry.flux_surface(rho, theta + eps)
            r_m, z_m = geometry.flux_surface(rho, theta - eps)
            dr_dth, dz_dth = (r_p - r_m) / (2 * eps), (z_p - z_m) / (2 * eps)
            numeric = abs(dr_drho * dz_dth - dr_dth * dz_drho)
            assert geometry.jacobian(rho, theta) == pytest.approx(numeric, rel=1e-6)


def test_jacobian_positive_off_axis():
    geometry = TokamakGeometry(6.0, 2.0, elongation=1.7, triangularity=0.33)
    rho = np.linspace(0.05, 1.0, 20)[:, None]
    theta = np.linspace(0.0, 2.0 * np.pi, 40)[None, :]
    assert np.all(geometry.jacobian(rho, theta) > 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"major_radius": -1.0, "minor_radius": 1.0},
        {"major_radius": 6.0, "minor_radius": 0.0},
        {"major_radius": 6.0, "minor_radius": 2.0, "elongation": 0.0},
        {"major_radius": 6.0, "minor_radius": 2.0, "triangularity": 1.0},
        {"major_radius": 1.0, "minor_radius": 2.0},  # R0 + shift - a <= 0
    ],
)
def test_invalid_geometry_rejected(kwargs):
    with pytest.raises(FusionbenchError):
        TokamakGeometry(**kwargs)

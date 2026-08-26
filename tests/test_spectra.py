import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from plasmakit import neutron_mean_energy, neutron_spectrum
from plasmakit.errors import PlasmakitError


def test_dt_mean_energy_shifted_above_cold():
    cold = neutron_mean_energy("DT", 1e-9)
    hot = neutron_mean_energy("DT", 10.0)
    assert hot > cold
    assert cold == pytest.approx(14_048.0, rel=1e-3)


def test_fwhm_scaling():
    for temperature in (1.0, 10.0, 25.0):
        assert neutron_spectrum("DT", temperature).fwhm == pytest.approx(
            177.0 * np.sqrt(temperature), rel=2e-2
        )
        assert neutron_spectrum("DDn", temperature).fwhm == pytest.approx(
            82.5 * np.sqrt(temperature), rel=2e-2
        )


def test_pdf_normalized():
    spec = neutron_spectrum("DT", 10.0)
    e = np.linspace(spec.mean_energy - 6 * spec.std, spec.mean_energy + 6 * spec.std, 4001)
    assert np.trapezoid(np.asarray(spec.pdf(e)), e) == pytest.approx(1.0, abs=1e-6)


def test_sample_statistics():
    spec = neutron_spectrum("DT", 10.0)
    rng = np.random.default_rng(42)
    samples = spec.sample(200_000, rng=rng)
    assert np.mean(samples) == pytest.approx(spec.mean_energy, rel=1e-3)
    assert np.std(samples) == pytest.approx(spec.std, rel=1e-2)


def test_neutron_std_matches_spectrum():
    from plasmakit.spectra import neutron_std

    for temperature in (1.0, 10.0, 25.0):
        assert neutron_std("DT", temperature) == neutron_spectrum("DT", temperature).std
    t = np.array([1.0, 10.0])
    out = neutron_std("DDn", t)
    assert out.shape == (2,)
    assert isinstance(neutron_std("DDn", 10.0), float)


def test_aneutronic_raises():
    with pytest.raises(PlasmakitError):
        neutron_spectrum("DDp", 10.0)
    with pytest.raises(PlasmakitError):
        neutron_mean_energy("DHe3", 10.0)


def test_nonpositive_temperature_raises():
    with pytest.raises(PlasmakitError):
        neutron_spectrum("DT", 0.0)


def test_mean_energy_array_contract():
    t = np.array([5.0, 10.0])
    out = neutron_mean_energy("DT", t)
    assert out.shape == (2,)
    assert isinstance(neutron_mean_energy("DT", 10.0), float)


@given(st.floats(min_value=0.5, max_value=50.0), st.floats(min_value=10_000.0, max_value=18_000.0))
def test_pdf_nonnegative_and_symmetric(temperature, energy):
    spec = neutron_spectrum("DT", temperature)
    offset = energy - spec.mean_energy
    left = spec.pdf(spec.mean_energy - offset)
    right = spec.pdf(spec.mean_energy + offset)
    assert left >= 0.0
    assert left == pytest.approx(right, rel=1e-9, abs=1e-300)

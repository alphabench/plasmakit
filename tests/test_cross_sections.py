import numpy as np
import pytest

from plasmakit.constants import KEV_TO_JOULE
from plasmakit.cross_sections import cross_section
from plasmakit.errors import ValidityRangeWarning
from plasmakit.reactivity import maxwellian_reactivity

BARN_TO_M2 = 1.0e-28


def test_dt_peak_about_5_barn_near_64_kev():
    e = np.linspace(40.0, 90.0, 501)
    sigma = np.asarray(cross_section("DT", e))
    peak = e[np.argmax(sigma)]
    assert 60.0 <= peak <= 70.0
    assert np.max(sigma) == pytest.approx(5.0 * BARN_TO_M2, rel=5e-2)


def test_positive_in_range():
    for rid, e_max in [("DT", 4700.0), ("DDn", 4900.0), ("DDp", 5000.0), ("DHe3", 900.0)]:
        e = np.linspace(1.0, e_max, 300)
        assert np.all(np.asarray(cross_section(rid, e)) > 0.0)


def test_dt_segment_continuity_at_550():
    low = cross_section("DT", 549.9)
    high = cross_section("DT", 550.1)
    assert abs(high - low) / low < 1e-2


def test_out_of_range_warns():
    with pytest.warns(ValidityRangeWarning):
        cross_section("DT", 0.1)
    with pytest.warns(ValidityRangeWarning):
        cross_section("DT", 5000.0)


def test_scalar_and_array_contract():
    assert isinstance(cross_section("DT", 64.0), float)
    arr = cross_section("DT", np.array([10.0, 64.0]))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2,)


def test_reactivity_consistent_with_cross_section_integral():
    """Numerically integrate sigma(E) over a Maxwellian and compare to Bosch-Hale <sigma*v>.

    <sigma*v> = sqrt(8/(pi*mu)) * (kT)^(-3/2) * integral sigma(E) E exp(-E/kT) dE
    with E the center-of-mass energy. Cross-validates the two independent
    parameterizations (Table IV vs Table VII).
    """
    from plasmakit.bosch_hale import REACTIVITY

    temperature = 10.0  # keV
    for rid in ("DT", "DDn", "DDp"):
        mu_kg = REACTIVITY[rid].mrc2 * KEV_TO_JOULE / (2.99792458e8) ** 2
        e = np.linspace(0.5, 550.0 if rid == "DT" else 1000.0, 20_000)
        sigma = np.asarray(cross_section(rid, e))
        kt_j = temperature * KEV_TO_JOULE
        e_j = e * KEV_TO_JOULE
        integrand = sigma * e_j * np.exp(-e / temperature)
        integral = np.trapezoid(integrand, e_j)
        sv = np.sqrt(8.0 / (np.pi * mu_kg)) * kt_j ** (-1.5) * integral
        assert sv == pytest.approx(maxwellian_reactivity(rid, temperature), rel=2e-2)

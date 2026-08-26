import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from fusionbench.errors import ValidityRangeWarning
from fusionbench.reactivity import maxwellian_reactivity

# Bosch & Hale, Nucl. Fusion 32 (1992) 611, Table VIII (converted to m^3/s)
TABLE_VIII = [
    ("DT", 1.0, 6.857e-27),
    ("DT", 2.0, 2.977e-25),
    ("DT", 10.0, 1.136e-22),
    ("DT", 20.0, 4.330e-22),
    ("DT", 50.0, 8.649e-22),
    ("DDn", 10.0, 6.023e-25),
    ("DDn", 20.0, 2.603e-24),
    ("DDp", 10.0, 5.781e-25),
    ("DHe3", 1.0, 3.057e-32),
    ("DHe3", 10.0, 2.126e-25),
]


@pytest.mark.parametrize(("rid", "temperature", "expected"), TABLE_VIII)
def test_table_viii_anchors(rid, temperature, expected):
    assert maxwellian_reactivity(rid, temperature) == pytest.approx(expected, rel=1e-2)


def test_monotonic_below_peak():
    t = np.linspace(1.0, 60.0, 200)
    sv = np.asarray(maxwellian_reactivity("DT", t))
    assert np.all(np.diff(sv) > 0)


def test_peak_location():
    t = np.linspace(40.0, 100.0, 601)
    sv = np.asarray(maxwellian_reactivity("DT", t))
    assert 60.0 <= t[np.argmax(sv)] <= 70.0


@pytest.mark.parametrize("temperature", [0.05, 500.0])
def test_out_of_range_warns(temperature):
    with pytest.warns(ValidityRangeWarning):
        maxwellian_reactivity("DT", temperature)


def test_scalar_and_array_contract():
    scalar = maxwellian_reactivity("DT", 10.0)
    assert isinstance(scalar, float)
    arr = maxwellian_reactivity("DT", np.array([[10.0, 20.0]]))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 2)
    assert arr[0, 0] == pytest.approx(scalar)


@given(st.floats(min_value=0.5, max_value=100.0))
def test_nonnegative_finite_in_range(temperature):
    sv = maxwellian_reactivity("DT", temperature)
    assert np.isfinite(sv)
    assert sv > 0.0

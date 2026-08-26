import numpy as np

from fusionbench.constants import (
    ALPHA_MASS_KEV,
    DEUTERON_MASS_KEV,
    KEV_TO_JOULE,
    NEUTRON_MASS_KEV,
    SPECIES_MASS_KEV,
    as_float64,
    scalar_like,
)


def test_kev_to_joule_exact():
    assert KEV_TO_JOULE == 1.602176634e-16


def test_species_masses_registered():
    assert set(SPECIES_MASS_KEV) == {"n", "p", "D", "T", "3He", "4He"}
    assert SPECIES_MASS_KEV["n"] == NEUTRON_MASS_KEV
    assert SPECIES_MASS_KEV["4He"] == ALPHA_MASS_KEV


def test_mass_ordering():
    assert NEUTRON_MASS_KEV < DEUTERON_MASS_KEV < ALPHA_MASS_KEV


def test_as_float64_scalar_and_array():
    assert as_float64(3).dtype == np.float64
    arr = as_float64([1.0, 2.0])
    assert arr.dtype == np.float64
    assert arr.shape == (2,)


def test_scalar_like_contract():
    result = np.float64(1.5)
    assert isinstance(scalar_like(result, 2.0, 3.0), float)
    out = scalar_like(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 3.0)
    assert isinstance(out, np.ndarray)

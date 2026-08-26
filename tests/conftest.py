import pytest

from plasmakit import PlasmaState


@pytest.fixture
def dt_plasma() -> PlasmaState:
    return PlasmaState(ion_temperature=10.0, ion_density=1.0e20, fuel={"D": 0.5, "T": 0.5})


@pytest.fixture
def dd_plasma() -> PlasmaState:
    return PlasmaState(ion_temperature=10.0, ion_density=1.0e20, fuel={"D": 1.0})

import numpy as np
import pytest

from fusionbench import PlasmaState
from fusionbench.errors import FusionbenchError


def test_basic_construction(dt_plasma):
    assert dt_plasma.fuel["D"] == 0.5
    assert dt_plasma.density("D") == 5.0e19
    assert dt_plasma.density("3He") == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ion_temperature": -1.0, "ion_density": 1e20},
        {"ion_temperature": 0.0, "ion_density": 1e20},
        {"ion_temperature": 10.0, "ion_density": -1e20},
        {"ion_temperature": 10.0, "ion_density": 1e20, "fuel": {"D": 0.5, "T": 0.6}},
        {"ion_temperature": 10.0, "ion_density": 1e20, "fuel": {"D": 1.5, "T": -0.5}},
        {"ion_temperature": 10.0, "ion_density": 1e20, "fuel": {"D": 0.5, "X": 0.5}},
        {"ion_temperature": np.array([10.0, 20.0]), "ion_density": np.array([1e20, 1e20, 1e20])},
    ],
)
def test_invalid_states_rejected(kwargs):
    with pytest.raises(FusionbenchError):
        PlasmaState(**kwargs)


def test_immutability(dt_plasma):
    with pytest.raises(AttributeError):
        dt_plasma.ion_temperature = 20.0
    with pytest.raises(TypeError):
        dt_plasma.fuel["D"] = 0.9


def test_array_state():
    state = PlasmaState(
        ion_temperature=np.array([5.0, 10.0, 20.0]), ion_density=1e20, fuel={"D": 1.0}
    )
    density = state.density("D")
    assert isinstance(density, np.ndarray) or density == 1e20


def test_to_dict_json_safe(dt_plasma):
    import json

    d = dt_plasma.to_dict()
    json.dumps(d)
    assert d["fuel"] == {"D": 0.5, "T": 0.5}

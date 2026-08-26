import json

import pytest

from fusionbench.errors import FusionbenchError
from fusionbench.tritium import (
    SECONDS_PER_DAY,
    TRITIUM_DECAY_CONSTANT,
    CycleHistory,
    TritiumCycle,
    atoms_to_kg,
    kg_to_atoms,
)


@pytest.fixture
def cycle() -> TritiumCycle:
    return TritiumCycle(
        burn_rate=1.0e20,
        tbr=1.1,
        fractional_burnup=0.05,
        startup_inventory=5.0,
        extraction_efficiency=0.95,
        processing_loss=0.001,
    )


def test_atom_kg_round_trip():
    assert kg_to_atoms(atoms_to_kg(1.0e25)) == pytest.approx(1.0e25, rel=1e-12)
    # 1 kg of tritium ~ 2e26 atoms
    assert kg_to_atoms(1.0) == pytest.approx(1.0e3 / 3.0160492 * 6.02214076e23, rel=1e-9)


@pytest.mark.parametrize(
    "overrides",
    [
        {"burn_rate": 0.0},
        {"tbr": -0.1},
        {"fractional_burnup": 0.0},
        {"fractional_burnup": 1.5},
        {"startup_inventory": -1.0},
        {"blanket_residence_days": 0.0},
        {"exhaust_residence_days": -1.0},
        {"processing_residence_days": 0.0},
        {"extraction_efficiency": 0.0},
        {"extraction_efficiency": 1.5},
        {"processing_loss": 1.0},
        {"processing_loss": -0.1},
        {"reserve_inventory": -1.0},
        {"decay_constant": -1.0},
    ],
)
def test_invalid_parameters(overrides):
    kwargs = dict(burn_rate=1e20, tbr=1.1, fractional_burnup=0.05, startup_inventory=5.0)
    kwargs.update(overrides)
    with pytest.raises(FusionbenchError):
        TritiumCycle(**kwargs)


def test_steady_state_matches_closed_forms(cycle):
    lam = TRITIUM_DECAY_CONSTANT
    tau_b = cycle.blanket_residence_days * SECONDS_PER_DAY
    tau_p = cycle.exhaust_residence_days * SECONDS_PER_DAY
    tau_e = cycle.processing_residence_days * SECONDS_PER_DAY
    n_b, f_b = cycle.burn_rate, cycle.fractional_burnup
    i_b = cycle.tbr * n_b * tau_b / (1.0 + lam * tau_b)
    i_p = (1.0 - f_b) / f_b * n_b * tau_p / (1.0 + lam * tau_p)
    i_e = (cycle.extraction_efficiency * i_b / tau_b + i_p / tau_p) * tau_e / (1.0 + lam * tau_e)
    i_s = ((1.0 - cycle.processing_loss) * i_e / tau_e - n_b / f_b) / lam
    steady = cycle.steady_state()
    assert steady["blanket"] == pytest.approx(float(atoms_to_kg(i_b)), rel=1e-12)
    assert steady["exhaust"] == pytest.approx(float(atoms_to_kg(i_p)), rel=1e-12)
    assert steady["processing"] == pytest.approx(float(atoms_to_kg(i_e)), rel=1e-12)
    assert steady["storage"] == pytest.approx(float(atoms_to_kg(i_s)), rel=1e-12)


def test_long_simulation_reaches_steady_state():
    cycle = TritiumCycle(burn_rate=1e20, tbr=0.9, fractional_burnup=0.05, startup_inventory=50.0)
    steady = cycle.steady_state()
    history = cycle.simulate(days=365.0 * 300.0, n_points=3001)
    for name in ("blanket", "exhaust", "processing"):
        assert history.inventory(name)[-1] == pytest.approx(steady[name], rel=1e-6)


def test_simulate_endpoint_exactness(cycle):
    coarse = cycle.simulate(days=100.0, n_points=2)
    fine = cycle.simulate(days=100.0, n_points=1001)
    for name in ("blanket", "exhaust", "processing", "storage"):
        assert coarse.inventory(name)[-1] == pytest.approx(fine.inventory(name)[-1], rel=1e-10)


def test_zero_decay_simulates_but_no_steady_state():
    cycle = TritiumCycle(
        burn_rate=1e20,
        tbr=1.0,
        fractional_burnup=0.05,
        startup_inventory=5.0,
        decay_constant=0.0,
    )
    history = cycle.simulate(days=1000.0)
    # TBR=1 lossless without decay conserves total inventory exactly
    assert history.total()[-1] == pytest.approx(history.total()[0], rel=1e-9)
    with pytest.raises(FusionbenchError):
        cycle.steady_state()


def test_simulate_validation(cycle):
    with pytest.raises(FusionbenchError):
        cycle.simulate(days=0.0)
    with pytest.raises(FusionbenchError):
        cycle.simulate(days=10.0, n_points=1)


def test_history_interface(cycle):
    history = cycle.simulate(days=10.0, n_points=11)
    assert isinstance(history, CycleHistory)
    assert history.times.shape == (11,)
    assert history.inventory("storage")[0] == pytest.approx(5.0)
    with pytest.raises(FusionbenchError):
        history.inventory("divertor")
    json.dumps(history.to_dict())


def test_provenance_and_to_dict(cycle):
    assert set(cycle.provenance.models) == {
        "abdou-1986",
        "abdou-2021",
        "lucas-unterweger-2000",
    }
    json.dumps(cycle.to_dict())
    assert cycle.provenance.inputs["tbr"] == 1.1

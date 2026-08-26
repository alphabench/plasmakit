import json
import os

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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


def test_accumulation_rate_lossless_limit():
    # eta=1, eps=0, lam=0: a = (TBR - 1) * N_b exactly
    cycle = TritiumCycle(
        burn_rate=1e20,
        tbr=1.15,
        fractional_burnup=0.05,
        startup_inventory=1.0,
        decay_constant=0.0,
    )
    expected = 0.15 * 1e20 * float(atoms_to_kg(1.0)) * SECONDS_PER_DAY
    assert cycle.accumulation_rate() == pytest.approx(expected, rel=1e-12)
    assert cycle.self_sufficient


def test_deficit_cycle_not_self_sufficient():
    cycle = TritiumCycle(burn_rate=1e20, tbr=0.9, fractional_burnup=0.05, startup_inventory=5.0)
    assert cycle.accumulation_rate() < 0.0
    assert not cycle.self_sufficient
    assert cycle.doubling_time() is None


def test_doubling_time_matches_linear_limit():
    cycle = TritiumCycle(
        burn_rate=1e20,
        tbr=1.15,
        fractional_burnup=0.05,
        startup_inventory=1.0,
        blanket_residence_days=1e-4,
        exhaust_residence_days=1e-4,
        processing_residence_days=1e-4,
        decay_constant=0.0,
    )
    a_atoms = 0.15 * 1e20
    expected_days = float(kg_to_atoms(1.0)) / a_atoms / SECONDS_PER_DAY
    # exact value exceeds the naive linear estimate by the pipeline-fill
    # (in-transit inventory) lag, small for these short residence times
    assert cycle.doubling_time() == pytest.approx(expected_days, rel=1e-3)
    assert cycle.doubling_time() > expected_days


def test_doubling_time_decreases_with_tbr():
    def doubling(tbr):
        return TritiumCycle(
            burn_rate=1e20, tbr=tbr, fractional_burnup=0.05, startup_inventory=1.0
        ).doubling_time()

    fast, slow = doubling(1.3), doubling(1.05)
    assert fast is not None and slow is not None
    assert fast < slow


def test_doubling_time_requires_startup():
    cycle = TritiumCycle(burn_rate=1e20, tbr=1.2, fractional_burnup=0.05, startup_inventory=0.0)
    with pytest.raises(FusionbenchError):
        cycle.doubling_time()


def test_required_startup_inventory_re_simulated():
    from dataclasses import replace

    cycle = TritiumCycle(
        burn_rate=1e20,
        tbr=0.95,
        fractional_burnup=0.05,
        startup_inventory=0.0,
        reserve_inventory=0.5,
    )
    needed = cycle.required_startup_inventory(days=365.0)
    assert needed > 0.0
    refit = replace(cycle, startup_inventory=needed)
    history = refit.simulate(days=365.0, n_points=4001)
    assert float(np.min(history.inventory("storage"))) >= 0.5 * (1.0 - 1e-6) - 1e-9


def test_required_startup_grows_with_horizon_for_deficit():
    cycle = TritiumCycle(burn_rate=1e20, tbr=0.9, fractional_burnup=0.05, startup_inventory=0.0)
    one_year = cycle.required_startup_inventory(days=365.0)
    ten_years = cycle.required_startup_inventory(days=3650.0)
    assert 0.0 < one_year < ten_years


def test_required_startup_tiny_for_strong_breeder():
    # even a strong breeder needs the pipeline-fill (in-transit) inventory,
    # which shrinks with the residence times but is never exactly zero
    cycle = TritiumCycle(
        burn_rate=1e20,
        tbr=2.0,
        fractional_burnup=0.5,
        startup_inventory=0.0,
        blanket_residence_days=1e-3,
        exhaust_residence_days=1e-3,
        processing_residence_days=1e-3,
    )
    needed = cycle.required_startup_inventory(days=365.0)
    assert 0.0 <= needed < 1e-3  # kg; ~1e-4 kg of in-transit fuel
    slower = TritiumCycle(
        burn_rate=1e20,
        tbr=2.0,
        fractional_burnup=0.5,
        startup_inventory=0.0,
        blanket_residence_days=1.0,
        exhaust_residence_days=1.0,
        processing_residence_days=1.0,
    )
    assert slower.required_startup_inventory(days=365.0) > needed


@settings(max_examples=15, deadline=None)
@given(
    st.floats(min_value=0.85, max_value=1.4),
    st.floats(min_value=0.005, max_value=0.5),
    st.floats(min_value=0.8, max_value=1.0),
    st.floats(min_value=0.0, max_value=0.1),
)
def test_inventories_nonnegative_from_required_start(tbr, f_b, eta, eps):
    cycle = TritiumCycle(
        burn_rate=1e20,
        tbr=tbr,
        fractional_burnup=f_b,
        startup_inventory=0.0,
        extraction_efficiency=eta,
        processing_loss=eps,
    )
    needed = cycle.required_startup_inventory(days=365.0)
    from dataclasses import replace

    history = replace(cycle, startup_inventory=needed).simulate(days=365.0)
    for name in ("blanket", "exhaust", "processing"):
        assert float(np.min(history.inventory(name))) >= -1e-12
    assert float(np.min(history.inventory("storage"))) >= -1e-9


@settings(max_examples=10, deadline=None)
@given(st.floats(min_value=0.9, max_value=1.3))
def test_required_startup_monotone_in_tbr(tbr):
    def needed(t):
        return TritiumCycle(
            burn_rate=1e20, tbr=t, fractional_burnup=0.02, startup_inventory=0.0
        ).required_startup_inventory(days=730.0)

    assert needed(tbr + 0.05) <= needed(tbr) * (1.0 + 1e-9) + 1e-12


def test_from_fusion_power():
    from fusionbench.constants import KEV_TO_JOULE
    from fusionbench.reactions import REACTIONS

    cycle = TritiumCycle.from_fusion_power(
        500.0, tbr=1.1, fractional_burnup=0.05, startup_inventory=5.0
    )
    expected = 500.0e6 / (REACTIONS["DT"].q_value * KEV_TO_JOULE)
    assert cycle.burn_rate == pytest.approx(expected, rel=1e-12)
    with pytest.raises(FusionbenchError):
        TritiumCycle.from_fusion_power(-1.0, tbr=1.1, fractional_burnup=0.05, startup_inventory=5.0)


def test_from_blanket_result():
    from fusionbench.blanket import Blanket, Layer
    from fusionbench.materials import eurofer97, li4sio4
    from fusionbench.neutronics import BlanketResult

    blanket = Blanket(
        layers=(
            Layer("first_wall", eurofer97(), 0.02),
            Layer("breeder", li4sio4(li6_enrichment=0.6), 0.5),
        ),
        major_radius=6.0,
        first_wall_radius=2.0,
    )
    result = BlanketResult.from_tallies(
        blanket=blanket,
        total_rate=2.0e20,
        particles=1000,
        batches=5,
        seed=1,
        h3_per_layer=[(0.02, 0.001), (1.05, 0.02)],
        heating_ev_per_layer=[(1.0e6, 1.0e4), (1.4e7, 1.0e5)],
        damage_energy_ev=(5.0e5, 1.0e4),
        wall_current=(np.array([14.07e6]), np.array([1.0]), np.array([0.05])),
    )
    cycle = TritiumCycle.from_blanket_result(result, fractional_burnup=0.05, startup_inventory=5.0)
    assert cycle.burn_rate == 2.0e20
    assert cycle.tbr == pytest.approx(1.07)
    assert cycle.self_sufficient


def test_uq_pattern_over_cycle_parameters():
    from fusionbench.distributions import Distribution
    from fusionbench.uncertainty import propagate

    def required(tbr, fractional_burnup):
        return TritiumCycle(
            burn_rate=1e20,
            tbr=tbr,
            fractional_burnup=fractional_burnup,
            startup_inventory=0.0,
        ).required_startup_inventory(days=365.0)

    result = propagate(
        required,
        {
            "tbr": Distribution.normal(1.05, 0.02),
            "fractional_burnup": Distribution.uniform(0.01, 0.05),
        },
        n_samples=64,
        seed=0,
    )
    assert np.isfinite(result.mean) and result.mean > 0.0
    repeat = propagate(
        required,
        {
            "tbr": Distribution.normal(1.05, 0.02),
            "fractional_burnup": Distribution.uniform(0.01, 0.05),
        },
        n_samples=64,
        seed=0,
    )
    assert np.array_equal(result.samples, repeat.samples)


def test_provenance_and_to_dict(cycle):
    assert set(cycle.provenance.models) == {
        "abdou-1986",
        "abdou-2021",
        "lucas-unterweger-2000",
    }
    json.dumps(cycle.to_dict())
    assert cycle.provenance.inputs["tbr"] == 1.1


# --- Transport integration ---------------------------------------------------

needs_data = pytest.mark.skipif(
    not os.environ.get("OPENMC_CROSS_SECTIONS"), reason="OPENMC_CROSS_SECTIONS not set"
)


@pytest.mark.transport
@needs_data
def test_fuel_cycle_from_transport():
    pytest.importorskip("openmc")
    from fusionbench.blanket import Blanket, Layer
    from fusionbench.materials import eurofer97, li4sio4
    from fusionbench.plasma import PlasmaState

    blanket = Blanket(
        layers=(
            Layer("first_wall", eurofer97(), 0.02),
            Layer("breeder", li4sio4(), 0.5),  # natural lithium: TBR < 1
        ),
        major_radius=9.0,
        first_wall_radius=2.9,
    )
    plasma = PlasmaState(ion_temperature=15.0, ion_density=1.0e20, fuel={"D": 0.5, "T": 0.5})
    result = blanket.run_neutronics(plasma, particles=5000, batches=5, seed=1, source_rate=1.0e20)
    assert result.tbr.value < 1.0
    cycle = TritiumCycle.from_blanket_result(result, fractional_burnup=0.02, startup_inventory=5.0)
    assert not cycle.self_sufficient
    assert cycle.doubling_time() is None
    one_year = TritiumCycle.from_blanket_result(
        result, fractional_burnup=0.02, startup_inventory=0.0
    ).required_startup_inventory(days=365.0)
    ten_years = TritiumCycle.from_blanket_result(
        result, fractional_burnup=0.02, startup_inventory=0.0
    ).required_startup_inventory(days=3650.0)
    assert 0.0 < one_year < ten_years

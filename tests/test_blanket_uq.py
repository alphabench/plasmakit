"""End-to-end uncertainty quantification through OpenMC transport."""

import os

import pytest

from plasmakit.blanket import Blanket, Layer
from plasmakit.distributions import Distribution
from plasmakit.materials import eurofer97, li4sio4
from plasmakit.neutronics import TallyValue
from plasmakit.optimization import optimize_surrogate
from plasmakit.plasma import PlasmaState
from plasmakit.uncertainty import propagate_transport

needs_data = pytest.mark.skipif(
    not os.environ.get("OPENMC_CROSS_SECTIONS"), reason="OPENMC_CROSS_SECTIONS not set"
)


def _tbr(ion_temperature: float, li6_enrichment: float, particles: int = 2000) -> TallyValue:
    blanket = Blanket(
        layers=(
            Layer("first_wall", eurofer97(), 0.02),
            Layer("breeder", li4sio4(li6_enrichment=li6_enrichment), 0.50),
        ),
        major_radius=9.0,
        first_wall_radius=2.9,
    )
    plasma = PlasmaState(
        ion_temperature=ion_temperature, ion_density=1.0e20, fuel={"D": 0.5, "T": 0.5}
    )
    result = blanket.run_neutronics(
        plasma, particles=particles, batches=5, seed=1, source_rate=1.0e20
    )
    return result.tbr


@pytest.mark.transport
@needs_data
def test_tbr_uncertainty_propagation():
    pytest.importorskip("openmc")
    result = propagate_transport(
        _tbr,
        {
            "ion_temperature": Distribution.lognormal(15.0, 2.0),
            "li6_enrichment": Distribution.uniform(0.4, 0.9),
        },
        n_samples=4,
        seed=0,
    )
    assert 0.1 < result.mean < 2.5
    assert result.std > 0.0
    assert result.extra_variance > 0.0  # transport tally noise is folded in
    summary = result.to_dict()
    assert summary["n_samples"] == 4


@pytest.mark.transport
@needs_data
def test_enrichment_optimization_via_surrogate():
    pytest.importorskip("openmc")

    def neg_tbr(li6_enrichment: float) -> float:
        return -_tbr(15.0, li6_enrichment).value

    result = optimize_surrogate(neg_tbr, {"li6_enrichment": (0.2, 0.95)}, n_train=4, seed=0)
    assert 0.2 <= result.best_parameters["li6_enrichment"] <= 0.95
    assert result.n_evaluations == 5
    assert result.best_value < 0.0  # a positive TBR was found

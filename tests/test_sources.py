import json

import numpy as np
import pytest

from fusionbench import NeutronSource, PlasmaState, maxwellian_reactivity
from fusionbench.errors import FusionbenchError
from fusionbench.rates import power_partition, reaction_rate_density
from fusionbench.spectra import neutron_mean_energy


def test_facade_matches_functions(dt_plasma):
    source = NeutronSource(dt_plasma)
    assert source.reactivity("DT") == maxwellian_reactivity("DT", 10.0)
    assert source.rate_density("DT") == reaction_rate_density(dt_plasma, "DT")
    assert source.mean_energy("DT") == neutron_mean_energy("DT", 10.0)
    assert source.power_density().total == power_partition(dt_plasma).total


def test_default_reaction_is_dt_for_dt_fuel(dt_plasma):
    source = NeutronSource(dt_plasma)
    assert source.reactivity() == source.reactivity("DT")
    assert source.spectrum().reaction.id == "DT"


def test_default_reaction_is_ddn_for_pure_d(dd_plasma):
    source = NeutronSource(dd_plasma)
    assert source.spectrum().reaction.id == "DDn"


def test_rate_density_sums_neutronic(dt_plasma):
    source = NeutronSource(dt_plasma)
    total = source.rate_density()
    parts = source.rate_density("DT") + source.rate_density("DDn")
    assert total == pytest.approx(parts, rel=1e-12)


def test_no_neutrons_raises():
    state = PlasmaState(ion_temperature=10.0, ion_density=1e20, fuel={"T": 1.0})
    with pytest.raises(FusionbenchError):
        NeutronSource(state).rate_density()


def test_spectrum_rejects_array_temperature():
    state = PlasmaState(
        ion_temperature=np.array([5.0, 10.0]), ion_density=1e20, fuel={"D": 0.5, "T": 0.5}
    )
    with pytest.raises(FusionbenchError):
        NeutronSource(state).spectrum()


def test_summary_and_json(dt_plasma):
    source = NeutronSource(dt_plasma)
    record = json.loads(source.to_json())
    assert record["summary"]["power_density"]["total"] > 0
    assert "bosch-hale-1992" in record["provenance"]["models"]
    assert record["provenance"]["inputs"]["plasma"]["fuel"] == {"D": 0.5, "T": 0.5}


def test_provenance_models(dt_plasma):
    provenance = NeutronSource(dt_plasma).provenance
    assert set(provenance.models) == {"bosch-hale-1992", "brysk-1973"}
    assert len(provenance.references) == 2

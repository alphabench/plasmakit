import os

import numpy as np
import pytest

from plasmakit.blanket import Blanket, Layer
from plasmakit.errors import PlasmakitError
from plasmakit.materials import beryllium, eurofer97, li4sio4, tungsten
from plasmakit.neutronics import (
    EV_TO_JOULE,
    SECONDS_PER_FULL_POWER_YEAR,
    WALL_LOAD_ENERGY_EDGES_EV,
    BlanketResult,
    TallyValue,
    nrt_dpa_rate,
    resolve_source_terms,
    wall_load_mw_per_m2,
)
from plasmakit.plasma import PlasmaState
from plasmakit.spatial import SpatialNeutronSource
from plasmakit.spectra import neutron_mean_energy, neutron_std


@pytest.fixture
def blanket() -> Blanket:
    return Blanket(
        layers=(
            Layer("first_wall", eurofer97(), 0.02),
            Layer("breeder", li4sio4(li6_enrichment=0.6), 0.5),
        ),
        major_radius=6.0,
        first_wall_radius=2.0,
    )


@pytest.fixture
def dt_state() -> PlasmaState:
    return PlasmaState(ion_temperature=15.0, ion_density=1.0e20, fuel={"D": 0.5, "T": 0.5})


def test_nrt_dpa_rate():
    # 1 keV/s of damage energy on a single atom with E_d = 40 eV -> 10 dpa/s
    assert nrt_dpa_rate(1000.0, 1.0) == pytest.approx(10.0)
    assert nrt_dpa_rate(1000.0, 1.0, displacement_energy_ev=90.0) == pytest.approx(
        0.8 * 1000.0 / 180.0
    )
    with pytest.raises(PlasmakitError):
        nrt_dpa_rate(1.0, 0.0)
    with pytest.raises(PlasmakitError):
        nrt_dpa_rate(1.0, 1.0, displacement_energy_ev=0.0)


def test_wall_load_single_bin(blanket):
    # all current in one bin at 14.07 MeV, 1 neutron per source particle
    energy = np.array([14.07e6])
    current = np.array([1.0])
    std = np.array([0.1])
    rate = 1.0e20
    load = wall_load_mw_per_m2(energy, current, std, rate, blanket.first_wall_area())
    expected = 14.07e6 * rate * EV_TO_JOULE / blanket.first_wall_area() / 1e6
    assert load.value == pytest.approx(expected)
    assert load.std_dev == pytest.approx(expected * 0.1)
    with pytest.raises(PlasmakitError):
        wall_load_mw_per_m2(energy, current, std, rate, 0.0)


def _synthetic_result(blanket: Blanket, **overrides) -> BlanketResult:
    kwargs = dict(
        blanket=blanket,
        total_rate=1.0e20,
        particles=1000,
        batches=5,
        seed=1,
        h3_per_layer=[(0.02, 0.001), (1.10, 0.02)],
        heating_ev_per_layer=[(1.0e6, 1.0e4), (1.4e7, 1.0e5)],
        damage_energy_ev=(5.0e5, 1.0e4),
        wall_current=(
            np.array([14.07e6]),
            np.array([1.0]),
            np.array([0.05]),
        ),
    )
    kwargs.update(overrides)
    return BlanketResult.from_tallies(**kwargs)


def test_from_tallies_arithmetic(blanket):
    result = _synthetic_result(blanket)
    assert result.tbr.value == pytest.approx(1.12)
    assert result.tbr.std_dev == pytest.approx(np.sqrt(0.001**2 + 0.02**2))
    assert result.tritium_production["breeder"].value == pytest.approx(1.10 * 1e20)
    assert result.energy_deposition["first_wall"].value == pytest.approx(1.0e6 * 1e20 * EV_TO_JOULE)
    n_atoms = eurofer97().atom_density * blanket.layer_volumes()[0]
    assert result.dpa.value == pytest.approx(0.8 * 5.0e5 * 1e20 / (2 * 40.0) / n_atoms)
    assert result.dpa_per_fpy.value == pytest.approx(result.dpa.value * SECONDS_PER_FULL_POWER_YEAR)
    assert set(result.energy_deposition) == {"first_wall", "breeder"}
    assert "nrt-1975" in result.provenance.models


def test_from_tallies_layer_count_mismatch(blanket):
    with pytest.raises(PlasmakitError):
        _synthetic_result(blanket, h3_per_layer=[(1.0, 0.1)])


def test_from_tallies_chains_source_provenance(blanket, dt_state):
    _, _, provenance = resolve_source_terms(dt_state, blanket, source_rate=1e20)
    result = _synthetic_result(blanket, source_provenance=provenance)
    assert result.provenance.inputs["source"]["models"] == list(provenance.models)


def test_resolve_0d_requires_rate(blanket, dt_state):
    with pytest.raises(PlasmakitError, match="source_rate"):
        resolve_source_terms(dt_state, blanket)


def test_resolve_0d_builds_axis_rings(blanket, dt_state):
    terms, rate, provenance = resolve_source_terms(dt_state, blanket, source_rate=1e20)
    assert rate == 1e20
    assert np.sum(terms.strength) == pytest.approx(1e20)
    assert np.all(terms.r == 6.0)
    assert np.all(terms.z == 0.0)
    assert set(terms.reaction_id) == {"DT", "DDn"}
    i_dt = terms.reaction_id.index("DT")
    assert terms.energy_mean[i_dt] == pytest.approx(float(neutron_mean_energy("DT", 15.0)))
    assert terms.energy_std[i_dt] == pytest.approx(float(neutron_std("DT", 15.0)))
    assert provenance is not None and "bosch-hale-1992" in provenance.models


def test_resolve_spatial_passthrough(blanket):
    from plasmakit.geometry import TokamakGeometry
    from plasmakit.profiles import PlasmaProfiles, RadialProfile

    spatial = SpatialNeutronSource.from_profiles(
        PlasmaProfiles(
            ion_temperature=RadialProfile.parabolic(15.0, 1.0),
            ion_density=RadialProfile.parabolic(1e20, 1e18),
        ),
        TokamakGeometry(major_radius=6.0, minor_radius=1.5),
        n_rho=8,
        n_theta=8,
    )
    terms, rate, provenance = resolve_source_terms(spatial, blanket)
    assert rate == pytest.approx(spatial.total_rate)
    assert np.sum(terms.strength) == pytest.approx(spatial.total_rate, rel=1e-9)
    assert provenance is spatial.provenance


def test_resolve_rejects_out_of_chamber_ring(blanket):
    from plasmakit.spatial import SourceTerms

    terms = SourceTerms(
        r=np.array([6.0, 8.5]),
        z=np.array([0.0, 0.0]),
        strength=np.array([1.0, 1.0]),
        energy_mean=np.array([14070.0, 14070.0]),
        energy_std=np.array([100.0, 100.0]),
        reaction_id=("DT", "DT"),
    )
    with pytest.raises(PlasmakitError, match="first wall"):
        resolve_source_terms(terms, blanket)


def test_tally_value_is_frozen():
    value = TallyValue(1.0, 0.1)
    with pytest.raises(AttributeError):
        value.value = 2.0


# --- OpenMC-dependent tests -------------------------------------------------


def test_build_model_structure(blanket, dt_state):
    openmc = pytest.importorskip("openmc")
    from plasmakit.neutronics import build_model

    model = build_model(blanket, dt_state, particles=100, batches=2, source_rate=1e20)
    assert isinstance(model, openmc.Model)
    surfaces = model.geometry.get_all_surfaces()
    tori = [s for s in surfaces.values() if isinstance(s, openmc.ZTorus)]
    assert len(tori) == len(blanket.layers) + 1
    for torus in tori:
        assert torus.b == torus.c  # circular cross-section
        assert torus.a == pytest.approx(600.0)  # R0 in cm
    vacuum = [s for s in surfaces.values() if s.boundary_type == "vacuum"]
    assert len(vacuum) == 1
    names = {t.name: t for t in model.tallies}
    assert set(names) == {"tritium", "heating", "damage", "wall_current"}
    assert names["tritium"].scores == ["H3-production"]
    assert names["damage"].scores == ["damage-energy"]
    assert sum(s.strength for s in model.settings.source) == pytest.approx(1.0)


needs_data = pytest.mark.skipif(
    not os.environ.get("OPENMC_CROSS_SECTIONS"), reason="OPENMC_CROSS_SECTIONS not set"
)


@pytest.mark.transport
@needs_data
def test_hcpb_tbr_band(dt_state):
    pytest.importorskip("openmc")
    blanket = Blanket(
        layers=(
            Layer("armor", tungsten(), 0.002),
            Layer("first_wall", eurofer97(), 0.02),
            Layer("multiplier", beryllium(), 0.05),
            Layer("breeder", li4sio4(li6_enrichment=0.60), 0.50),
            Layer("shield", eurofer97(), 0.10),
        ),
        major_radius=9.0,
        first_wall_radius=2.9,
    )
    result = blanket.run_neutronics(dt_state, particles=5000, batches=5, source_rate=1.0e20)
    # HCPB-like Be/Li4SiO4(60% Li-6) blankets: TBR ~ 1.1-1.2 (Hernandez et al.,
    # Fusion Eng. Des. 124 (2017) 882); band widened for this homogeneous
    # thick-shell idealization without ports or divertor.
    assert 1.0 < result.tbr.value < 1.5
    assert result.tbr.std_dev < 0.05
    # wall load near the uncollided estimate S * <E_n> / A (backscatter adds)
    analytic = 1.0e20 * 14.07e6 * EV_TO_JOULE / blanket.first_wall_area() / 1e6
    assert 0.5 * analytic < result.neutron_wall_load.value < 2.0 * analytic
    # energy bookkeeping: deposition cannot far exceed the source power
    total_deposited = sum(v.value for v in result.energy_deposition.values())
    source_power = 1.0e20 * 14.07e6 * EV_TO_JOULE
    assert 0.0 < total_deposited < 1.3 * source_power
    assert result.dpa.value > 0.0
    # internal consistency: tritium production sums to TBR * S
    total_tritium = sum(v.value for v in result.tritium_production.values())
    assert total_tritium == pytest.approx(result.tbr.value * 1.0e20, rel=1e-9)
    assert result.provenance.inputs["cross_sections"]
    assert len(WALL_LOAD_ENERGY_EDGES_EV) == 101

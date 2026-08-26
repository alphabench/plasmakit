import numpy as np
import pytest

from plasmakit.geometry import TokamakGeometry
from plasmakit.profiles import PlasmaProfiles, RadialProfile
from plasmakit.spatial import SpatialNeutronSource
from plasmakit.spectra import neutron_mean_energy, neutron_std


@pytest.fixture
def source() -> SpatialNeutronSource:
    profiles = PlasmaProfiles(
        ion_temperature=RadialProfile.parabolic(20.0, 1.0),
        ion_density=RadialProfile.parabolic(1.0e20, 1.0e18),
    )
    geometry = TokamakGeometry(6.0, 2.0, elongation=1.7, triangularity=0.33)
    return SpatialNeutronSource.from_profiles(profiles, geometry, n_rho=8, n_theta=16)


def test_source_terms_conserve_rate(source):
    terms = source.source_terms()
    assert np.sum(terms.strength) == pytest.approx(source.total_rate, rel=1e-12)
    assert len(terms.reaction_id) == terms.strength.size
    assert set(terms.reaction_id) == {"DT", "DDn"}


def test_source_terms_energies_match_spectra(source):
    terms = source.source_terms()
    for i in (0, len(terms.strength) // 2, -1):
        rid = terms.reaction_id[i]
        # recover the cell temperature from the ring's mean energy inverse
        # is awkward; instead verify consistency: mean/std correspond to the
        # same temperature through the Brysk closed forms.
        candidates = source.ion_temperature.ravel()
        means = np.asarray(neutron_mean_energy(rid, candidates))
        match = np.argmin(np.abs(means - terms.energy_mean[i]))
        t_cell = candidates[match]
        assert terms.energy_mean[i] == pytest.approx(float(neutron_mean_energy(rid, t_cell)))
        assert terms.energy_std[i] == pytest.approx(float(neutron_std(rid, t_cell)))


def test_max_sources_truncates_and_warns(source):
    with pytest.warns(UserWarning, match="discard"):
        terms = source.source_terms(max_sources=10)
    assert terms.strength.size <= 10
    full = source.source_terms()
    assert np.min(terms.strength) >= np.percentile(full.strength, 50)


def test_min_strength_fraction_drops_weak_rings(source):
    full = source.source_terms()
    with pytest.warns(UserWarning, match="discard"):
        trimmed = source.source_terms(min_strength_fraction=1e-3)
    assert trimmed.strength.size < full.strength.size
    assert np.sum(trimmed.strength) > 0.9 * source.total_rate


def test_to_vtk_round_trip(source, tmp_path):
    path = tmp_path / "source.vtk"
    source.to_vtk(path)
    text = path.read_text()
    n1, n2 = source.emissivity.shape
    assert f"DIMENSIONS {n1 + 1} {n2 + 1} 1" in text
    assert f"POINTS {(n1 + 1) * (n2 + 1)} double" in text
    assert f"CELL_DATA {n1 * n2}" in text
    for name in ("emissivity", "power_density", "ion_temperature", "ion_density", "volume"):
        assert f"SCALARS {name} double 1" in text
    # first emissivity value round-trips
    block = text.split("SCALARS emissivity double 1\nLOOKUP_TABLE default\n")[1]
    first = float(block.split("\n", 1)[0])
    assert first == pytest.approx(source.emissivity[0, 0], rel=1e-8)


def test_to_vtk_rz_pathway(tmp_path):
    rz = SpatialNeutronSource.from_rz(
        np.array([4.0, 6.0, 8.0]),
        np.array([-1.0, 0.0, 1.0]),
        np.full((2, 2), 10.0),
        np.full((2, 2), 1.0e20),
    )
    path = tmp_path / "rz.vtk"
    rz.to_vtk(path)
    assert "DIMENSIONS 3 3 1" in path.read_text()


def test_to_xarray(source):
    xr = pytest.importorskip("xarray")
    ds = source.to_xarray()
    assert isinstance(ds, xr.Dataset)
    assert tuple(ds["emissivity"].dims) == source.dims
    assert np.array_equal(ds["emissivity"].values, source.emissivity)
    assert np.array_equal(ds["R"].values, source.r)
    assert ds["emissivity"].attrs["units"] == "m^-3 s^-1"
    assert ds.attrs["total_rate"] == source.total_rate
    assert "bosch-hale-1992" in ds.attrs["provenance"]
    assert "emissivity_DT" in ds


def test_to_xarray_rz_pathway():
    pytest.importorskip("xarray")
    rz = SpatialNeutronSource.from_rz(
        np.array([4.0, 8.0]), np.array([-1.0, 1.0]), np.full((1, 1), 10.0), np.full((1, 1), 1e20)
    )
    ds = rz.to_xarray()
    assert tuple(ds["emissivity"].dims) == ("r", "z")


def test_to_openmc(source):
    openmc = pytest.importorskip("openmc")
    sources = source.to_openmc(max_sources=None)
    terms = source.source_terms()
    assert len(sources) == terms.strength.size
    first = sources[0]
    assert isinstance(first, openmc.IndependentSource)
    assert first.space.r.x[0] == pytest.approx(terms.r[0] * 100.0)
    assert first.energy.mean_value == pytest.approx(terms.energy_mean[0] * 1e3)
    assert sum(s.strength for s in sources) == pytest.approx(1.0, rel=1e-9)

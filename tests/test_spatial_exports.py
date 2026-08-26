import numpy as np
import pytest

from fusionbench.geometry import TokamakGeometry
from fusionbench.profiles import PlasmaProfiles, RadialProfile
from fusionbench.spatial import SpatialNeutronSource
from fusionbench.spectra import neutron_mean_energy, neutron_std


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

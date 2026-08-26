import numpy as np
import pytest

from plasmakit.distributions import Distribution
from plasmakit.errors import PlasmakitError
from plasmakit.estimation import fit


@pytest.fixture(scope="module")
def conjugate_posterior():
    # prior N(0,1), single observation y=2 of the identity model with noise 1:
    # analytic posterior is N(1, 1/2)
    return fit(
        lambda mu: mu,
        {"mu": Distribution.normal(0.0, 1.0)},
        observed=2.0,
        noise_std=1.0,
        n_samples=20_000,
        burn_in=2000,
        seed=0,
    )


def test_conjugate_mean_and_std(conjugate_posterior):
    assert conjugate_posterior.mean("mu") == pytest.approx(1.0, abs=0.05)
    assert conjugate_posterior.std("mu") == pytest.approx(np.sqrt(0.5), rel=0.1)


def test_conjugate_map(conjugate_posterior):
    assert conjugate_posterior.map_estimate["mu"] == pytest.approx(1.0, abs=1e-4)


def test_acceptance_rate_healthy(conjugate_posterior):
    assert 0.05 < conjugate_posterior.acceptance_rate < 0.95


def test_percentiles_and_to_dict(conjugate_posterior):
    p5 = conjugate_posterior.percentile("mu", 5)
    p95 = conjugate_posterior.percentile("mu", 95)
    assert p5 < 1.0 < p95
    record = conjugate_posterior.to_dict()
    assert record["mu"]["mean"] == pytest.approx(1.0, abs=0.05)


def test_uniform_prior_support_respected():
    posterior = fit(
        lambda x: x,
        {"x": Distribution.uniform(0.0, 1.0)},
        observed=5.0,  # far outside support: posterior piles at the upper edge
        noise_std=1.0,
        n_samples=2000,
        burn_in=500,
        seed=1,
    )
    samples = posterior.samples["x"]
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)
    assert posterior.mean("x") > 0.7


def test_vector_observations_tighten_posterior():
    single = fit(
        lambda mu: mu,
        {"mu": Distribution.normal(0.0, 1.0)},
        observed=1.0,
        noise_std=1.0,
        n_samples=5000,
        seed=2,
    )
    multiple = fit(
        lambda mu: np.full(10, mu),
        {"mu": Distribution.normal(0.0, 1.0)},
        observed=np.full(10, 1.0),
        noise_std=1.0,
        n_samples=5000,
        seed=2,
    )
    assert multiple.std("mu") < single.std("mu")


def test_seed_determinism():
    kwargs = dict(
        fn=lambda mu: mu,
        priors={"mu": Distribution.normal(0.0, 1.0)},
        observed=1.0,
        noise_std=1.0,
        n_samples=500,
        seed=5,
    )
    a = fit(**kwargs)
    b = fit(**kwargs)
    assert np.array_equal(a.samples["mu"], b.samples["mu"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"priors": {}, "observed": 1.0, "noise_std": 1.0},
        {"priors": {"x": Distribution.normal(0, 1)}, "observed": 1.0, "noise_std": 0.0},
        {
            "priors": {"x": Distribution.normal(0, 1)},
            "observed": 1.0,
            "noise_std": 1.0,
            "n_samples": 0,
        },
    ],
)
def test_invalid_inputs(kwargs):
    with pytest.raises(PlasmakitError):
        fit(lambda x: x, **kwargs)


def test_provenance(conjugate_posterior):
    assert "metropolis-hastings" in conjugate_posterior.provenance.models
    assert conjugate_posterior.provenance.inputs["noise_std"] == 1.0

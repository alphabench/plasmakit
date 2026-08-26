import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fusionbench.distributions import Distribution
from fusionbench.errors import FusionbenchError


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Distribution.normal(10.0, 0.0),
        lambda: Distribution.normal(10.0, -1.0),
        lambda: Distribution.lognormal(-5.0, 1.0),
        lambda: Distribution.lognormal(5.0, 0.0),
        lambda: Distribution.uniform(2.0, 1.0),
        lambda: Distribution.triangular(0.0, 2.0, 1.0),
        lambda: Distribution(kind="beta", parameters={"a": 1.0}),
        lambda: Distribution(kind="normal", parameters={"mu": 0.0, "std": 1.0}),
    ],
)
def test_invalid_specs_rejected(factory):
    with pytest.raises(FusionbenchError):
        factory()


def test_moments_match_spec():
    assert Distribution.normal(10.0, 2.0).mean == pytest.approx(10.0)
    assert Distribution.normal(10.0, 2.0).std == pytest.approx(2.0)
    # the key check of the lognormal mu/sigma conversion
    assert Distribution.lognormal(15.0, 2.0).mean == pytest.approx(15.0, rel=1e-12)
    assert Distribution.lognormal(15.0, 2.0).std == pytest.approx(2.0, rel=1e-12)
    assert Distribution.uniform(2.0, 6.0).mean == pytest.approx(4.0)
    assert Distribution.uniform(2.0, 6.0).std == pytest.approx(4.0 / np.sqrt(12.0))
    tri = Distribution.triangular(0.0, 1.0, 2.0)
    assert tri.mean == pytest.approx(1.0)


def test_ppf_monotonic_and_median():
    for dist in (
        Distribution.normal(10.0, 2.0),
        Distribution.lognormal(15.0, 2.0),
        Distribution.uniform(1.0, 3.0),
        Distribution.triangular(0.0, 0.5, 2.0),
    ):
        q = np.linspace(0.01, 0.99, 50)
        values = np.asarray(dist.ppf(q))
        assert np.all(np.diff(values) > 0)
    assert Distribution.normal(10.0, 2.0).ppf(0.5) == pytest.approx(10.0)
    assert Distribution.uniform(1.0, 3.0).ppf(0.5) == pytest.approx(2.0)


def test_logpdf_outside_support():
    assert Distribution.lognormal(15.0, 2.0).logpdf(-1.0) == -np.inf
    assert Distribution.uniform(1.0, 3.0).logpdf(0.5) == -np.inf


def test_sample_seeded_reproducibility():
    dist = Distribution.normal(0.0, 1.0)
    a = dist.sample(100, np.random.default_rng(7))
    b = dist.sample(100, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_to_dict_json_safe():
    record = Distribution.triangular(0.0, 0.5, 2.0).to_dict()
    json.dumps(record)
    assert record == {"kind": "triangular", "parameters": {"low": 0.0, "mode": 0.5, "high": 2.0}}


@settings(max_examples=25, deadline=None)
@given(
    st.floats(min_value=-100.0, max_value=100.0),
    st.floats(min_value=0.01, max_value=50.0),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_triangular_samples_in_bounds(low, span, mode_fraction):
    high = low + span
    mode = low + mode_fraction * span
    dist = Distribution.triangular(low, mode, high)
    samples = dist.sample(1000, np.random.default_rng(0))
    assert np.all(samples >= low - 1e-9)
    assert np.all(samples <= high + 1e-9)


@settings(max_examples=15, deadline=None)
@given(st.floats(min_value=0.5, max_value=100.0), st.floats(min_value=0.05, max_value=10.0))
def test_lognormal_sample_moments(mean, std):
    dist = Distribution.lognormal(mean, std)
    samples = dist.sample(50_000, np.random.default_rng(1))
    assert np.mean(samples) == pytest.approx(mean, rel=0.08)
    assert np.std(samples) == pytest.approx(std, rel=0.25)

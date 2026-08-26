import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fusionbench.distributions import Distribution
from fusionbench.errors import FusionbenchError
from fusionbench.neutronics import TallyValue
from fusionbench.uncertainty import propagate, propagate_transport, sobol_indices


def _identity(x):
    return x


def test_propagate_identity_analytic():
    for method in ("sobol", "random"):
        for vectorized in (False, True):
            result = propagate(
                _identity,
                {"x": Distribution.normal(10.0, 2.0)},
                n_samples=4096,
                seed=0,
                method=method,
                vectorized=vectorized,
            )
            assert result.mean == pytest.approx(10.0, rel=1e-2)
            assert result.std == pytest.approx(2.0, rel=5e-2)


def test_vectorized_matches_loop():
    params = {"x": Distribution.uniform(0.0, 1.0), "c": 3.0}

    def fn(x, c):
        return c * x**2

    loop = propagate(fn, params, n_samples=256, seed=1, vectorized=False)
    vect = propagate(fn, params, n_samples=256, seed=1, vectorized=True)
    assert np.allclose(loop.samples, vect.samples)


def test_fixed_parameters_passed_through():
    result = propagate(
        lambda x, offset: x + offset,
        {"x": Distribution.normal(0.0, 1.0), "offset": 100.0},
        n_samples=64,
        seed=0,
    )
    assert result.mean == pytest.approx(100.0, abs=0.5)


def test_seed_determinism():
    params = {"x": Distribution.lognormal(15.0, 2.0)}
    a = propagate(_identity, params, n_samples=128, seed=3)
    b = propagate(_identity, params, n_samples=128, seed=3)
    assert np.array_equal(a.samples, b.samples)
    c = propagate(_identity, params, n_samples=128, seed=4)
    assert not np.array_equal(a.samples, c.samples)


def test_sobol_rounds_to_power_of_two():
    result = propagate(_identity, {"x": Distribution.uniform(0.0, 1.0)}, n_samples=100, seed=0)
    assert result.samples.size == 128
    assert result.provenance.inputs["n_samples"] == 128


def test_no_free_parameters_raises():
    with pytest.raises(FusionbenchError):
        propagate(_identity, {"x": 1.0})


def test_bad_vectorized_shape_raises():
    with pytest.raises(FusionbenchError):
        propagate(
            lambda x: np.array([1.0]),
            {"x": Distribution.uniform(0.0, 1.0)},
            n_samples=8,
            vectorized=True,
        )


def test_percentiles_ordered():
    result = propagate(_identity, {"x": Distribution.normal(0.0, 1.0)}, n_samples=512, seed=0)
    assert result.percentile(5) < result.percentile(50) < result.percentile(95)
    assert result.to_dict()["p50"] == pytest.approx(result.percentile(50))


def test_provenance_records_specs():
    result = propagate(
        lambda x, c: x * c, {"x": Distribution.normal(1.0, 0.5), "c": 2.0}, n_samples=8, seed=0
    )
    specs = result.provenance.inputs["parameters"]
    assert specs["x"]["kind"] == "normal"
    assert specs["c"] == 2.0
    assert "sobol-qmc" in result.provenance.models


@settings(max_examples=20, deadline=None)
@given(st.floats(min_value=-5.0, max_value=5.0), st.floats(min_value=-10.0, max_value=10.0))
def test_linear_map_property(a, b):
    result = propagate(
        lambda x: a * x + b,
        {"x": Distribution.normal(2.0, 1.5)},
        n_samples=2048,
        seed=0,
        vectorized=True,
    )
    assert result.mean == pytest.approx(a * 2.0 + b, abs=0.05 * max(1.0, abs(a)))
    assert result.std == pytest.approx(abs(a) * 1.5, abs=0.08 * max(1.0, abs(a)))


# --- Sobol indices ----------------------------------------------------------

_PI = np.pi


def _ishigami(x1, x2, x3):
    return np.sin(x1) + 7.0 * np.sin(x2) ** 2 + 0.1 * x3**4 * np.sin(x1)


_ISHIGAMI_PARAMS = {
    "x1": Distribution.uniform(-_PI, _PI),
    "x2": Distribution.uniform(-_PI, _PI),
    "x3": Distribution.uniform(-_PI, _PI),
}


@pytest.fixture(scope="module")
def ishigami_indices():
    return sobol_indices(_ishigami, _ISHIGAMI_PARAMS, n_samples=4096, seed=0, vectorized=True)


def test_ishigami_first_order(ishigami_indices):
    assert ishigami_indices.first_order["x1"] == pytest.approx(0.3139, abs=0.02)
    assert ishigami_indices.first_order["x2"] == pytest.approx(0.4424, abs=0.02)
    assert ishigami_indices.first_order["x3"] == pytest.approx(0.0, abs=0.02)


def test_ishigami_total_order(ishigami_indices):
    assert ishigami_indices.total_order["x3"] == pytest.approx(0.2437, abs=0.02)
    assert ishigami_indices.total_order["x2"] == pytest.approx(0.4424, abs=0.02)
    for name in ("x1", "x2", "x3"):
        assert ishigami_indices.total_order[name] >= ishigami_indices.first_order[name] - 0.02


def test_ishigami_bootstrap_stds(ishigami_indices):
    for name in ("x1", "x2", "x3"):
        assert 0.0 < ishigami_indices.first_order_std[name] < 0.1
        assert 0.0 < ishigami_indices.total_order_std[name] < 0.1
    assert "saltelli-2010" in ishigami_indices.provenance.models


def test_additive_model_first_equals_total():
    indices = sobol_indices(
        lambda x, y: 2.0 * x + y,
        {"x": Distribution.normal(0.0, 1.0), "y": Distribution.normal(0.0, 1.0)},
        n_samples=2048,
        seed=0,
        vectorized=True,
    )
    # additive model: S_i == ST_i; variance split 4:1
    assert indices.first_order["x"] == pytest.approx(0.8, abs=0.05)
    assert indices.total_order["x"] == pytest.approx(0.8, abs=0.05)
    assert indices.first_order["y"] == pytest.approx(0.2, abs=0.05)


def test_constant_model_raises():
    with pytest.raises(FusionbenchError):
        sobol_indices(lambda x: 1.0, {"x": Distribution.uniform(0.0, 1.0)}, n_samples=64, seed=0)


# --- Transport-aware propagation --------------------------------------------


def test_propagate_transport_variance_decomposition():
    def fn(x):
        return TallyValue(value=2.0 * x, std_dev=0.5)

    result = propagate_transport(fn, {"x": Distribution.normal(5.0, 1.0)}, n_samples=64, seed=0)
    assert result.extra_variance == pytest.approx(0.25)
    between = float(np.var(result.samples, ddof=1))
    assert result.std == pytest.approx(np.sqrt(between + 0.25), rel=1e-12)
    assert result.mean == pytest.approx(10.0, rel=0.05)


def test_propagate_transport_zero_noise_matches_propagate():
    def fn(x):
        return TallyValue(value=x**2, std_dev=0.0)

    transport = propagate_transport(fn, {"x": Distribution.uniform(0.0, 1.0)}, n_samples=32, seed=1)
    plain = propagate(lambda x: x**2, {"x": Distribution.uniform(0.0, 1.0)}, n_samples=32, seed=1)
    assert np.allclose(transport.samples, plain.samples)
    assert transport.std == pytest.approx(plain.std)


def test_propagate_transport_type_error():
    with pytest.raises(FusionbenchError, match="TallyValue"):
        propagate_transport(lambda x: float(x), {"x": Distribution.uniform(0.0, 1.0)}, n_samples=4)

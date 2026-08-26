import json

import numpy as np
import pytest

from fusionbench.distributions import Distribution
from fusionbench.errors import FusionbenchError
from fusionbench.surrogates import GaussianProcess, Surrogate
from fusionbench.uncertainty import propagate


@pytest.fixture(scope="module")
def sin_gp() -> GaussianProcess:
    x = np.linspace(0.0, 2.0 * np.pi, 12)
    return GaussianProcess.train(x, np.sin(x), seed=0)


def test_gp_interpolates_training_points(sin_gp):
    x = np.linspace(0.0, 2.0 * np.pi, 12)
    mean, std = sin_gp.predict(x)
    assert np.allclose(mean, np.sin(x), atol=1e-6)
    assert np.all(std < 1e-3)


def test_gp_recovers_sin(sin_gp):
    grid = np.linspace(0.3, 2.0 * np.pi - 0.3, 100)
    mean, _ = sin_gp.predict(grid)
    assert np.max(np.abs(mean - np.sin(grid))) < 1e-3
    assert float(sin_gp.predict(np.array([1.0]))[0][0]) == pytest.approx(np.sin(1.0), rel=1e-3)


def test_gp_std_grows_away_from_data(sin_gp):
    _, std_inside = sin_gp.predict(np.array([np.pi]))
    _, std_outside = sin_gp.predict(np.array([4.0 * np.pi]))
    assert std_outside[0] > 10.0 * std_inside[0]


def test_gp_fixed_length_scale():
    x = np.linspace(0.0, 1.0, 8)
    gp = GaussianProcess.train(x, x**2, length_scale=0.5, noise=1e-8)
    assert np.allclose(gp.length_scales, [0.5])
    mean, _ = gp.predict(np.array([0.55]))
    assert mean[0] == pytest.approx(0.55**2, abs=1e-2)


def test_gp_to_dict_json_safe(sin_gp):
    record = sin_gp.to_dict()
    json.dumps(record)
    assert record["kernel"] == "rbf-ard"
    assert record["n_train"] == 12


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (np.zeros((1, 2)), np.zeros(1)),  # too few points
        (np.zeros((4, 2)), np.zeros(3)),  # shape mismatch
        (np.array([[np.inf, 0.0], [0.0, 1.0]]), np.zeros(2)),  # non-finite
    ],
)
def test_gp_invalid_inputs(x, y):
    with pytest.raises(FusionbenchError):
        GaussianProcess.train(x, y)


def test_gp_negative_noise_rejected():
    with pytest.raises(FusionbenchError):
        GaussianProcess.train(np.zeros((3, 1)), np.zeros(3), noise=-1.0)


def _quadratic(x, y):
    return (x - 1.0) ** 2 + 2.0 * y


@pytest.fixture(scope="module")
def quadratic_surrogate() -> Surrogate:
    return Surrogate.from_function(
        _quadratic,
        {"x": (-2.0, 4.0), "y": Distribution.uniform(-1.0, 1.0)},
        n_train=64,
        seed=0,
        vectorized=True,
    )


def test_surrogate_matches_truth(quadratic_surrogate):
    for x, y in [(0.0, 0.0), (1.5, 0.5), (-1.0, -0.5)]:
        assert quadratic_surrogate(x=x, y=y) == pytest.approx(_quadratic(x, y), abs=1e-3)
    mean, std = quadratic_surrogate.predict(x=1.0, y=0.0)
    assert mean == pytest.approx(0.0, abs=1e-3)
    assert std >= 0.0


def test_surrogate_propagate_matches_truth(quadratic_surrogate):
    params = {"x": Distribution.uniform(-1.0, 3.0), "y": Distribution.uniform(-0.5, 0.5)}
    on_surrogate = quadratic_surrogate.propagate(params, n_samples=2048, seed=1)
    on_truth = propagate(_quadratic, params, n_samples=2048, seed=1, vectorized=True)
    assert on_surrogate.mean == pytest.approx(on_truth.mean, rel=1e-2)
    assert on_surrogate.std == pytest.approx(on_truth.std, rel=5e-2)


def test_surrogate_provenance(quadratic_surrogate):
    assert "gp-rbf" in quadratic_surrogate.provenance.models
    assert quadratic_surrogate.provenance.inputs["n_train"] == 64
    json.dumps(quadratic_surrogate.provenance.inputs)

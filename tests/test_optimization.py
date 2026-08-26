import pytest

from fusionbench.errors import FusionbenchError
from fusionbench.optimization import optimize, optimize_surrogate


def _rosenbrock(x, y):
    return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2


def test_rosenbrock_minimum():
    result = optimize(_rosenbrock, {"x": (-2.0, 2.0), "y": (-1.0, 3.0)}, seed=0)
    assert result.best_parameters["x"] == pytest.approx(1.0, abs=1e-3)
    assert result.best_parameters["y"] == pytest.approx(1.0, abs=1e-3)
    assert result.best_value == pytest.approx(0.0, abs=1e-6)
    assert result.n_evaluations > 0
    assert result.success


def test_constrained_optimization():
    result = optimize(
        lambda x, y: x**2 + y**2,
        {"x": (-2.0, 2.0), "y": (-2.0, 2.0)},
        constraints=[lambda x, y: 1.0 - x],  # feasible iff x >= 1
        seed=0,
    )
    assert result.best_parameters["x"] == pytest.approx(1.0, abs=1e-2)
    assert result.best_parameters["y"] == pytest.approx(0.0, abs=1e-2)


def test_seed_determinism():
    a = optimize(_rosenbrock, {"x": (-2.0, 2.0), "y": (-1.0, 3.0)}, seed=3)
    b = optimize(_rosenbrock, {"x": (-2.0, 2.0), "y": (-1.0, 3.0)}, seed=3)
    assert a.best_parameters == b.best_parameters
    assert a.best_value == b.best_value


@pytest.mark.parametrize(
    ("bounds", "kwargs"),
    [
        ({}, {}),
        ({"x": (2.0, 1.0)}, {}),
        ({"x": (0.0, 1.0)}, {"method": "nelder-mead"}),
    ],
)
def test_invalid_inputs(bounds, kwargs):
    with pytest.raises(FusionbenchError):
        optimize(lambda x=0.0: x, bounds, **kwargs)


def test_provenance():
    result = optimize(_rosenbrock, {"x": (-2.0, 2.0), "y": (-1.0, 3.0)}, seed=0)
    assert "storn-price-1997" in result.provenance.models
    assert result.provenance.inputs["bounds"]["x"] == [-2.0, 2.0]
    assert result.surrogate_value is None


def test_optimize_surrogate_bowl():
    calls = {"n": 0}

    def bowl(x, y):
        calls["n"] += 1
        return (x - 3.0) ** 2 + (y + 1.0) ** 2

    result = optimize_surrogate(bowl, {"x": (0.0, 5.0), "y": (-3.0, 2.0)}, n_train=32, seed=0)
    assert result.best_parameters["x"] == pytest.approx(3.0, abs=0.2)
    assert result.best_parameters["y"] == pytest.approx(-1.0, abs=0.2)
    assert result.n_evaluations == 33  # 32 design points + 1 verification
    assert calls["n"] == 33
    assert result.surrogate_value == pytest.approx(result.best_value, abs=0.1)
    assert "gp-rbf" in result.provenance.models

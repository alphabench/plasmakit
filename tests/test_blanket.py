import json

import numpy as np
import pytest

from fusionbench.blanket import Blanket, Layer, torus_shell_volume
from fusionbench.errors import FusionbenchError
from fusionbench.geometry import TokamakGeometry
from fusionbench.materials import eurofer97, li4sio4


@pytest.fixture
def blanket() -> Blanket:
    return Blanket(
        layers=(
            Layer("first_wall", eurofer97(), 0.5),
            Layer("breeder", li4sio4(), 0.5),
        ),
        major_radius=6.0,
        first_wall_radius=2.0,
    )


def test_torus_shell_volume():
    assert torus_shell_volume(6.0, 2.0, 3.0) == pytest.approx(2.0 * np.pi**2 * 6.0 * 5.0)
    with pytest.raises(FusionbenchError):
        torus_shell_volume(6.0, 3.0, 2.0)
    with pytest.raises(FusionbenchError):
        torus_shell_volume(6.0, 2.0, 6.0)


def test_boundaries_and_volumes(blanket):
    assert blanket.boundaries() == (2.0, 2.5, 3.0)
    volumes = blanket.layer_volumes()
    assert volumes[0] == pytest.approx(torus_shell_volume(6.0, 2.0, 2.5))
    assert sum(volumes) == pytest.approx(torus_shell_volume(6.0, 2.0, 3.0))


def test_first_wall_area(blanket):
    assert blanket.first_wall_area() == pytest.approx(4.0 * np.pi**2 * 6.0 * 2.0)


@pytest.mark.parametrize(
    ("layers", "major_radius", "first_wall_radius"),
    [
        ((), 6.0, 2.0),
        ((Layer("a", eurofer97(), 0.5), Layer("a", li4sio4(), 0.5)), 6.0, 2.0),
        ((Layer("a", eurofer97(), 0.5),), -6.0, 2.0),
        ((Layer("a", eurofer97(), 0.5),), 6.0, 0.0),
        ((Layer("a", eurofer97(), 4.0),), 6.0, 2.0),  # build reaches R0
    ],
)
def test_invalid_blankets_rejected(layers, major_radius, first_wall_radius):
    with pytest.raises(FusionbenchError):
        Blanket(layers=layers, major_radius=major_radius, first_wall_radius=first_wall_radius)


def test_invalid_layers_rejected():
    with pytest.raises(FusionbenchError):
        Layer("", eurofer97(), 0.5)
    with pytest.raises(FusionbenchError):
        Layer("x", eurofer97(), 0.0)


def test_from_geometry_circular():
    geometry = TokamakGeometry(major_radius=6.0, minor_radius=2.0)
    blanket = Blanket.from_geometry([Layer("fw", eurofer97(), 0.1)], geometry)
    assert blanket.first_wall_radius == pytest.approx(2.0, rel=1e-6)
    with_gap = Blanket.from_geometry([Layer("fw", eurofer97(), 0.1)], geometry, gap=0.2)
    assert with_gap.first_wall_radius == pytest.approx(2.2, rel=1e-6)


def test_from_geometry_shaped_warns():
    geometry = TokamakGeometry(6.0, 2.0, elongation=1.7, triangularity=0.33)
    with pytest.warns(UserWarning, match="circular"):
        blanket = Blanket.from_geometry([Layer("fw", eurofer97(), 0.1)], geometry)
    # at least the tallest LCFS point (kappa * a), at most slightly more
    # (the top of the surface also sits a*delta inboard of R0)
    kappa_a = 1.7 * 2.0
    assert kappa_a <= blanket.first_wall_radius <= np.hypot(2.0 * 0.33, kappa_a) * 1.01


def test_to_dict_json_safe(blanket):
    record = blanket.to_dict()
    json.dumps(record)
    assert [entry["name"] for entry in record["layers"]] == ["first_wall", "breeder"]

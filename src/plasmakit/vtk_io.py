"""Minimal legacy-VTK output for structured 2-D grids.

Writes ASCII ``STRUCTURED_GRID`` files readable by ParaView and VisIt
using only NumPy — no VTK dependency.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from plasmakit.errors import PlasmakitError


def write_structured_grid(
    path: str | os.PathLike[str],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    cell_data: Mapping[str, npt.NDArray[np.float64]],
) -> None:
    """Write a planar structured grid with cell-centered scalar fields.

    Parameters
    ----------
    path : path-like
        Output file path (conventionally ``.vtk``).
    x, y : ndarray
        Grid-point (node) coordinates, shape ``(n1+1, n2+1)``; the third
        coordinate is written as 0.
    cell_data : mapping of str to ndarray
        Scalar fields on cells, each with shape ``(n1, n2)``.
    """
    if x.ndim != 2 or x.shape != y.shape:
        raise PlasmakitError("x and y must be 2-D arrays of identical shape")
    n1, n2 = x.shape[0] - 1, x.shape[1] - 1
    for name, values in cell_data.items():
        if values.shape != (n1, n2):
            raise PlasmakitError(
                f"cell field {name!r} has shape {values.shape}, expected {(n1, n2)}"
            )

    # VTK expects the first grid index to vary fastest.
    points = np.zeros(((n1 + 1) * (n2 + 1), 3))
    points[:, 0] = x.ravel(order="F")
    points[:, 1] = y.ravel(order="F")

    lines = [
        "# vtk DataFile Version 3.0",
        "plasmakit structured grid",
        "ASCII",
        "DATASET STRUCTURED_GRID",
        f"DIMENSIONS {n1 + 1} {n2 + 1} 1",
        f"POINTS {points.shape[0]} double",
        "\n".join(f"{p[0]:.9e} {p[1]:.9e} {p[2]:.1f}" for p in points),
        f"CELL_DATA {n1 * n2}",
    ]
    for name, values in cell_data.items():
        lines.append(f"SCALARS {name} double 1")
        lines.append("LOOKUP_TABLE default")
        lines.append("\n".join(f"{v:.9e}" for v in values.ravel(order="F")))
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")

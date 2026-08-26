"""Package exceptions and warnings."""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt


class PlasmakitError(Exception):
    """Base class for all plasmakit errors."""


class UnknownReactionError(PlasmakitError, KeyError):
    """Raised when a reaction identifier is not in the reaction registry."""


class ValidityRangeWarning(UserWarning):
    """Emitted when an input falls outside a parameterization's fitted range.

    The computed value is an extrapolation of the fit and its accuracy is
    not guaranteed by the reference publication.
    """


def warn_outside_range(
    values: npt.NDArray[np.float64],
    valid_range: tuple[float, float],
    model: str,
    quantity: str,
) -> None:
    """Emit :class:`ValidityRangeWarning` if any value falls outside a fit range."""
    lo, hi = valid_range
    if np.any((values < lo) | (values > hi)):
        warnings.warn(
            f"{quantity} outside the {model} fit range [{lo}, {hi}] keV; "
            "values there are extrapolations",
            ValidityRangeWarning,
            stacklevel=3,
        )

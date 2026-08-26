"""Reproducibility records for derived results."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

MODEL_REFERENCES: Mapping[str, str] = MappingProxyType(
    {
        "bosch-hale-1992": (
            "H.-S. Bosch and G.M. Hale, 'Improved formulas for fusion cross-sections "
            "and thermal reactivities', Nuclear Fusion 32 (1992) 611"
        ),
        "brysk-1973": (
            "H. Brysk, 'Fusion neutron energies and spectra', Plasma Physics 15 (1973) 611"
        ),
        "ciaaw-2021": (
            "T. Prohaska et al., 'Standard atomic weights of the elements 2021 "
            "(IUPAC Technical Report)', Pure and Applied Chemistry 94 (2022) 573"
        ),
        "abdou-1986": (
            "M.A. Abdou, E.L. Vold, C.Y. Gung, M.Z. Youssef and K. Shin, "
            "'Deuterium-tritium fuel self-sufficiency in fusion reactors', "
            "Fusion Technology 9 (1986) 250"
        ),
        "abdou-2021": (
            "M. Abdou, M. Riva, A. Ying, C. Day, A. Loarte, L.R. Baylor, P. Humrickhouse, "
            "T.F. Fuerst and S. Cho, 'Physics and technology considerations for the "
            "deuterium-tritium fuel cycle and conditions for tritium fuel self-sufficiency', "
            "Nuclear Fusion 61 (2021) 013001"
        ),
        "lucas-unterweger-2000": (
            "L.L. Lucas and M.P. Unterweger, 'Comprehensive review and critical evaluation "
            "of the half-life of tritium', Journal of Research of NIST 105 (2000) 541"
        ),
        "sobol-qmc": (
            "I.M. Sobol', 'On the distribution of points in a cube and the approximate "
            "evaluation of integrals', USSR Comput. Math. Math. Phys. 7 (1967) 86; "
            "scrambling: A.B. Owen, 'Scrambling Sobol and Niederreiter-Xing points', "
            "J. Complexity 14 (1998) 466"
        ),
        "saltelli-2010": (
            "A. Saltelli, P. Annoni, I. Azzini, F. Campolongo, M. Ratto and S. Tarantola, "
            "'Variance based sensitivity analysis of model output. Design and estimator for "
            "the total sensitivity index', Computer Physics Communications 181 (2010) 259"
        ),
        "metropolis-hastings": (
            "N. Metropolis, A.W. Rosenbluth, M.N. Rosenbluth, A.H. Teller and E. Teller, "
            "J. Chem. Phys. 21 (1953) 1087; W.K. Hastings, 'Monte Carlo sampling methods "
            "using Markov chains and their applications', Biometrika 57 (1970) 97"
        ),
        "gp-rbf": (
            "C.E. Rasmussen and C.K.I. Williams, 'Gaussian Processes for Machine Learning', "
            "MIT Press (2006)"
        ),
        "storn-price-1997": (
            "R. Storn and K. Price, 'Differential evolution - a simple and efficient "
            "heuristic for global optimization over continuous spaces', "
            "Journal of Global Optimization 11 (1997) 341"
        ),
        "nrt-1975": (
            "M.J. Norgett, M.T. Robinson and I.M. Torrens, 'A proposed method of "
            "calculating displacement dose rates', Nuclear Engineering and Design 33 (1975) 50"
        ),
        "openmc-2015": (
            "P.K. Romano, N.E. Horelik, B.R. Herman, A.G. Nelson, B. Forget and K. Smith, "
            "'OpenMC: A state-of-the-art Monte Carlo code for research and development', "
            "Annals of Nuclear Energy 82 (2015) 90"
        ),
        "miller-1998": (
            "R.L. Miller, M.S. Chu, J.M. Greene, Y.R. Lin-Liu and R.E. Waltz, "
            "'Noncircular, finite aspect ratio, local equilibrium model', "
            "Physics of Plasmas 5 (1998) 973"
        ),
    }
)
"""Full citations for every physics model identifier used in the package."""


@dataclass(frozen=True)
class Provenance:
    """Record of how a result was produced.

    Attributes
    ----------
    package : str
        Producing package name.
    version : str
        Package version.
    models : tuple of str
        Physics model identifiers used (e.g. ``"bosch-hale-1992"``).
    references : tuple of str
        Full citations corresponding to ``models``.
    inputs : dict
        JSON-safe record of the physical inputs.
    created : str
        ISO-8601 UTC creation timestamp.
    """

    package: str
    version: str
    models: tuple[str, ...]
    references: tuple[str, ...]
    inputs: dict[str, Any] = field(default_factory=dict)
    created: str = ""

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(asdict(self), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> Provenance:
        """Reconstruct a record from :meth:`to_json` output."""
        data = json.loads(s)
        data["models"] = tuple(data["models"])
        data["references"] = tuple(data["references"])
        return cls(**data)


def build_provenance(models: Sequence[str], inputs: dict[str, Any]) -> Provenance:
    """Create a :class:`Provenance` record for the given model identifiers.

    Parameters
    ----------
    models : sequence of str
        Identifiers that must exist in :data:`MODEL_REFERENCES`.
    inputs : dict
        JSON-safe description of the physical inputs.
    """
    from fusionbench import __version__

    return Provenance(
        package="fusionbench",
        version=__version__,
        models=tuple(models),
        references=tuple(MODEL_REFERENCES[m] for m in models),
        inputs=inputs,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

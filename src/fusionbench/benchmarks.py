"""Validation benchmarks against published reference values.

The :data:`CASES` registry is the single source of truth for reference
values: :func:`validate` runs it interactively and the test suite asserts
every case passes, so a physics regression fails both.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from fusionbench import bosch_hale, spectra
from fusionbench.plasma import PlasmaState
from fusionbench.provenance import Provenance, build_provenance
from fusionbench.rates import fusion_power_density
from fusionbench.reactions import REACTIONS
from fusionbench.reactivity import maxwellian_reactivity
from fusionbench.spectra import neutron_spectrum

_BOSCH_HALE = "H.-S. Bosch and G.M. Hale, Nucl. Fusion 32 (1992) 611"
_BRYSK = "H. Brysk, Plasma Phys. 15 (1973) 611"


@dataclass(frozen=True)
class BenchmarkCase:
    """One published reference value and the computation reproducing it.

    Attributes
    ----------
    name : str
        Human-readable case name.
    reference : str
        Citation for the reference value.
    reference_value : float
        Published value, in ``unit``.
    unit : str
        Unit of the reference value.
    rtol : float
        Relative tolerance for a pass.
    compute : callable
        Zero-argument function returning the package's value for comparison.
    """

    name: str
    reference: str
    reference_value: float
    unit: str
    rtol: float
    compute: Callable[[], float]


@dataclass(frozen=True)
class BenchmarkResult:
    """Outcome of one benchmark case."""

    case: BenchmarkCase
    computed_value: float
    relative_error: float
    passed: bool


@dataclass(frozen=True)
class BenchmarkReport:
    """Results of the full benchmark suite."""

    results: tuple[BenchmarkResult, ...]
    provenance: Provenance

    @property
    def passed(self) -> bool:
        """Whether every case passed."""
        return all(r.passed for r in self.results)

    def __str__(self) -> str:
        """Render the report as an aligned table with citations."""
        width = max(len(r.case.name) for r in self.results)
        lines = [
            f"{'case':<{width}}  {'reference':>12}  {'computed':>12}  {'rel err':>8}  status",
            "-" * (width + 48),
        ]
        for r in self.results:
            lines.append(
                f"{r.case.name:<{width}}  {r.case.reference_value:>12.4e}  "
                f"{r.computed_value:>12.4e}  {r.relative_error:>8.1e}  "
                f"{'PASS' if r.passed else 'FAIL'}"
            )
        lines.append("")
        lines.append("references:")
        for citation in sorted({r.case.reference for r in self.results}):
            lines.append(f"  - {citation}")
        return "\n".join(lines)

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize the report (cases, values, provenance) to JSON."""
        return json.dumps(
            {
                "results": [
                    {
                        "name": r.case.name,
                        "reference": r.case.reference,
                        "reference_value": r.case.reference_value,
                        "unit": r.case.unit,
                        "rtol": r.case.rtol,
                        "computed_value": r.computed_value,
                        "relative_error": r.relative_error,
                        "passed": r.passed,
                    }
                    for r in self.results
                ],
                "passed": self.passed,
                "provenance": json.loads(self.provenance.to_json()),
            },
            indent=indent,
        )


def _reactivity_case(reaction_id: str, temperature: float, value: float) -> BenchmarkCase:
    return BenchmarkCase(
        name=f"{reaction_id} reactivity at {temperature:g} keV",
        reference=f"{_BOSCH_HALE}, Table VIII",
        reference_value=value,
        unit="m^3/s",
        rtol=1e-2,
        compute=lambda: float(maxwellian_reactivity(reaction_id, temperature)),
    )


def _dt_peak_value() -> float:
    t = np.linspace(40.0, 100.0, 601)
    return float(np.max(np.asarray(maxwellian_reactivity("DT", t))))


def _dt_peak_location() -> float:
    t = np.linspace(40.0, 100.0, 601)
    sv = np.asarray(maxwellian_reactivity("DT", t))
    return float(t[np.argmax(sv)])


def _dt_power_density() -> float:
    state = PlasmaState(ion_temperature=10.0, ion_density=1.0e20, fuel={"D": 0.5, "T": 0.5})
    return float(fusion_power_density(state, reactions=["DT"]))


CASES: tuple[BenchmarkCase, ...] = (
    _reactivity_case("DT", 1.0, 6.857e-27),
    _reactivity_case("DT", 10.0, 1.136e-22),
    _reactivity_case("DT", 20.0, 4.330e-22),
    _reactivity_case("DT", 50.0, 8.649e-22),
    _reactivity_case("DDn", 10.0, 6.023e-25),
    _reactivity_case("DDn", 20.0, 2.603e-24),
    _reactivity_case("DDp", 10.0, 5.781e-25),
    _reactivity_case("DHe3", 10.0, 2.126e-25),
    BenchmarkCase(
        name="DT reactivity peak value",
        reference=f"{_BOSCH_HALE} (fit maximum; NRL Plasma Formulary)",
        reference_value=9.0e-22,
        unit="m^3/s",
        rtol=3e-2,
        compute=_dt_peak_value,
    ),
    BenchmarkCase(
        name="DT reactivity peak temperature",
        reference=f"{_BOSCH_HALE} (fit maximum)",
        reference_value=64.0,
        unit="keV",
        rtol=8e-2,
        compute=_dt_peak_location,
    ),
    BenchmarkCase(
        name="DT cold neutron energy",
        reference="Two-body kinematics, Q from CODATA 2018 masses (14.07 MeV nominal)",
        reference_value=14_070.0,
        unit="keV",
        rtol=3e-3,
        compute=lambda: float(REACTIONS["DT"].neutron_energy or 0.0),
    ),
    BenchmarkCase(
        name="DDn cold neutron energy",
        reference="Two-body kinematics, Q from CODATA 2018 masses (2.45 MeV nominal)",
        reference_value=2_450.0,
        unit="keV",
        rtol=3e-3,
        compute=lambda: float(REACTIONS["DDn"].neutron_energy or 0.0),
    ),
    BenchmarkCase(
        name="DT spectrum FWHM at 10 keV",
        reference=f"{_BRYSK} (FWHM ~= 177 sqrt(T) keV)",
        reference_value=177.0 * np.sqrt(10.0),
        unit="keV",
        rtol=2e-2,
        compute=lambda: neutron_spectrum("DT", 10.0).fwhm,
    ),
    BenchmarkCase(
        name="DDn spectrum FWHM at 10 keV",
        reference=f"{_BRYSK} (FWHM ~= 82.5 sqrt(T) keV)",
        reference_value=82.5 * np.sqrt(10.0),
        unit="keV",
        rtol=2e-2,
        compute=lambda: neutron_spectrum("DDn", 10.0).fwhm,
    ),
    BenchmarkCase(
        name="DT neutron power fraction",
        reference="Two-body kinematics: m_4He / (m_4He + m_n)",
        reference_value=0.799,
        unit="",
        rtol=1e-3,
        compute=lambda: float((REACTIONS["DT"].neutron_energy or 0.0) / REACTIONS["DT"].q_value),
    ),
    BenchmarkCase(
        name="DT power density (50/50, n=1e20 m^-3, T=10 keV)",
        reference="Hand calculation: (n^2/4) <sigma*v> Q",
        reference_value=8.0e5,
        unit="W/m^3",
        rtol=2e-2,
        compute=_dt_power_density,
    ),
)
"""Registry of benchmark cases, shared by :func:`validate` and the test suite."""


def run_case(case: BenchmarkCase) -> BenchmarkResult:
    """Execute one benchmark case and compare it against its reference."""
    computed = case.compute()
    relative_error = abs(computed - case.reference_value) / abs(case.reference_value)
    return BenchmarkResult(
        case=case,
        computed_value=computed,
        relative_error=relative_error,
        passed=bool(relative_error <= case.rtol),
    )


def validate(verbose: bool = True) -> BenchmarkReport:
    """Run every benchmark case against its published reference value.

    Parameters
    ----------
    verbose : bool
        Print the report table to stdout.

    Returns
    -------
    BenchmarkReport
        Per-case results; never raises on failure (check ``report.passed``).
    """
    report = BenchmarkReport(
        results=tuple(run_case(case) for case in CASES),
        provenance=build_provenance(
            models=[bosch_hale.MODEL_ID, spectra.MODEL_ID],
            inputs={"cases": len(CASES)},
        ),
    )
    if verbose:
        print(report)
    return report

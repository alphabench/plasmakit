"""Validation benchmarks against published reference values.

The :data:`CASES` registry is the single source of truth for reference
values: :func:`validate` runs it interactively and the test suite asserts
every case passes, so a physics regression fails both.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from fusionbench import bosch_hale, geometry, materials, neutronics, spectra, uncertainty
from fusionbench.blanket import Blanket, Layer
from fusionbench.distributions import Distribution
from fusionbench.geometry import TokamakGeometry
from fusionbench.materials import eurofer97, li4sio4, pbli, tungsten
from fusionbench.neutronics import nrt_dpa_rate
from fusionbench.plasma import PlasmaState
from fusionbench.profiles import PlasmaProfiles, RadialProfile
from fusionbench.provenance import Provenance, build_provenance
from fusionbench.rates import fusion_power_density
from fusionbench.reactions import REACTIONS
from fusionbench.reactivity import maxwellian_reactivity
from fusionbench.spatial import SpatialNeutronSource
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


def _identity_propagation() -> Any:
    from fusionbench.uncertainty import propagate

    return propagate(
        lambda x: x,
        {"x": Distribution.normal(10.0, 2.0)},
        n_samples=4096,
        seed=0,
        vectorized=True,
    )


def _ishigami_indices() -> Any:
    from fusionbench.uncertainty import sobol_indices

    def ishigami(x1: Any, x2: Any, x3: Any) -> Any:
        return np.sin(x1) + 7.0 * np.sin(x2) ** 2 + 0.1 * x3**4 * np.sin(x1)

    bounds = Distribution.uniform(-np.pi, np.pi)
    return sobol_indices(
        ishigami,
        {"x1": bounds, "x2": bounds, "x3": bounds},
        n_samples=4096,
        seed=0,
        vectorized=True,
    )


def _conjugate_posterior_mean() -> float:
    from fusionbench.estimation import fit

    posterior = fit(
        lambda mu: mu,
        {"mu": Distribution.normal(0.0, 1.0)},
        observed=2.0,
        noise_std=1.0,
        n_samples=20_000,
        burn_in=2000,
        seed=0,
    )
    return posterior.mean("mu")


def _gp_sin_recovery() -> float:
    from fusionbench.surrogates import GaussianProcess

    x = np.linspace(0.0, 2.0 * np.pi, 12)
    gp = GaussianProcess.train(x, np.sin(x), seed=0)
    return float(gp.predict(np.array([1.0]))[0][0])


def _demo_blanket() -> Blanket:
    return Blanket(
        layers=(
            Layer("first_wall", eurofer97(), 0.5),
            Layer("breeder", li4sio4(), 0.5),
        ),
        major_radius=6.0,
        first_wall_radius=2.0,
    )


def _flat_profiles(density_center: float = 1.0e20, density_edge: float = 1.0e20) -> PlasmaProfiles:
    return PlasmaProfiles(
        ion_temperature=RadialProfile.parabolic(10.0, 10.0),
        ion_density=RadialProfile.parabolic(density_center, density_edge),
    )


def _flat_profile_dt_rate() -> float:
    source = SpatialNeutronSource.from_profiles(
        _flat_profiles(), TokamakGeometry(major_radius=6.0, minor_radius=2.0)
    )
    return float(np.sum(source.emissivity_by_reaction["DT"] * source.volume))


def _parabolic_density_dt_rate() -> float:
    profiles = PlasmaProfiles(
        ion_temperature=RadialProfile.parabolic(10.0, 10.0),
        ion_density=RadialProfile.from_callable(lambda rho: 1.0e20 * (1.0 - 0.99 * rho**2)),
    )
    source = SpatialNeutronSource.from_profiles(
        profiles, TokamakGeometry(major_radius=6.0, minor_radius=2.0), n_rho=256
    )
    return float(np.sum(source.emissivity_by_reaction["DT"] * source.volume))


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
    BenchmarkCase(
        name="Circular torus volume (R0=6 m, a=2 m)",
        reference="Analytic torus volume V = 2 pi^2 R0 a^2",
        reference_value=2.0 * np.pi**2 * 6.0 * 4.0,
        unit="m^3",
        rtol=1e-3,
        compute=lambda: TokamakGeometry(major_radius=6.0, minor_radius=2.0).volume(),
    ),
    BenchmarkCase(
        name="Elliptical torus volume (kappa=1.7)",
        reference="Analytic torus volume V = 2 pi^2 R0 a^2 kappa",
        reference_value=2.0 * np.pi**2 * 6.0 * 4.0 * 1.7,
        unit="m^3",
        rtol=1e-3,
        compute=lambda: TokamakGeometry(
            major_radius=6.0, minor_radius=2.0, elongation=1.7
        ).volume(),
    ),
    BenchmarkCase(
        name="Flat-profile total DT rate (circular torus)",
        reference=f"Hand calculation: (n^2/4) <sigma*v> V with Table VIII value ({_BOSCH_HALE})",
        reference_value=0.25e40 * 1.136e-22 * 2.0 * np.pi**2 * 6.0 * 4.0,
        unit="1/s",
        rtol=1e-2,
        compute=_flat_profile_dt_rate,
    ),
    BenchmarkCase(
        name="Blanket torus-shell volume (R0=6, r 2->3 m)",
        reference="Pappus theorem, V = 2 pi^2 R0 (r_out^2 - r_in^2)",
        reference_value=2.0 * np.pi**2 * 6.0 * (9.0 - 4.0),
        unit="m^3",
        rtol=1e-9,
        compute=lambda: sum(_demo_blanket().layer_volumes()),
    ),
    BenchmarkCase(
        name="First-wall torus area (R0=6, r_fw=2 m)",
        reference="Torus surface area A = 4 pi^2 R0 r",
        reference_value=4.0 * np.pi**2 * 6.0 * 2.0,
        unit="m^2",
        rtol=1e-9,
        compute=lambda: _demo_blanket().first_wall_area(),
    ),
    BenchmarkCase(
        name="Pb-17Li lithium atom fraction",
        reference="Eutectic Pb-17Li (E. Mas de les Valls et al., J. Nucl. Mater. 376 (2008) 353)",
        reference_value=0.17,
        unit="",
        rtol=1e-6,
        compute=lambda: sum(
            fraction
            for species, fraction in pbli().atom_fractions().items()
            if species.startswith("Li")
        ),
    ),
    BenchmarkCase(
        name="Li4SiO4 natural Li-6 atom fraction",
        reference="CIAAW isotopic abundance of Li-6 (7.59 at%) over 4 Li per 9 atoms",
        reference_value=4.0 / 9.0 * 0.0759,
        unit="",
        rtol=1e-3,
        compute=lambda: li4sio4().atom_fractions()["Li6"],
    ),
    BenchmarkCase(
        name="Tungsten atom number density",
        reference="rho N_A / A with CRC density 19.30 g/cm^3, A = 183.84 g/mol",
        reference_value=6.322e28,
        unit="m^-3",
        rtol=1e-3,
        compute=lambda: tungsten().atom_density,
    ),
    BenchmarkCase(
        name="NRT displacements per 1 keV damage energy (E_d=40 eV)",
        reference=(
            "M.J. Norgett, M.T. Robinson and I.M. Torrens, Nucl. Eng. Des. 33 (1975) 50: "
            "0.8 E_dam / (2 E_d)"
        ),
        reference_value=10.0,
        unit="displacements",
        rtol=1e-9,
        compute=lambda: nrt_dpa_rate(1000.0, 1.0),
    ),
    BenchmarkCase(
        name="QMC propagation mean (identity, N(10, 2))",
        reference="Analytic: mean of the identity map equals the input mean",
        reference_value=10.0,
        unit="",
        rtol=1e-3,
        compute=lambda: _identity_propagation().mean,
    ),
    BenchmarkCase(
        name="QMC propagation std (identity, N(10, 2))",
        reference="Analytic: std of the identity map equals the input std",
        reference_value=2.0,
        unit="",
        rtol=1e-2,
        compute=lambda: _identity_propagation().std,
    ),
    BenchmarkCase(
        name="Ishigami first-order index S1",
        reference=(
            "Analytic (a=7, b=0.1): S1 = (1/2)(1 + b pi^4/5)^2 / V = 0.31391 "
            "(Saltelli et al., Comput. Phys. Commun. 181 (2010) 259)"
        ),
        reference_value=0.31391,
        unit="",
        rtol=5e-2,
        compute=lambda: _ishigami_indices().first_order["x1"],
    ),
    BenchmarkCase(
        name="MH posterior mean (conjugate normal-normal)",
        reference=(
            "Analytic conjugacy: prior N(0,1), observation 2.0 with noise 1 "
            "gives posterior N(1, 1/2)"
        ),
        reference_value=1.0,
        unit="",
        rtol=5e-2,
        compute=_conjugate_posterior_mean,
    ),
    BenchmarkCase(
        name="GP surrogate recovery of sin(1.0)",
        reference="Analytic: sin(1.0) = 0.841471 (GP trained on 12 points of sin over [0, 2 pi])",
        reference_value=float(np.sin(1.0)),
        unit="",
        rtol=1e-3,
        compute=_gp_sin_recovery,
    ),
    BenchmarkCase(
        name="Ishigami first-order index S2",
        reference="Analytic (a=7, b=0.1): S2 = (a^2/8) / V = 0.44241",
        reference_value=0.44241,
        unit="",
        rtol=5e-2,
        compute=lambda: _ishigami_indices().first_order["x2"],
    ),
    BenchmarkCase(
        name="Parabolic-density total DT rate",
        reference=(
            "Hand calculation: closed-form integral of n0^2(1 - c rho^2)^2 over a circular "
            "torus, factor (1 - c + c^2/3)"
        ),
        reference_value=0.25e40
        * 1.136e-22
        * 2.0
        * np.pi**2
        * 6.0
        * 4.0
        * (1.0 - 0.99 + 0.99**2 / 3.0),
        unit="1/s",
        rtol=1e-2,
        compute=_parabolic_density_dt_rate,
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
            models=[
                bosch_hale.MODEL_ID,
                spectra.MODEL_ID,
                geometry.MODEL_ID,
                materials.MODEL_ID,
                neutronics.MODEL_ID,
                uncertainty.MODEL_ID,
                uncertainty.SALTELLI_MODEL_ID,
            ],
            inputs={"cases": len(CASES)},
        ),
    )
    if verbose:
        print(report)
    return report

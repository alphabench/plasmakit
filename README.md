# fusionbench

A fusion nuclear physics engine for Python: it connects plasma conditions to
fusion reaction rates, neutron spectra, and power densities — and every
calculation is validated against published references you can re-run yourself
with `fusionbench.validate()`.

`fusionbench` is the plasma → neutron-source layer of fusion nuclear
engineering. It is a library, not a workflow: small immutable objects, pure
NumPy-vectorized functions, and first-class provenance so results are
citable and reproducible.

## Installation

```bash
pip install fusionbench
# or
uv add fusionbench
```

Requires Python ≥ 3.10. The only runtime dependency is NumPy.

## Quickstart

```python
import fusionbench as fb

plasma = fb.PlasmaState(
    ion_temperature=15.0,  # keV
    ion_density=1.0e20,  # m^-3
    fuel={"D": 0.5, "T": 0.5},
)

source = fb.NeutronSource(plasma)

source.rate_density()  # neutrons / m^3 / s
source.mean_energy()  # keV (~14,053 for D-T at 15 keV)
source.spectrum().fwhm  # thermal Doppler width, keV
source.power_density().neutron  # W / m^3 carried by neutrons
source.power_density().charged  # W / m^3 heating the plasma (alphas)

print(source.provenance.to_json())  # models, citations, inputs, version
```

Functional style works equally well, and everything is vectorized:

```python
import numpy as np
import fusionbench as fb

temperatures = np.linspace(1.0, 100.0, 500)  # keV
sigma_v = fb.maxwellian_reactivity("DT", temperatures)  # m^3/s, shape (500,)
sigma = fb.cross_section("DT", 64.0)  # m^2 at 64 keV (CM)
```

## Validate the physics

Every physics model in the package ships with a benchmark suite comparing
computed values against published reference values — the same cases the test
suite enforces in CI:

```python
import fusionbench as fb

report = fb.validate()
# case                          reference      computed   rel err  status
# DT reactivity at 10 keV      1.1360e-22    1.1362e-22   1.5e-04  PASS
# ...
report.passed  # True
report.to_json()  # citable machine-readable record
```

## What it computes

| Quantity | API | Model |
|---|---|---|
| Fusion cross sections σ(E) | `cross_section` | Bosch–Hale (1992) |
| Maxwellian reactivities ⟨σv⟩(T) | `maxwellian_reactivity` | Bosch–Hale (1992) |
| Reaction rate densities | `reaction_rate_density` | n·n·⟨σv⟩/(1+δ) |
| Fusion power density and neutron/charged split | `fusion_power_density`, `power_partition` | two-body kinematics |
| Neutron mean energy and thermally broadened spectrum | `neutron_spectrum`, `neutron_mean_energy` | Brysk (1973) |
| Neutron source summary with provenance | `NeutronSource` | all of the above |

Supported reactions: **D-T**, **D-D** (both branches), **D-³He**. Q values and
product birth energies are derived from CODATA 2018 nuclide masses, not
hand-copied constants.

## Units

One convention, everywhere. No unit-wrapper objects:

| Quantity | Unit |
|---|---|
| Ion temperature, particle energies, Q values | keV |
| Number density | m⁻³ |
| Cross section | m² |
| Reactivity ⟨σv⟩ | m³/s |
| Rate density | m⁻³ s⁻¹ |
| Power density | W/m³ |

## Validity ranges

The Bosch–Hale reactivity fits are accurate to ≤ 0.25% RMS inside their
fitted ranges (0.2–100 keV for D-T and D-D; 0.5–190 keV for D-³He). Outside
those ranges the package still returns the fit's value but emits a
`ValidityRangeWarning` — it never silently extrapolates without telling you.

## Scope and limitations (v0.1)

- 0-D Maxwellian plasmas (arrays of point values are supported; spatial
  geometry is not — yet).
- Thermonuclear reactivities only; no beam–target or non-Maxwellian effects.
- Neutron spectra are Gaussian (Brysk); relativistic/asymmetric corrections
  (Ballabio 1998) are planned.
- Secondary reactions (e.g. D-D triton burnup), radiation losses, transport,
  and blanket neutronics are out of scope for this release.

## References

- H.-S. Bosch and G.M. Hale, "Improved formulas for fusion cross-sections and
  thermal reactivities", *Nuclear Fusion* **32** (1992) 611.
- H. Brysk, "Fusion neutron energies and spectra", *Plasma Physics* **15**
  (1973) 611.

## Development

```bash
uv sync                 # install with dev dependencies
uv run pytest           # test suite (physics regressions + property tests)
uv run ruff check       # lint
uv run mypy             # strict type checking
```

## License

MIT

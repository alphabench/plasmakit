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

Requires Python ≥ 3.10. Runtime dependencies: NumPy and SciPy.

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

## Spatial sources

Real plasmas are not points. Build a spatially resolved neutron source from
radial profiles on parameterized tokamak flux surfaces (Miller geometry with
elongation, triangularity, and Shafranov shift), or from 2-D R-Z fields:

```python
import fusionbench as fb

profiles = fb.PlasmaProfiles(
    ion_temperature=fb.RadialProfile.parabolic(20.0, 1.0),  # keV, core -> edge
    ion_density=fb.RadialProfile.parabolic(1.0e20, 1.0e18),  # m^-3
)
geometry = fb.TokamakGeometry(
    major_radius=6.0, minor_radius=2.0, elongation=1.7, triangularity=0.33
)

source = fb.SpatialNeutronSource.from_profiles(profiles, geometry)

source.total_rate  # neutrons / s
source.total_fusion_power  # W
source.emissivity  # neutron emissivity field, m^-3 s^-1
```

Export it where you need it:

```python
source.to_openmc()  # weighted openmc.IndependentSource rings (needs openmc)
source.to_xarray()  # labeled xarray.Dataset  (pip install fusionbench[xarray])
source.to_vtk("source.vtk")  # ParaView/VisIt, no VTK dependency needed
```

The OpenMC export creates one axisymmetric ring source per cell and
reaction, each carrying the local thermally broadened (Brysk) energy
spectrum — so the hot core emits harder, wider neutrons than the edge, and
first-wall load calculations see a realistic source instead of an idealized
ring. OpenMC itself is not on PyPI; install it via conda-forge.

Users with equilibrium-code output can use
`SpatialNeutronSource.from_rz(r_edges, z_edges, T_field, n_field)` instead.

## Blanket neutronics

Close the loop from plasma to tritium breeding. Describe a layered
blanket with built-in, literature-cited materials and run an OpenMC
transport calculation against any fusionbench source:

```python
import fusionbench as fb
from fusionbench.materials import beryllium, eurofer97, li4sio4, tungsten

blanket = fb.Blanket(
    layers=(
        fb.Layer("armor", tungsten(), 0.002),
        fb.Layer("first_wall", eurofer97(), 0.02),
        fb.Layer("multiplier", beryllium(), 0.05),
        fb.Layer("breeder", li4sio4(li6_enrichment=0.60), 0.50),
        fb.Layer("shield", eurofer97(), 0.10),
    ),
    major_radius=9.0,
    first_wall_radius=2.9,
)

plasma = fb.PlasmaState(ion_temperature=15.0, ion_density=1.0e20, fuel={"D": 0.5, "T": 0.5})
result = blanket.run_neutronics(plasma, source_rate=1.0e20)  # needs openmc + nuclear data

result.tbr  # tritium breeding ratio, with MC uncertainty
result.neutron_wall_load  # MW/m^2
result.energy_deposition  # W per layer
result.tritium_production  # atoms/s per layer
result.dpa_per_fpy  # first-wall displacement dose per full-power year
result.provenance  # chains source models, nuclear data library, seeds
```

Layers are concentric circular torus shells (a documented approximation
for shaped plasmas — `Blanket.from_geometry` encloses a Miller LCFS
conservatively). Every quantity carries its Monte Carlo standard
deviation; DPA uses the NRT model (E_d = 40 eV, configurable). OpenMC
and a cross-section library are required only for `run_neutronics` —
install OpenMC from conda-forge and point `OPENMC_CROSS_SECTIONS` at a
data library.

## Tritium fuel cycle

TBR > 1 is necessary but not sufficient — a plant must also keep its
tritium inventory alive against decay, recirculation losses, and the
huge unburned-fuel loop. `TritiumCycle` is a linear compartment model
(blanket → processing → storage → plasma exhaust, after Abdou et al.)
solved exactly by matrix exponential:

```python
import fusionbench as fb

cycle = fb.TritiumCycle.from_blanket_result(  # chains from run_neutronics
    result,
    fractional_burnup=0.02,
    startup_inventory=5.0,  # kg
)

cycle.self_sufficient  # does storage accumulate at steady state?
cycle.doubling_time()  # days to bank a second plant's startup inventory
cycle.required_startup_inventory(days=3650.0)  # kg to survive a 10-year horizon
cycle.simulate(days=365.0 * 5).inventory("storage")  # kg vs time
```

`TritiumCycle.from_fusion_power(500.0, tbr=1.1, ...)` starts from a
plant power instead. Units: durations in days, rates in atoms/s,
inventories in kg. Because every method returns plain floats, the UQ
toolbox applies directly — for example, propagating TBR and burnup
uncertainty into the required startup inventory:

```python
result = fb.propagate(
    lambda tbr, fractional_burnup: fb.TritiumCycle(
        burn_rate=1e20, tbr=tbr, fractional_burnup=fractional_burnup, startup_inventory=0.0
    ).required_startup_inventory(),
    {
        "tbr": fb.Distribution.normal(1.05, 0.02),
        "fractional_burnup": fb.Distribution.uniform(0.01, 0.05),
    },
)
```

## Uncertainty quantification and optimization

Reactor parameters are never known exactly. Give any input a
distribution and propagate it through any model — from a fast
reactivity chain to a full transport calculation:

```python
import fusionbench as fb


def fusion_power(ion_temperature, ion_density):
    plasma = fb.PlasmaState(ion_temperature, ion_density, fuel={"D": 0.5, "T": 0.5})
    return fb.fusion_power_density(plasma)


result = fb.propagate(
    fusion_power,
    {
        "ion_temperature": fb.Distribution.lognormal(mean=15.0, std=2.0),
        "ion_density": fb.Distribution.normal(1.0e20, 5.0e18),
    },
    n_samples=10_000,
    vectorized=True,
)
result.mean, result.std, result.percentile(95)

indices = fb.sobol_indices(fusion_power, {...})  # which input drives the variance?
indices.first_order["ion_temperature"]
```

Through OpenMC transport, tally noise is folded into the total
uncertainty (`fb.propagate_transport`), so a TBR comes back as
parametric spread plus Monte Carlo statistics. The toolbox also
includes Bayesian parameter estimation (`fb.fit`, Metropolis–Hastings
with analytic-conjugacy validation), Gaussian-process surrogates
(`fb.Surrogate.from_function` — a drop-in emulator for expensive
models), and global optimization (`fb.optimize`, differential
evolution; `fb.optimize_surrogate` for the expensive case, e.g.
maximizing TBR over Li-6 enrichment in a handful of transport runs).

Every stochastic entry point is seeded and deterministic, and every
result carries provenance: distribution specs, sampler, estimator
citations (Saltelli 2010, Rasmussen–Williams 2006, Storn–Price 1997).

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
| Spatial emissivity fields and exportable sources | `SpatialNeutronSource` | Miller flux surfaces / R-Z mesh |

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

- Maxwellian plasmas: 0-D states, radial profiles on prescribed Miller
  flux surfaces, or user-supplied R-Z fields. Grad–Shafranov equilibria
  and EQDSK input are not read — yet.
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
- R.L. Miller, M.S. Chu, J.M. Greene, Y.R. Lin-Liu and R.E. Waltz,
  "Noncircular, finite aspect ratio, local equilibrium model",
  *Physics of Plasmas* **5** (1998) 973.
- M.A. Abdou et al., "Deuterium-tritium fuel self-sufficiency in fusion
  reactors", *Fusion Technology* **9** (1986) 250; M. Abdou et al.,
  *Nuclear Fusion* **61** (2021) 013001.

## Development

```bash
uv sync                 # install with dev dependencies
uv run pytest           # test suite (physics regressions + property tests)
uv run ruff check       # lint
uv run mypy             # strict type checking
```

## License

MIT

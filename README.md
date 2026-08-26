# PlasmaKit

[![PyPI](https://img.shields.io/pypi/v/plasmakit.svg)](https://pypi.org/project/plasmakit/)
[![Python](https://img.shields.io/pypi/pyversions/plasmakit.svg)](https://pypi.org/project/plasmakit/)
[![Downloads](https://api.pepy.tech/badge/plasmakit/month)](https://pepy.tech/projects/plasmakit)
[![License](https://img.shields.io/pypi/l/plasmakit.svg)](https://opensource.org/licenses/MIT)

**Fusion nuclear engineering, computed with receipts.**

PlasmaKit connects plasma conditions to fusion reaction rates, spatially
resolved neutron sources, blanket neutronics (TBR, wall load, damage),
uncertainty quantification, and the tritium fuel cycle — one Python API
spanning the whole plasma → neutron → blanket → tritium chain. Every physics
model is cited, every calculation is validated against a published or
analytic reference you can re-run yourself, and every result carries a
provenance record.

<p align="center">
  <strong>38 validated benchmarks</strong> · <strong>First-class provenance</strong> · <strong>Plasma → tritium in one chain</strong>
</p>

---

### Quick Install

```bash
pip install plasmakit
# or
uv add plasmakit
```

Requires Python ≥ 3.10. Runtime dependencies: NumPy and SciPy. OpenMC is
optional (transport only) and installed separately from conda-forge.

### 30-Second Example

```python
import plasmakit as pk

# A burning plasma
plasma = pk.PlasmaState(
    ion_temperature=15.0,  # keV
    ion_density=1.0e20,  # m^-3
    fuel={"D": 0.5, "T": 0.5},
)

source = pk.NeutronSource(plasma)
source.rate_density()  # 6.87e17 neutrons / m^3 / s
source.mean_energy()  # 14,053 keV (Brysk thermal shift included)
source.power_density().neutron  # W/m^3 carried by neutrons

# Prove the physics is right
pk.validate()  # 38 benchmark cases vs published references
```

---

## Table of Contents

- [Overview](#overview)
- [Validated Accuracy](#validated-accuracy)
- [Determinism and Provenance](#determinism-and-provenance)
- [Plasma States and Reactions](#plasma-states-and-reactions)
- [Spatial Neutron Sources](#spatial-neutron-sources)
- [Blanket Neutronics](#blanket-neutronics)
- [Tritium Fuel Cycle](#tritium-fuel-cycle)
- [Uncertainty Quantification and Optimization](#uncertainty-quantification-and-optimization)
- [End-to-End Worked Example](#end-to-end-worked-example)
- [API Reference](#api-reference)
- [Units](#units)
- [Building from Source](#building-from-source)
- [References](#references)
- [License](#license)
- [Changelog](#changelog)

---

## Overview

PlasmaKit is the *interface layer* of fusion nuclear engineering: it does not
replace plasma solvers, Monte Carlo transport codes, or systems codes — it
makes them composable. A `PlasmaState` becomes a spatially resolved neutron
source; the source drives an OpenMC blanket calculation; the blanket result
feeds a tritium fuel-cycle model; and the uncertainty toolbox wraps any step
of that chain.

### Key Features

- **Fusion reactions** — D-T, D-D (both branches), D-³He cross sections and
  Maxwellian reactivities via the Bosch–Hale (1992) parameterization; Q
  values and product energies derived from CODATA masses, never hand-copied.
- **Neutron spectra** — thermally broadened Gaussian (Brysk 1973) spectra
  with mean shift, FWHM, pdf, and seeded sampling.
- **Spatial sources** — radial profiles on parameterized Miller flux
  surfaces (elongation, triangularity, Shafranov shift) or user R-Z fields;
  exporters to OpenMC ring sources, xarray, and dependency-free VTK.
- **Blanket neutronics** — layered torus blankets with cited materials;
  OpenMC coupling returning TBR, neutron wall load, per-layer heating and
  tritium production, and NRT DPA, each with Monte Carlo uncertainty.
- **Tritium fuel cycle** — the Abdou compartment model solved exactly by
  matrix exponential: self-sufficiency, doubling time, required startup
  inventory.
- **UQ and optimization** — Sobol QMC propagation, Saltelli sensitivity
  indices, Metropolis–Hastings estimation, Gaussian-process surrogates, and
  differential-evolution optimization, all under one `fn(**params)`
  convention.
- **Validation as a feature** — `pk.validate()` re-runs the entire benchmark
  registry against published references, and the same registry gates CI.

---

## Validated Accuracy

The exact table `pk.validate()` prints — every physics model checked against
a published value, an analytic closed form, or a hand calculation, in one
seeded, deterministic run:

| Case | Reference | Computed | Rel. error |
|---|---|---|---|
| DT reactivity at 1 keV | 6.8570e-27 | 6.8569e-27 | 1.7e-05 |
| DT reactivity at 10 keV | 1.1360e-22 | 1.1362e-22 | 1.5e-04 |
| DT reactivity at 20 keV | 4.3300e-22 | 4.3302e-22 | 4.7e-05 |
| DT reactivity at 50 keV | 8.6490e-22 | 8.6491e-22 | 9.8e-06 |
| DDn reactivity at 10 keV | 6.0230e-25 | 6.0227e-25 | 5.7e-05 |
| DDn reactivity at 20 keV | 2.6030e-24 | 2.6027e-24 | 1.3e-04 |
| DDp reactivity at 10 keV | 5.7810e-25 | 5.7813e-25 | 4.7e-05 |
| DHe3 reactivity at 10 keV | 2.1260e-25 | 2.1261e-25 | 3.4e-05 |
| DT reactivity peak value | 9.0000e-22 | 8.9465e-22 | 5.9e-03 |
| DT reactivity peak temperature | 6.4000e+01 | 6.6600e+01 | 4.1e-02 |
| DT cold neutron energy | 1.4070e+04 | 1.4048e+04 | 1.6e-03 |
| DDn cold neutron energy | 2.4500e+03 | 2.4494e+03 | 2.5e-04 |
| DT spectrum FWHM at 10 keV | 5.5972e+02 | 5.6005e+02 | 5.9e-04 |
| DDn spectrum FWHM at 10 keV | 2.6089e+02 | 2.6096e+02 | 2.6e-04 |
| DT neutron power fraction | 7.9900e-01 | 7.9868e-01 | 4.0e-04 |
| DT power density (50/50, n=1e20 m⁻³, T=10 keV) | 8.0000e+05 | 8.0046e+05 | 5.7e-04 |
| Circular torus volume (R0=6 m, a=2 m) | 4.7374e+02 | 4.7374e+02 | 0.0 |
| Elliptical torus volume (κ=1.7) | 8.0536e+02 | 8.0536e+02 | 0.0 |
| Flat-profile total DT rate (circular torus) | 1.3454e+20 | 1.3456e+20 | 1.5e-04 |
| Parabolic-density total DT rate | 4.5300e+19 | 4.5306e+19 | 1.2e-04 |
| Blanket torus-shell volume (R0=6, r 2→3 m) | 5.9218e+02 | 5.9218e+02 | 0.0 |
| First-wall torus area (R0=6, r_fw=2 m) | 4.7374e+02 | 4.7374e+02 | 0.0 |
| Pb-17Li lithium atom fraction | 1.7000e-01 | 1.7000e-01 | 0.0 |
| Li4SiO4 natural Li-6 atom fraction | 3.3733e-02 | 3.3733e-02 | 2.1e-16 |
| Tungsten atom number density | 6.3220e+28 | 6.3222e+28 | 3.2e-05 |
| NRT displacements per 1 keV damage (E_d=40 eV) | 1.0000e+01 | 1.0000e+01 | 0.0 |
| Tritium storage decay over one half-life | 5.0000e-01 | 5.0000e-01 | 4.1e-11 |
| Tritium blanket steady-state inventory | 4.7525e-01 | 4.7525e-01 | 0.0 |
| Tritium mass conservation (TBR=1, lossless) | 1.0000e+00 | 1.0000e+00 | 2.1e-11 |
| Tritium net accumulation rate at steady state | -4.6917e-03 | -4.6917e-03 | 2.2e-11 |
| Tritium doubling time (linear limit) | 1.5407e+02 | 1.5409e+02 | 1.7e-04 |
| QMC propagation mean (identity, N(10, 2)) | 1.0000e+01 | 1.0000e+01 | 5.8e-08 |
| QMC propagation std (identity, N(10, 2)) | 2.0000e+00 | 1.9992e+00 | 3.8e-04 |
| Ishigami first-order index S1 | 3.1391e-01 | 3.1255e-01 | 4.3e-03 |
| Ishigami first-order index S2 | 4.4241e-01 | 4.4138e-01 | 2.3e-03 |
| MH posterior mean (conjugate normal-normal) | 1.0000e+00 | 1.0386e+00 | 3.9e-02 |
| GP surrogate recovery of sin(1.0) | 8.4147e-01 | 8.4147e-01 | 6.6e-06 |
| Differential-evolution Rosenbrock minimizer | 1.0000e+00 | 1.0000e+00 | 0.0 |

References include Bosch & Hale (Nucl. Fusion 32, 1992), Brysk (Plasma Phys.
15, 1973), Miller et al. (Phys. Plasmas 5, 1998), Norgett–Robinson–Torrens
(Nucl. Eng. Des. 33, 1975), Abdou et al. (Fusion Technology 9, 1986),
Saltelli et al. (Comput. Phys. Commun. 181, 2010), and analytic closed
forms. `pk.validate().to_json()` produces the same table as a citable
machine-readable record.

## Determinism and Provenance

- **Every stochastic entry point takes a seed** — QMC sampling, Sobol
  indices, MCMC chains, GP training restarts, differential evolution, and
  spectrum sampling are all bit-reproducible for a fixed seed.
- **Every derived result knows how it was made.** `result.provenance`
  carries the package version, the physics-model identifiers used (e.g.
  `bosch-hale-1992`, `brysk-1973`, `nrt-1975`), their full citations, the
  physical inputs, and — for transport — the nuclear-data library, particle
  count, and RNG seed. `provenance.to_json()` is a reproducible
  computational record; blanket results chain the source's provenance so the
  trail runs unbroken from the Bosch–Hale coefficients to the TBR.

---

## Plasma States and Reactions

The registry derives all energetics from CODATA 2018 nuclide masses at
import time — Q values and two-body product energies are computed, not
transcribed:

| ID | Reaction | Q (keV) | E_neutron (keV) | E_charged (keV) |
|---|---|---|---|---|
| `DT` | T(d,n)⁴He | 17589.3 | 14048.1 | 3541.1 |
| `DDn` | D(d,n)³He | 3268.9 | 2449.4 | 819.5 |
| `DDp` | D(d,p)T | 4032.7 | — | 4032.7 |
| `DHe3` | ³He(d,p)⁴He | 18353.1 | — | 18353.1 |

```python
import numpy as np
import plasmakit as pk

# Everything is vectorized: scalar in → float out, array in → array out
temperatures = np.linspace(1.0, 100.0, 500)  # keV
sigma_v = pk.maxwellian_reactivity("DT", temperatures)  # m^3/s, shape (500,)
sigma = pk.cross_section("DT", 64.0)  # m^2 at 64 keV (CM)

spec = pk.neutron_spectrum("DT", ion_temperature=10.0)
spec.mean_energy, spec.fwhm  # keV; FWHM ≈ 177·√T
spec.sample(10_000, rng=np.random.default_rng(0))  # seeded draws
```

Out-of-range inputs never extrapolate silently — a `ValidityRangeWarning`
names the model and its fitted range.

## Spatial Neutron Sources

Real plasmas are not rings. Build a cell-resolved neutron source from radial
profiles on shaped flux surfaces, or from 2-D R-Z fields out of an
equilibrium code:

```python
profiles = pk.PlasmaProfiles(
    ion_temperature=pk.RadialProfile.parabolic(20.0, 1.0),  # keV, core → edge
    ion_density=pk.RadialProfile.parabolic(1.0e20, 1.0e18),  # m^-3
)
geometry = pk.TokamakGeometry(
    major_radius=6.0, minor_radius=2.0, elongation=1.7, triangularity=0.33
)

source = pk.SpatialNeutronSource.from_profiles(profiles, geometry)
source.total_rate  # neutrons / s
source.total_fusion_power  # W
source.emissivity  # per-cell neutron emissivity field, m^-3 s^-1

source.to_openmc()  # weighted openmc.IndependentSource rings
source.to_xarray()  # labeled Dataset (pip install plasmakit[xarray])
source.to_vtk("source.vtk")  # ParaView/VisIt, no VTK dependency needed
```

The OpenMC export builds one axisymmetric ring per cell and reaction, each
carrying the local Brysk spectrum — the hot core emits harder, wider
neutrons than the edge, so wall-load calculations see a realistic source.
`SpatialNeutronSource.from_rz(r_edges, z_edges, T_field, n_field)` accepts
equilibrium-code output directly. Cell volumes are validated against
analytic torus integrals, and flat profiles reproduce the 0-D results
exactly.

## Blanket Neutronics

Layered torus blankets with a built-in, literature-cited materials registry:

| Factory | ρ (g/cm³) | Composition | Source |
|---|---|---|---|
| `tungsten()` | 19.30 | W (ao) | CRC Handbook, 97th ed. |
| `beryllium()` | 1.848 | Be (ao) | CRC Handbook, 97th ed. |
| `eurofer97()` | 7.798 | Fe 0.891, Cr 0.09, W 0.011, Mn, V, Ta, C (wo) | Mergia & Boukos, J. Nucl. Mater. 373 (2008) |
| `li4sio4(li6_enrichment)` | 2.40 | Li 4/9 (Li6/Li7 split), Si 1/9, O 4/9 (ao) | Knitter et al., J. Nucl. Mater. 442 (2013) |
| `pbli(li6_enrichment)` | 9.84 | Pb 0.83, Li 0.17 (ao), eutectic at 573 K | Mas de les Valls et al., J. Nucl. Mater. 376 (2008) |
| `water()` | 0.998 | H₂O (ao) | CRC Handbook, 97th ed. |
| `helium(density)` | 0.0057 | He (8 MPa, 673 K default) | ideal-gas estimate |

```python
from plasmakit.materials import beryllium, eurofer97, li4sio4, tungsten

blanket = pk.Blanket(
    layers=(
        pk.Layer("armor", tungsten(), 0.002),
        pk.Layer("first_wall", eurofer97(), 0.02),
        pk.Layer("multiplier", beryllium(), 0.05),
        pk.Layer("breeder", li4sio4(li6_enrichment=0.60), 0.50),
        pk.Layer("shield", eurofer97(), 0.10),
    ),
    major_radius=9.0,
    first_wall_radius=2.9,
)

result = blanket.run_neutronics(plasma, source_rate=7.1e20)  # needs openmc + data

result.tbr  # TallyValue: value ± Monte Carlo std
result.neutron_wall_load  # MW/m^2
result.energy_deposition  # W per layer
result.tritium_production  # atoms/s per layer
result.dpa_per_fpy  # first-wall displacement dose per full-power year
```

Layers are concentric circular torus shells (`Blanket.from_geometry`
conservatively encloses a shaped Miller LCFS). DPA uses the NRT model
(E_d = 40 eV, configurable). OpenMC is not on PyPI — install from
conda-forge and point `OPENMC_CROSS_SECTIONS` at a data library; everything
except `run_neutronics`/`to_openmc` works without it.

## Tritium Fuel Cycle

TBR > 1 is necessary but not sufficient — the plant must keep its inventory
alive against decay, recirculation losses, and the huge unburned-fuel loop.
`TritiumCycle` is the Abdou four-compartment linear model (blanket, exhaust,
processing, storage) solved **exactly** by matrix exponential — no ODE
tolerances:

```python
cycle = pk.TritiumCycle.from_blanket_result(
    result,
    fractional_burnup=0.02,
    startup_inventory=5.0,  # kg
)

cycle.self_sufficient  # storage accumulating?
cycle.accumulation_rate()  # kg/day at steady state
cycle.doubling_time()  # days to bank a 2nd startup inventory
cycle.required_startup_inventory(days=3650.0)  # kg to survive a 10-year horizon
cycle.simulate(days=365 * 5).inventory("storage")  # kg vs time
```

`TritiumCycle.from_fusion_power(2000.0, tbr=1.15, ...)` starts from plant
power instead. The model reproduces the classic Abdou result: at low
fractional burnup, recirculation losses can defeat even TBR = 1.15.

## Uncertainty Quantification and Optimization

One convention everywhere: models are callables invoked as `fn(**params)`,
so the same function drives propagation, sensitivity, fitting, surrogates,
and optimization.

```python
def fusion_power(ion_temperature, ion_density):
    plasma = pk.PlasmaState(ion_temperature, ion_density, fuel={"D": 0.5, "T": 0.5})
    return pk.fusion_power_density(plasma)


params = {
    "ion_temperature": pk.Distribution.lognormal(mean=15.0, std=2.0),
    "ion_density": pk.Distribution.normal(1.0e20, 5.0e18),
}

result = pk.propagate(fusion_power, params, n_samples=10_000, vectorized=True)
result.mean, result.std, result.percentile(95)

indices = pk.sobol_indices(fusion_power, params)  # which input drives the variance?
indices.first_order["ion_temperature"], indices.total_order["ion_temperature"]

posterior = pk.fit(fusion_power, priors=params, observed=8.0e5, noise_std=5.0e4)
posterior.mean("ion_temperature"), posterior.map_estimate

surrogate = pk.Surrogate.from_function(fusion_power, params, n_train=64)
best = pk.optimize(lambda x, y: (x - 3) ** 2 + y**2, {"x": (0, 5), "y": (-2, 2)})
```

Through transport, `pk.propagate_transport` folds the per-run Monte Carlo
tally variance into the total: `std² = parametric spread + mean tally
noise`. For expensive objectives, `pk.optimize_surrogate` runs a one-shot
QMC design → GP → optimize → verify loop and reports both the surrogate
prediction and the verified true value.

---

## End-to-End Worked Example

The full chain on a DEMO-scale machine (measured with OpenMC 0.15.3 on
ENDF/B-VII.1, 10k particles — reproduce with the blanket above):

```python
plasma = pk.PlasmaState(ion_temperature=15.0, ion_density=1.0e20, fuel={"D": 0.5, "T": 0.5})
result = blanket.run_neutronics(plasma, particles=10_000, source_rate=7.1e20)  # ~2 GW fusion
# result.tbr                      -> 1.259 ± 0.006
# result.neutron_wall_load        -> 0.245 MW/m^2 (analytic uncollided: 0.219)

cycle = pk.TritiumCycle.from_blanket_result(
    result,
    fractional_burnup=0.02,
    startup_inventory=5.0,
    extraction_efficiency=0.98,
    processing_loss=1e-4,
)
# cycle.self_sufficient              -> True (accumulating 66.9 g/day)
# cycle.required_startup_inventory() -> 20.3 kg for a 10-year horizon
# cycle.doubling_time()              -> 404 days
```

Every number above sits inside published DEMO-study ranges, and
`result.provenance` records the nuclear-data library, seeds, and the full
model chain that produced it.

---

## API Reference

All names below are importable from the top-level `plasmakit` namespace.

### Core physics

| Name | Kind | Purpose |
|---|---|---|
| `PlasmaState(ion_temperature, ion_density, fuel)` | class | Immutable 0-D (vectorizable) plasma state; `.density(species)`, `.to_dict()` |
| `REACTIONS` / `Reaction` | registry | Reaction metadata; `.q_value`, `.neutron_energy`, `.charged_energy`, `.product_energy(species)` |
| `cross_section(reaction, energy)` | function | σ(E_cm) in m² (Bosch–Hale) |
| `maxwellian_reactivity(reaction, T)` | function | ⟨σv⟩ in m³/s (Bosch–Hale) |
| `reaction_rate_density(state, reaction)` | function | m⁻³ s⁻¹ with the identical-reactant factor |
| `fusion_power_density(state)` / `power_partition(state)` | function | W/m³; `PowerPartition(.neutron, .charged, .total)` |
| `neutron_spectrum(reaction, T)` / `NeutronSpectrum` | function/class | Brysk Gaussian; `.mean_energy`, `.std`, `.fwhm`, `.pdf(E)`, `.sample(n, rng)` |
| `neutron_mean_energy` / `neutron_std` | function | Vectorized Brysk moments |
| `NeutronSource(plasma)` | class | 0-D façade: `.reactivity()`, `.rate_density()`, `.mean_energy()`, `.spectrum()`, `.power_density()`, `.provenance`, `.to_json()` |

### Spatial sources and geometry

| Name | Kind | Purpose |
|---|---|---|
| `RadialProfile` | class | Interpolated ρ-profile; `.parabolic(center, edge)`, `.from_callable(f)`, callable |
| `PlasmaProfiles(ion_temperature, ion_density, fuel)` | class | Profile plasma; `.state_at(rho)` |
| `TokamakGeometry(R0, a, elongation, triangularity, shafranov_shift)` | class | Miller surfaces; `.flux_surface(ρ, θ)`, `.jacobian`, `.volume()` |
| `SpatialNeutronSource` | class | `.from_profiles(profiles, geometry)`, `.from_rz(...)`; fields `.emissivity`, `.volume`, `.total_rate`, `.total_fusion_power`; `.source_terms()`, `.to_openmc()`, `.to_xarray()`, `.to_vtk(path)` |
| `SourceTerms` | class | Flattened ring sources: `.r`, `.z`, `.strength`, `.energy_mean`, `.energy_std`, `.reaction_id` |

### Blanket and transport

| Name | Kind | Purpose |
|---|---|---|
| `Material` / `MATERIALS` | class/registry | Density + composition; `.atom_fractions()`, `.atom_density`, `.to_openmc()` |
| `Layer(name, material, thickness)` | class | One blanket layer |
| `Blanket(layers, major_radius, first_wall_radius)` | class | `.from_geometry(...)`, `.layer_volumes()`, `.first_wall_area()`, `.run_neutronics(source, ...)` |
| `BlanketResult` | class | `.tbr`, `.neutron_wall_load`, `.energy_deposition`, `.tritium_production`, `.dpa`, `.dpa_per_fpy` — all `TallyValue(value, std_dev)` — plus `.provenance` |

### Tritium fuel cycle

| Name | Kind | Purpose |
|---|---|---|
| `TritiumCycle(burn_rate, tbr, fractional_burnup, startup_inventory, ...)` | class | `.from_fusion_power(MW, ...)`, `.from_blanket_result(result, ...)`, `.simulate(days)`, `.steady_state()`, `.accumulation_rate()`, `.self_sufficient`, `.doubling_time()`, `.required_startup_inventory(days)` |
| `CycleHistory` | class | `.times` (days), `.inventory(name)` (kg), `.total()`, `.to_dict()` |

### Uncertainty and optimization

| Name | Kind | Purpose |
|---|---|---|
| `Distribution` | class | `.normal(mean, std)`, `.lognormal(mean, std)`, `.uniform(low, high)`, `.triangular(low, mode, high)`; `.sample`, `.ppf`, `.logpdf`, `.mean`, `.std` |
| `propagate(fn, params, n_samples, seed, method, vectorized)` | function | Sobol-QMC propagation → `UncertainResult(.mean, .std, .percentile(q), .samples)` |
| `propagate_transport(fn, params, n_samples, seed)` | function | Same, for `TallyValue`-returning models; tally variance folded in |
| `sobol_indices(fn, params, n_samples, seed)` | function | Saltelli/Jansen → `SobolIndices(.first_order, .total_order, bootstrap stds)` |
| `fit(fn, priors, observed, noise_std, ...)` | function | Metropolis–Hastings → `Posterior(.mean, .std, .percentile, .map_estimate, .acceptance_rate)` |
| `GaussianProcess.train(x, y)` / `Surrogate.from_function(fn, params)` | class | RBF/ARD GP emulator; surrogate is a drop-in callable |
| `optimize(objective, bounds, constraints)` / `optimize_surrogate(fn, bounds)` | function | Differential evolution → `OptimizationResult(.best_parameters, .best_value, .n_evaluations, .surrogate_value)` |

### Everything else

`validate()` / `BenchmarkReport`, `Provenance`, and `TallyValue` round out
the public API. Package errors derive from `plasmakit.errors.PlasmakitError`.

---

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
| Lengths, areas, volumes | m, m², m³ |
| Fuel-cycle durations | days |
| Tritium inventories | kg |

## Building from Source

```bash
git clone git@github.com:alphabench/plasmakit.git
cd plasmakit
uv sync          # runtime + dev dependencies into .venv
```

### Verification Test

```bash
uv run ruff check && uv run ruff format --check          # style
uv run mypy                                              # strict typing
uv run pytest                                            # 300+ tests
uv run python -c "import plasmakit as pk; pk.validate()" # 38/38 PASS
```

For the OpenMC-coupled transport tests (marked `transport`), see
[CONTRIBUTING.md](CONTRIBUTING.md) — conda-forge OpenMC plus a targeted
~130 MB ENDF/B-VII.1 nuclide library.

## References

- H.-S. Bosch and G.M. Hale, *Nuclear Fusion* **32** (1992) 611 — cross
  sections and reactivities.
- H. Brysk, *Plasma Physics* **15** (1973) 611 — neutron spectra.
- R.L. Miller et al., *Physics of Plasmas* **5** (1998) 973 — flux-surface
  geometry.
- M.J. Norgett, M.T. Robinson and I.M. Torrens, *Nucl. Eng. Des.* **33**
  (1975) 50 — NRT displacement damage.
- P.K. Romano et al., *Annals of Nuclear Energy* **82** (2015) 90 — OpenMC.
- M.A. Abdou et al., *Fusion Technology* **9** (1986) 250 and *Nuclear
  Fusion* **61** (2021) 013001 — tritium fuel cycle and self-sufficiency.
- A. Saltelli et al., *Comput. Phys. Commun.* **181** (2010) 259 — Sobol
  sensitivity estimators.
- C.E. Rasmussen and C.K.I. Williams, *Gaussian Processes for Machine
  Learning*, MIT Press (2006).
- R. Storn and K. Price, *J. Global Optimization* **11** (1997) 341 —
  differential evolution.

## License

MIT — see [LICENSE](LICENSE).

## Changelog

Canonical history lives in [CHANGELOG.md](CHANGELOG.md).

### v0.1.0 — 2026-08-26

First public release, spanning the full plasma → tritium chain:

- 0-D physics core: `PlasmaState`, Bosch–Hale cross sections and
  reactivities, CODATA-derived reaction kinematics, Brysk neutron spectra,
  `NeutronSource` with first-class provenance.
- Spatially resolved sources: radial profiles on Miller flux surfaces or
  R-Z fields, with OpenMC / xarray / dependency-free VTK exporters.
- Blanket neutronics via OpenMC: cited materials registry, layered torus
  blankets, TBR / wall load / heating / tritium production / NRT DPA with
  Monte Carlo uncertainties and nuclear-data provenance.
- Uncertainty toolbox: `Distribution` specs, Sobol-QMC `propagate`,
  Saltelli `sobol_indices`, transport-aware propagation,
  Metropolis–Hastings `fit`, Gaussian-process surrogates, and
  differential-evolution `optimize` / `optimize_surrogate`.
- Tritium fuel cycle: exact matrix-exponential Abdou compartment model with
  self-sufficiency, doubling-time, and startup-inventory analyses.
- 38-case benchmark registry behind `plasmakit.validate()`, shared with CI.

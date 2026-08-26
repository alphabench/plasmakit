# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

### Added

- `PlasmaState`: immutable 0-D (vectorizable) plasma state with fuel composition.
- Fusion reaction registry (`REACTIONS`): D-T, D-D (both branches), D-3He with
  Q values and two-body product energies derived from nuclide masses.
- Cross sections `cross_section` and Maxwellian reactivities
  `maxwellian_reactivity` via the Bosch-Hale (1992) parameterization.
- Reaction rate densities, fusion power density, and neutron/charged power
  partition (`rates` module).
- Thermally broadened Gaussian neutron spectra (Brysk 1973 model) with
  mean energy, FWHM, pdf, and sampling.
- `NeutronSource` facade tying plasma state to reactivities, spectra, and power.
- First-class provenance records (`Provenance`) with model identifiers and
  citations, JSON round-trip.
- `fusionbench.validate()`: benchmark suite comparing computed values against
  published references, shared with the test suite.
- Spatially resolved neutron sources (`SpatialNeutronSource`): radial
  profiles (`RadialProfile`, `PlasmaProfiles`) on Miller-like tokamak flux
  surfaces (`TokamakGeometry`, Phys. Plasmas 5 (1998) 973) or 2-D R-Z
  fields, with cell-resolved emissivity, power density, and volumes
  validated against analytic torus integrals.
- Exporters: `to_openmc()` (weighted ring sources with local Brysk
  spectra; openmc via conda), `to_xarray()` (optional `[xarray]` extra),
  and `to_vtk()` (pure-NumPy legacy VTK writer).
- Vectorized `neutron_std` alongside `neutron_mean_energy`.

### Changed

- NumPy requirement raised to `>=2.0`.

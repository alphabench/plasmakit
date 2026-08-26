This file provides guidance to coding agents like Pier Code (piercode.com) when working with code in this repository.

## Project

PlasmaKit (`import plasmakit`, PyPI `plasmakit`) is a pure-Python fusion
nuclear engineering library spanning the plasma → neutron source → blanket →
tritium chain. Runtime deps are NumPy (≥2.0) and SciPy only; OpenMC is an
optional, conda-forge-only transport backend.

## Commands

```bash
uv sync                      # install runtime + dev deps into .venv
uv run pytest                # full suite (~300 tests; transport tests auto-skip)
uv run pytest tests/test_tritium.py            # one file
uv run pytest tests/test_rates.py::test_dt_neutron_power_fraction  # one test
uv run ruff check            # lint
uv run ruff format --check   # formatting (run `ruff format` to fix)
uv run mypy                  # strict typing on src/
uv run python -c "import plasmakit as pk; pk.validate()"  # 38-case physics benchmark
uv build                     # sdist + wheel
```

All four gates (ruff check, format, mypy, pytest) must pass before a commit.
Beware of shell pipelines masking exit codes (`pytest | tail` reports tail's
status) — check exit codes explicitly when gating.

### OpenMC transport tests

Tests marked `transport` need OpenMC plus nuclear data and are skipped
otherwise. OpenMC has no macOS-arm64 build anywhere; this machine has an
osx-64 (Rosetta) micromamba env already set up:

```bash
micromamba run -r ~/micromamba -n plasmakit-openmc pip install -e .
OPENMC_CROSS_SECTIONS=~/nuclear_data/plasmakit/cross_sections.xml \
  micromamba run -r ~/micromamba -n fusionbench-openmc python -m pytest tests/ -m transport
```

## Architecture

The package is a chain of thin layers; each module consumes the one before
through a small seam, and physics never lives in two places:

```
plasma.PlasmaState ──► reactivity/rates (Bosch–Hale ⟨σv⟩, power partition)
        │                      │
profiles.PlasmaProfiles ──► spatial.SpatialNeutronSource ──► SourceTerms (ring list)
        │   (state_at(ρ) reuses the 0-D rates unchanged)          │
geometry.TokamakGeometry (Miller surfaces, analytic volumes)      ▼
                                     blanket.Blanket ──► neutronics.run_neutronics (OpenMC)
                                                              │  BlanketResult (TallyValue ±σ)
                                                              ▼
                                     tritium.TritiumCycle.from_blanket_result
uncertainty/estimation/surrogates/optimization wrap ANY of the above via fn(**params)
```

Key seams to preserve:

- `PlasmaProfiles.state_at(rho)` returns an array-valued `PlasmaState`, so
  every 0-D physics function serves the spatial layer without modification.
- `SpatialNeutronSource.source_terms()` is the OpenMC-free flattening of a
  spatial source into weighted rings; all source physics is testable there
  without OpenMC. `to_openmc()`/`build_model()` only translate it.
- `BlanketResult.from_tallies(...)` does all transport post-processing on
  plain numbers — unit-testable with synthetic tallies.
- `tritium.TritiumCycle` is a linear (LTI) system solved exactly by the
  augmented matrix exponential; do not introduce ODE solvers or anything
  that breaks linearity without redesign.
- The Phase-4 toolbox (`propagate`, `sobol_indices`, `fit`, `Surrogate`,
  `optimize`) shares one convention: models are callables invoked as
  `fn(**params)` with parameter names as keywords.

## Non-negotiable conventions

- **Benchmark registry is the source of truth.** `benchmarks.CASES` drives
  both `plasmakit.validate()` and `tests/test_benchmarks.py`. Any physics
  change must keep all cases passing; any new physics model must add a
  `MODEL_ID` string, a full citation in `provenance.MODEL_REFERENCES`, and
  at least one benchmark case with a citable analytic or published value.
  `validate()` must stay runnable without OpenMC installed.
- **Provenance**: derived-result objects expose `.provenance` built via
  `provenance.build_provenance(models, inputs)`; every model id must exist
  in `MODEL_REFERENCES` (unknown ids raise). Blanket results chain the
  source's provenance.
- **OpenMC and xarray imports are function-local** (never module-level) so
  the package imports without them; their tests use
  `pytest.importorskip`, plus the `transport` marker + an
  `OPENMC_CROSS_SECTIONS` skipif for anything running actual transport.
  mypy overrides already ignore `openmc.*`/`xarray.*`/`scipy.*` stubs.
- **Units are fixed conventions, not wrapper objects**: keV, m, m⁻³, m²,
  m³/s, W/m³ everywhere — except `tritium.py`, where API durations are days
  and inventories kg (internally SI seconds/atoms). Document units in every
  public docstring.
- **Scalar/array contract**: public functions accept scalar or ndarray via
  `constants.as_float64` and return through `constants.scalar_like`
  (scalar in → float out, array in → same-shape array out).
- **Style**: frozen dataclasses with `__post_init__` validation raising
  `errors.PlasmakitError` and `MappingProxyType` freezing; noun-named
  modules; NumPy-style docstrings on the public API (ruff enforces);
  JSON-safe `to_dict()` on result objects; every stochastic entry point
  takes a seed and is deterministic for it. Out-of-range physics inputs
  warn via `errors.ValidityRangeWarning` (never silently extrapolate);
  pytest is configured to error on that warning unless a test opts in.
- **Git**: every commit message ends with the trailer
  `Co-authored-by: Pier Code <no-reply@piercode.com>`.

## Releases

Tag `vX.Y.Z` and publish a GitHub Release; `.github/workflows/release.yml`
gates on pytest plus a built-wheel `validate()` run, then publishes to PyPI
via trusted publishing (OIDC, environment `pypi` — no tokens). Version lives
in both `pyproject.toml` and `src/plasmakit/__init__.py.__version__`;
CHANGELOG.md is canonical (the README changelog section is a summary).

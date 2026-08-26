# Contributing to plasmakit

Thank you for considering a contribution.

## Development setup

```bash
git clone git@github.com:alphabench/plasmakit.git
cd plasmakit
uv sync           # installs runtime + dev dependencies into .venv
```

## Before opening a pull request

All four gates must pass:

```bash
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
```

## The benchmark convention

Every physics model in plasmakit is validated against published
reference values through the `plasmakit.benchmarks.CASES` registry,
which drives both `plasmakit.validate()` and the test suite. Changes
to physics code must keep `validate()` fully passing, and a new physics
model must arrive with:

- a `MODEL_ID` string and a full citation in
  `plasmakit.provenance.MODEL_REFERENCES`, and
- at least one benchmark case with a citable analytic or published
  reference value.

## Transport tests (optional)

Tests marked `transport` run real OpenMC calculations. They are skipped
unless OpenMC is importable and `OPENMC_CROSS_SECTIONS` points at a
nuclear-data library. OpenMC is not on PyPI — install it from
conda-forge, and fetch a targeted data library, e.g.:

```bash
micromamba create -n plasmakit-openmc -c conda-forge openmc pip
micromamba run -n plasmakit-openmc pip install -e . openmc-data-downloader
micromamba run -n plasmakit-openmc openmc_data_downloader \
  -l ENDFB-7.1-NNDC -e H He Li Be C O Si V Cr Mn Fe Ta W Pb -d ~/nuclear_data
export OPENMC_CROSS_SECTIONS=~/nuclear_data/cross_sections.xml
```

## Conventions

- Frozen dataclasses, `PlasmakitError` for validation, NumPy-style
  docstrings on the public API, noun-named modules.
- Units are documented per function: keV, m, m^-3, m^3/s, W/m^3 —
  except the tritium fuel cycle, where durations are days and
  inventories kg (documented in `plasmakit.tritium`).
- Every stochastic entry point takes a seed and must be deterministic
  for a fixed seed.

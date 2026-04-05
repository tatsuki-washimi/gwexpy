# Contributing to GWexpy

Thanks for your interest in contributing to GWexpy!

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

For coverage:

```bash
pytest --cov=gwexpy --cov-report=term-missing
```

**Optional stats dependencies note:**
If you want to run MIC-related test functionality locally, installing via `conda`/`mamba` is recommended:

```bash
mamba install -c conda-forge mictools
python scripts/install_minepy.py
```

## Style and linting

- Keep code readable and prefer small, focused functions.
- Add or update tests for behavior changes.
- Run `ruff check .` before submitting.

## Design Principles: Controlled Extension, No Silent Monkeypatching

GWexpy extends GWpy through two sanctioned mechanisms:

1. **Subclassing**: `gwexpy.TimeSeries`, `gwexpy.FrequencySeries`, etc. inherit from GWpy base classes.
   New methods (e.g. `.fit()`) are added via inheritance, not by modifying the upstream class directly.
2. **I/O registry injection**: On `import gwexpy`, `register_all()` is called to register GWexpy's
   format readers/writers into the GWpy I/O registry. This is the standard GWpy extension pattern
   and is limited to the I/O layer.

**What is prohibited:**

- Modifying `gwpy.types.Series` or other upstream classes at import time (beyond I/O registry entries).
- Adding attributes or methods to external objects in global scope.
- Relying on import side effects for non-I/O functionality — such behavior must be opt-in (e.g. `enable_series_fit()`).
- Keep import-time behavior predictable and document any compatibility shims when they are unavoidable.

## Documentation

To build docs locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html/docs
```

Then open `docs/_build/html/docs/index.html` (English/Japanese are under `docs/_build/html/docs/web/`).

## Submitting changes

1. Create a branch for your change.
2. Include tests and docs when relevant.
3. Open a pull request with a concise summary and motivation.

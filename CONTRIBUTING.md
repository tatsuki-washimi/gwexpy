# Contributing to GWExPy

Thanks for your interest in contributing to GWExPy!

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

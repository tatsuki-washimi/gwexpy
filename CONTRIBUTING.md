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

## Generated artifacts

Do not commit generated environments, caches, or docs build outputs.

- Forbidden tracked paths include `docs/.doctrees/`, `docs/_build/`, `scratch/.venv_docs/`, `.venv-ci/`, `.conda-envs/`, `.conda-pkgs/`, `.mypy_cache/`, `.ruff_cache/`, and `.pytest_cache/`.
- `pre-commit` and CI both enforce this guard with `scripts/check_forbidden_artifacts.py`.
- Keep local environments outside the repository when possible, and treat docs build outputs as disposable artifacts.

## Design Principles: Modular Extensibility

GWexpy is designed to extend GWpy and other core libraries in a way that remains safe and predictable. We avoid "silent monkeypatching" of upstream classes in favor of explicit, documented extension points.

**Recommended Extension Patterns:**

1. **Subclassing (Core Types)**: `gwexpy.TimeSeries`, `gwexpy.FrequencySeries`, etc. inherit from GWpy base classes. New methods (e.g., `.fit()`, `.mix_down()`) are added via inheritance, ensuring zero impact on other libraries using the base GWpy types.
2. **I/O Registry Injection**: We use the official `gwpy.io.registry` mechanism to add support for new file formats (e.g., `.tdms`, `.gbd`, `.root`). This is the sanctioned way to extend GWpy's multi-format support.
3. **MIMO/Matrix Abstractions**: For functionality involving multiple channels, we use distinct container classes (e.g., `TimeSeriesMatrix`) rather than adding complex logic to individual time series objects.

**Prohibited Behaviors:**

- **Implicit Modification**: Adding attributes or methods to `gwpy.*` or `astropy.*` classes at import time.
- **Global Scope Pollution**: Adding methods to built-in objects or external library objects globally.
- **Hidden Side Effects**: Any behavior that changes the output of external libraries simply by importing `gwexpy` (except for the authorized I/O registry entries).
- **Incompatible API Shims**: Modifying standard library functions or third-party API signatures without an explicit user opt-in.

## Documentation

To build docs locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html/docs
```

Then open `docs/_build/html/docs/index.html` (English/Japanese are under `docs/_build/html/docs/web/`).

### Jupyter Notebooks

Documentation and examples use Jupyter Notebooks (.ipynb). We categorize them to optimize CI performance:

- **Light** (Default): Fast execution. Tested via `papermill`. Cell outputs are stripped on commit.
- **Heavy**: Resource intensive or requires special env (GPU/LIGO VPN). Syntax-checked via `nbval`. Metadata tag: `"tags": ["ci-heavy"]`.
- **Display-only**: Pre-rendered results are the goal. CI skips execution. Metadata tag: `"tags": ["display-only"]`. Needs `.gitattributes` entry to keep outputs.

See [NOTEBOOK_POLICY.md](docs/NOTEBOOK_POLICY.md) for details. We use `nbstripout` via `pre-commit` to prevent committing large binary outputs for most notebooks.

#### Syntax & Indentation

To ensure notebooks are compatible with Sphinx/nbsphinx and the CI pipeline:

- **Avoid indentation errors**: Ensure all cells have valid Python syntax.
- **Warnings Blocks**: When using `with warnings.catch_warnings():`, ensure the entire cell content is correctly indented. Avoid mixing multiple warnings blocks or leaving trailing imports inside the block.
- **Automated Checks**: The `Notebook syntax check` in CI will block PRs with corrupted JSON or major syntax regressions. Run `nbformat` locally to verify if you suspect issues.
- **Dependencies**: Any new dependency used in a tutorial MUST be added to `requirements-dev.txt` and the relevant CI workflows (`test.yml`, `docs.yml`).

### Examples and Doctests

Interactive examples in docstrings must be self-contained and stable.

- **Randomness**: If an example uses random numbers, always set a seed:


  ```python
  >>> import numpy as np
  >>> np.random.seed(0)
  ```

- **Optional Dependencies**: If an example requires optional packages (matplotlib, interactive backends, GWpy plotting), annotate the code sample:

  ```python
  # doctest: +SKIP
  ```

  or prefer a non-visual minimal example.

- **Approximation**: When comparing numeric output or `repr` values, use `# doctest: +ELLIPSIS` for tolerant matches, or prefer explicit assertions in `pytest` tests (e.g., `np.allclose`).

- **Self-contained**: Include necessary imports and avoid external state (files, servers). If a heavy dataset is required, mark the example `# doctest: +SKIP` and provide a small synthetic example instead.

## Submitting changes

1. Create a branch for your change.
2. Include tests and docs when relevant.
3. Open a pull request with a concise summary and motivation.

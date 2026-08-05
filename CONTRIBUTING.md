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
- `pre-commit` and CI both enforce this guard with `scripts/check_forbidden_artifacts.py` and `scripts/check_repo_hygiene.py`.
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

### GWpy API Compatibility Principle

The patterns above govern *how* GWexpy extends GWpy. This governs *what* an
override is allowed to change when GWpy already provides the method.

Classify every such override into exactly one of three cases, and state which
one in the pull request:

1. **GWpy's behavior is reasonable** — match it. Do not diverge, even
   cosmetically.

2. **GWpy's behavior is unreasonable or inconvenient, but it returns a usable
   value** — add an option that selects the improved behavior, and **default
   that option to GWpy's behavior**. The improvement is opt-in. Silently
   changing a result that existing GWpy code already consumes is exactly what
   this rule exists to prevent.

3. **GWpy raises, or returns only `nan`, and a more appropriate result is
   constructible** — improve it **by default**. Nobody can depend on a value
   they never received, so there is nothing to break. The canonical example is
   computing a statistic over data containing `nan` by ignoring the `nan`
   (`nanmean`-style) rather than propagating it into a useless all-`nan`
   result. Offering GWpy's behavior behind an option as well is fine.

The line between (2) and (3) is whether GWpy's return value is usable. A
suboptimal-but-usable value is case (2), and its improvement must be opt-in;
an exception or an all-`nan` result is case (3), and its improvement is the
default.

This principle takes precedence when it conflicts with a general preference
for "more correct" behavior — including unit preservation. Dropping a physical
unit is undesirable, but if GWpy drops it and still returns a usable series,
that is case (2): preserve the unit behind an option, and default to GWpy.

`TimeSeries.rms` is the worked example. GWpy returns a dimensionless series
(case 2 → `unit=False` by default, `unit=True` opts into preservation) and
returns `nan` for any window containing a single `nan` (case 3 →
`ignore_nan=True` by default, `ignore_nan=False` restores GWpy's propagation).

## Third-Party Code and Licenses

GWexpy learns from neighbouring projects but does not vendor their code. Before adding a
dependency on, or an adapter for, an external project, read
[docs/developers/LICENSES_THIRD_PARTY.md](docs/developers/LICENSES_THIRD_PARTY.md). The rules
in short:

- **No vendoring.** Do not copy, paste, or transcribe source from another project, regardless
  of its licence. Reimplement from the published description instead.
- **Verify licences from the LICENSE file itself**, not from PyPI classifiers, README badges,
  or a previous document's claim. A project that ships no LICENSE file is all rights reserved,
  which is stricter than GPL — nothing may be reused from it.
- **Write "unverified" when you cannot confirm a licence.** Do not guess.
- **Prefer format compatibility over a package dependency.** Reading another tool's documented
  on-disk format with an existing base dependency keeps GWexpy's dependency surface small.

`pemcoupling` in particular carries no licence grant. Its domain concepts may be referenced;
its code, file structure, and implementations may not.

## Documentation

To build docs locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html/docs
```

Then open `docs/_build/html/docs/index.html` (English/Japanese are under `docs/_build/html/docs/web/`).

### Jupyter Notebooks

Documentation and examples use Jupyter Notebooks (.ipynb). We categorize them to optimize CI performance:

- **Light** (Default): Fast execution. Tested via `papermill`. Tracked notebooks under `docs/web/` and `examples/` are committed in a clean state with outputs stripped.
- **Heavy**: Resource intensive or requires special env (GPU/LIGO VPN). Syntax-checked via `nbval`. Metadata tag: `"tags": ["ci-heavy"]`. The same clean-source rule applies.
- **Display-only**: Pre-rendered results are the goal. CI skips execution. Metadata tag: `"tags": ["display-only"]`. Use this only for deliberate checked-in outputs that have been explicitly reviewed.

Published docs should be built from a temp executed notebook tree or equivalent generated artifacts, not from output-bearing source notebooks in Git.

See [NOTEBOOK_POLICY.md](docs/NOTEBOOK_POLICY.md) for details. `pre-commit` uses `scripts/strip_example_notebook_outputs.py` to clean tracked notebooks under `docs/web/` and `examples/`, and `scripts/check_repo_hygiene.py` enforces the same policy in CI.

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
4. Use CI workflow summaries/check logs to triage failures; do not auto-create GitHub issues for CI telemetry.
5. Keep the issue tracker for human-triaged work items only. Historical commented CI issues are a manual cleanup queue, not a future tracking model.

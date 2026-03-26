# Work Report: Remaining Integrated Plan (2026-01-26)

- Timestamp: 2026-01-26 23:35:05
- Model: GPT-5 (Codex CLI)
- Time taken: ~30 minutes
- Scope: Plan Step 1 (exception handling/type safety touch-ups) and Step 2 (tests/coverage)

## Summary

- Tightened typing in `extract_xml_channels` and removed the MyPy ignore override for that module.
- Confirmed existing exception-handling changes in NDS utilities and cache shutdown behavior.
- Ran targeted GUI/types tests with coverage; full-suite `pytest --cov` crashes under this environment.
- Full suite completes when pytest plugin autoload is disabled, suggesting a plugin-related abort.
- Further investigation shows pytest autoload still aborts, while manual plugin loading succeeds.
- Root cause: pytest-qt picked PySide6 by default while GUI conftest imports PyQt5, causing a Qt binding mix and abort.
- Fixed by forcing pytest-qt to use PyQt5 via pytest ini option.

## Files Changed

- `gwexpy/io/dttxml_common.py`
  - Added `ChannelInfo` TypedDict and explicit local typing.
  - Guarded `lower()` usage for optional strings.
- `pyproject.toml`
  - Removed `gwexpy.io.dttxml_common` from MyPy ignore overrides.
  - Added `qt_api = "pyqt5"` under pytest ini options.
  - Removed an empty `tool.mypy.overrides` block.
  - Added missing optional dependencies: `gpstime`, `nptdms`, `minepy`, `simpeg`, `sounddevice`, `numba`, `ligo.skymap`.
  - Minimized optional extras: `audio` now `librosa`/`pydub` only; `analysis` now `PyEMD`/`pywt` only; updated `all` accordingly.
- `docs/web/en/guide/installation.rst`
  - Updated `analysis` extra description to match PyEMD/pywt-only scope.
- `docs/web/ja/guide/installation.rst`
  - Updated `analysis` extra description to match PyEMD/pywt-only scope.
- `sitecustomize.py`
  - Attempted to set `PYTEST_QT_API` for pytest runs; not relied upon for console-script pytest.

## Tests and Checks

- `ruff check .`
- `mypy .`
- `pytest tests/gui tests/types --cov`
  - Result: 230 passed, 5 skipped
  - Total coverage: 37%
- `sphinx-build -b html docs docs/_build/html`
  - Result: build succeeded, 2 warnings (not in toctree: `docs/web/en/guide/tutorials/field_scalar_signal.md`, `docs/web/ja/guide/tutorials/field_scalar_signal.md`)
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest`
  - Result: 2457 passed, 221 skipped, 3 xfailed
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p pytest_cov.plugin --cov=gwexpy`
  - Result: 2457 passed, 221 skipped, 3 xfailed (coverage total 48%)
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p anyio.pytest_plugin -p nbmake.pytest_plugin -p pytestqt.plugin -p pytest_cov.plugin -p xdist.plugin -p zarr.testing -p ligo.skymap.tests.plugins.omp`
  - Result: 2457 passed, 221 skipped, 3 xfailed
- `PYTEST_QT_API=pyqt5 pytest`
  - Result: 2473 passed, 222 skipped, 3 xfailed
- `pytest`
  - Result: 2473 passed, 222 skipped, 3 xfailed (after `qt_api = "pyqt5"` in `pyproject.toml`)
- `pytest --cov`
  - Result: 2473 passed, 222 skipped, 3 xfailed (coverage total 72%)

## Issues/Notes

- Default `pytest` and `pytest --cov` abort immediately with `Sandbox(Signal(6))` even with `PYTHONFAULTHANDLER=1`.
- With `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, the full suite completes, indicating a third-party plugin autoload issue.
- Forcing `QT_API=pyside6` or `QT_API=pyqt5` does not avoid the abort.
- Setting `PYTEST_QT_API=pyqt5` avoids the abort; autoload run completes with GUI tests enabled.
- `sitecustomize.py` in repo root is not loaded for `pytest` console script, so `qt_api` in pytest config is the reliable fix.

## Knowledge Extraction

- No new reusable patterns or skill updates identified during this work.

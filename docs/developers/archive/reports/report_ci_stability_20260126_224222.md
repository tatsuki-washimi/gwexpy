# Work Report: CI Stability (Warnings + GUI Waits + Coverage)

## Summary
- Expanded pytest warning filters to suppress known third-party noise (Astropy, gwpy, igwn_auth_utils, matplotlib, numpy, dateparser, erfa) while keeping project-origin warnings visible.
- Stabilized GUI tests by raising/activating windows before `qtbot.waitExposed` in key fixtures and tests.
- Verified GUI and non-GUI test suites run cleanly (no warning output in split runs).
- Generated HTML coverage and summarized low-coverage areas in fields/timeseries/gui.
- Added the CI stability plan document to tracking.

## Key Changes
- Warning suppression additions: `pyproject.toml` (pytest filterwarnings).
- GUI wait stabilization:
  - `tests/gui/conftest.py`
  - `tests/gui/test_gui_main_window.py`
  - `tests/gui/test_gui_data_backend.py`
  - `tests/gui/integration/test_main_window_flow.py`
  - `tests/gui/test_gui_result_tab.py`

## Tests
- `pytest tests/gui`
- `pytest tests -k "not gui"`
- `pytest --cov=gwexpy --cov-report=html tests -k "not gui"`
- `pytest --cov=gwexpy --cov-append --cov-report=html tests/gui`
- `coverage report -m --include="gwexpy/fields/*,gwexpy/timeseries/*,gwexpy/gui/*"`

## Coverage Notes (selected)
- GUI (low):
  - `gwexpy/gui/engine.py` ~11%
  - `gwexpy/gui/excitation/generator.py` ~7%
  - `gwexpy/gui/ui/channel_browser.py` ~8%
  - `gwexpy/gui/nds/audio_thread.py` ~12%
- Timeseries (low):
  - `gwexpy/timeseries/_core.py` 0%
  - `gwexpy/timeseries/core.py` 0%
  - `gwexpy/timeseries/collections.py` ~36%
  - `gwexpy/timeseries/matrix_core.py` ~33%
- Fields (low):
  - `gwexpy/fields/demo.py` ~12%

HTML coverage report: `htmlcov/index.html`

## Warnings
- Split runs (`tests/gui` and `tests -k "not gui"`) completed without warning output after filter updates.
- Full `pytest` run in this environment can crash with Qt backend contention; split runs are recommended for stability.

## Metadata
- Model: GPT-5 (Codex CLI)
- Time taken: ~1-1.5 hours (estimated)

## Skills / Knowledge Notes
- No new skills created or updated.
- Note: When CI mixes Qt backends, split GUI/non-GUI test runs to avoid intermittent crashes.

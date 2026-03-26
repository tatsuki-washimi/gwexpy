# Static Analysis & Test Cleanup Plan (v1)

**Date**: 2026-01-25
**Target**: `v0.1.0b1` Post-Release Cleanup

## 1. Objectives

- **Clean Test Output**: Suppress known 3rd-party warnings in Pytest and fix deprecated GUI test calls.
- **Strengthen Typing**: Expand MyPy coverage to include `gwexpy.types.series_matrix_*`, which are core components.
- **Code Hygiene**: verify and clear remaining TODOs in fitting and spectrogram modules.

## 2. Detailed Roadmap

### Phase 1: Pytest Warning Cleanup

- [x] **Filter Warnings**: Update `pyproject.toml` `[tool.pytest.ini_options]` to ignore `DeprecationWarning` from `scitokens`.
- [x] **Fix GUI Tests**: Replace `qtbot.waitForWindowShown(window)` with `qtbot.waitExposed(window)` in `tests/gui/integration/test_main_window_flow.py` (Tests passed).

### Phase 2: MyPy Coverage Expansion

- [x] **Update Config**: Remove `gwexpy.types.seriesmatrix*` from MyPy ignores.
- [x] **Fix Errors**: Resolved type mismatch in `seriesmatrix_base.py` `__array_finalize__` using `typing.cast`.

### Phase 3: Code Hygiene

- [x] **Verify TODOs**: Confirmed no remaining TODOs in `gwexpy/fitting/core.py` and `gwexpy/spectrogram/matrix_core.py`.

## 3. Recommended Models & Skills

- **Model**: `Gemini 3 Pro` (Current) or `Claude 3.5 Sonnet`.
- **Skills**:
  - `fix_mypy`: For efficient type error resolution.
  - `test_gui`: Verification of GUI test fix.
  - `test_code`: Verification of warning suppression.

## 4. Effort Estimate

- **Total Time**: ~45 minutes.
- **Complexity**: Low-Medium (MyPy errors can be tricky).

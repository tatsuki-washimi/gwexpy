# Plan: Core Quality Improvement 2026-01-24

**Date:** 2026-01-24
**Author:** Antigravity Agent
**Status:** In Progress
**Model:** Gemini 3 Pro (High)

This plan outlines the specific steps for improving the core quality of the `gwexpy` repository, focusing on MyPy coverage, exception handling, and Python version modernization.

## Objectives & Goals

1.  **Strict Typing for Validation**: Enable MyPy checking for `gwexpy.types.series_matrix_core` and `gwexpy.types.series_matrix_indexing` by removing them from `pyproject.toml`'s ignore list.
2.  **Robust Exception Handling**: Audit and improve exception handling in non-GUI modules (`gwexpy/types/series_matrix_validation_mixin.py` and `gwexpy/plot/skymap.py`) to prevent silent failures.
3.  **Modern Python Cleanup**: Remove obsolete compatibility code for Python < 3.9 (e.g., `sys.version_info` checks) to streamline the codebase.

## Detailed Roadmap

### Phase 1: MyPy Coverage Expansion (gwexpy.types)

- **Target Modules**: `gwexpy.types.series_matrix_core`, `gwexpy.types.series_matrix_indexing`
- **Action**:
  1.  Remove these modules from the `ignore_errors = true` list in `pyproject.toml`.
  2.  Run `mypy .` and fix reported type errors.
  3.  Prioritize TypeDict/Protocol over `Any` or `cast`.
  4.  Verify pass with `pytest tests/types`.

### Phase 2: Exception Handling Audit

- **Target Modules**: `gwexpy/types/series_matrix_validation_mixin.py`, `gwexpy/plot/skymap.py`
- **Action**:
  1.  Search for `except Exception:` / `except:` blocks.
  2.  Replace with specific exceptions (e.g., `ValueError`, `KeyError`) where possible.
  3.  Ensure `logger.exception()` is used if re-raising is not appropriate.
  4.  Verify no behavioral regression with existing tests.

### Phase 3: Python 3.9+ Modernization

- **Target Modules**: `gwexpy/types/metadata.py` (and potentially others found by search)
- **Action**:
  1.  Search for `sys.version_info`.
  2.  Remove branches handling Python < 3.9.
  3.  Consolidate `typing_extensions` imports where standard `typing` suffices in 3.9+ (careful with 3.10+ features).

## Testing & Verification Plan

- **Linting**: `ruff check .` (ensure new changes pass).
- **Typing**: `mypy .` (must succeed for enabled modules).
- **Unit Tests**: `pytest tests/types tests/spectrogram tests/fitting` (regression testing).

## Models, Recommended Skills, and Effort Estimates

- **Model**: Gemini 3 Pro (High)
- **Skills**: `fix_mypy`, `lint`, `test_code`, `grep_search`
- **Effort Estimate**:
  - Phase 1: 30 mins (MyPy errors can be tricky)
  - Phase 2: 20 mins (Careful review of logic)
  - Phase 3: 15 mins (Search and delete)
  - Total: ~65 mins (High Quota consumption due to iterative type checking)

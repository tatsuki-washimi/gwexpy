# Task: Post-Review Cleanup and Coverage Expansion

**Status**: To Do
**Priority**: Medium
**Created**: 2026-01-24

## Context

Following the repository review (`repo_review_final_20260124.md`), the core codebase is solid. The next focus is on "cleaning up leftovers" and "expanding coverage" to peripheral modules.

## Objectives

1.  **[P3] Remove Stale TODOs in SpectrogramMatrix**
    - **Target**: `gwexpy/spectrogram/matrix.py` docstring.
    - **Action**: Remove the section describing Pickle limitation and the TODO about `__reduce__`, as it was implemented in commit `e027836`. Update the docstring to state that custom pickling matches standard behavior.

2.  **[P2] Expand MyPy to `gwexpy.types`**
    - **Target**: `pyproject.toml`
    - **Action**: Try removing `gwexpy.types.series_matrix_*` from the exclusion list one by one.
    - **Note**: These modules are heavily used by Matrix classes. Fixing typing here will solidify the foundation.

3.  **[P2] GUI Exception Auditing**
    - **Target**: `gwexpy/gui/streaming.py`, `engine.py`.
    - **Action**: Where `except Exception: pass` is found, insert `logger.warning(...)` or `logger.debug(...)` to aid future debugging.

4.  **[P3] Documentation Update**
    - Ensure the "Features" list in `README.md` or docs mentions the robust serialization support if applicable.

## Recommended Skills

- `replace_file_content` (for cleanup)
- `run_command` (MyPy checks)
- `analyze_code` (for debugging GUI exceptions)

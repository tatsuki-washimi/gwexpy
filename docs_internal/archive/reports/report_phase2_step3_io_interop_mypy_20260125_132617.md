---
title: "Phase 2 Step 3: IO + Interop MyPy Coverage Expansion"
date: 2026-01-25T13:26:17Z
models:
  - Codex (GPT-5)
description: |
  Removed MyPy ignore overrides for gwexpy.io.*, gwexpy.interop.*, and gwexpy.conftest,
  fixed resulting type errors, and validated with ruff/mypy/pytest.
---

# Scope

- Phase 2 Step 3 requested: type-safety hardening for `gwexpy.io` and related I/O mixins.
- Follow-up: expanded MyPy coverage by removing remaining overrides for `gwexpy.interop.*` and `gwexpy.conftest`.

# Changes

- `pyproject.toml`
  - Removed `ignore_errors = true` overrides for:
    - `gwexpy.io.*`
    - `gwexpy.interop.*`
    - `gwexpy.conftest`
- `gwexpy/io/utils.py`
  - Fixed optional `SeriesMatrix` import pattern without assigning to a type.
  - Used `TYPE_CHECKING` + `cast` so `meta` access is type-safe while keeping runtime logic unchanged.
- `gwexpy/types/series_matrix_io.py`
  - Updated I/O mixin typing to use `IndexLike` and aligned `shape3D` return type with core mixin (`tuple[int, ...]`).
- `gwexpy/interop/polars_.py`
  - Annotated `t0`/`dt` as `float` to match inferred float assignments.
- `gwexpy/interop/mne_.py`
  - Removed duplicate helper definitions that caused MyPy `no-redef`.
  - Added a small `Protocol` + `cast` to allow `to_matrix()` alignment without treating mappings as mutable.
- Ruff-driven cleanups (no logic changes):
  - `gwexpy/fields/scalar.py`, `gwexpy/types/metadata.py`, `gwexpy/types/typing.py`
- `scripts/verify_scalarfield_physics.py`
  - Guarded axis-unit checks when axis indices may be `None` (MyPy-safe).

# Verification

- `mypy gwexpy/io`
- `mypy gwexpy/interop`
- `mypy .`
- `ruff check .`
- `pytest tests/io`
- `pytest tests/interop`

# Notes

- Some tests are skipped depending on optional dependencies; this is expected.
- Joblib emits a warning and falls back to serial mode on this machine (permission-related).

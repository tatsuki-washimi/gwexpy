# Optional Backend & Packaging Audit — Implementation Plan

**Issue:** #251  
**Branch:** `claude/optional-backend-packaging-uDtm8`  
**Date:** 2026-04-25  
**Author:** Claude (claude-sonnet-4-6)

---

## Objectives & Goals

Verify that package extras, optional imports, CI gates, and backend-unavailable behavior are
consistent across the codebase. Eliminate drift between:

- `pyproject.toml` `[project.optional-dependencies]`
- Runtime import guards (`require_optional` / raw `try/except`)
- Test skip logic (`pytest.importorskip`, `pytest.mark.skipif`)
- CI gate coverage (`run_gate.py`)
- User-facing documentation

---

## Current State (findings from audit)

### A. pyproject.toml extras (declared)

| Extra | Key packages |
|-------|-------------|
| `analysis` | scikit-learn, statsmodels, pmdarima, dcor, hurst, PyWavelets, EMD-signal |
| `fitting` | iminuit, emcee, corner |
| `control` | control |
| `seismic` | obspy, mth5, mtpy, mt_metadata |
| `gw` | lalsuite, gwdatafind, gwosc, dqsegdb2, dttxml, gwinc, ligo.skymap |
| `io` | nptdms |
| `plotting` | pygmt |
| `audio` | pydub, tinytag |
| `gui` | PyQt5, pyqtgraph, qtpy, sounddevice |
| `dev` | ruff, mypy, pytest, nbmake, nbval, freezegun, … |
| `all` | union of above (excluding `gui`, `dev`) |

### B. Drift issues identified

1. **`_optional.py` `_EXTRA_MAP` has phantom extras** — entries map to extra names that don't
   exist in `pyproject.toml`: `"interop"`, `"bio"`, `"stats"`, `"eda"`. Affected packages:
   `torch`, `jax`, `dask`, `zarr`, `xarray`, `netCDF4`, `cupy`, `mne`, `neo`, `librosa`,
   `pyroomacoustics`, `PySpice`, `skrf`, `polars`, `joblib`.

2. **`requirements-dev.txt` tail is hand-edited** — packages from the `analysis` extra
   (`scikit-learn`, `statsmodels`, `PyWavelets`, etc.) are appended outside the
   `pip-compile`-generated block ("# Added for CI stability v3"), creating a dual source of truth.

3. **Raw `try/except ImportError` blocks bypass `require_optional()`** — found in:
   - `gwexpy/timeseries/io/audio.py`, `seismic.py`, `tdms.py`, `zarr_.py`, `win.py`,
     `netcdf4_.py`, `ats.py`, `csv_enhanced.py`
   - `gwexpy/noise/wave.py`, `obspy_.py`
   - `gwexpy/interop/cupy_.py`, `emg3d_.py`, `gwinc_.py`, `meep_.py`, `mne_.py`,
     `mt_.py`, `openems_.py`
   - `gwexpy/fitting/core.py`, `models.py`
   These may produce unclear errors (e.g. no `pip install gwexpy[extra]` hint).

4. **`io-optional` CI gate is narrow** — only covers 4 test files; many optional-dep reader
   tests are in `pr-fast` ignores and have no gate at all.

5. **`gui` entry point `gwexpy.gui` is in `[project.scripts]`** but `gui` is not in `all` and
   is labelled "experimental". The script will fail on clean install without `.[gui]`.

6. **`netCDF4` / `xarray`** — used in `timeseries/io/netcdf4_.py` but mapped to `"seismic"`
   extra in `_optional.py`; those packages are not listed in the `seismic` extra in
   `pyproject.toml`. They have no declared extra at all.

7. **`zarr`** — used via `timeseries/io/zarr_.py`, has an `io-zarr` CI gate and env-var guard
   (`GWEXPY_ALLOW_ZARR`), but no declared extra in `pyproject.toml`.

---

## Detailed Roadmap

### Phase 1 — Inventory & Gap Analysis (no code changes yet)

**Goal:** produce a complete mapping of all optional packages → extras → guards → tests → CI gate.

Tasks:
- [ ] 1.1 Walk all `gwexpy/**/*.py` files that contain `try:` / `except ImportError` or call
      `require_optional()`. Record: module path, package guarded, guard style, extra expected.
- [ ] 1.2 Walk all `tests/` files that use `pytest.importorskip` or `pytest.mark.skipif` on
      an import. Record: test file, package skipped, which CI gate runs that test.
- [ ] 1.3 Build a master table: `package → pyproject extra → _optional.py entry → test skip →
      CI gate`. Identify every row with a gap or mismatch.
- [ ] 1.4 Verify wheel smoke: `python -m build --wheel --no-isolation` succeeds clean.

Deliverable: annotated gap table (inline in the PR description or as a separate audit artifact).

---

### Phase 2 — Fix `_optional.py` phantom extras

**Goal:** every package in `_EXTRA_MAP` maps to a real extra in `pyproject.toml`, or is
explicitly documented as "no declared extra, install manually".

Tasks:
- [ ] 2.1 For each phantom extra (`interop`, `bio`, `stats`, `eda`):
  - Option A: add the extra to `pyproject.toml` (if there is user value).
  - Option B: change the install hint to a bare `pip install <pkg>` with a comment explaining
    why it has no extra.
  - Decision: keep `interop`/`bio`/`stats`/`eda` as **undeclared** for now (too niche / heavy
    dependencies); update `_EXTRA_MAP` to use `None` and fall back to bare `pip install <pkg>`.
- [ ] 2.2 Fix `netCDF4` / `xarray` mapping: add `"netcdf"` extra to `pyproject.toml` with
      `netCDF4` and `xarray`, or simply move the install hint to bare install and document.
- [ ] 2.3 Fix `zarr` mapping: add `"zarr"` extra, or leave undeclared with a clear hint.
- [ ] 2.4 Regenerate `requirements-dev.txt` with `pip-compile` to remove the hand-edited tail.
      The `analysis` packages belong in CI environment setup, not in the dev requirements.

---

### Phase 3 — Standardize import guards

**Goal:** every optional import that has a user-facing error message uses `require_optional()`
(or an equivalent lazy-import function) so the error includes a `pip install` hint.

Tasks:
- [ ] 3.1 Audit the raw `try/except ImportError` blocks listed in §B-3. For each:
  - If the block is at module-import time (top of file): keep as lazy sentinel (`None`) and
    raise via `require_optional()` inside the calling function.
  - If the block is inside a function: replace with `require_optional(name)`.
- [ ] 3.2 Ensure no optional import causes `ImportError` at `import gwexpy` time. Verify with:
  ```
  python -c "import gwexpy" --no site
  ```
  in a minimal venv that has only the core dependencies.
- [ ] 3.3 For GUI modules: `gwexpy/gui/` imports are already norecursedirs for pytest and
      excluded from `mypy`. Confirm that `import gwexpy` does not trigger a GUI import.

---

### Phase 4 — CI gate coverage

**Goal:** every optional backend has at least one test that is skipped (not failed) when the
dep is absent, and covered by a CI gate when the dep is present.

Tasks:
- [ ] 4.1 Extend `io-optional` gate to include seismic (`obspy`, `mth5`) guard tests.
- [ ] 4.2 Add `interop-optional` gate (or extend `interop-contract`) to cover
      `tests/interop/test_errors_and_optional.py` and guard tests for cupy/jax/torch.
- [ ] 4.3 Verify `gui` entry-point script: add a test or note in docs that `gwexpy.gui` requires
      `pip install gwexpy[gui]`.
- [ ] 4.4 Review CI workflow YAML (`.github/workflows/`) to confirm gates are wired to the
      correct jobs. (Out of scope if CI files are not present locally — flag for human review.)

---

### Phase 5 — Documentation

**Goal:** user-facing docs name the required extra for every optional feature.

Tasks:
- [ ] 5.1 Add or update a "Optional dependencies" page in `docs/` that lists extras and the
      features they unlock.
- [ ] 5.2 Ensure `gwexpy/interop/_optional.py` docstring / inline comments are accurate after
      Phase 2 changes.
- [ ] 5.3 Update `CHANGELOG.md` with a note about the packaging fixes.

---

### Phase 6 — Validation

Tasks:
- [ ] 6.1 `pip install -e ".[dev,test,docs]"` succeeds in a clean conda env.
- [ ] 6.2 `python scripts/ci/run_gate.py pr-fast` passes.
- [ ] 6.3 `python scripts/ci/run_gate.py io-contract` passes (includes wheel build smoke).
- [ ] 6.4 `python scripts/ci/run_gate.py io-optional` passes.
- [ ] 6.5 `python scripts/ci/run_gate.py interop-contract` passes.
- [ ] 6.6 `ruff check gwexpy/ tests/` — clean.
- [ ] 6.7 `mypy gwexpy/` — clean (no new errors).

---

## Testing & Verification Plan

| Scenario | Verification method |
|----------|---------------------|
| Core install (no extras) doesn't raise on `import gwexpy` | `python -c "import gwexpy"` in minimal venv |
| Each optional dep raises clear `ImportError` with install hint | `tests/io/test_optional_deps.py` + new equivalent in `tests/interop/` |
| Wheel build is clean | `run_gate.py io-contract` |
| `pr-fast` gate green | `run_gate.py pr-fast` |
| `io-optional` gate green | `run_gate.py io-optional` |
| `interop-contract` gate green | `run_gate.py interop-contract` |

---

## Models, Recommended Skills, and Effort Estimates

| Phase | Recommended agent/model | Est. effort |
|-------|------------------------|-------------|
| 1 — Inventory | Explore agent (fast) | 30 min |
| 2 — Fix `_optional.py` | claude-sonnet-4-6 | 1–2 h |
| 3 — Standardize guards | claude-sonnet-4-6 | 2–3 h |
| 4 — CI gates | claude-sonnet-4-6 | 1 h |
| 5 — Docs | claude-sonnet-4-6 | 30 min |
| 6 — Validation | Local run + agent | 30 min |
| **Total** | | **~6–7 h** |

No physics-sensitive changes; `verify_physics` is not required. No `gwexpy/fields/` changes.

---

## Risk Notes

- **`requirements-dev.txt` regeneration** (Phase 2.4): `pip-compile` may pin different
  versions. Review the diff carefully before committing to avoid breaking CI.
- **Raw `try/except` refactoring** (Phase 3): some patterns guard entire module-level
  registrations (e.g., `_registration.py`). Those must remain at module level as sentinels;
  only the *error message* needs to improve.
- **CI YAML** (Phase 4.4): may require human action if the workflow files are not accessible
  from this branch.

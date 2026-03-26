# Extra Lib Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the four review findings in external-library interop code without broadening scope beyond correctness, metadata integrity, and API contract consistency.

**Architecture:** Apply targeted fixes in the affected interop modules first, then tighten tests so they reflect real upstream data layouts and metadata semantics. Keep public APIs stable except where current behavior is already incorrect, and prefer regression tests that fail on the current implementation before touching production code.

**Tech Stack:** Python, pytest, NumPy, h5py, GWexpy interop modules, GWexpy field/series containers

---

## Objectives & Goals

- Correct pyroomacoustics RIR indexing so source/microphone selection matches upstream layout.
- Correct openEMS axis0 handling so time/frequency coordinates use physical metadata instead of synthetic indices.
- Make meshio cell-data handling safe for real meshes.
- Restore `FrequencySeries.from_mtspec*` return-type consistency when CI data is present.

## File Map

- Modify: `gwexpy/interop/pyroomacoustics_.py`
- Modify: `gwexpy/interop/openems_.py`
- Modify: `gwexpy/interop/meshio_.py`
- Modify: `gwexpy/interop/multitaper_.py`
- Modify: `tests/interop/test_interop_pyroomacoustics.py`
- Modify: `tests/interop/test_interop_openems.py`
- Modify: `tests/interop/test_interop_meshio.py`
- Modify: `tests/interop/test_interop_multitaper.py`
- Optional doc note if behavior changes need recording: `docs/developers/plans/extra_lib.md`

## Detailed Roadmap

**Status:** Completed on 2026-03-26

**Completion Summary**

| Phase | Result | Commit | Verification |
|---|---|---|---|
| Phase 4 | multitaper return-type contract fixed with dict gating | `63a0fb31` | `26/26` tests passed |
| Phase 3 | meshio unsafe `cell_data` interpolation rejected explicitly | `3af0e143` | `18/18` tests passed |
| Phase 1 | pyroomacoustics `rir[mic][source]` indexing corrected | `4b039498` | `35/35` tests passed |
| Phase 2 | openEMS physical axis0 metadata read from HDF5 attrs with fallback preserved | `7f96e805` | `35/35` tests passed |
| Phase 5 | integrated regression + lint | n/a | `114/114` tests passed, `ruff` clean |

### Phase 0: Lock Reproductions

**Files:**
- Modify: `tests/interop/test_interop_pyroomacoustics.py`
- Modify: `tests/interop/test_interop_openems.py`
- Modify: `tests/interop/test_interop_meshio.py`
- Modify: `tests/interop/test_interop_multitaper.py`

- [x] Add a pyroomacoustics regression test with real upstream layout assumptions: `room.rir[mic][src]`.
- [x] Add an openEMS regression test where TD datasets carry `attrs["time"]` and FD datasets carry physical frequency metadata; assert `axis0` stores those values.
- [x] Add a meshio regression test where `cell_data` length differs from `mesh.points` length in the normal way; assert current code fails or is rejected.
- [x] Add a multitaper regression test asserting `from_mtspec(FrequencySeries, mt_with_ci)` returns `FrequencySeries`, not `FrequencySeriesDict`.
- [x] Run targeted tests and confirm they fail on current code.

Run:
```bash
pytest tests/interop/test_interop_pyroomacoustics.py -v
pytest tests/interop/test_interop_openems.py -v
pytest tests/interop/test_interop_meshio.py -v
pytest tests/interop/test_interop_multitaper.py -v
```

### Phase 1: Fix pyroomacoustics RIR Semantics

**Files:**
- Modify: `gwexpy/interop/pyroomacoustics_.py`
- Modify: `tests/interop/test_interop_pyroomacoustics.py`

- [x] Update `from_pyroomacoustics_rir` to interpret `room.rir` as outer=`mic`, inner=`source`.
- [x] Update the all-pairs, single-source, and single-mic branches so naming and selected data remain consistent under the corrected layout.
- [x] Update `from_pyroomacoustics_field(..., mode="rir")` to gather RIRs by microphone for the requested source.
- [x] Keep `mic_signals`, `source`, and STFT paths untouched unless the new tests expose coupling bugs.
- [x] Run only pyroomacoustics interop tests.

Run:
```bash
pytest tests/interop/test_interop_pyroomacoustics.py -v
```

### Phase 2: Fix openEMS Physical Axis Metadata

**Files:**
- Modify: `gwexpy/interop/openems_.py`
- Modify: `tests/interop/test_interop_openems.py`

- [x] Inspect actual openEMS dump metadata conventions used in this repo’s intended format and decide the attribute names to support first.
- [x] Change `_read_openems_td` to read physical time values from dataset attrs when available, falling back to integer indices only as a compatibility fallback.
- [x] Change `_read_openems_fd` to read physical frequencies from attrs when available, falling back to integer indices only as a compatibility fallback.
- [x] Ensure returned `axis0` units match domain expectations: seconds for TD, hertz for FD when metadata is available.
- [x] Add explicit tests for both metadata-present and fallback cases.
- [x] Run openEMS tests.

Run:
```bash
pytest tests/interop/test_interop_openems.py -v
```

### Phase 3: Make meshio Cell Data Safe

**Files:**
- Modify: `gwexpy/interop/meshio_.py`
- Modify: `tests/interop/test_interop_meshio.py`

- [x] Decide the minimal safe behavior for `cell_data`: either convert using cell centroids or reject unsupported cell-data interpolation with a clear error.
- [x] Prefer the smaller safe change unless centroid calculation is already straightforward with current `mesh.cells` usage.
- [x] Update `_get_field_data` and/or `from_meshio` so point-based interpolation never silently mixes point coordinates with cell values.
- [x] Replace the current unrealistic passing test with one that matches actual meshio semantics.
- [x] Run meshio tests.

Run:
```bash
pytest tests/interop/test_interop_meshio.py -v
```

### Phase 4: Restore multitaper API Contracts

**Files:**
- Modify: `gwexpy/interop/multitaper_.py`
- Modify: `tests/interop/test_interop_multitaper.py`

- [x] Gate dict returns on `cls` being `FrequencySeriesDict`-compatible; otherwise return the main `FrequencySeries`.
- [x] Apply the same return-type rule to `from_mtspec_array`.
- [x] Add explicit tests for `FrequencySeries` and `FrequencySeriesDict` callers in both CI and non-CI paths.
- [x] Verify no existing tests rely on the incorrect dict return for `FrequencySeries`.
- [x] Run multitaper tests.

Run:
```bash
pytest tests/interop/test_interop_multitaper.py -v
```

### Phase 5: Cross-Module Verification

**Files:**
- Modify as needed only if regressions appear during verification

- [x] Run the four targeted interop test modules together.
- [x] If any classmethod wiring paths are affected, run adjacent interop suites for LAL/PyCBC/gwinc/scikit-rf/Finesse only if failures suggest shared helper breakage.
- [x] Run lint on changed files.
- [x] Summarize any behavior changes that should be reflected in docs.

Run:
```bash
pytest \
  tests/interop/test_interop_pyroomacoustics.py \
  tests/interop/test_interop_openems.py \
  tests/interop/test_interop_meshio.py \
  tests/interop/test_interop_multitaper.py -v

ruff check \
  gwexpy/interop/pyroomacoustics_.py \
  gwexpy/interop/openems_.py \
  gwexpy/interop/meshio_.py \
  gwexpy/interop/multitaper_.py \
  tests/interop/test_interop_pyroomacoustics.py \
  tests/interop/test_interop_openems.py \
  tests/interop/test_interop_meshio.py \
  tests/interop/test_interop_multitaper.py
```

## Testing & Verification Plan

- Start with regression-first targeted tests for each finding.
- Prefer isolated module-level pytest runs after each fix.
- Finish with one combined pytest invocation over all four touched test modules.
- Run `ruff check` on all changed source and test files.
- If openEMS metadata support requires assumptions not captured in current fixtures, document the accepted attribute names in tests.

## Risk Notes

- The pyroomacoustics fix changes user-visible selection behavior for multi-source/multi-mic cases; tests must lock naming as well as values.
- The openEMS fix is the riskiest because upstream metadata conventions may vary; fallback behavior must remain explicit and tested.
- The meshio change should bias toward safety over convenience. Rejecting ambiguous cell-data interpolation is acceptable if centroids are not already reliable.
- The multitaper fix is low-risk but can break callers who accidentally relied on the wrong return type; tests should prove the intended contract.

## Models, Recommended Skills, and Effort Estimates

- Recommended primary model: current coding model in this session is sufficient.
- Recommended skills during execution:
  - `systematic-debugging`
  - `test-driven-development`
  - `verification-before-completion`
- Estimated effort:
  - Phase 0-1: 20-35 min
  - Phase 2: 20-40 min
  - Phase 3: 20-45 min
  - Phase 4-5: 15-25 min
  - Total: roughly 1.5-2.5 hours depending on openEMS metadata handling and meshio cell-data decision

## Notes for Execution

- Keep commits scoped by finding, not by file type.
- Do not broaden the plan into metadata enhancements beyond the four confirmed review issues unless tests reveal a directly related blocker.
- If openEMS or meshio real-world semantics are still ambiguous after reading current fixtures and docs in-repo, stop and write down the exact ambiguity before implementing a guess.

## Final Outcome

- All four review findings were fixed.
- Targeted and integrated verification completed successfully.
- Result snapshot:
  - `multitaper`: `26/26` tests, commit `63a0fb31`
  - `meshio`: `18/18` tests, commit `3af0e143`
  - `pyroomacoustics`: `35/35` tests, commit `4b039498`
  - `openEMS`: `35/35` tests, commit `7f96e805`
  - Integrated: `114/114` tests passed, `ruff` clean

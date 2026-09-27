# PR #751 MyPy Follow-Up Qualification

**Date:** 2026-09-27
**Follow-up to:** [cross-format post-fix qualification](2026-09-27-public-io-cross-format-post-fix-report.md)
**Starting commit:** `1d5f559ea54876a0e28cd7c2eef06ca1e71b3f4f`
**Source/test Git tree evaluated:** `5658fa0562b409aa6e96c39769e0a4849f5d1a96`

## Scope

PR Fast identified three MyPy errors in NetCDF matrix index validation. After validating an index as a nonnegative Python or NumPy integer, the code now converts it to Python `int` before using it as a dictionary key. This aligns the runtime representation with the declared `dict[int, object]` types and preserves the accepted inputs and validation behavior.

This follow-up does not change or add findings to the frozen 74-scenario baseline audit. It also does not change the post-fix matrix classifications: 36 baseline defects remain `FIX_VERIFIED`, and 38 other scenarios remain `NOT_REQUALIFIED`.

## Qualification

| Check | Result |
| --- | ---: |
| NetCDF matrix validation and reader tests | 70 passed |
| I/O contract gate | 1,797 passed, 35 skipped, 1 deselected |
| I/O conformance gate | 71 passed, 7 skipped; generated contract display: 6 blocked / 6 total |
| Full MyPy (`gwexpy`) | 396 source files passed |
| PR Fast MyPy command | 397 source files passed |
| Ruff check (`gwexpy/`, `tests/`) | passed |
| Ruff format check (`netcdf4_.py`) | passed |
| Full pytest | 13,425 passed, 278 skipped, 6 xfailed in 731.26 seconds |
| Full pytest JUnit counters | 13,709 tests, 284 skipped (including xfails), 0 failures, 0 errors |
| `git diff --check` for the source fix | passed |

The full pytest used the same Zarr-enabled environment as the original qualification. A case-level JUnit report was generated and its aggregate counters were cross-checked against pytest's summary. The machine-readable follow-up record preserves those counts and the exact command; the bulky per-case XML is not included in the repository.

## Commands and environment

The checks ran in the `gwexpy` conda environment with Python 3.11.14 and the editable package loaded from this worktree.

```text
rtk proxy conda run -n gwexpy python -m pytest -q tests/io/test_netcdf4_matrix_validation.py tests/io/test_netcdf4_reader.py
rtk proxy conda run -n gwexpy python scripts/ci/run_gate.py io-contract
rtk proxy conda run -n gwexpy python scripts/ci/run_gate.py io-conformance
rtk proxy conda run -n gwexpy mypy gwexpy
rtk proxy conda run -n gwexpy mypy gwexpy tests/docs/test_tutorial_notebook_quality.py --ignore-missing-imports
rtk proxy conda run -n gwexpy ruff check gwexpy/ tests/
rtk proxy conda run -n gwexpy ruff format --check gwexpy/timeseries/io/netcdf4_.py
conda run -n gwexpy env GWEXPY_ALLOW_ZARR=1 PYTHONPATH=. python -m pytest -q --tb=short --junitxml=docs/developers/reports/2026-09-27-public-io-cross-format-post-fix-evidence/mypy-followup-full-pytest-zarr-enabled.xml
git diff --check
```

The `6 blocked / 6 total` line is the conformance gate's generated contract display; it did not fail the gate.

## Tree binding

The source/test tree evaluated by these checks is recorded above. The complete final tree, including this follow-up report and machine-readable record, is bound by the `Qualified-Tree` trailer on the final evidence-only commit. That trailer is authoritative for the complete PR tree and avoids a self-referential tree hash in the report.

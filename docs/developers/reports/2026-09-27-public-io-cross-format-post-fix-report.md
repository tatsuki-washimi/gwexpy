# Public I/O Cross-Format Post-Fix Qualification

**Date:** 2026-09-27
**Baseline:** GWexpy 0.2.4 at `ae10234a1e37853508c54901bf7c9e80878f25aa`
**Post-fix source:** branch `codex/fix-public-io-audit-20260927`, based on the same baseline, grouped into nine runtime-plus-regression commits
**Worktree source content SHA-256:** `c192c52eb2b3adab7a388f199de7000913fd4609f352a9b867b68287b12943f2`

The Git tree SHA for the complete qualified commit tree is recorded in the `Qualified-Tree` trailer of the final audit/evidence-only commit.

## Relationship to the frozen audit

The baseline audit remains the immutable record of the 74 scenarios measured against `ae10234a1e37853508c54901bf7c9e80878f25aa`. Its [report](2026-09-27-public-io-cross-format-audit.md), [matrix](2026-09-27-public-io-cross-format-audit/runtime-characterization-matrix.json), evidence, reproductions, and ZIP were not changed.

This post-fix record qualifies the 36 baseline defect-confirming scenarios against public regression assertions. The remaining 38 baseline scenarios are retained by ID and baseline status, marked `NOT_REQUALIFIED`, and are not presented as post-fix observations.

## Qualification result

| Check | Result |
| --- | ---: |
| Targeted public I/O regression suite | 319 passed, 1 skipped |
| I/O contract gate | 1,797 passed, 35 skipped, 1 deselected |
| I/O conformance gate | 71 passed, 7 skipped; contract display: 6 blocked / 6 total |
| Optional dependency gate | 81 passed, 1 skipped |
| Zarr gate | 60 passed |
| GWpy HDF5 compatibility and collection tests | 72 passed |
| Full pytest | 13,425 passed, 278 skipped, 6 xfailed |
| Ruff check (`gwexpy`, `tests`) | passed |
| Ruff format check | 18 changed Python files passed |
| MyPy changed production files | 11 files passed |
| `git diff --check` | passed |

The targeted test run is preserved as [JUnit XML](2026-09-27-public-io-cross-format-post-fix-evidence/targeted-regressions.xml). The post-fix matrix contains all 74 baseline IDs and links each verified defect to its public regression test.

## Fix boundaries qualified

The nine rows below map one-to-one to nine runtime-plus-regression commits. The review follow-up `HDF5-MANIFEST-PAYLOAD-001` is recorded separately and included in the HDF5 commit; it is not part of the frozen 74-scenario baseline count.

| # | Commit boundary | Commit subject | Finding IDs | Post-fix assertion |
| --- | --- | --- | --- | --- |
| 1 | NetCDF matrix topology and heterogeneous dtype | `fix(netcdf): reject malformed matrix topology and unsafe dtype casts` | `NC-MATRIX-001`–`007`; `NC-DTYPE-001` | Invalid or conflicting cells fail before a misleading matrix is returned; unsafe heterogeneous conversion is rejected rather than truncating values. |
| 2 | NetCDF irregular axes | `fix(netcdf): reject irregular legacy time axes` | `NC-AXIS-001`, `002` | Numeric and datetime gaps are rejected rather than regularized. |
| 3 | NetCDF matrix unit preservation | `fix(netcdf): preserve matrix cell units` | `NC-MATRIX-UNIT-001`, `002` | Written and legacy-read cell units remain `V`. |
| 4 | Zarr values, axes, units, and multi-store dtypes | `fix(zarr): preserve values, axes, units, and multi-store dtypes` | `ZARR-DTYPE-001`, `002`; `ZARR-AXIS-001`; `ZARR-UNIT-001`; `ZARR-MULTISTORE-DTYPE-001`; `ZARR-UINT64-001` | Exact integer and complex values are retained, units are preserved, and inconsistent axes or unsafe conversion fail explicitly. |
| 5 | Zarr auto and optional-dependency routes | `fix(zarr): preserve public auto routes without optional dependencies` | `ZARR-AUTO-001`; `OPTIONAL-002`, `003` | Public Zarr routes identify stores and raise the backend-specific `ImportError` when optional packages are absent; six absent-package routes were checked in a separate process. |
| 6 | TDMS timing | `fix(tdms): reject invalid waveform increments` | `TDMS-TIME-001`–`005`, `010` | Public readers reject absent, boolean, zero, negative, NaN, and infinite increments. |
| 7 | GL500 GBD headers | `fix(gbd): reject malformed GL500 header fields` | `GBD-HEADER-001`, `003`–`005` | Public readers reject malformed required metadata within the tested GL500 firmware 1.00–1.21 scope. |
| 8 | HDF5 manifest collection integrity | `fix(hdf5): fail closed on invalid manifest collection payloads` | `HDF5-TS-COLL-001`, `HDF5-FS-COLL-001`, `HDF5-SPEC-COLL-001`, `HDF5-HIST-COLL-001`; review follow-up `HDF5-MANIFEST-PAYLOAD-001` | Manifest-backed reads fail closed on unreadable, missing, or substituted payloads; manifest-free discovery remains tolerant. |
| 9 | Audio registry metadata | `fix(audio): preserve registry tag provenance` | `AUDIO-METADATA-001` | WAV and FLAC registry routes retain available tag metadata in provenance. |

## HDF5 review follow-up: `HDF5-MANIFEST-PAYLOAD-001`

This correctness defect was discovered during fix review and is outside the original 74-scenario baseline matrix. At baseline SHA `ae10234a1e37853508c54901bf7c9e80878f25aa`, public `HistogramDict.read` and `HistogramList.read` accepted unrelated root-level `values`/`edges` datasets after the manifest entry's canonical payload was removed, returning `[999.0, 999.0, 999.0]`. The independent [baseline reproduction](2026-09-27-public-io-cross-format-post-fix-evidence/baseline-hdf5-payload-fallback.py) preserves the setup and observation.

The post-fix public regression requires rejection in eight parameterized cases: TimeSeries, FrequencySeries, Spectrogram, and Histogram, each through Dict and List readers. They are preserved in [JUnit evidence](2026-09-27-public-io-cross-format-post-fix-evidence/targeted-regressions.xml) under `tests/io/test_hdf5_manifest_collection_integrity.py::test_manifest_backed_group_read_does_not_fallback_to_other_dataset`.

## Environment and limits

Qualification ran with Python 3.11.14, GWexpy 0.2.4 editable from this worktree, GWpy 4.0.2, NumPy 1.26.4, xarray 2026.2.0, netCDF4 1.7.4, Zarr 3.1.5, npTDMS 1.11.0, ObsPy 1.5.0, SciPy 1.12.0, pydub 0.25.1, TinyTag 2.3.0, h5py 3.16.0, MTH5 0.6.8, and `/usr/bin/ffmpeg`.

The qualification environment contains the optional packages. In addition, a fresh process in the audit `base` environment confirmed that `zarr`, `xarray`, and `netCDF4` are absent before GWexpy imports. Six public auto routes (TimeSeries, TimeSeriesDict, and Matrix Zarr reads; Matrix Zarr read/write; Matrix NetCDF read/write) returned the expected backend-specific `ImportError`. The [reproducible probe](2026-09-27-public-io-cross-format-post-fix-evidence/probe_missing_zarr_public_routes.py) and [machine-readable observations](2026-09-27-public-io-cross-format-post-fix-evidence/optional-missing-zarr-public-routes.json) record this run. The conformance gate's `6 blocked / 6 total` is its generated contract display and did not fail the test gate.

The GWpy override inventory's source locations were synchronized after class and method line numbers moved; its 1,150 case records and behavioral evidence were preserved. The `gwexpy` conda environment was reinstalled editable from this worktree so notebook kernels and distribution metadata both report 0.2.4. The frozen baseline bundle remains unchanged.

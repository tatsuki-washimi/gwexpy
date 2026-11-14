# Issue #887 — HDF5 axis preservation

Status: in-progress

## Objectives & Goals

Prevent silent coordinate changes when a `TimeSeries` or `FrequencySeries` axis is reported regular by GWpy but cannot be reconstructed exactly from HDF5 `x0`/`dx` metadata. Reuse the existing `xindex` attribute layout; preserve the existing compact layout for axes exactly reproduced by affine metadata. Keep changes within HDF5 and the existing reader/writer paths. Do not alter v0.2.10 release policy (#872).

## Detailed Roadmap (by Phase)

1. **Baseline reproduction — completed**
   - Run the #887 100-sample, 1 ns-jitter time axis and 2,000,000-sample `geomspace(1, 10)` frequency axis against current `origin/main` (`071cbfcba`).
   - Record regularity, HDF5 attributes, coordinate equality and maximum absolute error.
2. **Implementation — in progress**
   - Compare an explicit axis against GWpy `Index.define(x0, dx, n)` exactly.
   - Route mismatches through existing `xindex` storage and suppress misleading `dx` metadata.
   - Keep TimeSeries exact-HDF5 read handling from affine-merging explicit axes.
   - Apply exact-coordinate bounds selection to native HDF5 `TimeSeries`, `TimeSeriesDict`, and `TimeSeriesList` reads; retain GWpy coverage/padding for regular members in mixed dictionaries.
3. **Verification — in progress**
   - Add TimeSeries and FrequencySeries round-trip cases for coordinates, values, dtype, units, names, real/complex data, caller attrs, and bounded collection reads.
   - Run focused and related HDF5 tests, Ruff checks, and applicable CI.
   - Request human scientific/data-integrity review through the PR label.
4. **PR — pending**
   - Push the implementation branch and open `[AGENT:verify_physics]` against `main` with numerical evidence, audit manifest, validation results, and `needs-physics-review` label.

## Baseline and Expected Results

| Axis | GWpy `xindex.regular` | Baseline HDF5 layout | Baseline max abs coordinate error | Fixed result |
|---|---:|---|---:|---|
| `arange(100) * 1e-6 + alternating 1e-9 s` | `True` | `x0`, `dx`, no `xindex` | `9.80000000000116e-08 s` | `xindex`, exact (0 s) |
| `geomspace(1, 10, 2_000_000) Hz` | `True` | `x0`, `dx`, no `xindex` | `6.697413581347061 Hz` | `xindex`, exact (0 Hz) |

Both examples also preserved their values, `float64` dtype, data unit, axis unit, and name after the fix. Existing exact-affine layouts remain covered by regular TimeSeries and FrequencySeries HDF5 tests.

## Testing & Verification Plan

- Baseline / after numerical script: use GWexpy's `main` conda environment, h5py-backed files in `$TMPDIR`; compare `np.array_equal` and max absolute coordinate error.
- Focused: `conda run -n gwexpy python -m pytest tests/timeseries/test_hdf5_irregular_axis.py tests/frequencyseries/test_hdf5_irregular_axis.py -q`.
- Related HDF5 regression suite: TimeSeries layout/exact-t0/family, FrequencySeries layout/GWpy interop, and general GWpy HDF5 compatibility.
- Style: `ruff check` and `ruff format --check` on changed Python files in the `gwexpy` environment.
- CI: GitHub Actions checks on the opened PR; record the completed check summary here and in the PR.
- Omitted unless CI or source changes require them: full test suite and non-HDF5 I/O suites. MyPy is run on changed production modules.

## Models, Recommended Skills, and Effort Estimates

- Implementation route: GPT-6 Luna, medium effort (routine, bounded implementation; deterministic tests available).
- Independent code review: GPT-6 Sol, medium effort, routed with `hmodel` as `final_independent_review`.
- Skills used: `setup_plan`, `gwexpy_conda_jobs`, `lint_check`, `verify_physics`, `code-reviewer`, `pr-writer`, `github-mcp-secret-filter`.
- Estimated total wall time: 30–45 minutes; quota: medium. Main uncertainty is interaction with the TimeSeries transactional writer and its reader merge path.

## Scientific Acceptance Notes

The change does not alter GWpy's global regularity classification, sample values, axis units, data units, or transforms. It selects the existing explicit-coordinate HDF5 representation only when affine metadata fails an exact readback-coordinate check. Human review is requested because previously written files could silently encode a different scientific axis. No new API, dependency, HDF5 attribute name, or release gate is added.

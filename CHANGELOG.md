# Changelog

## Unreleased

### Behaviour-visible bug fixes

- **statistics**: `compute_student_t_nu()` / `TimeSeries.student_t_spectrogram()`
  now return a GPS time axis (the input `TimeSeries`'s `t0` plus the STFT
  relative time) instead of relative-to-start seconds. `fftlength`, `stride`
  (and `overlap`), `sample_rate`, `window`, and `frange` are now validated:
  non-finite/non-positive values raise `ValueError` instead of an opaque
  downstream failure, and `stride > fftlength` now raises `ValueError`
  explicitly rather than silently running a gapped analysis that skips
  samples between segments (`scipy.signal.stft` accepts a negative
  `noverlap` without erroring). The underlying `scipy.signal.stft` call now
  passes `window="hann", detrend=False, boundary="zeros", padded=True`
  explicitly instead of relying on scipy's defaults. Partially addresses
  #465 (DC/Nyquist bin bias is tracked separately).

- **statistics**: `compute_student_t_nu()` now fits the DC (index 0) and,
  when the segment length is even, Nyquist (the last one-sided bin)
  frequency bins using only their real FFT coefficient, instead of
  concatenating real and imaginary parts as if they were independent for
  those bins too. A real-valued signal's DC/Nyquist FFT coefficients are
  purely real, so the previous re+im concatenation fed a constant-zero
  imaginary half into the fit, collapsing the estimated Student-t `nu`
  toward 0 and producing a systematic false non-Gaussianity detection at
  DC regardless of the actual input. This is a GWexpy-specific
  correctness fix, not a GWpy compatibility change (GWpy has no
  equivalent Student-t fit API); DC/Nyquist are identified structurally
  (index 0 / the last one-sided bin), not via a floating-point frequency
  comparison, computed from the *effective* segment length (`scipy.signal.stft`
  silently shrinks `nperseg` to `len(ts)` when the requested value exceeds
  it, which can flip its parity relative to the requested `fftlength *
  sample_rate`). `compute_student_t_nu()` now also rejects complex-valued
  input with `ValueError` (previously silent, and `scipy.signal.stft`
  ignores `return_onesided=True` for complex input, which would have
  broken this bin classification). Completes #465.

## [0.1.10] - 2026-07-18

This is a bugfix release covering numerical regularization, axis regularity,
histogram weights, and TimeSeries resampling dtype and boundary contracts.

### Behaviour-visible bug fixes

- **numerics**: `TimeSeriesMatrix.partial_correlation_matrix()` now defaults
  to `eps="auto"` instead of `1e-8`, and `WhitenTransform` now defaults to
  `eps="auto"` instead of `1e-12`. `LaplaceGram.normalize_per_sigma()` now
  defaults to `eps="auto"` instead of `1e-30`. These defaults scale their
  regularization to the input, so strain-scale partial correlations and
  whitening no longer collapse toward zero. Pass the former explicit value to
  retain prior behavior. `partial_correlation_matrix(eps=None)` continues to
  disable its added ridge for compatibility. All three APIs now reject
  non-finite input and invalid epsilon values with `ValueError` (#482).

- **types**: `AxisDescriptor.regular` now compares represented adjacent
  intervals with a narrowly bounded interval-scale tolerance. It accepts
  ordinary floating-point regular grids while rejecting materially unequal
  large-offset float32 intervals that could yield an incorrect `delta` (#492).

- **histogram**: `Histogram.fill()` now preserves fractional scalar weights
  when histogram coordinates use an integer dtype, including flow-bin values
  and variance accumulation (#489).

- **timeseries**: time-bin `resample()` now implements `closed="right"` as an
  include-lowest first bin followed by right-closed bins. It also validates
  enum options, widths, and offsets before constructing the bin grid (#491).

- **timeseries**: `TimeSeries.asfreq()` preserves integer source dtype and
  exact values when an integral `fill_value` is representable. Out-of-range
  integral fills now raise `ValueError` rather than causing a lossy dtype
  promotion; fractional, complex, NaN, and infinite fills still promote to a
  dtype that can represent them (#490).

## [0.1.9] - 2026-07-11

This is a bugfix release. It completes the GWF read-path NaN-padding
harmonization deferred from 0.1.8 (#481) and fixes two fitting bugs
surfaced by the #461 follow-up audit: an off-by-one in `fit_series`'s
`sigma` cropping at exact `x_range` boundaries, and an unvalidated
`run_mcmc(n_walkers=...)` that let emcee's internal error leak through
(#466).

### Behaviour-visible bug fixes

- GWF-specific reads (`TimeSeries.read`/`TimeSeriesDict.read`/`TimeSeriesMatrix.read`
  with `format="gwf"`) now default `gap="pad"` to `NaN` padding instead of `0.0`,
  completing the harmonization with `SeriesMatrix.append`'s NaN default shipped in
  0.1.8. Code that relied on zero-filled GWF gaps should pass an explicit `pad=`
  value. Padding a missing region (an inter-file gap or a requested interval
  that extends beyond the available data) with `NaN` on a non-floating-point
  GWF channel dtype now raises a clear `ValueError` instead of silently
  corrupting the data with an out-of-range integer fill value or leaking
  NumPy's opaque ``cannot convert float NaN to integer`` (#481).

### Bug fixes

- **fitting**: `fit_series(..., sigma=..., x_range=...)` no longer crashes with
  a spurious `Sigma length mismatch` `ValueError` when `x_range`'s upper bound
  exactly matches a data bin edge. The sigma array is now cropped using the
  same index range that `Series.crop()` actually used for the fitted data (#466).
- **fitting**: `FitResult.run_mcmc()` now validates `n_walkers >= 2 * ndim`
  before invoking `emcee.EnsembleSampler`, raising a clear `ValueError` with
  the required minimum instead of letting emcee's internal `RuntimeError`
  surface with no gwexpy-level context (#466).

### Development

- Move maintainer-only `.harness/` AI workflow files out of the public
  repository and skip private harness sync tests when the harness is absent
  (#483).

## [0.1.8] - 2026-07-04

This is a bugfix and I/O-hardening release. It fixes the `from_obspy`
crash on `obspy.Stream` input (#452) and hardens the SeriesMatrix / I/O
layer: gap handling, metadata aliasing, key round-trips, and reader
failure modes surfaced by the #443 follow-up audit are fixed with explicit
contracts and regression tests (#450).

### Behaviour-visible bug fixes

- `SeriesMatrix.append(gap="pad")` (and the matrix multi-source merge paths
  built on it) now pads gaps with `NaN` instead of `0.0` by default, matching
  the missing-data convention already used by the multi-source readers.
  Code that relied on zero-filled gaps should pass an explicit `pad=` value
  or handle NaNs before reductions/FFT/filtering. Padding a gap with `NaN`
  on a non-floating matrix dtype now raises `ValueError` instead of silently
  inserting an invalid fill value.
- **io/netcdf4**: A failed per-channel series construction no longer degrades
  to a plain dict (which deferred an opaque `AttributeError` downstream); the
  reader now raises a clear `ValueError` at the source.
- **io/netcdf4**: Time-coordinate detection now prefers an explicitly named
  coordinate (`time`/`Time`/`TIME`/`t`) and warns when it has to fall back to
  a datetime64 coordinate — more loudly when several candidates make the
  choice ambiguous. Pass `time_coord=` to select the axis explicitly.
- **io/zarr**: Reading a store without a stored epoch now emits a
  `UserWarning` (the silent default was GPS epoch 1980); a new
  `t0_override=` argument sets the epoch explicitly.
- **io/sdb**: Non-numeric values dropped by numeric coercion are no longer
  silently lost; the reader warns with the affected column and count.

### Bug fixes

- **interop/obspy**: `TimeSeries.from_obspy` no longer crashes with an opaque
  `AttributeError` when given an `obspy.Stream`: a single-trace Stream is
  converted directly, and a multi-trace Stream raises a clear `TypeError`
  pointing to the new `TimeSeriesDict.from_obspy` / `to_obspy` methods.
  The `from_neo` (Block/Segment), `from_torch` (non-1D tensors) and
  `from_root` (container inputs) converters gained equivalent clear-error
  guards (#452).
- `SeriesMatrix` metadata aliasing is now fully closed out: `rows`/`cols`/
  `meta`/`attrs` are deep-copied in `_get_meta_for_constructor` (used by
  `crop`/`append`/`diff`/`pad`/`interpolate`), in the matrix math ops
  (`trace`, `matmul`, `schur`, `abs`, …) and in `__array_ufunc__`, so
  mutating a derived object's metadata no longer corrupts the source.
  This extends the `astype`/`real`/`imag` fix shipped in 0.1.6 (#442) to the
  remaining construction paths.
- **io/hdf5**: `SeriesMatrix` row/column keys are now JSON-encoded on write
  (`key_format="json"`, mirroring the NetCDF4/Zarr encoders), so tuple and
  numeric keys survive HDF5 round-trips instead of collapsing to strings.
  Legacy string-keyed files remain readable.
- **io/dttxml**: JSON key decoding now recursively converts nested lists to
  tuples (matching the Zarr decoder) so round-trips preserve tuple keys and
  hashability.
- **interop**: `to_xarray()` (and `to_xarray_frequencyseries()`) no longer
  persist an unset `channel`/`name` as the literal string `"None"`, which
  previously round-tripped back into a bogus `Channel("None")`. The
  `channel` metadata now survives `to_dict`/`from_dict`,
  `to_xarray`/`from_xarray` and `to_hdf5`/`from_hdf5` round-trips.

### Behaviour changes

- **interop**: `TimeSeries.from_dict`/`from_json`/`from_xarray`/`from_pandas`/
  `from_hdf5_dataset`/`from_netcdf4`/`from_polars` (and the underlying interop
  helpers) now accept explicit `channel`/`unit`/`name`/`t0`/`dt` keyword
  arguments to supply metadata the source object cannot carry; an explicit
  argument always takes priority over a stored value (`user > source`).
- **interop**: when `t0`/`dt` cannot be recovered or inferred, the converters
  (`from_dict`, `from_pandas`, `from_hdf5`, `from_netcdf4`, `from_polars`) now
  fall back to `t0=0`/`dt=1` with a `UserWarning` instead of silently, matching
  the Zarr reader. The fallback values are unchanged, so this is a backward-
  compatible (SemVer MINOR) addition; pass `t0=`/`dt=` explicitly to silence it.

### Dependencies

- Added upper version bounds to I/O backends to guard against major-version
  API breaks: `h5py>=3.0,<4` (core), `obspy<2`, `nptdms<2`, `netCDF4<2`,
  `zarr>=2,<4` (extras). These caps stay within the currently tested major
  series but can affect dependency resolution in environments that already
  pin newer majors (#450, D1).

### Tests

- **interop**: Added regression tests for the `resolve_timing`/`resolve_meta`
  helpers, `channel` round-trips across dict/json/xarray/hdf5/netcdf4, the
  `"None"`-string guard, falsy-zero `t0=0.0` handling, user-supplied metadata
  overrides, and the new missing-timing `UserWarning` for each converter.
- Added matrix key round-trip contracts across HDF5/NetCDF4/Zarr, multi-source
  reader and gap-padding coverage, SDB/Zarr reader warning tests, and
  io-conformance generators/validators for the SDB and Zarr backends.

### Known limitations

- GWF-specific read padding still follows its existing zero-padding path
  (`gwexpy/timeseries/_gwf_io.py`); harmonizing GWF padding with the
  matrix/core NaN convention is deferred to a follow-up issue.

## [0.1.7] - 2026-06-27

This is a numerical-robustness hardening release. A Phase 1 audit across the
statistics, fitting, and spectral modules surfaced a set of silent-failure and
degenerate-input bugs; this release adds explicit input-contract guards and a
matching suite of regression tests so that invalid inputs raise clear errors
instead of producing Inf/NaN or silently wrong results.

### Bug fixes

- **fitting/core**: LSQ cost classes now validate `dy` — zero, negative,
  non-finite, or complex elements raise `ValueError` instead of causing silent
  fit failures or Inf/NaN results (#469).
- **fitting/gls**: GLS classes gained covariance / inverse-covariance
  conditioning guards and PSD checks (#457 via #472).
- **fitting/models**: Added degenerate-parameter guards to model shape
  functions (#455 via #471).
- **statistics**: Guarded degenerate and non-finite inputs across
  `rayleigh_test`, `gauch`, `dq_flag`, and `student_t_indicator` — all-NaN
  inputs, `p=0` false vetoes, Inf-corrupted distributions, an `IndexError`,
  and mis-sized segments are now handled explicitly (#459 via #470).
- **statistics/roc**: `calculate_roc` and `evaluate_detection_performance`
  enforce their input contract — empty classes, shape mismatch, non-finite
  scores, and tied-FPR bias now raise `ValueError`, and sklearn-style
  `{-1, +1}` labels are handled correctly (#468).
- **spectral**: Guarded `bootstrap_spectrogram` edge cases — float truncation,
  NaN/Inf energy, `rebin_width` validation, covariance mean-imputation, and
  zero-width confidence-interval warnings (#460 via #473).

### Tests

- Added input-contract regression suites covering the fixes above:
  `tests/fitting/test_lsq_cost_dy_contract.py`,
  `tests/fitting/test_gls_contracts.py`,
  `tests/fitting/test_models_domain_contract.py`,
  `tests/statistics/test_degenerate_input_contract.py`,
  `tests/statistics/test_roc_input_contract.py`,
  `tests/spectral/test_bootstrap_spectrogram_contract.py`.

### Maintenance

- Centralized gwexpy provisioning across CI workflows and fixed nightly drift
  via a shared `setup-gwexpy` composite action (#454).
- Added release-note tooling (`tools/gen_release_notes.py`,
  `tools/publish_releases.sh`) that generates standardized GitHub Release notes
  from `CHANGELOG.md`.

### Documentation

- Added the Phase 1 numerical-robustness sweep and supplement reports under
  `tech_notes/` (#462).

## [0.1.6] - 2026-06-11

This is a bugfix and maintenance release: plotting/I/O follow-up fixes
(#440, #441, #442 via #443), a development dependency sweep (#431), and
FrequencySeries collection registry audit tests (#438).

### Bug fixes

- Fixed subplot geometry calculation in `Plot` so the expansion count always
  matches `_expand_args`: all arguments are now counted regardless of order
  (a leading Spectrogram or matrix no longer hides later containers), the
  duplicated counting loops were unified into a single helper, and a leading
  matrix keeps its grid geometry only when it is the sole argument (#440).
- All TimeSeries readers now handle a list or tuple of paths: formats with
  well-defined merge semantics (tdms, ats, csv, netcdf4, gbd, ndscope HDF5,
  zarr, sdb, win, dttxml) concatenate channels along time with NaN gap
  padding via a shared multi-source helper, while self-contained formats
  (wav, audio) raise a clear `ValueError` instead of an opaque backend
  `TypeError` (#441).
- GWF alias registration no longer swallows unexpected errors silently:
  expected missing-backend lookups return `None` as before, anything else
  emits a warning (#442).
- Pickling a `SeriesMatrix` now emits a warning listing any attrs entries
  that had to be dropped because they cannot be pickled, instead of
  dropping them silently (#442).
- `SeriesMatrix.astype()`, `.real`, `.imag`, `.conj()`, `.transpose()`/`.T`
  and `.reshape()` now deep-copy `attrs` like `.copy()` does, so mutating
  the result's attrs no longer leaks into the source matrix (#442).
- Multi-file NetCDF4/Zarr reads into `TimeSeriesMatrix` now preserve the
  matrix row/column keys instead of collapsing them through the dict-reader
  shortcut, and NetCDF4 gained a dedicated matrix writer.
- Multi-file matrix segment merging now passes `pad=np.nan`, so gaps between
  files are NaN-padded instead of raising.

### Maintenance

- Updated the development dependency group (24 packages) in
  `requirements-dev.txt` (#431).

### Tests

- Added plot geometry tests for mixed-container argument orders, single
  2D/3D/4D matrices, and parity with `_expand_args` expansion counts.
- Added multi-source reader tests covering merge, gap padding, overlap
  errors, empty-list rejection, and clear single-file-only errors.
- Added regression tests for pickle attrs warnings and attrs independence
  of derived matrices.
- **frequencyseries/io**: Added registry-backend audit tests and a developer
  note for FrequencySeries collection read/write fallback (#438). No
  behaviour change.

## [0.1.5] - 2026-06-10

This is a patch release focused on plotting and I/O hotfixes.

### Bug fixes

- Fixed `TimeSeriesDict.plot()` so multi-channel dictionaries are expanded into
  separate subplots instead of producing a blank or invalid figure (#432).
- Fixed ObsPy-backed seismic readers so `TimeSeriesDict` keys are stable string
  trace names (e.g. `"IU.ANMO.00.BHZ"`), enabling reliable string-based lookup
  (#435).
- Added support for passing a list or tuple of miniSEED paths to
  `TimeSeriesDict.read(..., format="mseed")` (#433).
- Fixed `gwexpy.frequencyseries` import-time I/O registration so FrequencySeries
  read formats are visible through the GWpy default I/O registry (#437).

### Tests

- Added regression tests for `TimeSeriesDict.plot()`.
- Added seismic I/O tests for string keys, list-of-path miniSEED input, and
  empty-list rejection.
- Added subprocess-isolated FrequencySeries I/O registration tests.

### Documentation

- Clarified that GWexpy is an independent package built on top of GWpy and is
  not an official component of the GWpy project.
- Updated the README installation notes to reflect that the conda-forge
  feedstock is available, while conda-forge packages may lag the latest PyPI
  release.

### Deferred

- Broad FrequencySeries collection read/write registry-backend migration is
  deferred to #438.
- Dependency sweep (#431) is deferred to the v0.2.0-prep lane.

## [0.1.4] - 2026-05-20

### Added

- **io/conformance**: Added the first contract-driven I/O conformance baseline for `gwf`, `hdf.ndscope`, `hdf5`, `csv`, `txt`, and `wav`.
- **io/contracts**: Added v3 public I/O contract policy fields for fixture generation, coverage status, CI jobs, and missing optional dependency behavior.
- **ci/io**: Added the `io-conformance` gate and expanded I/O gate documentation.
- **time**: Added opt-in `dtype=` output modes for `to_gps()`. The default
  remains GWpy-compatible, while `dtype=float` / `dtype="float"` return plain
  float values and `dtype="quantity"` returns seconds quantities for direct
  `.times` comparisons.

### Fixed

- **io/dttxml**: Fixed `load_dttxml_products()` so DTTXML `TS` entries remain raw dict payloads and do not collide with `TimeSeries.get()`.
- **io/gwf**: Provisioned the GWF backend for the PR fast gate and tolerated backend-specific GWF channel metadata variance.

### Documentation

- **feedback**: Updated the README, docs hub pages, footer links, roadmap, and
  troubleshooting pages to point lightweight bug reports and feature requests
  to the public feedback form. Security reports remain directed to the
  repository security policy.

### Tests

- **io/conformance**: Added deterministic fixture generators and read/write round-trip coverage for the v0.1.4 blocking format baseline.
- **io/dttxml**: Added regression coverage for DTTXML `TS` dict parsing through `read_timeseriesdict_dttxml()`.
- **netcdf**: Added a fixture-generation contract that requires generated
  NetCDF fixtures to expose an explicit time coordinate (#393).
- **timeseries/gwf**: Added regression coverage for multi-channel GWF
  list-source reads and padded gap reads with `parallel > 1`.

### Known Issues

- **io/zarr**: The optional `io-zarr` gate can hang in environments where Zarr 3.1.5 stalls during basic `create_array()` fixture generation. Zarr remains outside the v0.1.4 base blocking gate and is tracked for optional-backend hardening.

## [0.1.3] - 2026-05-12

### Fixed

- **timeseries/gwf**: Fixed multi-file GWF reads for `TimeSeries` and
  `TimeSeriesDict` inputs.
- **timeseries/matrix**: Fixed ndscope HDF5 auto-detection for
  `TimeSeriesMatrix.read()`.
- **io/contracts**: Aligned public I/O docs and contract metadata with current
  autodetection behavior.
- **frequencyseries/csv**: Added a dedicated CSV fast path that preserves the
  original frequency column values.
- **timeseries/zarr**: Fixed matrix round-trip coverage under zarr 3 and
  removed timeout-prone fixture behavior.
- **plot/geomap**: Treat PyGMT installations without a loadable GMT shared
  library as an unavailable optional backend instead of failing at import time.

### Known Issues

- **netcdf**: The bundled NetCDF fixture can fail the TimeSeries reader
  time-coordinate contract in some cases (#393). Generated NetCDF round-trip
  coverage still passes; users relying on NetCDF fixtures should verify that
  their files expose an explicit time coordinate.

## [0.1.2] - 2026-05-08

### Narrow v0.1.2 hotfix scope

- **io/gwpy4**: Narrow compatibility hotfixes for public I/O proxy imports and GWF list/dict read paths.
- **io/formats**: Targeted reader auto-identify and compatibility fixes for histogram HDF5, ATS/MTH5, audio, seismic, SegmentTable span CSV, and FrequencySeries DTT XML flows.
- **integration**: Narrow landing updates include only the minimal #369 landing/demo import hunk required for this track.
- **release status**: Version metadata and release notes are finalized for `v0.1.2`, but tag creation, PyPI publication, Zenodo publication, fresh release smoke reruns, and conda-forge refresh are still pending.

### Packaging & Optional Dependencies (issue #251)

- **packaging**: Added `netcdf4` extra (`netCDF4`, `xarray`) and `zarr` extra (`zarr`) to `pyproject.toml`; both are now included in the `all` convenience extra.
- **packaging**: Removed the experimental `gwexpy.gui` package, console script, and `gui` extra from the first PyPI distribution; GUI work remains source/development-only until the post-release stabilization track is complete.
- **packaging**: Tightened first-release artifact hygiene by excluding top-level tests, docs sample data, and package-internal Sphinx helper shims from built distributions.
- **packaging**: Removed hand-edited tail from `requirements-dev.txt`; `analysis` extras are now managed exclusively through `pyproject.toml`.
- **interop**: Fixed `_optional.py` `_EXTRA_MAP` — phantom extras (`interop`, `bio`, `stats`, `eda`) replaced with `None` entries that fall back to bare `pip install <package>`; `netCDF4`/`xarray` now point to `netcdf4` extra; `zarr` points to `zarr` extra.
- **io**: `ensure_dependency()` error hint corrected to `pip install 'gwexpy[<extra>]'` instead of `pip install <pkg>[<extra>]`.
- **io**: `_import_pydub`, `_import_obspy`, `_import_nptdms`, `_import_zarr`, `_import_xarray` error messages now include `pip install 'gwexpy[<extra>]'` hints.
- **fitting**: `gwexpy.fitting.__getattr__` error messages now suggest `pip install 'gwexpy[fitting]'`.
- **ci**: `io-optional` gate extended with `test_seismic_public_io.py`; `test_optional_deps.py` augmented with `gwexpy[extra]` hint assertions and `TestSeismicImportGuard`.
- **docs**: Installation guide updated with `netcdf4`, `zarr` extras and clarified `gui` is not in `all`.

### Infrastructure & CI

- **ci**: Comprehensive stabilization of the CI pipeline, resolving all `ModuleNotFoundError` and `SyntaxError` regressions.
- **ci**: Added mandatory **Notebook syntax validation** to the primary test workflow to proactively catch corrupted `.ipynb` files.
- **ci**: Restored and standardized scientific dependencies (`control`, `statsmodels`, `scikit-learn`, etc.) across all GitHub Actions environments.
- **docs**: Performed a global "reset-and-rewrap" of tutorial notebooks to fix indentation errors in `warnings.catch_warnings()` blocks.

### Added

- **fields**: `VectorField` and `TensorField` now support initialization directly from NumPy ndarrays (5D for VectorField, 6D for TensorField), automatically creating the component `ScalarField`s without breaking backward-compatible dictionary initialization.

### Changed

- **fields**: `ScalarField` binary arithmetic now fails fast with `ValueError` when operands have mismatched time/frequency domains, spatial domains, or coordinate grids. Align fields explicitly before arithmetic; future regridding/interpolation APIs will track explicit grid-alignment workflows.
- **plot**: `FieldPlot` labels now avoid empty unit brackets for unitless metadata and expose the latest scalar colorbar via the public `last_field_colorbar` attribute. Explicit `label=""` colorbar labels remain supported.

### Documentation

- **docs**: Unified the Class Index into five major categories (Core, Field, Signal, Analysis, Utilities) with standardized Japanese translations (e.g., "時系列行列" for `TimeSeriesMatrix`).
- **docs**: Redesigned major guidance pages (`io_formats`, `numerical_stability`, `time_utilities`, `architecture`) using judgment tables and decision-driven structures.
- **docs**: Refined visual aesthetics with custom CSS for modern typography, responsive tables, and card-based navigation in the Sphinx RTD theme.
- **docs**: Integrated SEO/OGP metadata, sitemaps, and automated "Last updated" timestamps.

### Infrastructure

- **ci**: Implemented a weekly documentation health check (`docs-weekly-health.yml`) to monitor broken links, terminology consistency, and JA/EN synchronization.
- **ci**: Standardized notebook testing pipeline with `papermill` for full execution (Light) and `nbval` for syntax validation (Heavy).
- **ci**: Integrated `nbstripout` into pre-commit hooks to manage repository size and diff clarity.
- **pre-commit**: Added a GitHub Actions PR template with automated quality gate checklist.

## [0.1.1] - 2026-04-28

### Added

- **SegmentTable**: New factory methods `read()` and `read_csv()` for initializing from external files.
- **SegmentTable**: Support for the iterable protocol (`__iter__`) and `RowProxy` for direct row-wise processing.
- **Tutorials**: Comprehensive new notebooks for `SegmentTable`, `Noise Generation`, and `Spectral Fitting`.
- **Infrastructure**: Automated tutorial execution testing via `pytest --nbmake` and GitHub Actions.
- **analysis/coupling**: `CouplingFunctionAnalysis` — `from_time_windows()`, `from_time_windows_batch()`, `bkg_window` パラメータ追加 (Phase 1).
- **analysis/coupling_result**: `CouplingResult` — `to_csv()`, `from_csv()`, `to_txt()`, `from_txt()`, `to_summary_csv()` によるファイルエクスポート (Phase 2).
- **analysis/coupling_result**: `CouplingResult` — `plot_significance()`, `plot_asdgram()`, `plot_snrgram()` 可視化メソッド追加 (Phase 3).
- **analysis/coupling_result**: `CouplingResultCollection` — 複数結果の集約コンテナ (Phase 2).
- **analysis/stats**: `SpectralStats` — スペクトル統計コンテナ（`spectral_stats()` より取得） (Phase 2).
- **analysis/response**: `ResponseFunctionResult` — `plot_projection_summary()`, `plot_response_matrix()` 可視化メソッド追加 (Phase 3).
- **analysis**: `ResponseFunctionResult`, `ResponseFunctionAnalysis`, `estimate_response_function`, `detect_step_segments` を `gwexpy.analysis` から公開 (Phase 4).
- **docs**: Sphinx API リファレンスに `coupling_result`, `response`, `threshold`, `stats` モジュール追加 (Phase 4).
- **tutorials**: `case_coupling_analysis.ipynb` / `case_response_analysis.ipynb` に Phase 1–3 の利用例を追補 (Phase 4).

### Changed

- **SegmentTable**: `add_series_column()` now accepts a simple `loader(segment)` callable for intuitive lazy loading.
- **noise/peaks**: Renamed `lorentzian_line()` parameter `fwhm` to `gamma` for consistency with implementation.

### Fixed

- **fitting/highlevel**: Resolved frequency bin alignment between PSD and covariance matrix in `fit_bootstrap_spectrum`.
- **fitting/highlevel**: Removed unsupported `stride` parameter from `fit_bootstrap_spectrum`.
- **table/segment_plot**: Fixed `TypeError` when an existing `Axes` object is provided to `segments()`.

### Previously Unreleased (merged into 0.1.1)

- **interop/multitaper**: `from_mtspec` / `from_mtspec_array` が `cls` パラメータを
  無視して CI 付き入力でも常に `FrequencySeriesDict` を返していた問題を修正。
- **interop/meshio**: `cell_data` のみを持つ `meshio.Mesh` を `from_meshio` に渡した場合の
  誤った補間経路を廃止し、明確な `ValueError` を送出。
- **interop/pyroomacoustics**: `room.rir` のインデックス順序（マイク ↔ ソース）を修正。
- **interop/openems**: HDF5 データセットの `"Time"` / `"frequency"` 属性の優先使用を修正。

## [0.1.0] - 2026-03-15

### Release Summary

First stable release of GWexpy for SoftwareX publication. This release focuses on API stability, GWpy compatibility, and reproducible commissioning workflows.

### Changed

- **Version**: Updated from `0.1.0b2` to `0.1.0` (stable release)
- **GWpy API UX Compatibility**: Aligned key spectral API call conventions with GWpy 4.x usage patterns.
  - `TimeSeries.transfer_function` now accepts GWpy-style positional calls:
    - `transfer_function(other, fftlength, overlap, window, average, ...)`
  - `TimeSeriesDict` / `TimeSeriesList` now accept positional spectral args for:
    - `csd`, `coherence`, `csd_matrix`, `coherence_matrix`
    - positional `(fftlength, overlap)` is supported in addition to keyword usage
  - Mixed positional+keyword specification of `fftlength`/`overlap` now raises clear `TypeError`.
- **Authors**: Removed email from `pyproject.toml` to prevent spam (contact via GitHub Issues or paper)

### Added

- **Compatibility policy doc**:
  - `docs/developers/compatibility/gwpy/API_UX_POLICY_20260303.md`
- **GWpy compatibility tests**:
  - `tests/timeseries/test_transfer_function_compat.py`
  - `tests/timeseries/test_collections_spectral_compat.py`
  - `tests/timeseries/test_fft_param_compat.py`
  - Includes edge-case checks for positional/keyword conflicts and invalid numeric `other` in collection APIs.
- **CI workflow for compatibility gate**:
  - `.github/workflows/test-compat-gwpy.yml`
  - Runs focused GWpy-compat tests plus `tests/timeseries`, with pinned `numpy<2.0` and `astropy<7.0`.
- **Publication materials**:
  - Paper source: `docs/gwexpy-paper/main.tex`
  - Publication preparation plan: `docs/developers/plans/for_paper_publication.md`

## [0.1.0b2] - 2026-02-23

### Changed

- **API Unification**: Standardized all spectral analysis function signatures to use time-based parameters (`fftlength`/`overlap` in seconds) instead of sample-count-based parameters (`nperseg`/`noverlap`). This aligns gwexpy with GWpy conventions and improves user experience.
  - **Affected Functions**:
    - `gwexpy.spectral.bootstrap_spectrogram()` - now accepts `fftlength` and `overlap` (seconds)
    - `gwexpy.fitting.fit_bootstrap_spectrum()` - now accepts `fftlength` and `overlap` (seconds)
    - `gwexpy.spectrogram.Spectrogram.bootstrap()` and `.bootstrap_asd()` - now accept `fftlength` and `overlap` (seconds)
    - `gwexpy.fields.signal.*` spectral functions (spectral_density, compute_psd, freq_space_map, coherence_map) - now accept `fftlength` and `overlap` (seconds)
    - `gwexpy.timeseries.TimeSeriesMatrix` spectral methods (\_vectorized_psd, \_vectorized_csd, \_vectorized_coherence) - now accept `fftlength` and `overlap` (seconds)
  - **Migration Note**: Using deprecated `nperseg` or `noverlap` parameters will raise `TypeError` with guidance to use `fftlength` and `overlap` instead. No deprecation period - breaking change applies immediately.
  - **New Module**: `gwexpy.utils.fft_args` provides helper functions for parameter validation and conversion:
    - `parse_fftlength_or_overlap()` - converts time values (float, int, Quantity) to seconds and samples
    - `check_deprecated_kwargs()` - detects and rejects deprecated parameters
    - `get_default_overlap()` - returns window-appropriate default overlap values (GWpy-compatible)
  - **GWpy Compatibility**: All functions now follow GWpy conventions for time-based FFT parameters, improving interoperability and reducing API confusion.

### Improved

- **Numerical Stability**: Implemented a comprehensive numerical hardening strategy for low-amplitude gravitational-wave data (O(1e-21)).
  - **Adaptive Whitening**: `whiten()` now uses an adaptive `eps` relative to input variance, preventing signal destruction in quiet channels.
  - **Robust ICA**: `ica_fit()` includes internal standardization and relative tolerances to handle high-dynamic-range data.
  - **Safe Logging**: Visualization tools now use dynamic floor calculation to prevent `-inf` or clipped values in dB plots.
  - **Machine Precision**: Numerical constants now adapt to float32/float64 machine precision.

### Fixed

- **GBD**: Apply amplifier range scaling when reading Graphtec `.gbd` so analog channels are correctly converted from raw counts to volts, and treat `Alarm`/`AlarmOut`/`Pulse*`/`Logic*` as digital status channels (0/1, dimensionless). Digital channel mapping can be overridden via `digital_channels=...`.

## [0.1.0b1] - 2026-02-01

### Initial Public Release

- This is the first public beta release of `gwexpy`. All previous development history (up to internal version 0.4.0) is consolidated here.

### Important Notes

- **gwpy Compatibility**: This release is compatible with `gwpy>=3.0.0,<4.0.0`. gwpy 4.0.0 introduced breaking API changes that are not yet supported. Users should ensure they have gwpy 3.x installed.

### Refactored

- **Exception Handling**: Eliminated broad `except Exception` patterns in NDS, GUI, and IO modules. Replaced with specific exception types (`OSError`, `ValueError`, `KeyError`, etc.) for more predictable error handling and better debugging.
- **GUI Architecture**: Improved separation of concerns between UI and core logic layers in GUI components.

### Added

- **Core Data Structures**:
  - `TimeSeries`, `FrequencySeries`, `Spectrogram` classes with metadata management.
  - `TimeSeriesMatrix`, `FrequencySeriesMatrix`, `SpectrogramMatrix` for multi-channel data handling.
  - `ScalarField`, `VectorField`, `TensorField` for 4D experimental domain semantics.
- **Numerical Semantics**:
  - Strict unit propagation for calculus methods.
  - Fixed DC component handling in integration to prevent singularities.
- **Advanced Signal Processing**:
  - functional Short-Time Laplace Transform (`stlt`).
  - High-performance resampling with various aggregation methods.
  - Whitening and standardization models.
- **Interoperability**:
  - Support for various file formats: TDMS, GBD, WIN, ATS, SDB/SQLite, WAV.
  - Integration with ML frameworks (Torch/TensorFlow) and ROOT (CERN).
  - MTH5 support for magnetotelluric data.
- **Fitting & Statistics**:
  - Comprehensive `fitting` module with `iminuit` and `emcee` (MCMC) support.
  - Statistical aggregation and interpolation for matrix structures.
- **GUI**:
  - Interactive GUI for real-time streaming data visualization and analysis.

### Improved

- **Type Safety**: Comprehensive type annotation expansion across the codebase:
  - Added strict type hints to GUI (UI layer, NDS modules, streaming, engine).
  - Enhanced `TimeSeriesMatrix` mixin with Protocol-based type-safe `super()` calls.
  - Introduced `TypedDict` definitions for structured data in IO and GUI modules.
  - Expanded MyPy coverage to include `gui/nds/` and `gui/ui/` directories.
- **CI Stability**:
  - Replaced deprecated `qtbot.waitForWindowShown()` with `qtbot.waitExposed()` in GUI tests.
  - Added warning filters to suppress third-party deprecation warnings (NumPy, pandas) in test configuration.
  - Refined MyPy exclude patterns for better coverage-exclusion balance.
- Optimized ROOT/NumPy vectorization.
- Refactored `noise` module for better maintenance.

### Fixed

- **GUI Tests**: Resolved flaky test issues related to window visibility timing.
- **Type Errors**: Fixed various MyPy errors including uninitialized attributes and missing return type annotations.
- Fixed unit propagation in complex matrix operations.
- Corrected IFFT amplitude scaling for one-sided spectra.

[Unreleased]: https://github.com/tatsuki-washimi/gwexpy/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/tatsuki-washimi/gwexpy/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/tatsuki-washimi/gwexpy/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/tatsuki-washimi/gwexpy/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/tatsuki-washimi/gwexpy/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/tatsuki-washimi/gwexpy/compare/v0.1.0b2...v0.1.0
[0.1.0b2]: https://github.com/tatsuki-washimi/gwexpy/compare/v0.1.0b1...v0.1.0b2
[0.1.0b1]: https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.1.0b1

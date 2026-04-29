# Changelog

## [Unreleased]

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

[Unreleased]: https://github.com/tatsuki-washimi/gwexpy/compare/v0.1.0b2...HEAD
[0.1.0b2]: https://github.com/tatsuki-washimi/gwexpy/compare/v0.1.0b1...v0.1.0b2
[0.1.0b1]: https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.1.0b1

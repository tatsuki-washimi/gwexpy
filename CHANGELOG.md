# Changelog

## [0.4.0] - 2026-01-16

### Added
- **Numerical Semantics (A2 Fixed)**:
    - `FrequencySeries`: Added strict unit propagation for calculus methods (`differentiate`, `integrate`).
    - `FrequencySeries`: Fixed DC component handling in time-integration to prevent singularities (returns 0 instead of NaN/Inf).
    - `Spectral`: Enforced strict unit differentiation between PSD (`V²/Hz`) and Power Spectrum (`V²`).
    - `Fitting`: Implemented `Fitter` class and `GLS` solver with automatic unit propagation for model evaluation.
    - `Astro`: Validated range calculation physics and unit consistency.
- **Transformations**:
    - `FrequencySeriesMatrix.ifft`: Now strictly preserves amplitude by correcting the one-sided spectral scaling factor (×0.5).

## [Unreleased] - 2025-12-19

### Added
- **Fitting**: Enhanced `fitting` module with `iminuit` integration.
  - Implemented `FitResult` class for analyzing fit results.
  - Added MCMC support using `emcee` with `corner` plot generation (`FitResult.plot_corner`).
  - Added `[fitting]` extra dependency group.
- **TimeSeries Imputation**: Added `max_gap` parameter to `impute()` methods (and `gwexpy.timeseries.preprocess.impute_timeseries`). Allows preventing interpolation across large gaps.
  - Supported in `TimeSeries`, `TimeSeriesDict`, `TimeSeriesList`, `TimeSeriesMatrix`.
  - When `max_gap` is specified, edge extrapolation is also disabled for safety.
- **New File Reader Support**:
  - **TDMS (.tdms)**: Support for National Instruments TDMS format via `npTDMS`.
  - **GBD (.gbd)**: Support for Graphtec Data Logger format.
  - **WIN (.win)**: Support for NIED Hi-net seismic format (patched `obspy` reader).
  - **WAV (.wav)**: Support for WAVE audio format (multi-channel).
  - **ATS (.ats)**: Support for Metronix ADU magnetotelluric logger (direct binary reader `ats` and `mth5` integration `ats.mth5`).
  - **SDB (.sdb/.sqlite)**: Support for Davis Vantage Pro2 Weather Station data (WeeWX SQLite database) with automatic unit conversion.
- **Time Conversion**: Added `leap` parameter to `gwexpy.interop.gps_to_datetime_utc` to handle leap seconds:
  - `leap='raise'` (default): Raises `LeapSecondConversionError`.
  - `leap='floor'`: Clamps to `59.999999`s.
  - `leap='ceil'`: Rounds to next minute `00.000000`s.

- **SpectrogramMatrix Refactor**:
  - `SpectrogramMatrix` now inherits from `SeriesMatrix`, enabling powerful analysis methods: `crop`, `append`, `pad`, `interpolate`, and statistical aggregations (`mean`, `std`, etc).
  - Robust support for 3D `(Batch, Time, Freq)` and 4D `(Row, Col, Time, Freq)` data structures.
  - Added `to_series_1Dlist()` and `to_series_2Dlist()` conversion methods.

### Improved
- **ROOT Interoperability**: Significantly optimized `to_th1d`, `to_th2d`, and `from_root` using vectorization, improving performance for large arrays.
- **Noise Module**: Refactored `gwexpy.noise.magnetic` and `gwexpy.noise.peaks` for better maintenance and accuracy.
- **TimeSeries Analysis**: Separated analysis methods into `_analysis.py` for better code organization.
- **TimePlaneTransform**: Implemented `linear` interpolation for `at_time(t, method='linear')`.
- **STLT**: Implemented functional Short-Time Laplace Transform (`TimeSeries.stlt`) using `scipy.signal.stft`.
- **Resampling**: Enhanced `TimeSeries.resample` to support fast aggregation methods `median`, `min`, `max`, and `std` using `scipy` or `pandas` optimizations.
- **MTH5 Interop**: Added experimental `to_mth5` and `from_mth5` functions in `gwexpy.interop.mt_` (requires optional `mth5` package).
- **I/O Error Handling**: Standardized `NotImplementedError` for unsupported formats.
  - Now raises `gwexpy.interop.errors.IoNotImplementedError` with specific hints and references.

### Fixed
- **SpectrogramMatrix**: Fixed unit propagation in arithmetic operations and `append` compatibility checks.

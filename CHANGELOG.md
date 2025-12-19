# Changelog

## [Unreleased] - 2025-12-19

### Added
- **TimeSeries Imputation**: Added `max_gap` parameter to `impute()` methods (and `gwexpy.timeseries.preprocess.impute_timeseries`). Allows preventing interpolation across large gaps.
  - Supported in `TimeSeries`, `TimeSeriesDict`, `TimeSeriesList`, `TimeSeriesMatrix`.
  - When `max_gap` is specified, edge extrapolation is also disabled for safety.
- **Time Conversion**: Added `leap` parameter to `gwexpy.interop.gps_to_datetime_utc` to handle leap seconds:
  - `leap='raise'` (default): Raises `LeapSecondConversionError`.
  - `leap='floor'`: Clamps to `59.999999`s.
  - `leap='ceil'`: Rounds to next minute `00.000000`s.

### Improved
- **TimePlaneTransform**: Implemented `linear` interpolation for `at_time(t, method='linear')`.
- **STLT**: Implemented functional Short-Time Local Transform (`TimeSeries.stlt`) using `scipy.signal.stft`.
- **Resampling**: Enhanced `TimeSeries.resample` to support fast aggregation methods `median`, `min`, `max`, and `std` using `scipy` or `pandas` optimizations.
- **MTH5 Interop**: Added experimental `to_mth5` and `from_mth5` functions in `gwexpy.interop.mt_` (requires optional `mth5` package).
- **I/O Error Handling**: Standardized `NotImplementedError` for unsupported formats (WIN, SDB, etc.).
  - Now raises `gwexpy.interop.errors.IoNotImplementedError` with specific hints and references.
  - Removed premature dependency checks for unimplemented formats to avoid confusing `ImportError`.

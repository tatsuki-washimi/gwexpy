# Changelog

## [0.1.0b1] - 2026-01-25

### Initial Public Release

- This is the first public beta release of `gwexpy`. All previous development history (up to internal version 0.4.0) is consolidated here.

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

- Optimized ROOT/NumPy vectorization.
- Refactored `noise` module for better maintenance.

### Fixed

- Fixed unit propagation in complex matrix operations.
- Corrected IFFT amplitude scaling for one-sided spectra.

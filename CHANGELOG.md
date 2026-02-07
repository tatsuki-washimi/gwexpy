# Changelog

## [Unreleased]

### Added

- **Field Signal Processing Methods**: Implemented 23 new methods for `ScalarField` and `FieldDict` based on GWpy's `TimeSeries` and `FrequencySeries` APIs to achieve feature parity with standard gravitational wave data analysis workflows.
  - **High Priority Methods (10)**: Core signal processing essential for GW data analysis:
    - `detrend(type='linear')` - Remove polynomial trends (linear/constant)
    - `taper(side='leftright', duration=1.0)` - Apply Tukey window to endpoints
    - `crop(start, end, copy=True)` - Extract time segments
    - `pad(pad_width, **kwargs)` - Extend data with padding
    - `zpk(zeros, poles, gain, **kwargs)` - Zero-Pole-Gain filter
    - `whiten(fftlength=None, **kwargs)` - Normalize ASD for matched filtering
    - `convolve(fir, **kwargs)` - Time-domain FIR convolution
    - `inject(other, alpha=1.0)` - Add simulated signals
    - `csd(other, **kwargs)` - Cross-spectral density analysis
    - `coherence(other, **kwargs)` - Frequency-coherence analysis
    - `spectrogram(stride, fftlength=None, **kwargs)` - Time-frequency representation
  - **Low Priority Methods (12)**: Utility and mathematical operations:
    - `value_at(t)` - Extract values at specific times
    - `abs()`, `sqrt()` - Element-wise operations
    - `mean()`, `median()`, `std()`, `rms()` - Statistical operations with axis support
    - `is_compatible(other)` - Validate field compatibility
    - `is_contiguous(other, tol=None)` - Test segment adjacency
    - `append(other, gap='raise', **kwargs)` - Concatenate fields in time
    - `prepend(other, **kwargs)` - Prepend another field
  - **Medium Priority Methods (5)**: Advanced analysis methods with straightforward implementation:
    - `autocorrelation(maxlag=None, **kwargs)` - Compute autocorrelation function to reveal periodic structures
    - `correlate(other, maxlag=None, **kwargs)` - Time-domain cross-correlation for time-delay estimation
    - `interpolate(sample_rate, kind='cubic')` - High-quality interpolation resampling (better than FFT for calibrated data)
    - `rayleigh_spectrum(fftlength=None, **kwargs)` - Rayleigh statistic vs frequency to detect non-Gaussian features
    - `rayleigh_spectrogram(stride, fftlength=None, **kwargs)` - Time-frequency Rayleigh statistic for transient feature detection
  - **FieldDict Support**: All methods are also available on `FieldDict`, applying operations to all components
  - **Documentation**: Added comprehensive user guide in English and Japanese (`docs/web/{en,ja}/user_guide/fields_new_methods.rst`) with examples and complete workflow demonstrations, including new medium-priority methods
  - **Note**: Remaining medium-priority advanced analysis methods (q_transform, heterodyne, gate, transfer_function) are deferred for future implementation due to design complexity

### Changed

- **API Unification**: Standardized all spectral analysis function signatures to use time-based parameters (`fftlength`/`overlap` in seconds) instead of sample-count-based parameters (`nperseg`/`noverlap`). This aligns gwexpy with GWpy conventions and improves user experience.
  - **Affected Functions**:
    - `gwexpy.spectral.bootstrap_spectrogram()` - now accepts `fftlength` and `overlap` (seconds)
    - `gwexpy.fitting.fit_bootstrap_spectrum()` - now accepts `fftlength` and `overlap` (seconds)
    - `gwexpy.spectrogram.Spectrogram.bootstrap()` and `.bootstrap_asd()` - now accept `fftlength` and `overlap` (seconds)
    - `gwexpy.fields.signal.*` spectral functions (spectral_density, compute_psd, freq_space_map, coherence_map) - now accept `fftlength` and `overlap` (seconds)
    - `gwexpy.timeseries.TimeSeriesMatrix` spectral methods (_vectorized_psd, _vectorized_csd, _vectorized_coherence) - now accept `fftlength` and `overlap` (seconds)
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

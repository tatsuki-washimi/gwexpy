# gwexpy Numerical Logic (A2) Contract

This document defines the strict numerical and semantic contracts for the `gwexpy` library. All core components (TimeSeries, FrequencySeries, Spectrogram, etc.) must adhere to these rules to ensure physical correctness and reproducibility.

## 1. Axis Alignment and Resampling

- **Strict Alignment**: Binary operations (addition, subtraction, multiplication, division) between two series objects (e.g., `TimeSeries`, `FrequencySeries`) require strictly matching axes.
- **Mismatch Handling**: If `dx` (sample rate/frequency resolution) or `x0` (start time/frequency) do not match, a `ValueError` is raised. Automatic interpolation is **forbidden** for these operations to prevent unintended numerical artifacts.
- **Explicit Resampling**: Users must explicitly call `.resample()` or `.interp()` to align axes before performing binary operations.

## 2. Unit Consistency

- **Physical Units**: All numerical operations must preserve or correctly transform physical units using `astropy.units`.
- **Unit Compatibility**: Operations like addition and subtraction require units to be strictly compatible (convertible).
- **Dimensionless Outputs**: Operations that normalize data (e.g., `whiten()`, `standardize()`) must return results with `dimensionless_unscaled` units.
- **Power Spectra**: PSDs calculated with `scaling='density'` must have units of `[Y]^2 / Hz`. PSDs with `scaling='spectrum'` must have units of `[Y]^2`.

## 3. Metadata Propagation

- **Inheritance**: Derived objects (from slicing, cropping, or single-input transformations) must inherit metadata:
    - `epoch` (GPS start time)
    - `unit` (Physical unit)
    - `name` (Channel/Series name)
- **SeriesMatrix**: In `TimeSeriesMatrix` or `SpectrogramMatrix`, operations across the time/frequency axis must preserve the metadata of individual elements where possible.

## 4. Signal Processing and Spectral Analysis

- **FFT Scaling**: FFT and IFFT must follow consistent scaling conventions (typically the GWpy/LIGO convention for one-sided spectra).
- **DC Handling**: The DC component (f=0) should be handled gracefully, often set to 0 in whitening or filtering to avoid offsets, but preserved in raw FFTs.
- **Bootstrap Statistics**: High-level estimation functions (like `bootstrap_spectrogram`) must return error estimates (e.g., quantiles) that are shape-consistent with the main estimation.

## 5. Fitting and Modeling

- **Unit-Aware Models**: Model evaluation (e.g., `FitResult.model(x)`) should handle input units and return results with the appropriate physical units.
- **Robust Fitting**: Fitting routines must handle `NaN` values (typically by ignoring them or requiring imputation) and provide reliable parameter recovery within stated bounds.
- **Parameter Access**: Best-fit parameters must be accessible via `.params` with both `.value` and `.error` attributes, while remaining compatible with float operations.

## 6. Wave 3 Baseline Numerical Contracts

Issue #273 starts the Wave 3 numerical and analysis contract pass. This first
baseline is intentionally docs/test-only. It records current public behavior and
separates behavior changes that need physics or statistical review.

### Spectral Wrapper Units And Axes

- `TimeSeries.psd(..., scaling="density")` is contracted as a density spectrum
  with units equivalent to `[Y]^2 / Hz`.
- `TimeSeries.psd(..., scaling="spectrum")` is contracted as a power spectrum
  with units equivalent to `[Y]^2`.
- `TimeSeries.asd()` is contracted as an amplitude density with units
  equivalent to `[Y] / sqrt(Hz)`.
- `TimeSeries.csd(other)` is contracted as a cross-density spectrum with units
  equivalent to `self.unit * other.unit / Hz`.
- `TimeSeries.spectrogram(..., scaling="density")` is contracted as a density
  spectrogram with the same frequency bins as the corresponding PSD call for
  the same segment length.
- Frequency axes for these wrappers must start at DC and be monotonically
  increasing in Hz.

### Transient FFT Axis Gap

`TimeSeries.fft(mode="transient")` returns an amplitude spectrum, not a density
spectrum. The current implementation passes `self.dt.value` directly to
`numpy.fft.rfftfreq()`. For `TimeSeries` objects whose `dt` unit is not seconds,
this can produce frequency bins with the wrong numerical scale while still
labeling them as Hz. The Wave 3 baseline records this as a strict xfail instead
of changing behavior in the first contract PR.

Any fix must be split from the baseline PR because it changes numerical
frequency axes for non-second time units.

### Matrix Spectral Baseline

`TimeSeriesMatrix` vectorized spectral helpers are now covered for:

- constructor-provided non-second `dt` that the matrix constructor normalizes to
  a second-valued axis;
- preservation of matrix row and column keys;
- preservation of per-element unit, name, and channel metadata for vectorized
  FFT outputs;
- `_vectorized_asd()` returning the square root of the vectorized PSD on the
  same frequency axis.

Explicit non-second matrix axes, such as `xunit="ms"`, still expose the same
raw-`dt.value` frequency-axis gap in vectorized FFT/PSD/CSD/coherence helpers.
This pass records the FFT form of that gap as a strict xfail.

Changing vectorized PSD/ASD/CSD/coherence normalization remains review-sensitive
because it can alter physical PSD and coherence values.

### Window Normalization Helper

`get_window_normalized()` currently returns only the raw SciPy window array.
Although its docstring mentions an ENBW return value, no tuple is returned in
the current public behavior. This mismatch is a documented contract gap. A later
PR should either update the docstring or introduce a new return contract with
compatibility notes.

### Bruco And SpectralStats Baseline

- `FastCoherenceEngine.compute_coherence()` returns a plain `ndarray` over the
  cached frequency axis. A zero auxiliary signal returns all-zero coherence.
- Identical finite input currently produces coherence values equal to one within
  floating-point tolerance. Values are expected to remain finite and within
  numerical tolerance of `[0, 1]`.
- `SpectralStats.significance()` currently delegates division to the underlying
  `FrequencySeries` values. A zero sigma bin therefore emits a runtime warning
  and returns infinite significance for that bin. Rejecting, clipping, or
  regularizing zero sigma is a future statistical-behavior decision.

## 7. Review-Sensitive Follow-Ups

Do not combine these changes with docs/test-only contract PRs:

- Fix transient FFT frequency bins for non-second `TimeSeries.dt`.
- Fix `TimeSeriesMatrix` vectorized spectral frequency bins for explicit
  non-second matrix axes.
- Change Bruco/FastCoherenceEngine PSD or coherence normalization, including
  DC/Nyquist one-sided scaling.
- Change transfer-function epsilon semantics or default zero-denominator
  handling.
- Change PSD/ASD/CSD/coherence floors, clipping, or finite-value rejection.
- Change `SpectralStats.significance()` zero-sigma behavior.
- Change fitting sigma/covariance validation, GLS conditioning behavior, or
  complex-data support.

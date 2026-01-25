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

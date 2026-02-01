# Algorithm Fix Report - 2026-02-01

**Author**: Gemini CLI
**Date**: 2026-02-01
**Context**: Implementation of P2/P3 fixes identified in `merged_validation_report.md`.

## Summary

This report documents the fixes and improvements applied to the `gwexpy` library to address algorithm validation findings. The focus was on Transfer Function regularization, Bayesian fitting stability, DTT XML parsing reliability, dimensionality reduction consistency, and comprehensive numerical stability/documentation improvements covering noise models and signal processing utilities.

## 1. Transfer Function Regularization ([P2])

*   **Issue**: `transfer_function(mode='transient')` could produce numerical instability (Infs/NaNs) when the denominator (reference signal FFT) contained zeros or near-zero values.
*   **Fix**:
    *   Added an optional `epsilon` parameter to `TimeSeries.transfer_function`.
    *   Updated `_divide_with_special_rules` in `gwexpy/timeseries/_signal.py` to implement L2 regularization when `epsilon > 0`.
    *   **Formula**:
        *   Steady mode: $H(f) = \frac{\text{CSD}(f)}{\text{PSD}(f) + \epsilon}$
        *   Transient mode: $H(f) = \frac{\text{FFT}_B(f) \cdot \text{FFT}_A^*(f)}{|\text{FFT}_A(f)|^2 + \epsilon}$

## 2. Dimensionality Reduction Consistency ([P1/P3])

*   **3D Shape Restoration ([P1])**:
    *   Updated `pca_fit` and `ica_fit` in `gwexpy/timeseries/decomposition.py` to store the original matrix shape in `input_meta`.
    *   Updated inverse transforms to correctly reshape reconstructed data back to `(channels, cols, time)`.
*   **ZCA Whitening Warnings ([P3])**:
    *   Added a `UserWarning` in `whiten` (`gwexpy/signal/preprocessing/whitening.py`) when `n_components` is specified with `method='zca'`, as this breaks the channel-preserving property of ZCA.

## 3. DTT XML Parsing Reliability ([P1])

*   **Issue**: The standard `dttxml` package often discards imaginary parts of complex transfer functions (specifically subtype 6), leading to phase information loss.
*   **Fix**:
    *   Implemented a native DTT XML parser `load_dttxml_native` in `gwexpy/io/dttxml_common.py`.
    *   The native parser correctly decodes Base64 streams for `floatComplex` and `doubleComplex` types, preserving full phase information.
    *   Updated `read_frequencyseriesdict_dttxml` and `read_frequencyseriesmatrix_dttxml` to support a `native=True` parameter.
    *   Added `UserWarning` when potential phase loss is detected while using the non-native parser.

## 4. Bayesian Fitting & Time Series Modeling ([P3])

*   **MCMC Stability ([P3])**:
    *   Enforced fixed covariance assumption in `run_mcmc`.
    *   Implemented **Cholesky decomposition** (`scipy.linalg.solve_triangular`) for $\chi^2$ calculations, replacing the less stable `pinv` approach.
*   **ARIMA Timing Assumptions ([P3])**:
    *   Updated `ArimaResult.forecast` docstring to explicitly state assumptions: equally-spaced data without gaps, and use of GPS/TAI continuous seconds (no leap seconds).
    *   Clarified that `n_obs` reflects the internally differenced series length when $d > 0$.

## 5. Statistical & Physical Refinements ([P2/P3])

*   **Step Detection Robustness ([P2])**:
    *   Updated `detect_step_segments` in `gwexpy/analysis/response.py` to set the default `freq_tolerance` to 2.0 Hz.
    *   Added documentation explaining that tolerance should be at least $2 \cdot \Delta f$ to avoid false boundaries due to spectral fluctuations.
*   **SigmaThreshold Assumptions ([P2])**:
    *   Updated `SigmaThreshold` in `gwexpy/analysis/coupling.py` with detailed documentation on Gaussian approximation validity (requires $K \ge 10$ averages).
    *   Added `_check_gaussian_validity` to issue a warning when `n_avg < 10`.
*   **k-space Conventions ([P3])**:
    *   Updated `ScalarField.fft_space` docstring to clarify the **angular wavenumber** convention ($k = 2\pi / \lambda$) and the sign convention used for descending spatial axes ($dx < 0$) to preserve physical phase factors.
*   **Laplace Overflow Guardrail ([P2])**:
    *   Added overflow checks in `TimeSeries.laplace` and `stlt` to raise `ValueError` if the exponent exceeds 700.
*   **Robust Zero-Frequency Handling ([P3])**:
    *   Updated `gwexpy/noise/obspy_.py` and `gwexpy/noise/colored.py` to use boolean masks `data[f == 0] = np.nan` (or `inf`/`0.0`) instead of only checking the first element. This ensures robustness against unsorted frequency arrays.
*   **Landau Model Clarification ([P3])**:
    *   Updated docstring in `gwexpy/fitting/models.py` to explicitly state that the Landau function uses the Moyal approximation and that parameter `A` is a scaling factor, not the peak amplitude ($Peak \approx 0.607 A$).
*   **Hurst Optimization ([P3])**:
    *   Refactored `local_hurst` in `gwexpy/timeseries/hurst.py` to move the `MockTS` class definition to module level, reducing object creation overhead in loops.
*   **Lorentzian Normalization ([P3])**:
    *   Clarified Peak Normalization (ASD form) in `gwexpy/noise/peaks.py`.

## 6. Validation

### Tests Performed
The following test suites were executed to verify the changes:

1.  **Transfer Function Regularization**: Verified Infs/NaNs suppression.
2.  **Fitting & MCMC**: `pytest tests/test_fitting*.py` (Passed).
3.  **Spectral & Bootstrap**: `pytest tests/spectral/test_bootstrap_advanced.py` (Passed).
4.  **DTT XML & Fields**: Manual verification of docstring accuracy and native parser logic.
5.  **Noise Models**: Validated f=0 handling with unsorted arrays.

### Result
All identified P1/P2/P3 algorithm findings have been addressed through code modifications, parameter tuning, or rigorous documentation of assumptions. The changes improve numerical robustness and provide better guidance to users without breaking existing functionality.

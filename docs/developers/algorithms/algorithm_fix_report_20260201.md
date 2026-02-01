# Algorithm Fix Report - 2026-02-01

**Author**: Gemini CLI
**Date**: 2026-02-01
**Context**: Implementation of P2/P3 fixes identified in `merged_validation_report.md`.

## Summary

This report documents the fixes and improvements applied to the `gwexpy` library to address algorithm validation findings. The focus was on Transfer Function regularization, Bayesian fitting stability, and various documentation/check improvements.

## 1. Transfer Function Regularization ([P2])

*   **Issue**: `transfer_function(mode='transient')` could produce numerical instability (Infs/NaNs) when the denominator (reference signal FFT) contained zeros or near-zero values.
*   **Fix**:
    *   Added an optional `epsilon` parameter to `TimeSeries.transfer_function`.
    *   Updated `_divide_with_special_rules` in `gwexpy/timeseries/_signal.py` to implement L2 regularization when `epsilon > 0`.
    *   **Formula**:
        *   Steady mode: $H(f) = \frac{\text{CSD}(f)}{\text{PSD}(f) + \epsilon}$
        *   Transient mode: $H(f) = \frac{\text{FFT}_B(f) \cdot \text{FFT}_A^*(f)}{|\text{FFT}_A(f)|^2 + \epsilon}$

## 2. Bayesian Fitting Improvements ([P3])

*   **Issue**:
    *   `run_mcmc` assumed a fixed covariance matrix without explicit checks or documentation.
    *   Use of `np.linalg.pinv` for $\chi^2$ calculation in GLS fitting poses numerical stability risks for ill-conditioned matrices.
*   **Fix**:
    *   **Fixed Covariance Assumption**: Added `ValueError` in `run_mcmc` if `cov_inv` is callable (parameter-dependent), enforcing the fixed covariance assumption required for omitting the $\log|\Sigma|$ term.
    *   **Cholesky Decomposition**:
        *   Updated `FitResult` to store the original covariance matrix `cov`.
        *   Updated `GeneralizedLeastSquares` (used by Minuit) and `run_mcmc` (used by emcee) to prefer **Cholesky decomposition** (`scipy.linalg.solve_triangular` with `np.linalg.cholesky`) over `pinv` when the covariance matrix is available and positive-definite.
        *   This calculates $r^T \Sigma^{-1} r$ as $\|L^{-1} r\|^2$ where $\Sigma = LL^T$, improving stability.

## 3. Minor Documentation & Checks ([P3])

*   **Lorentzian Normalization**:
    *   Updated docstring in `gwexpy/noise/peaks.py` to clarify that `lorentzian_line` uses **Peak Normalization** (ASD form), not Area Normalization.
*   **Geomagnetic Unit Heuristics**:
    *   Added `logging.info` in `gwexpy/noise/magnetic.py` when `geomagnetic_background` automatically converts units (Tesla to pT) based on amplitude magnitude.
*   **Block Bootstrap**:
    *   Updated `gwexpy/spectral/estimation.py`:
        *   Added support for `block_size='auto'`, which estimates the optimal block size based on the spectrogram overlap ratio ($\lceil \text{duration} / \text{stride} \rceil$).
        *   Added a warning if significant variation in segment levels is detected, indicating potential non-stationarity which violates block bootstrap assumptions.
        *   Fixed missing imports (`FrequencySeries`, `get_window`, `logging`) in `estimation.py`.

## 4. Validation

### Tests Performed
The following test suites were executed to verify the changes:

1.  **Transfer Function Regularization**:
    *   Verified with a temporary script that `epsilon` effectively suppresses Infs/NaNs in transient mode and produces expected near-zero outputs for uncorrelated/zero inputs.
2.  **Fitting & MCMC**:
    *   `pytest tests/test_fitting.py` (Passed)
    *   `pytest tests/test_fitting_gls.py` (Passed)
    *   `pytest tests/test_fitting_mcmc.py` (Passed)
3.  **Spectral & Bootstrap**:
    *   `pytest tests/spectral/test_bootstrap_advanced.py` (Passed after import fix)
    *   `pytest tests/signal` (Passed)
    *   `pytest tests/spectral` (Passed)
    *   `pytest tests/spectrogram` (Passed)

### Result
All relevant tests passed. The changes improve numerical robustness and provide better guidance to users without breaking existing functionality.

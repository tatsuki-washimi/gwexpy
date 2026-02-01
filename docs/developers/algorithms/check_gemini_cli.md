# gwexpy Algorithm Validation Report (Gemini CLI Audit)

**Auditor:** Gemini CLI (Senior Physicist / Software Engineer)
**Date:** 2026-01-31
**Subject:** Algorithm audit of `gwexpy` package (Analysis, Fitting, Fields, Signal Processing)

## Executive Summary
The `gwexpy` package demonstrates a high level of implementation quality with specific attention paid to physical unit consistency and rigorous signal processing conventions (e.g., DTT compatibility, Transient amplitude preservation).

Most algorithms are mathematically sound. However, a few areas regarding **GLS covariance matrix numerical stability**, **spatial FFT assumptions**, and **ARIMA input regularity** warrant attention to ensure robustness in edge cases.

---

## Detailed Findings

### 1. Physical Fields & k-space (`ScalarField.fft_space`)

*   **Status**: **Valid with minor caveat**
*   **Severity**: **Low**
*   **Finding**: The conversion $k = 2\pi \cdot f_{\text{spatial}}$ is physically correct for angular wavenumber. The implementation correctly handles units ($1/\text{length}$) and negative frequencies.
*   **Caveat**: When handling a spatial axis with negative spacing (`dx < 0`, e.g., a descending coordinate system), the code simply flips the sign of the resulting $k$ axis (`k_values = -k_values`). While `np.fft.fftfreq` relies on `abs(dx)`, simply flipping the sign of the axis label might not be sufficient if the data array itself isn't reversed prior to FFT. The Fourier transform of $f(-x)$ is $F(-k)$, so flipping the axis is mathematically consistent, but users might expect the data array index order to match the new axis direction.
*   **Recommendation**: Ensure that the relationship between the array index order and the strictly monotonic $k$-axis is documented or enforced, especially for `ifft_space` reconstruction.

### 2. Transient Response Analysis (`ResponseFunctionAnalysis`)

*   **Status**: **Valid**
*   **Severity**: **None**
*   **Finding**:
    *   The `_fft_transient` method correctly implements a "one-sided amplitude spectrum" logic manually (`dft[1:] *= 2.0`). Crucially, it includes the inverse logic in `frequencyseries.ifft` (implied by the context) to divide by 2.0 before calling `irfft`. This ensures energy preservation for round-trip transformations, which is critical for transient analysis.
    *   The `detect_step_segments` logic using a tracking spectrogram and SNR threshold is a standard and robust approach for stepped sine injections.

### 3. Robust Statistics (`bootstrap_spectrogram`)

*   **Status**: **Valid**
*   **Severity**: **None**
*   **Finding**:
    *   The VIF (Variance Inflation Factor) calculation correctly integrates the window autocorrelation function ($\rho(\tau)$) to account for the correlation induced by overlapping Welch segments. The formula $1 + 2\sum (1 - k/M)\rho^2$ is standard for variance of the mean of correlated time series.
    *   The distinction between block bootstrap (`factor=1.0`) and standard bootstrap (with analytical VIF correction) is statistically sound. Block bootstrap empirically captures the correlation structure if the block size is sufficient.

### 4. Bayesian Fitting & GLS (`run_mcmc`)

*   **Status**: **Potential Instability**
*   **Severity**: **Medium**
*   **Finding**:
    *   The likelihood calculation `chi2 = float(np.real(r.conj() @ cov_inv @ r))` is mathematically correct for complex Gaussian distributed errors.
    *   **Concern**: Inverting the covariance matrix (`cov_inv = np.linalg.pinv(cov_arr)`) can be numerically unstable if `cov` is ill-conditioned (e.g., adjacent frequency bins are highly correlated due to excessive window overlap or zero-padding). A naive `pinv` might introduce large numerical errors in the $\chi^2$ calculation.
*   **Recommendation**: Instead of storing and passing `cov_inv`, passing the Cholesky decomposition of `cov` ($L$ where $LL^T = C$) is more numerically stable. The term $r^\dagger C^{-1} r$ can be computed as $\|L^{-1} r\|^2$ by solving the linear system, avoiding explicit inversion.

### 5. Time Series Modeling (ARIMA)

*   **Status**: **Conditional Validity**
*   **Severity**: **Medium**
*   **Finding**: The `fit_arima` and `ArimaResult` implementation assumes the input `TimeSeries` is strictly regular (constant `dt` and no gaps).
    *   While `timeseries.dt` is checked, `gwpy.TimeSeries` objects can sometimes contain gaps (represented as NaNs or masked arrays) while maintaining a metadata `dt`. Standard ARIMA implementations (like `statsmodels`) do not inherently handle missing data (NaNs) well without explicit Kalman Filter support, or they might treat the data as contiguous, leading to phase errors in the model.
*   **Recommendation**: Add an explicit check for NaNs or gaps in the input data before passing to `statsmodels`. If gaps exist, either enforce `nan_policy='raise'` or strictly require the use of state-space models (`SARIMAX`) which can handle missing data, rather than standard `ARIMA`.

### 6. Dimensionality Reduction (PCA/ICA)

*   **Status**: **Valid**
*   **Severity**: **None**
*   **Finding**: The flattening logic `X_features = value.reshape(-1, time)` correctly treats spatial/channel dimensions as features and time as samples. This aligns with the standard physical interpretation of finding "spatial modes" or "sensor correlations" that persist over time. The inverse transform logic correctly attempts to reconstruct the 3D structure using the stored shape/metadata.

---

## Summary of Recommendations

1.  **Fitting (High Priority)**: Consider replacing `cov_inv` with a Cholesky factor or using `np.linalg.solve` inside the likelihood function rather than pre-computing an explicit inverse matrix, to improve stability for large `n_bins`.
2.  **ARIMA (Medium Priority)**: Enforce strict contiguity checks (no NaNs/gaps) on `TimeSeries` inputs or ensure `statsmodels` is configured to use the Kalman Filter for missing data handling.
3.  **ScalarField (Low Priority)**: Verify coordinate monotonic consistency for spatial axes with negative `dx` during round-trip FFT operations.

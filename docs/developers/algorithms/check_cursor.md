# gwexpy Algorithm Validation Report

**Auditor**: LLM-based cross-check (senior physicist / software auditor)  
**Date**: 2025-01-31  
**Scope**: Physical/statistical validity and numerical stability of custom algorithms in `gwexpy`.

---

## 1. Physical Fields & k-space (`ScalarField.fft_space`)

### 1.1 k-coordinate transformation

**Severity**: — (Validated OK)

**Finding**: The angular wavenumber formula \(k = 2\pi \cdot \text{fftfreq}(n, d=|\mathrm{d}x|)\) is correct. It satisfies \(k = 2\pi/\lambda\) and matches the standard Fourier convention for spatial FFT.

**Source**: `gwexpy/fields/scalar.py` L414–415

**Recommendation**: None. The implementation correctly uses `abs(dx_value)` for `fftfreq` and flips the sign when the original axis is descending.

---

### 1.2 Unit handling

**Severity**: — (Validated OK)

**Finding**: The unit mapping \(k_{\mathrm{unit}} = 1 / \mathrm{dx}_{\mathrm{unit}}\) is correct. For example, if `dx` is in m, then `k` is in rad/m.

---

### 1.3 Inverse FFT (`ifft_space`)

**Severity**: — (Validated OK)

**Finding**: The relation \(\mathrm{d}x = 2\pi / (n \cdot |\mathrm{d}k|)\) is correct and consistent with `fftfreq`. The handling of descending axes (`dk < 0`) is appropriate.

**Source**: `gwexpy/fields/scalar.py` L478–483

---

## 2. Transient Response Analysis

### 2.1 Automatic step detection (`detect_step_segments`)

**Severity**: **Medium** (Unstable under specific conditions)

**Finding**: The default `freq_tolerance=1.0` Hz can be inconsistent with the spectrogram's frequency resolution. For `fftlength=1.0` s, \(\Delta f = 1/\mathrm{fftlength} = 1\) Hz, so bin-to-bin jumps are exactly at the tolerance. Frequency estimates can sit between bins, leading to false segment boundaries or missed transitions.

**Source**: `gwexpy/analysis/response.py` L229

**Recommendation**: Set `freq_tolerance` as a multiple of the frequency resolution, e.g. `freq_tolerance = max(1.0, 1.5 * (1.0 / fftlength))`, or make it a fraction of the current peak frequency (e.g. 1–5%).

---

### 2.2 Amplitude preservation in `_fft_transient`

**Severity**: — (Validated OK)

**Finding**: One-sided spectrum normalization is correct: `rfft(x)/nfft` with bins 1:-1 (or 1:) multiplied by 2. This recovers the amplitude of a real sinusoid and matches the GWpy convention.

**Source**: `gwexpy/timeseries/_spectral_fourier.py` L207–219

---

## 3. Robust Statistics

### 3.1 Variance Inflation Factor (VIF)

**Severity**: — (Validated OK)

**Finding**: The formula
\[
\mathrm{factor} = \sqrt{1 + 2 \sum_{k=1}^{M-1} \left(1 - \frac{k}{M}\right) |\rho_{\mathrm{window}}(kS)|^2}
\]
is consistent with standard overlap corrections (e.g. Bendat & Piersol, Percival & Walden). The window autocorrelation and weighting are implemented correctly.

**Source**: `gwexpy/spectral/estimation.py` L159–234

---

### 3.2 VIF inference when `nperseg` is unknown

**Severity**: **Medium** (Unstable under specific conditions)

**Finding**: When `nperseg` is not provided, the code uses `dummy_step=100` and `dummy_nperseg = int(round(100 * overlap_ratio))` to infer overlap. This may not match the actual `nperseg`/`noverlap` used to build the spectrogram, especially for different segment lengths (e.g. 512 vs 1024), so the VIF can be inaccurate.

**Source**: `gwexpy/spectral/estimation.py` L355–361

**Recommendation**: Store `nperseg` and `noverlap` (or `stride`) in spectrogram metadata and reuse them when available. Fall back to inference only when metadata is missing, and document the limitation.

---

### 3.3 Block bootstrap logic

**Severity**: — (Validated OK)

**Finding**: Moving block bootstrap is implemented correctly: `num_possible_blocks = n_time - block_size + 1`, concatenation of blocks, and truncation to `n_time` follow standard practice. Block bootstrap correctly sets `factor=1.0` for the VIF, since dependence is captured by the blocks.

---

## 4. Bayesian Fitting

### 4.1 Log-likelihood with GLS covariance

**Severity**: — (Validated OK)

**Finding**: For GLS, \(\chi^2 = r^{\dagger} W r\) with \(W = \mathrm{cov}^{-1}\) is correct. For real residuals, \(r^{\dagger} = r^T\), and `np.real(val)` is redundant but safe. For complex residuals (not currently supported in GLS mode), the Hermitian form remains valid; the `NotImplementedError` for complex + cov in `fit_series` avoids that path.

**Source**: `gwexpy/fitting/core.py` L464–470

---

### 4.2 Covariance inverse stability

**Severity**: **Low** (Optimization suggestion)

**Finding**: `np.linalg.pinv` is used for the covariance inverse, which is appropriate for near-singular matrices. For poorly conditioned matrices, numerical issues can still appear in the quadratic form.

**Recommendation**: Optionally add a condition-number check or a regularization step (e.g. add a small diagonal term) when `np.linalg.cond(cov) > 1e10`.

---

## 5. Time Series Modeling (ARIMA)

### 5.1 GPS-aware forecast timing

**Severity**: — (Validated OK)

**Finding**: The forecast start time \(\mathrm{forecast\_t0} = t_0 + n_{\mathrm{obs}} \cdot \mathrm{dt}\) is correct. It places the first forecast step immediately after the last observation, in line with statsmodels' 0-based indexing.

**Source**: `gwexpy/timeseries/arima.py` L139–145

---

### 5.2 `t0` / `epoch` consistency

**Severity**: **Low** (Documentation / robustness)

**Finding**: `fit_arima` uses `timeseries.t0`. GWpy-derived series typically expose `x0` or `epoch`; `t0` is used as an alias. Code assumes `t0` exists.

**Recommendation**: Add an explicit fallback, e.g. `t0 = getattr(timeseries, 't0', None) or getattr(timeseries, 'x0', None) or getattr(timeseries, 'epoch', None)`, and raise a clear error if none is available.

---

## 6. Dimensionality Reduction (PCA/ICA)

### 6.1 Flattening and reconstruction consistency

**Severity**: **High** (Logical inconsistency)

**Finding**: The original shape `(channels, cols, time)` is flattened to `(channels*cols, time)` for sklearn. The inverse transform yields `(samples, features)` and is reshaped to `(features, 1, samples)` = `(channels*cols, 1, time)`. The original layout `(channels, cols)` is not restored. For `cols > 1`, the output shape is `(channels*cols, 1, time)` instead of `(channels, cols, time)`.

**Source**: `gwexpy/timeseries/decomposition.py` L265–325

**Recommendation**: Store the original shape `(channels, cols)` in `pca_res.preprocessing` (e.g. `input_shape`) and use it in the inverse:

```python
# In pca_fit: store
preprocessing["input_shape"] = (matrix.shape[0], matrix.shape[1])

# In pca_inverse_transform: restore
orig_shape = pca_res.preprocessing.get("input_shape")
if orig_shape:
    X_rec_3d = X_rec_val.T.reshape(*orig_shape, -1)
else:
    X_rec_3d = X_rec_val[:, None, :]
```

Apply the same pattern for ICA inverse transform.

---

### 6.2 PCA/ICA scaler application

**Severity**: — (Validated OK)

**Finding**: `_fit_scaler` and `_inverse_scaler` are applied consistently. The scaler operates on the flattened feature representation, which is appropriate for PCA/ICA.

---

## 7. Additional Custom Algorithms (Extended Audit)

### 7.1 Noise Models (`noise/peaks.py`, `noise/magnetic.py`, `noise/colored.py`)

#### Lorentzian / Voigt / Gaussian peaks

**Severity**: — (Validated OK)

**Finding**: The Lorentzian formula \(\mathrm{ASD}(f) = A \cdot \gamma / \sqrt{(f-f_0)^2 + \gamma^2}\) is correct and peaks at amplitude \(A\). The Voigt profile uses Faddeeva (`wofz`) with correct normalization: \(z = ((f-f_0) + i\gamma)/(\sigma\sqrt{2})\), and peak factor at \(z_0 = i\gamma/(\sigma\sqrt{2})\). The Gaussian and Schumann resonance (PSD sum of Lorentzians) are implemented correctly.

**Source**: `gwexpy/noise/peaks.py` L27–154, `gwexpy/noise/magnetic.py` L15–88

---

#### Power-law noise (`colored.py`)

**Severity**: **Low** (Edge case)

**Finding**: At \(f=0\), the code checks `f_vals[0] == 0` and sets `data[0] = np.inf` (exponent > 0) or `0.0` (exponent < 0). This assumes the frequency array is sorted with the first element being the minimum. If frequencies are passed in arbitrary order, zeros in the middle may be mishandled.

**Source**: `gwexpy/noise/colored.py` L62–67

**Recommendation**: Use `mask = (f_vals == 0)` and apply the zero-frequency handling to all such indices.

---

### 7.2 ObsPy Seismic Quantity Conversion (`noise/obspy_.py`)

**Severity**: **Medium** (Edge case)

**Finding**: The conversion formulas \(\mathrm{velocity} = \mathrm{accel}/(2\pi f)\) and \(\mathrm{displacement} = \mathrm{accel}/(2\pi f)^2\) are physically correct. However, the \(f=0\) handling uses `if len(f) > 0 and f[0] == 0`, which assumes the frequency array is sorted and that only the first element can be zero. If \(f=0\) appears elsewhere (e.g. after interpolation), those bins remain unhandled and can produce `inf` or `nan`.

**Source**: `gwexpy/noise/obspy_.py` L228–243

**Recommendation**: Replace with `zero_mask = (f == 0)` and set `data[zero_mask] = np.nan` for all zero-frequency bins.

---

### 7.3 Fitting Models (`fitting/models.py`)

#### Landau model

**Severity**: **Low** (Documentation)

**Finding**: The implementation uses the Moyal distribution approximation: \(f(x) = A \exp(-0.5(\lambda + e^{-\lambda}))\) with \(\lambda = (x-\mu)/\sigma\). This is a valid Landau approximation, but the docstring says "Landau distribution" without clarifying that it is the Moyal form. The peak at \(\lambda=0\) is \(A e^{-0.5} \approx 0.607A\), so the parameter \(A\) is not the peak height.

**Source**: `gwexpy/fitting/models.py` L75–82

**Recommendation**: Document that this is the Moyal approximation and clarify the relationship between \(A\) and the peak amplitude.

---

#### Power-law model at x=0

**Severity**: **Low** (User responsibility)

**Finding**: \(f(x) = A x^\alpha\) yields `inf` for \(\alpha < 0\) and \(x = 0\), and `0` for \(\alpha > 0\). Fitting with data including \(x=0\) can cause numerical issues when \(\alpha < 0\).

**Recommendation**: Document this behavior and suggest excluding \(x=0\) or adding a small offset when fitting power laws.

---

### 7.4 BifrequencyMap Inverse (`frequencyseries/bifrequencymap.py`)

**Severity**: — (Validated OK)

**Finding**: The pseudo-inverse via `np.linalg.pinv` is appropriate for covariance matrices. The axis swap (f1↔f2) is correct for the Moore–Penrose inverse. The unit \(1/\mathrm{unit}\) for covariance (variance) gives \(1/\mathrm{unit}^2\), which is correct for \(\chi^2 = r^T \mathrm{cov}^{-1} r\).

**Source**: `gwexpy/frequencyseries/bifrequencymap.py` L123–163

---

### 7.5 Imputation with max_gap (`signal/preprocessing/imputation.py`)

**Severity**: — (Validated OK)

**Finding**: The max_gap constraint is implemented correctly. After interpolation, gaps larger than `gap_threshold` are identified using `np.diff(valid_times) > gap_threshold - 1e-12`, and interpolated values within those gaps are reverted to NaN. The use of `1e-12` avoids floating-point comparison issues. Edge NaN handling (before first and after last valid point) is consistent.

**Source**: `gwexpy/signal/preprocessing/imputation.py` L230–249

---

### 7.6 DTT Normalization (`signal/normalization.py`)

**Severity**: — (Validated OK)

**Finding**: The DTT ENBW formula \((f_s \cdot N) / (\sum w)^2\) and the Scipy-to-DTT conversion ratio \((\sum w^2 \cdot N) / (\sum w)^2\) are consistent with LIGO/DTT conventions. The comment about power_sum / dc_gain² in the docstring matches the derivation.

**Source**: `gwexpy/signal/normalization.py` L16–131

---

### 7.7 Fast Coherence Engine (`analysis/bruco.py`)

**Severity**: — (Validated OK)

**Finding**: The Welch coherence formula \(|\mathrm{CSD}|^2 / (\mathrm{PSD}_1 \cdot \mathrm{PSD}_2)\) is correct. The density scaling factor \(2/(f_s \cdot \sum w^2)\) is applied consistently to PSD and CSD, so it cancels in the coherence. The segmentation and windowing follow standard Welch practice.

**Source**: `gwexpy/analysis/bruco.py` L76–113

---

### 7.8 Geomagnetic Background (`noise/magnetic.py`)

**Severity**: **Low** (Robustness)

**Finding**: The heuristic `if amplitude_1hz < 1e-9: amplitude_1hz *= 1e12` assumes Tesla input and converts to pT. This can fail if the user passes values in other units (e.g. nT or pT). The default unit `pT/Hz^(1/2)` is documented.

**Recommendation**: Prefer explicit unit handling (e.g. `u.Quantity`) or document the assumed input unit and conversion logic.

---

### 7.9 Hurst Exponent / local_hurst (`timeseries/hurst.py`)

**Severity**: — (Validated OK)

**Finding**: The Hurst calculation delegates to external packages (`hurst`, `hurst_exponent`, `exp_hurst`). The `local_hurst` sliding-window logic, time indexing (`t0_val + mid * dt_val` for centered windows), and TimeSeries construction with irregular `times` are consistent. The MockTS wrapper for segment-level Hurst calls is appropriate.

**Source**: `gwexpy/timeseries/hurst.py` L185–358

---

## Summary Table

| Area                         | Severity | Status                                      |
|-----------------------------|----------|---------------------------------------------|
| k-space transformation      | —        | Validated OK                                |
| Transient step detection    | Medium   | `freq_tolerance` resolution-dependent       |
| Transient FFT amplitude     | —        | Validated OK                                |
| VIF formula                 | —        | Validated OK                                |
| VIF inference               | Medium   | May mismatch spectrogram parameters         |
| Block bootstrap             | —        | Validated OK                                |
| GLS log-likelihood          | —        | Validated OK                                |
| Covariance stability        | Low      | Optional regularization suggested           |
| ARIMA GPS timing            | —        | Validated OK                                |
| ARIMA t0 fallback           | Low      | Add `epoch`/`x0` fallback                    |
| PCA/ICA shape restoration   | High     | Original 3D shape not restored              |
| **Noise peaks (Lorentz/Voigt)** | —     | Validated OK                                |
| **Power-law at f=0**        | Low      | Assumes sorted frequencies                  |
| **ObsPy f=0 handling**      | Medium   | Only checks f[0], misses zeros elsewhere    |
| **Landau/Moyal model**      | Low      | Document Moyal approximation                |
| **BifrequencyMap inverse**  | —        | Validated OK                                |
| **Imputation max_gap**      | —        | Validated OK                                |
| **DTT normalization**       | —        | Validated OK                                |
| **Fast coherence (bruco)**  | —        | Validated OK                                |
| **Geomagnetic background**  | Low      | Implicit unit conversion heuristic          |
| **Hurst / local_hurst**     | —        | Validated OK                                |

---

## References

- Bendat, J. S., & Piersol, A. G., *Random Data: Analysis and Measurement Procedures*.
- Percival, D. B., & Walden, A. T., *Spectral Analysis for Physical Applications*.
- Ingram, A. (2019), "Error formulae for the energy-dependent cross-spectrum".

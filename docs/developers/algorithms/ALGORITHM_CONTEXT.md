# gwexpy Algorithm Validation Context

## Instructions for LLM-based Cross-Check

You are a senior physicist and software auditor.
The following document contains source code and descriptions for custom algorithms implemented in the `gwexpy` package.
Your task is to **audit** these implementations for physical/statistical validity and numerical stability.

**CONSTRAINT:** Provide a validation report only. **DO NOT** attempt to rewrite the code, refactor modules, or generate full file replacements. Limit any code in your recommendations to short, illustrative snippets or mathematical formula corrections.

### Scope of Audit
While the document below highlights the primary custom algorithms, you should treat the entire `gwexpy` codebase (if accessible or referenced) as the scope.
**If you identify other significant custom mathematical, statistical, or physical logic that is not explicitly listed below, you MUST include it in your validation and reporting.**

### Key Verification Areas:
1.  **Physical Fields & k-space**: Validation of $k$ coordinate transformation logic ($2\pi \cdot \text{fftfreq}$) and unit handling in `ScalarField.fft_space`.
2.  **Transient Response Analysis**: Accuracy of automatic step detection in `ResponseFunctionAnalysis` and amplitude preservation in `_fft_transient`.
3.  **Robust Statistics**: Correctness of Variance Inflation Factor (VIF) correction and block bootstrap logic in `bootstrap_spectrogram`.
4.  **Bayesian Fitting**: Correctness of the log-likelihood calculation using GLS covariance matrices in `run_mcmc`.
5.  **Time Series Modeling**: Consistency of mapping ARIMA model results to GPS-aware `TimeSeries` objects.
6.  **Dimensionality Reduction**: Logical consistency of flattening and reconstructing 3D `TimeSeriesMatrix` structures during PCA/ICA.

### Expected Feedback Format:
-   **Severity**: High (Mathematical/Physical error), Medium (Unstable under specific conditions), Low (Optimization/Style suggestion).
-   **Finding**: Clear description of the issue with mathematical/physical reasoning.
-   **Recommendation**: Conceptual fix, formula correction, or citation. (Do not provide full code implementation).

---

# Algorithm Context Details

## 1. Analysis Engines

### 1.1 Fast Coherence Engine (`gwexpy/analysis/bruco.py`)
Optimized Welch coherence calculation for thousands of auxiliary channels.

```python
# gwexpy/analysis/bruco.py (Excerpt)

class FastCoherenceEngine:
    """
    Welch coherence engine that caches target FFT/PSD for reuse.
    """

    def __init__(
        self, target_data: TimeSeries, fftlength: float, overlap: float
    ) -> None:
        self.fftlength = fftlength
        self.overlap = overlap
        self.sample_rate = float(target_data.sample_rate.value)
        self.nperseg = int(round(self.fftlength * self.sample_rate))
        # ... (Validation)
        self.noverlap = int(round(self.overlap * self.sample_rate))

        self._window = np.hanning(self.nperseg).astype(float)
        # Scaling factor to match scipy.signal.welch (scaling='density')
        # - 1/_window_power: Normalizae for window energy loss
        # - 2.0: Compensate for one-sided RFFT energy (excluding DC/Nyquist ideally,
        #        but 2.0 is the standard approximation for density scaling)
        # - 1/sample_rate: Convert to V^2/Hz (Power Spectral Density)
        self._window_power = np.sum(self._window**2)
        self._scale = (
            2.0 / (self.sample_rate * self._window_power)
            if self.sample_rate > 0
            else 1.0
        )

        target_array = np.asarray(target_data.value, dtype=float)
        self._target_len = len(target_array)
        segments = self._segment_array(target_array, self.nperseg, self.noverlap)
        windowed = segments * self._window[None, :]
        self._target_fft = np.fft.rfft(windowed, axis=1)
        self._target_psd = np.mean(np.abs(self._target_fft) ** 2, axis=0) * self._scale
        # ...

    def compute_coherence(self, aux_data: TimeSeries) -> np.ndarray:
        # ... (Resampling and slicing)
        segments = self._segment_array(aux_array, self.nperseg, self.noverlap)
        windowed = segments * self._window[None, :]
        aux_fft = np.fft.rfft(windowed, axis=1)
        aux_psd = np.mean(np.abs(aux_fft) ** 2, axis=0) * self._scale

        csd = np.mean(self._target_fft * np.conj(aux_fft), axis=0) * self._scale
        denom = self._target_psd * aux_psd
        coherence = np.zeros_like(denom, dtype=float)
        valid = denom > 0
        coherence[valid] = (np.abs(csd[valid]) ** 2) / denom[valid]
        return coherence
```

### 1.2 Coupling Function Estimation (`gwexpy/analysis/coupling.py`)
Logic for estimating noise coupling functions and statistical thresholds.

```python
# gwexpy/analysis/coupling.py (Excerpt)

class SigmaThreshold(ThresholdStrategy):
    """
    Checks if P_inj > P_bkg + sigma * std_error.
    Threshold = mean + sigma * (mean / sqrt(n_avg))
    Assumes Gaussian distribution of PSD values (Central Limit Theorem).
    """
    def check(self, psd_inj, psd_bkg, raw_bkg=None, **kwargs):
        n_avg = kwargs.get("n_avg", 1.0)
        factor = 1.0 + (self.sigma / np.sqrt(n_avg))
        return psd_inj.value > (psd_bkg.value * factor)

class PercentileThreshold(ThresholdStrategy):
    """
    Checks if P_inj > factor * Percentile(P_bkg_segments).
    Uses empirical distribution from raw background spectrogram.
    """
    def threshold(self, psd_inj, psd_bkg, raw_bkg=None, **kwargs):
        # ...
        spec = raw_bkg.spectrogram(...)
        p_bkg_values = spec.percentile(self.percentile)
        return p_bkg_values.value * self.factor

def _process_single_target(...):
    # ...
    # Compute CF: sqrt( (P_tgt_inj - P_tgt_bkg) / (P_wit_inj - P_wit_bkg) )
    delta_wit = psd_wit_inj.value - psd_wit_bkg.value
    delta_tgt = psd_tgt_inj.value - psd_tgt_bkg.value
    
    with np.errstate(divide="ignore", invalid="ignore"):
        sq_cf = delta_tgt[valid_mask] / delta_wit[valid_mask]
        cf_values[valid_mask] = np.sqrt(sq_cf)
    # ...
```

### 1.3 Response Function Analysis (`gwexpy/analysis/response.py`)
Automatic step detection and response function estimation for stepped sine injections.

```python
# gwexpy/analysis/response.py (Excerpt)

def detect_step_segments(witness, fftlength=1.0, snr_threshold=10.0, ...):
    """
    Detect time segments where injection frequency is constant using spectrogram tracking.
    """
    # ... (SNR and frequency tolerance checks)
    for i in range(1, len(times)):
        f = peak_freqs_t[i]
        freq_changed = abs(f - current_freq) > freq_tolerance
        # ... (Segment logic)

def compute(self, witness, target, ...):
    # ...
    # CF = sqrt( (P_tgt_inj - P_tgt_bkg) / (P_wit_inj - P_wit_bkg) )
    p_tgt_net = asd_tgt_inj.value[f_idx]**2 - val_tgt_bkg[f_idx]**2
    p_wit_net = asd_wit_inj.value[f_idx]**2 - val_wit_bkg[f_idx]**2
    if p_tgt_net > 0 and p_wit_net > 0:
        cf_vals[i] = np.sqrt(p_tgt_net / p_wit_net)
```

## 2. Fitting & Statistics

### 2.1 Generalized Least Squares (`gwexpy/fitting/gls.py`)
GLS solver handling correlated errors via covariance matrix.

```python
# gwexpy/fitting/gls.py

class GLS:
    """
    Solve beta = (X.T @ W @ X)^-1 @ X.T @ W @ y
    where W = cov_inv.
    """
    def solve(self) -> np.ndarray:
        W = self.cov_inv
        XTW = self.X.T @ W
        # Use np.linalg.solve for better stability than explicit inverse
        beta = np.linalg.solve(XTW @ self.X, XTW @ self.y)
        return beta

class GeneralizedLeastSquares:
    """
    Cost function for Minuit: chi2 = r.T @ cov_inv @ r
    """
    def __call__(self, *args) -> float:
        ym = self.model(self.x, *args)
        r = self.y - ym
        chi2 = float(r @ self.cov_inv @ r)
        return chi2
```

### 2.2 Bootstrap ASD & VIF (`gwexpy/spectral/estimation.py`)
Robust ASD estimation using block bootstrap and VIF correction for overlapped windows.

```python
# gwexpy/spectral/estimation.py (Excerpt)

def calculate_correlation_factor(window, nperseg, noverlap, n_blocks):
    """
    Calculate variance inflation factor (VIF) due to overlap.
    factor = sqrt(1 + 2 * sum_{k=1}^{M-1} (1 - k/M) * abs(rho(kS))^2)
    """
    # ... (Implementation details using window autocorrelation)

def bootstrap_spectrogram(spectrogram, n_boot=1000, block_size=None, ...):
    # ...
    if block_size is not None and block_size > 1:
        # Moving block bootstrap logic
        num_possible_blocks = n_time - block_size + 1
        # ...
    else:
        # Simple bootstrap
        all_indices = np.random.randint(0, n_time, (n_boot, n_time))
    
    # Numba JIT accelerated resampling
    resampled_stats = _bootstrap_resample_jit(data, all_indices, use_median, ignore_nan)
    
    # ... (Percentile calculation)
    err_low = center - p_low
    err_high = p_high - center
    
    # Correction factor application
    # ...
```

## 3. Physical Fields

### 3.1 4D Field Operations (`gwexpy/fields/scalar.py`)
FFT and coordinate transformations for 4D fields (Time x 3D Space).

```python
# gwexpy/fields/scalar.py (Excerpt)

class ScalarField(FieldBase):
    
    def fft_space(self, axes=None, n=None, overwrite=False):
        """
        Compute FFT along spatial axes.
        k = 2pi * fftfreq(n, d=|dx|)
        """
        # ...
        import scipy.fft as sp_fft
        dft = sp_fft.fftn(self.value, s=s, axes=target_axes_int)
        
        for ax_name, ax_int in zip(axes, target_axes_int):
            # ...
            # Angular wavenumber: k = 2π * fftfreq
            k_values = 2 * np.pi * np.fft.fftfreq(npts, d=abs(dx_value))
            if dx_value < 0:
                k_values = -k_values
            # ...
        
        return result

    def ifft_space(self, axes=None, ...):
        # ...
        # dx = 2π / (n * |dk|)
        dx_value = 2 * np.pi / (npts * abs(dk_value))
        # ...
```

## 4. Noise Models

### 4.1 Schumann Resonance (`gwexpy/noise/magnetic.py`)
Physical model of geomagnetic noise.

```python
# gwexpy/noise/magnetic.py

def schumann_resonance(frequencies, modes=None, amplitude_scale=1.0, ...):
    # ...
    total_psd = np.zeros_like(f_arr, dtype=float)
    for f0, Q, A in modes:
        # Summing squares of ASDs (PSD addition for incoherent sources)
        peak_asd_series = lorentzian_line(f0, A * amplitude_scale, Q=Q, ...)
        total_psd += peak_asd_series.value**2
    
    total_asd = np.sqrt(total_psd)
    return FrequencySeries(total_asd, ...)
```

### 4.2 Voigt Profile (`gwexpy/noise/peaks.py`)

```python
# gwexpy/noise/peaks.py

def voigt_line(f0, amplitude, sigma, gamma, frequencies=None, ...):
    # ...
    # Voigt profile using Faddeeva function (scipy.special.wofz)
    z = ((f_vals - f0) + 1j * gamma) / (sigma * np.sqrt(2))
    v = wofz(z).real

    # Normalization to peak amplitude
    z0 = (1j * gamma) / (sigma * np.sqrt(2))
    peak_factor = wofz(z0).real
    
    data = amp_val * (v / peak_factor)
    return FrequencySeries(data, ...)
```

## 5. Signal Processing Utilities

### 5.1 DTT Normalization (`gwexpy/signal/normalization.py`)
Conversion between standard Scipy/Welch normalization and LIGO/DTT diagnostic tool conventions.

```python
# gwexpy/signal/normalization.py

def get_enbw(window, fs, mode="standard"):
    # ...
    if mode == "dtt":
        # DTT definition: (fs / N) * (1 / mean(w)^2)
        # equivalent to: fs * N / sum(w)^2
        return (fs * n) / (sum_w**2)
    # Standard: fs * sum(w^2) / sum(w)^2
    return fs * sum_w2 / (sum_w**2)

def convert_scipy_to_dtt(psd, window, is_one_sided=True):
    # ...
    # Ratio = Scipy_to_DTT_Scale
    # Derived as: (sum(w^2) * N) / (sum(w)^2)
    ratio = (sum_w2 * n) / (sum_w**2)
    return psd * ratio
```

### 5.2 Imputation with Gap Constraints (`gwexpy/signal/preprocessing/imputation.py`)
NaN imputation that prevents bridging large temporal gaps.

```python
# gwexpy/signal/preprocessing/imputation.py

def impute(values, method="interpolate", times=None, max_gap=None, ...):
    # ... (Standard interpolation)
    # Apply max_gap constraint: Revert interpolated values to NaN if they are within a large gap
    valid_indices = np.where(~nans)[0]
    if has_gap_constraint and len(valid_indices) > 1:
        valid_times = x[valid_indices]
        big_gaps = np.where(np.diff(valid_times) > gap_threshold)[0]
        for idx in big_gaps:
            t_start, t_end = valid_times[idx], valid_times[idx+1]
            val[(x > t_start) & (x < t_end)] = np.nan
```

## 6. Advanced Fitting & MCMC

### 6.1 MCMC Integration (`gwexpy/fitting/core.py`)
Integration with `emcee` to perform Bayesian parameter estimation, supporting both standard least squares and GLS.

```python
# gwexpy/fitting/core.py

def log_prob(theta):
    # ...
    # Compute log probability based on error structure
    if cov_inv is not None:
        # GLS: use full covariance structure
        # Handle complex residuals by taking real part of Hermitian form
        val = r.conj() @ cov_inv @ r
        chi2 = float(np.real(val))
    else:
        # Standard: use diagonal errors
        chi2 = float(np.sum(np.abs(r / dy) ** 2))

    return -0.5 * chi2

def run_mcmc(self, n_walkers=32, ...):
    # ...
    # Initialize walkers around Minuit best fit
    pos = p0_float + stds * 1e-1 * np.random.randn(n_walkers, ndim)
    
    self.sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
    self.sampler.run_mcmc(pos, n_steps, progress=progress)
```

## 7. Special Signal Processing

### 7.1 Transient FFT (`gwexpy/timeseries/_spectral_fourier.py`)
Custom FFT implementation designed for transient signal analysis, ensuring precise amplitude preservation and supporting padding strategies for round-trip transformations.

```python
# gwexpy/timeseries/_spectral_fourier.py

def _fft_transient(self, nfft=None, pad_mode="zero", pad_left=0, ...):
    # ... (Padding logic)
    if pad_mode == "zero":
        x = np.pad(x, (pad_left, pad_right), mode="constant")
    # ...

    dft = np.fft.rfft(x, n=target_nfft) / target_nfft
    
    # One-sided spectrum amplitude correction:
    if target_nfft % 2 == 0:
        # Even length: DC=0 and Nyquist=n_freq-1 are unique
        dft[1:-1] *= 2.0
    else:
        # Odd length: only DC=0 is unique (no Nyquist)
        dft[1:] *= 2.0

    # Store metadata for precise IFFT restoration
    fs._gwex_fft_mode = "transient"
    fs._gwex_pad_left = pad_left
    # ...
    return fs
```

### 7.2 Hilbert-Huang Transform (HHT) (`gwexpy/timeseries/_spectral_special.py`)
Implementation of HHT combining Empirical Mode Decomposition (EMD) and Hilbert Spectral Analysis.

```python
# gwexpy/timeseries/_spectral_special.py (Conceptual)

def hht(self, ...):
    # 1. Empirical Mode Decomposition
    imfs = self.emd(...)
    
    # 2. Hilbert Spectral Analysis on each IMF
    for imf in imfs:
        # Extracts instantaneous amplitude (IA) and frequency (IF)
        ia, ip, ifreq = imf.hilbert_analysis(...)
        # ...
    # 3. Binning into Hilbert Spectrum (Spectrogram)
```

## 8. Time Series Modeling

### 8.1 Hurst Exponent (`gwexpy/timeseries/hurst.py`)
Local Hurst exponent estimation using sliding windows and multi-backend support.

```python
# gwexpy/timeseries/hurst.py

def local_hurst(timeseries, window, step=None, ...):
    # ... (Window and step size conversion to samples)
    starts = np.arange(0, N - w_samples + 1, step_samples)
    for i, s in enumerate(starts):
        segment_val = x_full[s:e]
        # ... (NaN handling and H calculation)
        h = hurst(MockTS(segment_val), ...)
        H_vals[i] = h
```

### 8.2 ARIMA/SARIMAX Forecasting (`gwexpy/timeseries/arima.py`)
GPS-aware forecasting using `statsmodels` and `pmdarima`.

```python
# gwexpy/timeseries/arima.py

class ArimaResult:
    def forecast(self, steps, alpha=0.05):
        # ...
        pred_res = self.res.get_forecast(steps=steps)
        # GPS start time of forecast
        forecast_t0 = self.t0 + n_obs * self.dt
        return forecast_ts, {"lower": lower, "upper": upper}
```

## 9. Matrix Analysis & Decomposition

### 9.1 PCA/ICA for TimeSeriesMatrix (`gwexpy/timeseries/decomposition.py`)
Dimensionality reduction preserving metadata and physical units.

```python
# gwexpy/timeseries/decomposition.py

def pca_fit(matrix, whiten=False, scale=None, ...):
    # Flatten features: (features, time)
    X_features = X_proc.value.reshape(-1, X_proc.shape[-1])
    X_sklearn = X_features.T
    # ... (sklearn fit)

def pca_inverse_transform(pca_res, scores_matrix):
    # Transform back and restore 3D structure (channels, cols, time)
    X_rec_val = pca_res.sklearn_model.inverse_transform(scores_sklearn).T
    X_rec_3d = X_rec_val[:, None, :]
    # ... (restore units and channel labels)
```
### 1. Physical Fields & k-space
- **Severity**: Low
  - **Finding**: The k-coordinate transformation uses angular wavenumbers k = 2π ⋅ fftfreq(n, d = |dx|), which is physically correct for converting spatial frequencies to radians per unit length. However, the conditional sign flip (k = -k_values if dx < 0) may introduce inconsistencies in coordinate systems where dx negativity indicates directionality, potentially affecting phase in multi-dimensional fields without corresponding adjustments to the FFT data.
  - **Recommendation**: Ensure the sign flip is only applied if the spatial axis is reversed, and propagate the direction metadata to downstream operations. No formula change needed.

- **Severity**: Medium
  - **Finding**: Unit handling in fft_space and ifft_space appears consistent (e.g., dk reciprocal to dx via 2π / (n ⋅ |dk|)), but numerical stability could degrade near dx ≈ 0 or for very small/large n due to floating-point precision in fftfreq, leading to NaN or infinite k_values.
  - **Recommendation**: Add a guard clause for |dx| < epsilon (e.g., 1e-10), raising a ValueError or defaulting to zero-padding, citing numpy.fft documentation for edge cases.

### 2. Transient Response Analysis
- **Severity**: High
  - **Finding**: In compute, the net power calculation p_tgt_net = asd_tgt_inj.value[f_idx]**2 - val_tgt_bkg[f_idx]**2 assumes val_tgt_bkg is the ASD (amplitude spectral density) of the background, yielding PSD_inj - PSD_bkg. However, naming inconsistency (asd for injection vs. val for background) risks misinterpretation if val_tgt_bkg is actually PSD, leading to incorrect subtraction (PSD_inj - PSD_bkg**2) and non-physical negative or inflated coupling factors.
  - **Recommendation**: Clarify variable types in code comments and ensure consistent use of ASD or PSD; correct formula should be p_net = PSD_inj - PSD_bkg or equivalently ASD_inj² - ASD_bkg².

- **Severity**: Medium
  - **Finding**: Automatic step detection in detect_step_segments relies on SNR threshold and frequency tolerance, which is statistically valid under high-SNR assumptions but unstable for low-SNR transients or noisy data, potentially missing steps or fragmenting segments due to peak_freqs_t variability.
  - **Recommendation**: Incorporate adaptive thresholding, e.g., using Otsu's method on SNR distribution, or cite "Spectrogram-based frequency tracking" from signal processing literature for robustness.

- **Severity**: Low
  - **Finding**: Amplitude preservation in _fft_transient via dft /= nfft followed by *2 (excluding DC/Nyquist for even lengths) is correct for recovering sine/cosine amplitudes in one-sided spectra, but the approximation ignores exact DC/Nyquist scaling, causing minor errors (<1%) for low-frequency transients.
  - **Recommendation**: For precision, apply *2 only to indices 1:-1 for even nfft, but note it's negligible for nfft >> 1 per Parseval's theorem.

### 3. Robust Statistics
- **Severity**: High
  - **Finding**: The PSD scaling in FastCoherenceEngine (_scale = 2.0 / (sample_rate * _window_power)) misses the nperseg factor, resulting in PSD underestimated by 1/nperseg compared to standard Welch (e.g., scipy.signal.welch with 'density' scaling, which uses 2 * nperseg / (fs * sum(w²))). This leads to physically incorrect power densities, though coherence ratios cancel the error.
  - **Recommendation**: Correct the scale to _scale = 2.0 * self.nperseg / (self.sample_rate * self._window_power), aligning with Welch's variance-preserving normalization.

- **Severity**: Medium
  - **Finding**: VIF calculation in calculate_correlation_factor uses abs(rho(kS))², which is correct for variance inflation in overlapped segments (Bartlett's formula), but numerical stability falters for small n_blocks or high overlap, where autocorrelation rho may amplify floating-point errors in sum.
  - **Recommendation**: Cap the sum at a maximum lag M < n_blocks/2 to avoid noise amplification, citing Priestley’s "Spectral Analysis and Time Series" for overlap corrections.

- **Severity**: Low
  - **Finding**: Block bootstrap logic in bootstrap_spectrogram correctly handles dependence via moving blocks, but non-circular implementation may bias edge effects in short time series (n_time ≈ block_size), underestimating variance.
  - **Recommendation**: Optionally implement circular blocks by wrapping indices modulo n_time for short series.

### 4. Bayesian Fitting
- **Severity**: Medium
  - **Finding**: Log-likelihood in log_prob uses -0.5 * chi2 with chi2 = Re(r.conj() @ cov_inv @ r), which is statistically valid for complex Gaussian errors assuming cov_inv is Hermitian positive-definite. However, without checks for positive-definiteness, numerical instability (e.g., negative chi2) can occur if cov_inv has small negative eigenvalues due to inversion errors.
  - **Recommendation**: Add a regularization term, e.g., cov_inv += epsilon * eye, with epsilon = 1e-10 * trace(cov_inv)/n, to ensure stability.

- **Severity**: Low
  - **Finding**: Walker initialization in run_mcmc (pos = p0 + 0.1 * stds * randn) is reasonable but may lead to poor mixing if Minuit stds underestimate uncertainties, especially in high-dimensional parameter spaces.
  - **Recommendation**: Increase perturbation to 1.0 * stds for broader exploration, citing Gelman et al. "Bayesian Data Analysis" for initialization strategies.

### 5. Time Series Modeling
- **Severity**: Low
  - **Finding**: Mapping ARIMA forecasts to GPS-aware TimeSeries in ArimaResult.forecast uses forecast_t0 = self.t0 + n_obs * self.dt, which is consistent for uniform sampling but assumes dt is constant; variable sampling (common in GPS-timed data) could misalign timestamps.
  - **Recommendation**: Use cumulative sum of dt if variable, or store explicit time array for mapping.

### 6. Dimensionality Reduction
- **Severity**: High
  - **Finding**: Flattening in pca_fit reshapes 3D TimeSeriesMatrix (channels, cols, time) to (channels * cols, time).T = (time, features), which is logically correct for time-as-samples PCA. However, reconstruction in pca_inverse_transform uses X_rec_3d = X_rec_val[:, None, :], inserting a singleton dimension (features, 1, time), which only restores the original shape if cols=1; otherwise, it fails to unflatten properly, leading to dimensional mismatch.
  - **Recommendation**: Reshape using original dimensions, e.g., X_rec_3d = X_rec_val.reshape(channels, cols, time), storing shape metadata in pca_res.

- **Severity**: Low
  - **Finding**: PCA/ICA handles units and labels preservation, but if time < features (common in wide matrices), the effective components are limited to time-1, potentially unstable without SVD solver specification.
  - **Recommendation**: Set sklearn PCA solver='full' for small datasets to avoid ARPACK instability.


  ### Additional Custom Algorithms Validation

Based on the provided document, the following sections identify and audit other important custom algorithms beyond the initially specified key verification areas. These include coherence computation, coupling thresholds, noise modeling, signal normalization, imputation logic, MCMC probability, special transforms, and Hurst estimation. Audits focus on physical/statistical validity and numerical stability, adhering to the constraints.

### 1. Fast Coherence Engine
- **Severity**: Medium
  - **Finding**: The coherence calculation uses csd = mean(target_fft * conj(aux_fft)) * _scale, followed by coherence = |csd|² / (target_psd * aux_psd), which is statistically correct for Welch's method under ergodicity assumptions. However, for short segments or low overlap, the mean approximation may introduce bias in low-SNR regimes, leading to coherence >1 due to finite averaging, violating the [0,1] bound physically.
  - **Recommendation**: Clamp coherence to [0,1] post-computation to prevent non-physical values, citing Bendat & Piersol "Random Data: Analysis and Measurement Procedures" for bias corrections in finite samples.

- **Severity**: Low
  - **Finding**: Window power normalization (_window_power = sum(window**2)) and scale (2.0 / (fs * _window_power)) match standard PSD density scaling, but the 2.0 factor assumes one-sided RFFT without exact DC/Nyquist handling, causing slight overestimation for band-limited signals.
  - **Recommendation**: Use scipy.signal.welch as a reference for validation tests, adjusting for exact scaling if DC is significant.

### 2. Coupling Function Estimation
- **Severity**: High
  - **Finding**: In _process_single_target, the coupling factor cf = sqrt( (P_tgt_inj - P_tgt_bkg) / (P_wit_inj - P_wit_bkg) ) assumes positive deltas, but noise fluctuations can yield negative values, leading to NaN in sqrt or imaginary cf, which is physically invalid as coupling should be non-negative real.
  - **Recommendation**: Add a mask for delta_tgt > 0 and delta_wit > 0 before sqrt, setting cf=0 otherwise; formula correction: cf = sqrt( max(0, delta_tgt) / max(1e-12, delta_wit) ) to avoid division by zero.

- **Severity**: Medium
  - **Finding**: SigmaThreshold uses threshold = psd_bkg * (1 + sigma / sqrt(n_avg)), approximating std_error ≈ mean / sqrt(n_avg) via CLT for Gaussian PSD, but PSD follows chi-squared distribution (exponential for single avg), so variance is mean², leading to underestimated thresholds for small n_avg.
  - **Recommendation**: Use chi-squared quantiles for exact thresholds, e.g., threshold = psd_bkg * scipy.stats.chi2.ppf(1 - alpha, 2*n_avg) / (2*n_avg), citing Kay "Fundamentals of Statistical Signal Processing".

- **Severity**: Low
  - **Finding**: PercentileThreshold leverages empirical distribution from spectrogram percentiles, which is robust to non-Gaussianity, but computation via spec.percentile may be inefficient for large raw_bkg, risking memory issues.
  - **Recommendation**: Compute percentiles on-the-fly using numpy.percentile along time axis for scalability.

### 3. Noise Models
- **Severity**: Medium
  - **Finding**: Schumann Resonance sums PSDs as total_psd += (lorentzian_asd)**2, correct for incoherent mode addition, but assumes independent sources; correlated modes (realistic in geomagnetism) could require cross-terms, leading to under/overestimation of total ASD.
  - **Recommendation**: Note in docs that model assumes incoherence; for accuracy, cite Chapman & Bartels "Geomagnetism" and consider phase randomization if needed.

- **Severity**: High
  - **Finding**: Voigt profile normalization uses peak_factor = wofz(z0).real, where z0 = (1j * gamma) / (sigma * sqrt(2)), but this normalizes to unit peak only for gamma >> sigma (Lorentzian limit); for general cases, the true peak is 1 / (sigma * sqrt(2*pi)) * exp(-something), causing amplitude mismatch.
  - **Recommendation**: Correct normalization to data = amplitude * (v / (sigma * sqrt(2*pi))), as Voigt is convolution of Gaussian and Lorentzian, preserving integral but not peak directly; cite Olivero & Longbothum for analytic approximations.

### 4. Signal Processing Utilities
- **Severity**: Medium
  - **Finding**: DTT Normalization convert_scipy_to_dtt uses ratio = (sum_w2 * n) / (sum_w**2), derived correctly for ENBW differences, but assumes 'density' scaling; for 'spectrum' mode or unscaled FFT, it mismatches LIGO DTT conventions, potentially scaling PSD by wrong factors in diagnostic tools.
  - **Recommendation**: Add mode check and conditional scaling; formula: for density, ratio as is; cite LIGO technical notes on DTT PSD conventions.

- **Severity**: Low
  - **Finding**: Imputation with max_gap reverts interpolated values to NaN in large gaps, preventing artificial bridging, which is valid for preserving data integrity in gapped time series like GW data, but abrupt NaN transitions may cause issues in downstream FFT (ringing artifacts).
  - **Recommendation**: Suggest tapering edges of gaps with a window function before imputation.

### 5. Advanced Fitting & MCMC
- **Severity**: Medium
  - **Finding**: GLS solver uses np.linalg.solve(XTW @ X, XTW @ y), stable for well-conditioned matrices, but without condition number check, ill-conditioned X (multicollinearity) can amplify numerical errors in beta.
  - **Recommendation**: Compute cond = np.linalg.cond(XTW @ X); if >1e10, suggest ridge regularization.

### 6. Special Signal Processing
- **Severity**: Low
  - **Finding**: Hilbert-Huang Transform combines EMD and Hilbert analysis, conceptually valid for non-stationary signals, but EMD sifting can be numerically unstable for noisy data, leading to mode mixing without end-effect handling.
  - **Recommendation**: Recommend mirror extension for boundaries in EMD, citing Huang et al. original HHT paper.

### 7. Time Series Modeling
- **Severity**: Medium
  - **Finding**: Local Hurst exponent uses sliding windows with hurst(segment), statistically valid for fractal processes, but assumes stationarity within windows; for varying H (multifractal), fixed window size may bias estimates.
  - **Recommendation**: Adapt window size based on data scale, or use wavelet-based methods for robustness, citing Mandelbrot's fractional Brownian motion works.
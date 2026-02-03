# Validated Algorithms

The following algorithms have been validated through cross-verification
by 12 different AI models (2026-02-01). This page documents the verified
implementations with references.

## k-space Computation

**Function**: {meth}`gwexpy.fields.ScalarField.fft_space`

**Consensus**: 10/12 AI models confirmed correctness

The angular wavenumber computation uses the standard physics definition:

$$
k = 2\pi \cdot \text{fftfreq}(n, d)
$$

This satisfies $k = 2\pi / \lambda$ and is consistent with:

- Press et al., *Numerical Recipes* (3rd ed., 2007), §12.3.2
- NumPy `fftfreq` documentation
- GWpy FrequencySeries (Duncan Macleod et al., SoftwareX 13, 2021)

The `2π` factor is correctly applied, and units are properly set as
`1/dx_unit` (rad/length).


## Amplitude Spectrum (Transient FFT)

**Function**: `TimeSeries._fft_transient`

**Consensus**: Validated with clear rebuttal of incorrect critiques

The transient FFT returns an **amplitude spectrum**, not a density spectrum:

$$
\text{amplitude} = \text{rfft}(x) / N
$$

with one-sided values (excluding DC and Nyquist) doubled.

This convention allows direct reading of sinusoidal peak amplitudes.
The suggestion to multiply by `dt` applies to density spectra
(V/√Hz), which is a different use case.

**References**:

- Oppenheim & Schafer, *Discrete-Time Signal Processing* (3rd ed., 2010), §8.6.2
- SciPy `rfft` documentation


## VIF (Variance Inflation Factor)

**Function**: {func}`gwexpy.spectral.estimation.calculate_correlation_factor`

**Consensus**: 8/12 AI models confirmed correctness

The VIF formula follows Percival & Walden (1993):

$$
\text{VIF} = \sqrt{1 + 2 \sum_{k=1}^{M-1} \left(1 - \frac{k}{M}\right) |\rho(kS)|^2}
$$

**Important**: This is NOT the regression VIF (1/(1-R²)) used for
multicollinearity diagnosis. The name collision caused confusion,
but the implementation is correct for spectral analysis.

**References**:

- Percival, D.B. & Walden, A.T., *Spectral Analysis for Physical Applications*
  (1993), Ch. 7.3.2, Eq.(56)
- Bendat, J.S. & Piersol, A.G., *Random Data* (4th ed., 2010)


## Forecast Timestamp (ARIMA)

**Function**: {meth}`gwexpy.timeseries.arima.ArimaResult.forecast`

**Consensus**: Validated with rebuttal of incorrect concerns

The forecast start time is computed as:

$$
t_{\text{forecast}} = t_0 + n_{\text{obs}} \times \Delta t
$$

This assumes equally-spaced data without gaps. GPS times follow the
LIGO/GWpy convention using TAI continuous seconds.

The leap-second concern raised by some models does not apply to
GPS/TAI time systems used in gravitational wave data analysis.

**References**:

- GWpy TimeSeries.epoch documentation
- LIGO GPS time convention (LIGO-T980044)
- Box, G.E.P. & Jenkins, G.M., *Time Series Analysis* (1976), Ch. 4


# Assumptions and Conventions

The following sections document important assumptions and conventions
that users should be aware of when using these algorithms.


## MCMC Fixed Covariance Assumption

**Function**: {meth}`gwexpy.fitting.FitResult.run_mcmc`

**Assumption**: The covariance matrix Σ is **parameter-independent (fixed)**.

Under this assumption, the log determinant term `log|Σ|` is constant
and can be correctly omitted from the log likelihood:

$$
\log p(y|\theta) = -\frac{1}{2} r^T \Sigma^{-1} r + \text{const}
$$

If Σ depends on model parameters θ, the full log likelihood including
`log|Σ|` must be used, which requires modifying the implementation.

**Complex Data**: For complex-valued data, the Hermitian form
`r.conj() @ cov_inv @ r` assumes **circular complex Gaussian**
distribution (equal variance for real and imaginary parts).

**References**:

- Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006), Ch. 2.2
- Gelman et al., *Bayesian Data Analysis* (3rd ed., 2013), §14.2


## Angular vs Cycle Wavenumber

**Function**: {meth}`gwexpy.fields.ScalarField.fft_space`

**Convention**: The `k` axis returns **angular wavenumber** [rad/length],
NOT cycle wavenumber [1/length].

- Angular wavenumber: $k = 2\pi / \lambda$
- Cycle wavenumber: $\nu = 1 / \lambda$
- Conversion: $\nu = k / (2\pi)$


## Sign Convention for Descending Axes

**Function**: {meth}`gwexpy.fields.ScalarField.fft_space`

**Convention**: If the spatial axis is descending (dx < 0), the k-axis is
**sign-flipped** to preserve physical consistency with the phase factor
convention $e^{+ikx}$.

This ensures that positive `k` corresponds to waves propagating in
the positive x direction, regardless of the data storage order.

**Reference**: Jackson, *Classical Electrodynamics* (3rd ed., 1998), §4.2


## VIF Application in Bootstrap

**Function**: {func}`gwexpy.spectral.estimation.bootstrap_spectrogram`

**Conditional Behavior**:

- **Standard bootstrap** (`block_size=None`): VIF correction is applied
  to account for Welch overlap correlation.
- **Block bootstrap** (`block_size` specified): VIF correction is
  **disabled** (factor=1.0) to prevent double correction.

This prevents over-inflation of variance estimates when both block
structure and analytical VIF would be applied simultaneously.


## Bootstrap Stationarity Assumption

**Function**: {func}`gwexpy.spectral.estimation.bootstrap_spectrogram`

**Assumption**: The input data is a **stationary process**.

Moving Block Bootstrap assumes the statistical properties of the data
do not change over time. For non-stationary data (glitches, transients,
drifting noise floors), the confidence intervals may be biased.

**Recommended** `block_size`: At least the stride length (segment spacing)
for time-correlated spectrograms.

**References**:

- Künsch, H.R., "The jackknife and the bootstrap for general stationary
  observations", Ann. Statist. 17(3), 1989
- Politis, D.N. & Romano, J.P., "The stationary bootstrap",
  J. Amer. Statist. Assoc. 89(428), 1994


## About the Validation

These validations were performed as part of a comprehensive algorithm
audit using 12 different AI models:

- ChatGPT 5.2 Pro (Deep Research)
- Claude Opus 4.5 (Antigravity, IDE)
- Copilot (IDE)
- Cursor
- Felo
- Gemini 3 Pro (Antigravity, CLI, Web)
- Grok
- NotebookLM
- Perplexity

The full validation report is available in the developer documentation.

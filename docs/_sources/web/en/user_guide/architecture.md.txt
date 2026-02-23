# Architecture & Physics

This section details the custom algorithms, physical models, and mathematical logic implemented in `gwexpy`. It serves as a reference for advanced users and developers who wish to understand the underlying mechanics of the package's analytical tools.

## Physical Fields & Fourier Space

### 4D Field Operations

`gwexpy` provides robust handling of multi-dimensional data, specifically 4D fields representing time and 3D spatial dimensions (e.g., `ScalarField`). A core capability is the seamless transformation between the spatial domain and the wavenumber ($k$) domain (reciprocal space).

The `fft_space` and `ifft_space` methods handle these transformations using multi-dimensional Fast Fourier Transforms (FFT). Crucially, the package correctly manages the coordinate transformations, ensuring that physical units and grid spacing are accurately translated into the $k$-space.

```python
# The angular wavenumber k is computed based on the spatial resolution dx:
k_values = 2 * np.pi * np.fft.fftfreq(npts, d=abs(dx_value))
```

This guarantees that applications relying on physical wavelengths or spatial frequencies receive correctly scaled dimensional outputs.

## Spectral Analysis and Statistics

### Robust ASD Estimation (Bootstrap Spectrogram)

Standard Amplitude Spectral Density (ASD) estimation can be sensitive to non-stationary noise. `gwexpy` implements a robust estimation method via the `bootstrap_spectrogram` function.

This method uses bootstrap resampling (including Moving Block Bootstrap for time-correlated data) across the time axis of a spectrogram to estimate the median or mean ASD along with confidence intervals.

**Variance Inflation Factor (VIF)**
When calculating statistics from overlapping data segments (like in Welch's method), the segments are correlated, artificially reducing the apparent variance. `gwexpy` correctly applies a Variance Inflation Factor (VIF) based on the overlapping window's autocorrelation function.

```math
\text{factor} = \sqrt{1 + 2 \sum_{k=1}^{M-1} (1 - k/M) |\rho_{\text{window}}(k \Delta)|^2}
```

This ensures that the confidence intervals returned by the bootstrap method accurately reflect the true statistical uncertainty of the data.

### Transient Signal FFT

For analyzing transient signals (like injections or physical impulses), standard windowing techniques often distort the signal's true amplitude. `gwexpy` provides `_fft_transient`, an FFT method optimized for transients. It uses predictable padding strategies to maintain the temporal structure and accurately preserves the amplitude in the one-sided spectrum.

### Fast Coherence Engine (BruCo)

The `FastCoherenceEngine` calculates Welch coherence between a single target channel and potentially thousands of auxiliary channels efficiently. By pre-calculating and caching the target signal's FFT and PSD, it avoids redundant computations, significantly speeding up large-scale noise audits (like the "Brute-force Coherence" tool).

## Transient Response and System Analysis

### Response Function Analysis

The framework includes tools for analyzing a system's response to injected signals (e.g., stepped sine excitations). The `detect_step_segments` function automatically tracks the injection frequency in a spectrogram to identify stable segments of excitation.

The coupling function (transfer function magnitude) is then estimated by comparing the signal power during the injection to the background power, ensuring that background noise does not artificially inflate the transfer function estimate.

```math
\text{CF}(f) = \sqrt{\frac{P_{\text{tgt,inj}}(f) - P_{\text{tgt,bkg}}(f)}{P_{\text{wit,inj}}(f) - P_{\text{wit,bkg}}(f)}}
```

## Bayesian Fitting & Modeling

### Generalized Least Squares (GLS) and MCMC

When fitting models to data with correlated errors (where the noise covariance matrix is known), `gwexpy` supports Generalized Least Squares (GLS) solving. The fitting core integrates natively with the `emcee` package to perform Markov Chain Monte Carlo (MCMC) parameter estimation.

For GLS, the log-likelihood leverages the full inverse covariance matrix ($\Sigma^{-1}$), accommodating heteroscedastic and correlated errors.

```math
\log p(y|\theta) \propto - \frac{1}{2} r^T \Sigma^{-1} r
```

For complex-valued data (common in transfer functions), the framework utilizes the Hermitian form $r^\dagger \Sigma^{-1} r$ to correctly process both magnitude and phase residuals.

### Dimensionality Reduction (PCA/ICA)

Principal Component Analysis (PCA) and Independent Component Analysis (ICA) can be applied directly to multi-channel data representations like `TimeSeriesMatrix`. `gwexpy` manages the underlying dimensionality changes (flattening the 3D structure (Channels $\times$ Columns $\times$ Time) into a 2D feature matrix for scikit-learn, and reconstructing it back) while preserving essential metadata like GPS timestamps and physical units.

## Noise Models

`gwexpy` includes built-in generators for specific physical noise models, useful for simulation and testing.

- **Schumann Resonance**: Simulates the background geomagnetic noise using a sum of independent Lorentzian profiles corresponding to the Earth-ionosphere cavity modes.
- **Voigt Profile**: Generates spectral peaks combining Gaussian (Doppler broadening) and Lorentzian (collision broadening) characteristics using the Faddeeva function, common in atomic physics and high-Q mechanical resonances.

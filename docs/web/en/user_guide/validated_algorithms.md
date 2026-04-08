# Validated Algorithms

The numerical algorithms implemented in `gwexpy` have undergone a rigorous validation process to ensure scientific accuracy and reliability. This page lists the validated implementations and their physical/mathematical baselines.

## Validation Criteria

An algorithm labeled as "Validated" meets at least one of the following criteria:

1.  **Literature Cross-Check**: The formula corresponds 1-to-1 with established textbooks or peer-reviewed papers.
2.  **Cross-Library Verification**: Computational results match those of standard libraries (GWpy, SciPy, LALSuite) under identical conditions.
3.  **Numerical Experimentation**: Convergence to analytical solutions for known test data within theoretical error bounds.
4.  **Independent Peer Review**: Code and logic have been audited by experts other than the core developers.

---

## List of Validated Algorithms

| Algorithm | Target API | Validation Method | Usage | Related Tutorial |
| :--- | :--- | :--- | :--- | :--- |
| **k-space calculation** | `.fft_space()` | Literature / SciPy | Spatial correlation, wavenumbers | [Field Intro](tutorials/field_scalar_intro.ipynb) |
| **Transient FFT** | `._fft_transient()` | Literature / NumPy | Short-duration burst analysis | — |
| **VIF Correction** | `calculate_correlation_factor()` | Literature (Percival) | Spectral error estimation | [Bootstrap Guide](tutorials/spectral_bootstrap.ipynb) |
| **Forecast Timing** | `ArimaResult.forecast()` | LIGO Timing Convention | Trend prediction | — |
| **Adaptive Whitening** | `.whiten(eps="auto")` | Numerical Experiment | Robust signal extraction | [Stability](numerical_stability.md) |

---

## Algorithm Details and Physical Basis

### 1. k-space Calculation
**Target**: {meth}`gwexpy.fields.ScalarField.fft_space`

Angular wavenumber calculation follows the standard physics definition $k = 2\pi / \lambda$.

$$
k = 2\pi \cdot \text{fftfreq}(n, d)
$$

**Basis**:
- Press et al., *Numerical Recipes* (3rd ed., 2007), §12.3.2
- Verified against NumPy `fftfreq` documentation.

---

### 2. Transient FFT (Amplitude Spectrum)
**Target**: `TimeSeries._fft_transient`

Uses an amplitude-preserving convention rather than density, allowing direct reading of peak amplitudes for pulsed signals.

$$
\text{amplitude} = \text{rfft}(x) / N
$$

**Basis**:
- Oppenheim & Schafer, *Discrete-Time Signal Processing* (3rd ed., 2010), §8.6.2
- Verified with numerical experiments on sine waves with known amplitudes.

---

### 3. VIF (Variance Inflation Factor)
**Target**: {func}`gwexpy.spectral.estimation.calculate_correlation_factor`

Corrects for the reduction in effective sample size caused by overlapping segments in methods like Welch's PSD.

$$
\text{VIF} = \sqrt{1 + 2 \sum_{k=1}^{M-1} \left(1 - \frac{k}{M}\right) |\rho(kS)|^2}
$$

**Basis**:
- Percival, D.B. & Walden, A.T., *Spectral Analysis for Physical Applications* (1993), Eq.(56)
- **Note**: This is distinct from the VIF ($1/(1-R^2)$) used in multicollinearity diagnostics.

---

### 4. Forecast Timestamp (ARIMA)
**Target**: {meth}`gwexpy.timeseries.arima.ArimaResult.forecast`

Calculates future timestamps while maintaining GPS continuity according to LIGO conventions.

**Basis**:
- LIGO GPS Time Convention (LIGO-T980044)
- Verified consistency as continuous TAI seconds (excluding leap seconds).

---

## Assumptions and Constraints

Users should be aware of the following statistical and physical assumptions:

- **MCMC Fixed Covariance**: `run_mcmc` assumes the noise covariance $\Sigma$ is independent of the parameters.
- **Stationarity for Bootstrap**: `bootstrap_spectrogram` assumes the input data satisfies the stationary process assumption.
- **Angular Wavenumber Convention**: `fft_space` returns angular wavenumbers [rad/length]. Conversion to cycles [1/length] requires division by $2\pi$.

## Audit Trail

Detailed unit test results, literature comparison scripts, and past review logs are maintained in the `docs_internal/verification/` directory.

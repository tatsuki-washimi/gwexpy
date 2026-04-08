# Validated Algorithms

> [!NOTE]
> **Who should read this page?**
> - Researchers who want to verify the mathematical and physical validity of the methods.
> - Developers who need to know the numerical tolerances between GWexpy and external libraries (SciPy, LALSuite, etc.).
> - Users who need to check the data characteristics (stationarity, Gaussianity, etc.) assumed by each algorithm.

The numerical algorithms implemented in `gwexpy` have undergone a rigorous validation process to ensure scientific accuracy and reliability.

## Validation Criteria and Numerical Precision

Algorithms labeled as "Validated" meet the following standards and achieve specific precision (tolerance) benchmarks:

- **Tolerance**: We generally use $10^{-12}$ (relative error) as the passing criterion. This indicates no significant logic differences beyond standard double-precision rounding errors.
- **Invariance**: All algorithms are verified for "Scale Invariance," ensuring results remain consistent even after scaling data (e.g., multiplying by 1000).

## Validated Algorithms Summary Table

| Algorithm | Target API | External Baseline | Numerical Tol | Physical Assumptions | Related Tutorial |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **k-space calculation** | `.fft_space()` | SciPy / FFTW | $10^{-15}$ | Uniform Grid | [Field Intro](tutorials/field_scalar_intro.ipynb) |
| **Transient FFT** | `._fft_transient()` | NumPy | $10^{-15}$ | No window (Rect) | — |
| **VIF Correction** | `calculate_correlation_factor()` | Literature (Percival)| — | Stationary Process | [Bootstrap Guide](tutorials/case_bootstrap_gls_fitting.ipynb) |
| **Forecast Timing** | `ArimaResult.forecast()` | LIGO Convention | $10^{-9}$ (s) | No leap seconds (GPS) | — |
| **Adaptive Whitening** | `.whiten(eps="auto")` | Numerical Exp | $10^{-12}$ | Locally Quasi-stationary | [Stability](numerical_stability.md) |

---

## Detailed Algorithm Basis and Assumptions

### 1. k-space Calculation
**Target**: {meth}`gwexpy.fields.ScalarField.fft_space`

Angular wavenumber calculation follows the standard physics definition $k = 2\pi / \lambda$.

**Assumptions**:
- Spatial coordinates ($x, y$) must be **Uniformly Spaced**.
- For non-uniform grids, a correct wavenumber axis cannot be obtained without prior interpolation.

**Reference**: Press et al., *Numerical Recipes* (3rd ed., 2007), §12.3.2

---

### 2. Transient FFT (Amplitude Spectrum)
**Target**: `TimeSeries._fft_transient`

Uses an amplitude-preserving convention rather than density, allowing direct reading of time-domain peak amplitudes.

**Assumptions**:
- No window function is applied to the input signal (assumes Rectangular window).
- For data with windows already applied, separate correction for coherent gain is required.

---

### 3. VIF (Variance Inflation Factor)
**Target**: {func}`gwexpy.spectral.estimation.calculate_correlation_factor`

Corrects for the reduction in effective sample size caused by overlapping segments in spectral estimation methods like Welch's.

**Assumptions**:
- The data must be **Weakly Stationary**.
- VIF may under- or over-estimate variance if non-stationary glitches or step responses are present.

**Reference**: Percival, D.B. & Walden, A.T., *Spectral Analysis for Physical Applications* (1993), Eq.(56)

---

### 4. Forecast Timestamp (ARIMA)
**Target**: {meth}`gwexpy.timeseries.arima.ArimaResult.forecast`

Maintains GPS continuity according to LIGO conventions for future timestamp prediction.

**Assumptions**:
- The time system must be **GPS Time (no leap seconds)**.
- Using this in time systems with leap seconds (like UTC) will result in a 1-second offset in future predictions.

---

## Audit Trail

Detailed unit test results, literature comparison scripts, and past review logs are maintained in the `docs_internal/verification/` directory.
All changes are merged only after passing the `verify-physics` gate in the CI environment.

## Related Documents

- {doc}`numerical_stability` - Numerical Stability (Precision Management)
- {doc}`glossary` - Glossary of Algorithms
- {doc}`../reference/api/signal` - Signal Processing API Reference

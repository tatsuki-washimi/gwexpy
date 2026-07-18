---
myst:
  html_meta:
    description: "Review GWexpy validated algorithms, numerical tolerances, assumptions, audit evidence, and bridges to theory, API, and tutorials."
---

# Validated Algorithms

:::{note}
**Who should read this page?**
- Researchers who want to verify the mathematical and physical validity of the methods.
- Developers who need to know the numerical tolerances between GWexpy and external libraries (SciPy, LALSuite, etc.).
- Users who need to check the data characteristics (stationarity, Gaussianity, etc.) assumed by each algorithm.
:::

:::{important}
**Advanced validation/theory companion**

This page intentionally stays under the user guide so readers can discover it from feature docs, but it is not an onboarding page. Treat it as the advanced/theory landing for audit-oriented reading across the API reference, theory notes, and targeted tutorials. For first-use guidance, start with [Getting Started](../tutorials/getting_started.md), [Prerequisites and Conventions](prerequisites_and_conventions.md), or the linked tutorials for each method.
:::

The algorithms listed on this page have public implementation tests, theory references, or audit records. Those evidence types are distinct and their scope is stated for each algorithm below.

(validated-algorithms-validation-criteria)=
## Validation Criteria and Numerical Precision

The algorithms listed on this page are assessed against the evidence recorded in their own sections.

- **Tolerance**: Numerical tolerance is algorithm- and test-specific. The relevant test or audit entry states the comparison and acceptance condition where one is available.
- **Invariance**: Scale-invariance checks apply only to algorithms whose evidence explicitly records such a check; they are not a package-wide guarantee.

(validated-en-objective-evidence)=
## Objective Evidence

The summaries on this page are backed by the public evidence trail collected here.

The evidence table is intentionally column-dense so the audit trail stays on one page. On mobile, horizontal scrolling is expected and should align with the shared table CSS used across the docs.

| Evidence type | Public source | What it shows |
| :--- | :--- | :--- |
| Audit scope | {ref}`Validation summary <validated-algorithms-summary-table>` | Which algorithms were reviewed and which numerical questions were checked |
| Consolidated findings | {ref}`Per-algorithm sections <validated-algorithms-detail-sections>` | Cross-reviewed findings, severity, and agreement across independent audits |
| Fix history | {ref}`Audit Trail <validated-algorithms-audit-trail>` | What changed after the review findings were merged and addressed |
| Field-specific physics review | {ref}`k-space calculation <validated-en-k-space>` | Physics and axis-consistency review for `ScalarField` FFT behavior |

(validated-en-source-references)=
## Source References

Use the source keys below in the per-algorithm sections so the bibliography and source trail stay centralized on this page.

| Key | Source | Used for |
| :--- | :--- | :--- |
| `S1` | Press et al., *Numerical Recipes* (3rd ed., 2007), §12.3.2 | k-space calculation and FFT-axis interpretation |
| `S2` | Percival, D.B. & Walden, A.T., *Spectral Analysis for Physical Applications* (1993), Eq.(56) | Welch overlap / VIF correction |
| `S3` | [Prerequisites and Conventions](prerequisites_and_conventions.md) | Time-system assumptions and FFT-related conventions referenced by validation notes |
| `S4` | [Numerical Stability](numerical_stability.md) | Adaptive stabilization guidance for whitening and related preprocessing paths |
| `S5` | {ref}`Objective Evidence <validated-en-objective-evidence>` and the Audit Trail below | Repository-backed implementation evidence for transient FFT behavior and complex GLS / MCMC handling |

(validated-algorithms-summary-table)=
## Validated Algorithms Summary Table

| Algorithm | Primary API | API page | Evidence | Related tutorial |
| :--- | :--- | :--- | :--- | :--- |
| **k-space calculation** | `ScalarField.fft_space()` | [Fields API](../reference/api/fields) / [ScalarField](../reference/api/fields) | {ref}`Objective Evidence <validated-en-objective-evidence>` | [Field Intro](../how-to/containers/field_scalar_intro) |
| **Transient FFT** | `TimeSeries.fft(mode="transient")` / `TimeSeries._fft_transient` | [Time Series API](../reference/api/timeseries) / [TimeSeries](../reference/api/timeseries) | {ref}`Objective Evidence <validated-en-objective-evidence>` | [Signal Extraction](../how-to/case-studies/case_signal_extraction) |
| **VIF correction** | `calculate_correlation_factor()` | [Spectral API](../reference/api/spectral) / [Spectral Estimation](../reference/api/spectral) | {ref}`Objective Evidence <validated-en-objective-evidence>` | [Bootstrap Guide](../how-to/case-studies/case_bootstrap_gls_fitting) |
| **Forecast timing** | `ArimaResult.forecast()` | [Time Series API](../reference/api/timeseries) / [TimeSeries](../reference/api/timeseries) | {ref}`Objective Evidence <validated-en-objective-evidence>` | [Advanced ARIMA](../how-to/fitting/advanced_arima) |
| **MCMC / GLS likelihood** | `fit_series()` / `GeneralizedLeastSquares` | [Fitting API](../reference/api/fitting) / [gwexpy.fitting](../reference/api/fitting) | {ref}`Objective Evidence <validated-en-objective-evidence>` | [Bootstrap Guide](../how-to/case-studies/case_bootstrap_gls_fitting) |
| **Adaptive whitening** | `whiten()` / `WhiteningModel` | [Preprocessing API](../reference/api/preprocessing) | {ref}`Objective Evidence <validated-en-objective-evidence>` | [ML Preprocessing Case Study](../how-to/case-studies/case_ml_preprocessing) |

---

(validated-algorithms-how-to-read)=
## How to Read This Page

- If you want the shared prerequisites, time-system assumptions, and FFT conventions first, start with [Prerequisites and Conventions](prerequisites_and_conventions.md).
- If you want hands-on usage before reading the audit notes, use the exact tutorial linked in the summary table rather than the general tutorial index.
- If you want the implementation-facing APIs first, start with [Fields API](../reference/api/fields), [Time Series API](../reference/api/timeseries), [Spectral API](../reference/api/spectral), [Fitting API](../reference/api/fitting), and [Preprocessing API](../reference/api/preprocessing).
- If you want the repository-backed evidence behind the summaries, continue to the {ref}`Objective Evidence <validated-en-objective-evidence>`, {ref}`Source References <validated-en-source-references>`, and Audit Trail sections below.

(validated-algorithms-detail-sections)=
## Detailed Algorithm Basis and Assumptions

(validated-en-k-space)=
### 1. k-space Calculation
**Target**: {meth}`gwexpy.fields.ScalarField.fft_space`

Angular wavenumber calculation follows the standard physics definition $k = 2\pi / \lambda$.

**Assumptions**:
- Spatial coordinates ($x, y$) must be **Uniformly Spaced**.
- For non-uniform grids, a correct wavenumber axis cannot be obtained without prior interpolation.

**Source reference**: {ref}`S1 <validated-en-source-references>`

**Related tutorial**
- [Field Intro](../how-to/containers/field_scalar_intro)

**Related API**
- [Fields API](../reference/api/fields)
- [ScalarField](../reference/api/fields)

**Related theory**
- [Prerequisites and Conventions](prerequisites_and_conventions.md)
- [Architecture and Data Flow](architecture.md)

---

(validated-en-transient-fft)=
### 2. Transient FFT (Amplitude Spectrum)
**Target**: `TimeSeries._fft_transient`

Uses an amplitude-preserving convention rather than density, allowing direct reading of time-domain peak amplitudes.

**Assumptions**:
- No window function is applied to the input signal (assumes Rectangular window).
- For data with windows already applied, separate correction for coherent gain is required.

**Source reference**: {ref}`S5 <validated-en-source-references>`

**Related tutorial**
- [Signal Extraction tutorial](../how-to/case-studies/case_signal_extraction)

**Related API**
- [Time Series API](../reference/api/timeseries)
- [TimeSeries](../reference/api/timeseries)

**Related theory**
- [Prerequisites and Conventions](prerequisites_and_conventions.md)
- [Numerical Stability](numerical_stability.md)

---

(validated-en-vif)=
### 3. VIF (Variance Inflation Factor)
**Target**: {func}`gwexpy.spectral.estimation.calculate_correlation_factor`

Corrects for the reduction in effective sample size caused by overlapping segments in spectral estimation methods like Welch's.

Here, VIF is not meant as the regression-style Variance Inflation Factor used in multicollinearity diagnostics. In the codebase it is exposed as `calculate_correlation_factor()` because the API is describing the concrete operation: computing the overlap-induced correlation correction factor.

**Assumptions**:
- The data must be **Weakly Stationary**.
- VIF may under- or over-estimate variance if non-stationary glitches or step responses are present.

**Source reference**: {ref}`S2 <validated-en-source-references>`

**Related tutorial**
- [Bootstrap GLS fitting case study](../how-to/case-studies/case_bootstrap_gls_fitting)

**Related API**
- [Spectral API](../reference/api/spectral)
- [Spectral Estimation](../reference/api/spectral)

**Related theory**
- [Prerequisites and Conventions](prerequisites_and_conventions.md)

---

(validated-en-arima-forecast)=
### 4. Forecast Timestamp (ARIMA)
**Target**: {meth}`gwexpy.timeseries.arima.ArimaResult.forecast`

Maintains GPS continuity according to LIGO conventions for future timestamp prediction.

A representative update rule is `forecast_t0 = t0 + n_obs * dt`, where

- `t0`: start GPS time of the original series
- `n_obs`: number of observed samples used for the fit
- `dt`: sampling interval in seconds

This is the quantity mapping assumed when forecast timestamps are extended forward.

**Assumptions**:
- The time system must be **GPS Time (no leap seconds)**.
- Using this in time systems with leap seconds (like UTC) will result in a 1-second offset in future predictions.

**Source reference**: {ref}`S3 <validated-en-source-references>`

**Related tutorial**
- [Advanced ARIMA tutorial](../how-to/fitting/advanced_arima)

**Related API**
- [Time Series API](../reference/api/timeseries)
- [TimeSeries](../reference/api/timeseries)

**Related theory**
- [Prerequisites and Conventions](prerequisites_and_conventions.md)

(validated-en-mcmc-gls)=
### 5. MCMC / GLS Likelihood
**Target**: `run_mcmc`, `GLS`

For complex-valued residuals, the MCMC likelihood path assumes a Hermitian quadratic form `r.conj() @ cov_inv @ r` and then takes the real part. In practice, this means the API can accept complex residual vectors without silently dropping the imaginary component.

**Assumptions**:
- `cov_inv` should be close to Hermitian positive-definite.
- The implementation is intended for a circular-complex-Gaussian-style treatment of residuals rather than a naive "real-part only" model.

**Source reference**: {ref}`S5 <validated-en-source-references>`

**Related tutorial**
- [Bootstrap GLS fitting case study](../how-to/case-studies/case_bootstrap_gls_fitting)

**Related API**
- [Time Series API](../reference/api/timeseries)
- [Fitting API](../reference/api/fitting)
- [gwexpy.fitting](../reference/api/fitting)

**Related theory**
- [Prerequisites and Conventions](prerequisites_and_conventions.md)
- [Numerical Stability](numerical_stability.md)

---

(validated-en-adaptive-whitening)=
### 6. Adaptive Whitening
**Target**: `TimeSeriesMatrix.whiten_channels(eps="auto")`, `whiten_matrix()`, `gwexpy.signal.preprocessing.whiten()`, `WhiteningModel`

Adaptive whitening uses an automatically chosen stabilization parameter so that PSD-based normalization remains well behaved in the presence of very small bins or local numerical underflow.

**Assumptions**:
- The data are treated as locally quasi-stationary over the FFT segments used for whitening.
- The adaptive `eps` path is intended to stabilize small denominators, not to compensate for a badly chosen whitening window or clearly non-stationary bursts.

**Source reference**: {ref}`S4 <validated-en-source-references>`

**Related tutorial**
- [ML Preprocessing Case Study](../how-to/case-studies/case_ml_preprocessing)

**Related API**
- [Preprocessing API](../reference/api/preprocessing)

**Related theory**
- [Numerical Stability](numerical_stability.md)
- [Prerequisites and Conventions](prerequisites_and_conventions.md)

---

(validated-algorithms-audit-trail)=
## Audit Trail

The public audit trail is summarized on this page so readers do not need to follow
developer-only or historical review artifacts. The validation summary table above
records the reviewed algorithm families, while each per-algorithm section records
the public-facing assumptions, implementation status, and related theory links.

(validated-algorithms-related-documents)=
## Related Documents for Validation

- [Numerical Stability](numerical_stability.md) - Precision management
- [Prerequisites and Conventions](prerequisites_and_conventions.md) - Shared assumptions for time systems and FFT conventions
- [Verification and Quality Signals](verification_and_quality.md) - Public notebook, I/O, and coverage evidence map
- [API Reference](../reference/index.md) - Entry point to all API pages
- [Fields API](../reference/api/fields)
- [Time Series API](../reference/api/timeseries)
- [Spectral API](../reference/api/spectral)
- [Fitting API](../reference/api/fitting)
- [Preprocessing API](../reference/api/preprocessing)

(validated-algorithms-next-to-read)=
## Next to Read

- [Prerequisites and Conventions](prerequisites_and_conventions.md) - Shared assumptions behind the validation notes
- [Numerical Stability](numerical_stability.md) - Stabilization behavior referenced by adaptive whitening
- [Architecture and Data Flow](architecture.md) - Container and API design context behind the validated methods
- [Tutorial Index](../tutorials/index.md) - Task-oriented notebooks that show the methods in use

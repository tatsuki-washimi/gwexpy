# Validated Algorithms

:::{note}
**Who should read this page?**
- Researchers who want to verify the mathematical and physical validity of the methods.
- Developers who need to know the numerical tolerances between GWexpy and external libraries (SciPy, LALSuite, etc.).
- Users who need to check the data characteristics (stationarity, Gaussianity, etc.) assumed by each algorithm.
:::

:::{important}
**Advanced validation/theory companion**

This page intentionally stays under the user guide so readers can discover it from feature docs, but it is not an onboarding page. Treat it as an audit-oriented companion to the API reference, theory notes, and targeted tutorials. For first-use guidance, start with [Getting Started](getting_started.md), [Prerequisites and Conventions](prerequisites_and_conventions.md), or the linked tutorials for each method.
:::

The numerical algorithms implemented in `gwexpy` have undergone a rigorous validation process to ensure scientific accuracy and reliability.

## Validation Criteria and Numerical Precision

Algorithms labeled as "Validated" meet the following standards and achieve specific precision (tolerance) benchmarks:

- **Tolerance**: We generally use $10^{-12}$ (relative error) as the passing criterion. This indicates no significant logic differences beyond standard double-precision rounding errors.
- **Invariance**: All algorithms are verified for "Scale Invariance," ensuring results remain consistent even after scaling data (e.g., multiplying by 1000).

(validated-en-objective-evidence)=
## Objective Evidence

The summaries on this page are backed by repository artifacts you can inspect directly.

| Evidence type | Source | What it shows |
| :--- | :--- | :--- |
| Audit scope | [ALGORITHM_CONTEXT](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/ALGORITHM_CONTEXT.md) | Which algorithms were reviewed and which numerical questions were checked |
| Consolidated findings | [merged_validation_report](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/merged_validation_report.md) | Cross-reviewed findings, severity, and agreement across independent audits |
| Fix history | [algorithm_fix_report_20260201](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/algorithm_fix_report_20260201.md) | What changed after the review findings were merged and addressed |
| Field-specific physics review | [scalarfield_physics_review_20260120](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs_internal/tech_notes/scalarfield_physics_review_20260120.md) | Physics and axis-consistency review for `ScalarField` FFT behavior |

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

## Validated Algorithms Summary Table

| Algorithm | Primary API | API page | Evidence | Related tutorial |
| :--- | :--- | :--- | :--- | :--- |
| **k-space calculation** | `ScalarField.fft_space()` | [Fields API](../reference/api/fields.rst) / [ScalarField](../reference/ScalarField.md) | {ref}`Objective Evidence <validated-en-objective-evidence>` | [Field Intro](tutorials/field_scalar_intro.ipynb) |
| **Transient FFT** | `TimeSeries.fft(mode="transient")` / `TimeSeries._fft_transient` | [Time Series API](../reference/api/timeseries.rst) / [TimeSeries](../reference/TimeSeries.md) | {ref}`Objective Evidence <validated-en-objective-evidence>` | [Signal Extraction](tutorials/case_signal_extraction.ipynb) |
| **VIF correction** | `calculate_correlation_factor()` | [Spectral API](../reference/api/spectral.rst) / [Spectral Estimation](../reference/Spectral.md) | {ref}`Objective Evidence <validated-en-objective-evidence>` | [Bootstrap Guide](tutorials/case_bootstrap_gls_fitting.ipynb) |
| **Forecast timing** | `ArimaResult.forecast()` | [Time Series API](../reference/api/timeseries.rst) / [TimeSeries](../reference/TimeSeries.md) | {ref}`Objective Evidence <validated-en-objective-evidence>` | [Advanced ARIMA](tutorials/advanced_arima.ipynb) |
| **MCMC / GLS likelihood** | `fit_series()` / `GeneralizedLeastSquares` | [Fitting API](../reference/api/fitting.rst) / [gwexpy.fitting](../reference/fitting.md) | {ref}`Objective Evidence <validated-en-objective-evidence>` | [Bootstrap Guide](tutorials/case_bootstrap_gls_fitting.ipynb) |
| **Adaptive whitening** | `whiten()` / `WhiteningModel` | [Preprocessing API](../reference/api/preprocessing.rst) | {ref}`Objective Evidence <validated-en-objective-evidence>` | [ML Preprocessing Case Study](tutorials/case_ml_preprocessing.ipynb) |

---

## How to Read This Page

- If you want the shared prerequisites, time-system assumptions, and FFT conventions first, start with [Prerequisites and Conventions](prerequisites_and_conventions.md).
- If you want hands-on usage before reading the audit notes, use the exact tutorial linked in the summary table rather than the general tutorial index.
- If you want the implementation-facing APIs first, start with [Fields API](../reference/api/fields.rst), [Time Series API](../reference/api/timeseries.rst), [Spectral API](../reference/api/spectral.rst), [Fitting API](../reference/api/fitting.rst), and [Preprocessing API](../reference/api/preprocessing.rst).
- If you want the repository-backed evidence behind the summaries, continue to the {ref}`Objective Evidence <validated-en-objective-evidence>`, {ref}`Source References <validated-en-source-references>`, and Audit Trail sections below.

## Detailed Algorithm Basis and Assumptions

(validated-en-k-space)=
### 1. k-space Calculation
**Target**: {meth}`gwexpy.fields.ScalarField.fft_space`

Angular wavenumber calculation follows the standard physics definition $k = 2\pi / \lambda$.

**Assumptions**:
- Spatial coordinates ($x, y$) must be **Uniformly Spaced**.
- For non-uniform grids, a correct wavenumber axis cannot be obtained without prior interpolation.

**Source reference**: {ref}`S1 <validated-en-source-references>`

**Related API pages**
- [Fields API](../reference/api/fields.rst)
- [ScalarField](../reference/ScalarField.md)

---

(validated-en-transient-fft)=
### 2. Transient FFT (Amplitude Spectrum)
**Target**: `TimeSeries._fft_transient`

Uses an amplitude-preserving convention rather than density, allowing direct reading of time-domain peak amplitudes.

**Assumptions**:
- No window function is applied to the input signal (assumes Rectangular window).
- For data with windows already applied, separate correction for coherent gain is required.

**Source reference**: {ref}`S5 <validated-en-source-references>`

**Related API pages**
- [Time Series API](../reference/api/timeseries.rst)
- [TimeSeries](../reference/TimeSeries.md)
- [Signal Extraction tutorial](tutorials/case_signal_extraction.ipynb)

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

**Related API pages**
- [Spectral API](../reference/api/spectral.rst)
- [Spectral Estimation](../reference/Spectral.md)
- [Bootstrap GLS fitting case study](tutorials/case_bootstrap_gls_fitting.ipynb)

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

**Related API pages**
- [Time Series API](../reference/api/timeseries.rst)
- [TimeSeries](../reference/TimeSeries.md)
- [Advanced ARIMA tutorial](tutorials/advanced_arima.ipynb)

(validated-en-mcmc-gls)=
### 5. MCMC / GLS Likelihood
**Target**: `run_mcmc`, `GLS`

For complex-valued residuals, the MCMC likelihood path assumes a Hermitian quadratic form `r.conj() @ cov_inv @ r` and then takes the real part. In practice, this means the API can accept complex residual vectors without silently dropping the imaginary component.

**Assumptions**:
- `cov_inv` should be close to Hermitian positive-definite.
- The implementation is intended for a circular-complex-Gaussian-style treatment of residuals rather than a naive "real-part only" model.

**Source reference**: {ref}`S5 <validated-en-source-references>`

**Related API pages**
- [Time Series API](../reference/api/timeseries.rst)
- [Fitting API](../reference/api/fitting.rst)
- [gwexpy.fitting](../reference/fitting.md)
- [Bootstrap GLS fitting case study](tutorials/case_bootstrap_gls_fitting.ipynb)

---

(validated-en-adaptive-whitening)=
### 6. Adaptive Whitening
**Target**: `whiten()`, `WhiteningModel`, `.whiten(eps="auto")`

Adaptive whitening uses an automatically chosen stabilization parameter so that PSD-based normalization remains well behaved in the presence of very small bins or local numerical underflow.

**Assumptions**:
- The data are treated as locally quasi-stationary over the FFT segments used for whitening.
- The adaptive `eps` path is intended to stabilize small denominators, not to compensate for a badly chosen whitening window or clearly non-stationary bursts.

**Source reference**: {ref}`S4 <validated-en-source-references>`

**Related API pages**
- [Preprocessing API](../reference/api/preprocessing.rst)
- [Numerical Stability](numerical_stability.md)
- [ML Preprocessing Case Study](tutorials/case_ml_preprocessing.ipynb)

---

## Audit Trail

The links below point to the audit scope, merged findings, and fix history used to support the summaries on this page.

- [ALGORITHM_CONTEXT](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/ALGORITHM_CONTEXT.md) - Overview of the audited algorithms
- [merged_validation_report](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/merged_validation_report.md) - Consolidated severity and agreement across independent audit passes
- [check_claude_antigravity](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/check_claude_antigravity.md) - Audit notes for VIF, GLS, and `_fft_transient`
- [algorithm_fix_report_20260201](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs/developers/algorithms/algorithm_fix_report_20260201.md) - Change history around ARIMA and MCMC behavior
- [scalarfield_physics_review_20260120](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs_internal/tech_notes/scalarfield_physics_review_20260120.md) - Physics and FFT review notes for `ScalarField`
- [report_scalarfield_physics_verification_20260122_222000](https://github.com/tatsuki-washimi/gwexpy/blob/main/docs_internal/archive/reports/report_scalarfield_physics_verification_20260122_222000.md) - Archived follow-up verification notes for field FFT behavior

## Related Documents

- [Numerical Stability](numerical_stability.md) - Precision management
- [Prerequisites and Conventions](prerequisites_and_conventions.md) - Shared assumptions for time systems and FFT conventions
- [Glossary](glossary.rst) - Glossary of algorithms
- [Fields API](../reference/api/fields.rst)
- [Time Series API](../reference/api/timeseries.rst)
- [Spectral API](../reference/api/spectral.rst)
- [Fitting API](../reference/api/fitting.rst)
- [Preprocessing API](../reference/api/preprocessing.rst)

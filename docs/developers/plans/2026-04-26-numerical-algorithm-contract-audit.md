# Numerical Algorithm Contract Audit

Date: 2026-04-26
Issue: #249, "Audit numerical algorithm contracts and documentation"
Mode: audit-first only; no runtime behavior changes in this pass.

## Core Contract: Numerical Equivalence

GWexpy maintains a strict **Numerical Equivalence** contract with GWpy:

1. **Drop-in replacement**: Default numerical outputs must be bit-identical to the upstream GWpy implementation.
2. **No silent "fixes"**: Improving precision or stability must not change default results. Such changes must be opt-in via parameters (e.g., `stable=True`).
3. **Verification**: All numerical audits must prioritize verifying parity before proposing enhancements.

## Ground Rules And Inputs

- `.agent/AGENTS.md` was reviewed. The relevant local requirements are physical
  consistency with `astropy.units`, explicit time-domain vs frequency-domain
  conventions, metadata preservation, finite/zero-handling audits, and human
  review for physics/data-model changes.
- `.agent/analysis/SKILL.md` and `.agent/validation/SKILL.md` are not present on
  the audited branch.
- `docs/developers/plans/numerical_hardening_plan.md` is not present in the
  current developer plan directory. It was archived under
  `docs_internal/archive/plans/`, where it records the earlier numerical
  hardening work as completed.
- Related existing contracts/docs reviewed:
  - `docs/developers/contracts/numerical.md`
  - `docs/developers/algorithms/ALGORITHM_CONTEXT.md`
  - `docs/developers/algorithms/algorithm_fix_report_20260201.md`
  - `docs/web/en/reference/FFT_Conventions.md`
  - `docs/web/en/user_guide/numerical_stability.md`
  - `docs/web/en/user_guide/validated_algorithms.md`
  - `docs/web/en/reference/fitting.md`

## Scope Inventory

| Area | Implementation files | Existing test coverage seen | Contract documentation seen |
| --- | --- | --- | --- |
| FFT / PSD / ASD / CSD / spectrogram | `gwexpy/timeseries/_spectral.py`, `_spectral_fourier.py`, `_spectral_special.py`, `gwexpy/timeseries/matrix_spectral.py`, `gwexpy/signal/normalization.py`, `gwexpy/signal/spectral/*` | `tests/timeseries/test_fft.py`, `test_fft_param_compat.py`, `test_laplace.py`, `test_hht*.py`, `test_spectral_matrix.py`, `tests/signal/test_normalization.py`, `test_spectral_*` | FFT conventions, numerical contract, algorithm context |
| Filtering / resampling | `gwexpy/timeseries/_resampling.py`, `gwexpy/signal/filter_design.py`, `gwexpy/signal/window.py`, GWpy delegated filters | `tests/timeseries/test_resampling.py`, `tests/signal/test_filter_design.py`, `test_window.py` | Numerical contract, GWpy compatibility docs |
| Coherence / coupling / transfer functions | `gwexpy/analysis/bruco.py`, `coupling.py`, `response.py`, `stats.py`, `gwexpy/timeseries/_signal.py` | `tests/signal/test_coherence.py`, `tests/timeseries/test_transfer_function_compat.py`, matrix coherence tests | Algorithm context, physics models, fitting/reference docs |
| Imputation / preprocessing | `gwexpy/timeseries/preprocess.py`, `gwexpy/signal/preprocessing/{imputation,standardization,whitening,ml}.py` | `tests/timeseries/test_preprocess_*.py`, `tests/signal/test_imputation.py`, `test_preprocessing.py`, `test_whitening.py` | Numerical stability, preprocessing reference |
| Fitting / statistics | `gwexpy/fitting/{core,gls,highlevel,models}.py`, `gwexpy/analysis/stats.py`, `stat_info.py`, `threshold.py` | `tests/fitting/test_*.py`, coupling threshold tests through analysis paths | Fitting reference, validated algorithms, physics models |

## Current Contract Summary

### FFT / PSD / ASD

- Unit behavior:
  - `TimeSeries.fft(mode="gwpy")` delegates to GWpy and wraps the result while
    preserving GWpy units/axis metadata.
  - `TimeSeries.fft(mode="transient")` returns an amplitude spectrum, not a
    density spectrum. It preserves the input unit, not `unit / sqrt(Hz)`.
  - `psd`, `asd`, `csd`, `spectrogram`, `rayleigh_spectrogram` mostly delegate
    to GWpy. The local `csd` wrapper forces unit `self.unit * other.unit / Hz`.
  - `laplace(normalize="integral")` returns `input_unit * s`;
    `normalize="mean"` returns `input_unit`.
  - `hht(output="spectrogram", weight="ia2")` returns `input_unit**2`;
    `weight="ia"` returns `input_unit`.
- Axis/domain assumptions:
  - FFT, Laplace, CWT, STLT and Hilbert/HHT paths require regular sampling.
  - Transient FFT frequency bins are intended to use one-sided real FFT bins,
    but the implementation passes `self.dt.value` directly to
    `np.fft.rfftfreq(target_nfft, d=...)` and then labels the result as Hz.
    This is only correct when `dt.value` is expressed in seconds. For example,
    `dt=1*u.ms` would be passed as `d=1` instead of `d=0.001`, producing bins
    that are off by a factor of 1000 while still carrying `u.Hz`.
  - `TimeSeriesMatrix` vectorized spectral paths in
    `gwexpy/timeseries/matrix_spectral.py` repeat the same raw-`dt.value`
    assumption. `_vectorized_fft()` uses it in `np.fft.rfftfreq(...)`, while
    `_vectorized_psd()`, `_vectorized_csd()`, and `_vectorized_coherence()`
    derive SciPy's `fs` as `1.0 / self.dt.value`; `_vectorized_asd()` inherits
    the PSD axis and normalization. Non-second `dt` units therefore risk wrong
    frequency axes and sample-rate-dependent PSD/ASD/CSD/coherence scaling in
    matrix outputs.
  - `cepstrum` stores quefrency values in the `frequencies` field with
    `axis_type="quefrency"`, which is a non-standard axis encoding that should
    be explicitly documented wherever exposed.
- Normalization convention:
  - Transient FFT uses `rfft(x) / N` and doubles positive one-sided bins except
    DC and, for even `N`, Nyquist.
  - PSD/ASD delegates inherit GWpy/Welch conventions unless a local analysis
    engine reimplements the estimate.
  - `gwexpy.signal.normalization` exposes SciPy-standard and DTT-style helper
    factors, but these helpers are not wired into the main `TimeSeries.psd`
    wrappers.
  - `get_window_normalized()` is documented as returning both `w` and `enbw`,
    but the implementation currently returns only the raw SciPy window `w`.
    Treat the docstring/API mismatch as an observed contract gap before relying
    on it in public normalization guidance.
- Finite/zero handling:
  - Transient FFT does not explicitly reject NaN/Inf input.
  - `laplace` guards sigma-induced exponential overflow.
  - `cepstrum(eps=None)` can produce `-inf` for zero spectra by design; this
    should be documented as an explicit contract or default `eps` policy.
- Optional dependencies:
  - DCT/cepstrum/Laplace/STLT/CWT/Hilbert-related transforms require SciPy or
    PyWavelets/PyEMD depending on method and raise/import through optional
    dependency helpers.
- Metadata preservation:
  - Wrappers generally preserve `epoch`, `name`, `channel`, `unit`, and frequency
    axes. Special transforms add local metadata attributes, but coverage is not
    uniform across all outputs.
  - Matrix spectral outputs preserve matrix-level rows/cols/meta, but the
    vectorized FFT/PSD/ASD/CSD/coherence paths set `freqs * u.Hz` from raw
    `.dt.value`/`fs` calculations. That makes the numerical frequency axis part
    of the same matrix axis and metadata contract tracked in #244 and #246.

### Filtering / Resampling

- Unit behavior:
  - Numeric-rate `resample(rate)` delegates to GWpy and should preserve the
    signal unit through GWpy.
  - Time-bin `_resample_time_bin` preserves the input unit for mean/sum/min/max/
    median/std/callable aggregations.
  - Standardization/whitening wrappers intentionally return dimensionless
    results in the TimeSeries preprocessing API.
- Axis/domain assumptions:
  - Numeric resampling requires regular sampling.
  - Time-bin resampling accepts time-like or dimensionless rules and uses
    left-closed bins with configurable label/origin/align behavior.
  - `asfreq` does not interpolate; it reindexes with exact/ffill/bfill/nearest
    semantics.
- Normalization convention:
  - Signal resampling is GWpy-defined.
  - Time-bin aggregation is statistical aggregation, not anti-alias filtering.
- Finite/zero handling:
  - Time-bin aggregation has `nan_policy` and `min_count`; Inf handling is not
    separately specified.
  - `_generate_target_grid` divides by `target_dt_sec`; zero/negative dt should
    be contracted and tested explicitly.
- Optional dependencies:
  - Median aggregation uses SciPy if available on some paths; other operations
    are NumPy/GWpy.
- Metadata preservation:
  - Tests cover unit/t0/dt preservation for several resampling and preprocessing
    paths, but less so for sample-rate-changing GWpy delegated resampling.

### Coherence / Coupling / Transfer Functions

- Unit behavior:
  - `FastCoherenceEngine.compute_coherence()` returns a plain ndarray coherence
    curve; units cancel.
  - Coupling function values use `target_unit / witness_unit` where possible,
    otherwise dimensionless.
  - `transfer_function()` derives output unit as numerator unit divided by
    denominator unit from the underlying CSD/PSD or FFT outputs.
- Axis/domain assumptions:
  - Bruco coherence uses target sample rate and resamples/truncates/pads aux
    channels to the cached target length.
  - Coupling compares frequency grids with `np.allclose` on x-index values, then
    skips mismatches.
  - Response function selects the closest frequency bin to each injection
    frequency.
  - Transient transfer can downsample one side automatically, crop to the
    intersection, and then truncate to equal length.
- Normalization convention:
  - Bruco implements a local Hann-window Welch-like estimate with scale
    `2.0 / (sample_rate * sum(window**2))`.
  - Coupling and response use PSD/ASD differences:
    `sqrt((target_inj - target_bkg) / (witness_inj - witness_bkg))`, with ASD
    powers squared in response.
  - Steady transfer is documented as CSD/PSD and is tested against GWpy for the
    compatibility call shape.
- Finite/zero handling:
  - Bruco sets coherence to zero when the denominator is not positive; it does
    not explicitly clip to `[0, 1]` or reject non-finite input.
  - Coupling/response leave non-valid bins as NaN and require positive excess
    powers.
  - Transfer has explicit zero-denominator semantics for NaN/Inf and optional
    epsilon regularization, but negative/non-finite epsilon is not separately
    contracted.
- Optional dependencies:
  - Bruco currently requires SciPy at module import.
  - Response parallelism optionally requires joblib.
- Metadata preservation:
  - Coupling/result objects retain frequency axes and labels. Bruco returns
    ndarrays for speed, so metadata preservation is an outer API responsibility.

### Imputation / Preprocessing

- Unit behavior:
  - Low-level `gwexpy.signal.preprocessing.imputation.impute` operates on arrays
    and strips Quantity units from `times`.
  - TimeSeries wrappers reconstruct `TimeSeries` outputs and preserve t0/dt,
    unit, name, and channel in covered tests.
  - Standardization and whitening wrappers intentionally produce dimensionless
    data; inverse-transform models return plain arrays.
- Axis/domain assumptions:
  - Imputation is primarily 1D along the selected axis; `times` must be 1D and
    strictly increasing after sorting.
  - Standardization uses a specified axis and ignores NaN in center/scale.
  - Whitening expects a 2D sample-by-feature matrix at the low level; matrix
    wrappers map domain axes.
- Normalization convention:
  - Robust standardization uses `1.4826 * MAD`.
  - Whitening uses covariance SVD and adds an epsilon to eigenvalues.
- Finite/zero handling:
  - Imputation treats NaN as missing but does not define Inf as missing.
  - Standardization preserves NaN and replaces zero scale with 1.0.
  - Whitening has `_resolve_eps` validation but the main `whiten()` path accepts
    explicit numeric `eps` without finite/non-negative validation.
- Optional dependencies:
  - Pandas is optional for ffill/bfill acceleration/parity; NumPy fallback
    exists.
- Metadata preservation:
  - TimeSeries preprocessing metadata tests are strong for impute/standardize
    and partial for whitening/matrix wrappers.

### Fitting / Statistics

- Unit behavior:
  - `fit_series()` strips x/y values for minimization and stores y/x units for
    `FitResult.model()`.
  - GLS covariance input is treated as numeric; BifrequencyMap support extracts
    `.value`.
  - Statistical `SpectralStats.significance()` returns a FrequencySeries-like
    ratio where units should cancel if input units match.
- Axis/domain assumptions:
  - FrequencySeries uses `frequencies.value`; TimeSeries uses `times.value`;
    otherwise `xindex.value`.
  - Covariance matrices must be `(n, n)` after cropping.
- Normalization convention:
  - Real/complex least squares use chi-square-style sums over residual/sigma.
  - `GeneralizedLeastSquares` uses Cholesky when covariance is positive-definite;
    otherwise it falls back to `r @ cov_inv @ r`.
- Finite/zero handling:
  - Sigma/dy zero and non-finite values are not explicitly rejected before
    division in least-squares costs.
  - `GLS` uses `np.linalg.inv` for direct covariance inversion without condition
    diagnostics.
  - `SpectralStats.significance()` divides by sigma without a zero/finite policy.
- Optional dependencies:
  - Fitting imports require `iminuit` and SciPy.
  - Voigt model requires SciPy; other model functions are NumPy-only.
- Metadata preservation:
  - FitResult preserves enough metadata for plotting/model unit output, but
    covariance axes and physical covariance units are not documented as a public
    contract.

## Gaps And Review Candidates

### Physics Review Required Before Behavior Changes

1. Bruco/FastCoherenceEngine normalization and bin semantics.
   The local scale applies the one-sided factor `2.0` to all RFFT bins, while
   comments acknowledge that DC/Nyquist are special. Coherence ratios can hide a
   common scale mismatch, but PSD/CSD comparability to SciPy/GWpy should be
   established with direct tests before changing behavior.

2. Coupling/response excess-power definition and bin selection.
   Coupling uses PSD differences; response uses ASD squared differences and
   nearest-bin lookup. The physical contract should state whether nearest-bin
   selection is acceptable, whether a tolerance is required, and how negative
   excess power should be reported.

3. Transfer-function epsilon semantics.
   The docstring defines special zero division and epsilon regularization, but
   no current contract says whether epsilon is additive in PSD units, amplitude
   units, or dimensionless user input. Any default or validation change needs
   domain review.

4. DTT normalization helpers.
   `convert_scipy_to_dtt()` contains a stale derivation immediately overwritten
   by the final ratio. Existing tests validate the final ratio for Hann, but the
   public contract should be reviewed against current DTT/diaggui convention
   before using it in production PSD paths.

5. GLS/covariance validity policy.
   Enforcing symmetric positive-definite covariance, Hermitian complex
   covariance, or pseudo-inverse fallback affects statistical interpretation and
   should be reviewed before runtime behavior changes.

### Test / Documentation Gaps That Can Be Split Into Small PRs

1. PSD/ASD wrapper contract tests.
   Add focused tests that assert GWpy-equivalent units, frequency axes, epoch,
   and density/spectrum units for `psd`, `asd`, `csd`, and `spectrogram` wrappers.

2. Transient FFT non-second `dt` axis tests.
   Add an explicit regression candidate for `TimeSeries.fft(mode="transient")`
   with `dt=1*u.ms` or another non-second time unit. The expected frequency bins
   should match `np.fft.rfftfreq(nfft, d=dt.to_value(u.s)) * u.Hz`, not
   `np.fft.rfftfreq(nfft, d=dt.value) * u.Hz`.

3. Matrix spectral non-second `dt` axis and metadata tests.
   Add focused tests for `TimeSeriesMatrix.fft()`, `psd()`, `asd()`, `csd()`,
   and `coherence()` covering non-second `dt` units. The checks should verify
   frequency bins/sample-rate-derived normalization, rows/cols/meta
   preservation, and per-element metadata consistency in line with the #244
   SeriesMatrix invariants and #246 `to_matrix()` axis/unit contract.

4. Matrix spectral ASD mutation and cleanup follow-up.
   `_vectorized_asd()` mutates the PSD object in place via
   `psd.value[:] = asd_data` before returning it, and contains duplicate dead
   code after the first `return`. Add a focused follow-up test or cleanup PR to
   decide whether ASD should return a freshly constructed object, document any
   intentional mutation contract, and remove the unreachable duplicate block.

5. `get_window_normalized()` return-contract test.
   The docstring promises both `w` and `enbw`, while the implementation returns
   only `w`. Add a test or documentation-only follow-up that either changes the
   documented return contract to one value or restores the advertised ENBW
   return after confirming the intended API.

6. Bruco coherence reference tests.
   Add tests comparing `FastCoherenceEngine` against `scipy.signal.coherence` or
   a documented reference on simple signals, including DC/Nyquist expectations
   and coherence bounds.

7. Resampling boundary tests.
   Add tests for zero/negative time-bin rules, `agg="count"` if intended, Inf
   propagation, and metadata preservation through numeric GWpy delegated
   `resample()`.

8. Transfer-function edge tests.
   Add tests for documented zero-denominator cases, non-finite denominator
   behavior, epsilon validation expectations, unit ratio, and transient
   downsample/intersection metadata.

9. Imputation/preprocessing docs and tests for Inf and Quantity times.
   State whether Inf is valid data or missing data. Add tests that Quantity times
   with `max_gap` are converted consistently and inverse transforms intentionally
   return arrays.

10. Whitening epsilon validation.
   Either route `whiten()` through `_resolve_eps()` or document why explicit
   negative/non-finite `eps` is accepted. Add direct tests for the chosen
   contract.

11. Fitting sigma/covariance validation docs.
   Add tests/docs for sigma zero, non-finite sigma, singular covariance, and
   shape/axis behavior after `x_range` cropping.

12. `SpectralStats.significance()` zero-sigma behavior.
   Add a contract and tests for sigma zero/non-finite bins. This can be
   documentation-only first, then behavior if approved.

## Recommended PR Order After Issue #249

1. P0 docs/test-only: codify PSD/ASD/CSD/spectrogram wrapper contracts,
   transient FFT axis/unit metadata, and non-second `dt` frequency-axis tests
   for scalar and matrix spectral paths, including the matrix ASD mutation/dead
   code cleanup question. Low risk, no behavior change.
2. P0 docs/test-only with physics review notes: add Bruco reference tests that
   quantify current normalization and DC/Nyquist behavior without changing it.
3. P1 docs/test-only: add transfer-function edge-case tests for the already
   documented zero/epsilon/unit behavior.
4. P1 docs/test-only: resampling/preprocessing boundary contract tests
   (`target_dt <= 0`, Inf policy, Quantity `max_gap`, metadata).
5. P1 docs/test-only: normalization helper return-contract tests/docs for
   `get_window_normalized()` before presenting ENBW helper behavior as public
   API.
6. P1 docs/test-only: fitting/statistics numerical contract tests for sigma,
   covariance, and zero-sigma significance.
7. P2 implementation PRs, only after review: Bruco normalization/clipping,
   transfer epsilon validation, whitening `_resolve_eps()` adoption, and GLS
   covariance diagnostics.

## Audit Notes

- This file intentionally records observed behavior and gaps only.
- No code behavior was changed as part of this audit.
- Related PRs #256, #257, #258, and #259 are now merged; no follow-up runtime
  code from those PRs was modified as part of this audit.

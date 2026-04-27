# Wave 3 Fitting Contract Audit

Date: 2026-04-28
Issue: #277, "Audit fitting API contracts and metadata propagation"
Mode: audit-first; docs and regression tests only.

## Scope For This Slice

This first #277 slice records current fitting behavior that can be baselined
without changing numerical or physical semantics:

- `fit_series()` `x_range` cropping for `FrequencySeries` inputs;
- full-length `sigma` arrays with `x_range` boundaries that fall exactly on
  samples;
- zero, `nan`, and `inf` `sigma` values;
- ndarray covariance shape handling after `x_range` cropping;
- singular ndarray covariance with a zero diagonal element;
- metadata retained on `FitResult` for fit data, full plotting data, units, and
  x-axis kind.

The tests are concentrated in `tests/fitting/test_fitting_contracts.py` so a
future validation or metadata PR can update one focused contract baseline.

## Contracts Recorded

### `x_range` Cropping

- `fit_series(series, ..., x_range=(2.0, 8.0))` delegates data selection to
  `FrequencySeries.crop()`. For the exact grid covered in this slice, the upper
  boundary sample at `8.0` is excluded and the fit receives frequencies
  `[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]`.
- `FitResult.x` and `FitResult.y` store the cropped fit arrays.
- `FitResult.x_data` and `FitResult.y_data` store the original full-range
  plotting arrays.
- `FitResult.x_fit_range` stores the caller-provided tuple unchanged.
- `FitResult.x_kind`, `FitResult.unit`, and `FitResult.x_unit` retain the
  current frequency-series metadata used by plotting and unit propagation.

### Sigma Handling

- Scalar and already-fit-length `sigma` inputs remain the existing supported
  paths covered by older tests.
- A full-length `sigma` array with `x_range` ending exactly on a sample boundary
  currently uses `np.searchsorted(..., side="right")` for sigma cropping while
  `FrequencySeries.crop()` excludes the upper boundary in this case. The result
  is a `ValueError` with a sigma length mismatch.
- Zero sigma is not rejected or regularized before fitting. It is preserved in
  `FitResult.dy`, can produce NumPy divide warnings, and currently leaves the
  Minuit result invalid with `nan` chi-square for the deterministic case in the
  contract test.
- `nan` sigma is preserved in `FitResult.dy` and currently leaves the Minuit
  result invalid for the deterministic case in the contract test.
- `inf` sigma is preserved in `FitResult.dy`; this slice does not assert a
  validity policy for the optimizer result.

### Covariance Handling

- ndarray covariance input is interpreted as the covariance for the active fit
  data. It is not automatically cropped when `x_range` is provided.
- Passing a full-series covariance with cropped fit data currently raises a
  covariance shape `ValueError`.
- Passing an already-cropped covariance with shape matching the cropped fit data
  succeeds; `FitResult.cov`, `FitResult.cov_inv`, and `FitResult.dy` all use
  the cropped shape.
- Singular ndarray covariance currently uses `np.linalg.pinv()` in the
  `fit_series()` path. A zero diagonal element is preserved as zero plotting
  uncertainty, the pseudo-inverse remains finite for the covered case, and
  `GeneralizedLeastSquares` falls back from Cholesky to the inverse-covariance
  path.

## Deferred Behavior Changes

These are intentionally not included in this docs/test-only slice:

- Harmonize `sigma` x-range cropping with `FrequencySeries.crop()` boundary
  semantics.
- Automatically crop ndarray covariance using `x_range`, or require callers to
  pass fit-range covariance with clearer documentation/errors.
- Reject, mask, or regularize zero, `nan`, or `inf` sigma values.
- Validate covariance finite-ness, symmetry, positive definiteness, Hermitian
  structure for complex covariance, or conditioning thresholds.
- Change `np.linalg.pinv()` fallback behavior, covariance regularization, or
  Cholesky fallback policy.
- Change optimizer failure handling, partial-fit result policy, or error
  messages beyond documenting current behavior.
- Change unit propagation through model inputs, fit outputs, residuals, plots,
  bootstrap, or MCMC helpers.
- Change complex-valued GLS support, bootstrap/MCMC shape/reproducibility
  behavior, optional dependency semantics, or corner-plot outputs.

## Follow-Up Slices For #277

1. Fitting validation policy: decide sigma and covariance finite/zero/singular
   handling, with explicit physics/statistics review before runtime changes.
2. Unit propagation: model input quantities, parameter units, result model
   output units, residual arrays, and plot labels.
3. Bootstrap and high-level GLS: `fit_bootstrap_spectrum()` frequency-range
   cropping, `BifrequencyMap` covariance metadata, confidence interval shapes,
   and optional dependency behavior.
4. MCMC and plotting: sample shape, random-state/reproducibility policy,
   parameter intervals, `plot_corner()`, and fit-band outputs.
5. Failed and partial fits: Minuit validity, Hesse/covariance availability,
   parameter errors, and public error-message contracts.

## Verification

Focused contract check:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q tests/fitting/test_fitting_contracts.py
```

Related fitting regression checks:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q tests/fitting/test_fitting_contracts.py tests/fitting/test_fitting_core.py tests/test_fitting.py tests/test_fitting_gls.py
```

Changed-file hygiene:

```bash
rtk ruff check tests/fitting/test_fitting_contracts.py
rtk ruff format --check tests/fitting/test_fitting_contracts.py
rtk git diff --check
```

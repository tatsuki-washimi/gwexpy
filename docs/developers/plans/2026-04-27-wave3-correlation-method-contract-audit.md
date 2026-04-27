# Wave 3 Correlation Method Contract Audit

Date: 2026-04-27
Issue: #286, "Audit statistics correlation and numerical primitive contracts"
Mode: audit-first; docs and regression tests only.

## Scope For This Slice

This #286 follow-up slice records the current scalar `TimeSeries` correlation
contracts without changing runtime behavior:

- `TimeSeries.correlation()` method dispatch for Pearson, Kendall, MIC,
  distance correlation, and fastmi.
- Alias parity for `pcc()`, `ktau()`, `fastmi()`, and `distance_correlation()`.
- Input preparation behavior for positional length cropping, sample-rate
  mismatch warning/resampling, and scalar-only construction.
- Edge behavior for constant Pearson input, NaN handling, and lack of
  timestamp/`t0` alignment.
- Scalar `partial_correlation()` no-control fallback, residual method,
  precision method, control length cropping, and invalid-method errors when
  controls are present.

The tests are concentrated in
`tests/statistics/test_correlation_method_contracts.py` so future behavior
changes can update one contract baseline deliberately.

## Contracts Recorded

### Scalar Dispatch And Aliases

- `correlation(..., method="pearson")` delegates to SciPy Pearson correlation
  and returns the statistic.
- `correlation(..., method="kendall")` delegates to SciPy Kendall tau and
  returns the statistic.
- `correlation(..., method="fastmi")` delegates to the internal copula/probit
  fastmi estimator and returns a non-negative mutual-information estimate.
- `correlation(..., method="mic")` uses `minepy.MINE` when available. The
  contract test mocks `minepy` and verifies kwargs and cropped arrays.
- `correlation(..., method="distance")` uses `dcor.distance_correlation` when
  available. The contract test mocks `dcor` and verifies alias dispatch.
- Unknown methods raise `ValueError("Unknown correlation method: ...")`;
  `spearman` is therefore recorded as unsupported in this slice.
- `pcc()`, `ktau()`, and `fastmi()` are aliases for the matching
  `correlation()` methods.

### Input Preparation

- Scalar correlation crops both series by position to the shorter length before
  computing statistics.
- The crop does not align by timestamps or `t0`; non-overlapping `t0` values can
  still produce a perfect correlation when the first samples match by position.
- If `sample_rate` differs, `other` is resampled to `self.sample_rate` and a
  `UserWarning` is emitted.
- Multidimensional input is rejected before scalar correlation in current
  construction paths.

### Edge And NaN Behavior

- Constant Pearson input returns `nan` and emits SciPy
  `ConstantInputWarning`.
- Pearson with NaN input currently raises SciPy's
  `ValueError("array must not contain infs or NaNs")`.
- Kendall with NaN input currently returns `nan`.
- fastmi with NaN input currently emits NumPy runtime cast warnings before
  raising `IndexError` from the grid lookup path.

### Scalar Partial Correlation

- With `controls=None`, `partial_correlation()` falls back to Pearson
  correlation before checking the requested partial-correlation method. An
  unknown method therefore still returns Pearson when no controls are present.
- `method="residual"` residualizes `self` and `other` against the controls via
  least squares with an intercept, then applies Pearson to the residuals.
- `method="precision"` stacks `self`, `other`, and controls, computes
  covariance, inverts or pseudo-inverts it, and returns the precision-matrix
  partial correlation.
- Controls are cropped by position together with `self` and `other`.
- Unknown partial-correlation methods raise `ValueError` only once controls are
  present.

## Deferred Behavior Changes

These are intentionally not included in this docs/test-only slice:

- Matrix `TimeSeriesMatrix.correlation_vector()` DataFrame shape, dtype,
  sorting, label, and optional-dependency contracts.
- Matrix `TimeSeriesMatrix.partial_correlation_matrix()` covariance,
  shrinkage, dtype, sorting, and failure-mode contracts.
- Spearman support or alias decisions.
- Timestamp-aware alignment or changes to positional cropping.
- NaN or constant-input policy changes.
- Optional-dependency install hints or fallback semantics for `minepy` and
  `dcor`.
- fastmi estimator interpretation, NaN handling, warning policy, grid behavior,
  or output scaling.
- Partial-correlation covariance regularization or invalid-method ordering.

## Verification

Focused contract check:

```bash
rtk proxy pytest tests/statistics/test_correlation_method_contracts.py -q
```

Related regression checks:

```bash
rtk proxy pytest \
  tests/test_correlation.py \
  tests/timeseries/test_statistics.py \
  tests/statistics/test_correlation_method_contracts.py \
  -q
```

Changed-file hygiene:

```bash
rtk ruff check tests/statistics/test_correlation_method_contracts.py
rtk ruff format --check tests/statistics/test_correlation_method_contracts.py
rtk git diff --check
```

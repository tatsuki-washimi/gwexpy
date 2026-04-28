# Wave 3 Numerical Primitive Contract Audit

Date: 2026-04-27
Issue: #286, "Audit statistics correlation and numerical primitive contracts"
Mode: audit-first; docs and regression tests only.

## Scope For This Slice

This first #286 slice focuses on low-level numerical primitives and statistical
bounds that can be baselined without changing behavior:

- `safe_epsilon()` non-finite-input fallback behavior;
- `AutoScaler` non-finite-input handling as a known gap;
- `safe_log_scale()` signed input and zero-floor behavior;
- `calculate_roc()` output bounds, monotonic false-positive rates, AUC bounds,
  and degenerate-class behavior.

This slice does not change numerical primitives, statistical thresholds,
correlation algorithms, optional dependency behavior, or public return classes.

## Contracts Recorded

### Numerical Primitives

- `safe_epsilon(data, abs_tol=...)` currently returns the absolute tolerance
  when non-finite input causes `np.std(data)` to become non-finite. The path may
  emit NumPy runtime warnings.
- `safe_log_scale()` uses absolute values before applying the logarithm. Signed
  values with equal magnitude therefore map to equal log-scale values.
- `safe_log_scale()` clips zeros to a dynamic floor of
  `max(SAFE_FLOOR, max(abs(data)) * 10 ** (-dynamic_range_db / 10))`.
- `AutoScaler` does not currently reject or regularize non-finite input. The
  desired finite-regularized behavior is recorded as a strict xfail instead of
  being changed in this audit PR.

### ROC Bounds

- `calculate_roc()` returns finite false-positive and true-positive rates in
  `[0, 1]`.
- Returned false-positive rates are sorted monotonically before AUC
  calculation.
- AUC is bounded in `[0, 1]` for the covered deterministic case.
- If either positive or negative examples are absent, the current contract is a
  diagonal ROC curve and AUC `0.5`.

## Deferred Behavior Changes

These are intentionally not included in this docs/test-only slice:

- Reject non-finite input in `safe_epsilon()` or `AutoScaler`.
- Change `AutoScaler` to use `nanstd()` or finite-only scale estimation.
- Change `safe_log_scale()` signed-value behavior or dynamic-floor formula.
- Change ROC threshold ordering, endpoint policy, or degenerate-class policy.
- Change correlation methods, p-value semantics, optional dependency behavior,
  or statistical thresholds.

## Follow-Up Slices For #286

1. Correlation methods: Pearson/Kendall/Spearman, MIC, distance correlation,
   partial correlation, labels, alignment, NaN, and constant-input policy.
2. Matrix analysis outputs: correlation/coherence-adjacent matrix methods,
   labels, metadata, and dtype contracts.
3. Statistical helpers: GauCh, Rayleigh, Student-t, DQ flag helpers, and
   `SpectralStats` edge behavior.
4. Optional dependencies: dcor, minepy/MIC, SciPy, and install hints.

## Verification

Suggested focused command:

```bash
rtk pytest -q \
  tests/numerics/test_primitive_contracts.py \
  tests/statistics/test_statistical_bounds_contracts.py \
  tests/numerics/test_scaling.py \
  tests/signal/test_normalization.py \
  tests/analysis/test_numerical_contracts.py
```

Suggested hygiene checks:

```bash
rtk ruff check \
  tests/numerics/test_primitive_contracts.py \
  tests/statistics/test_statistical_bounds_contracts.py
rtk git diff --check
```

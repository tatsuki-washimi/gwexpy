# Wave 3 Numerical Contract Baseline

Date: 2026-04-27
Issue: #273, "Add numerical algorithm contract regression tests"
Mode: audit-first; docs and regression tests only.

## Scope And Non-Goals

This pass starts Wave 3 by adding a shared numerical baseline before targeted
audits for #285, #286, #277, #278, #284, #288, and #282.

The scope is deliberately limited to behavior that can be tested without
changing runtime modules:

- spectral wrapper units and frequency-axis agreement;
- transient FFT non-second `dt` axis behavior as a known gap;
- matrix vectorized spectral axis and metadata behavior, plus the explicit
  non-second matrix-axis gap;
- current `get_window_normalized()` return shape;
- Bruco coherence edge behavior for zero and identical inputs;
- current `SpectralStats.significance()` zero-sigma behavior.

This pass does not alter numerical algorithms, thresholds, normalization
constants, public return classes, or metadata propagation.

## Inputs Reviewed

- `.agent/AGENTS.md`
- `docs/developers/contracts/numerical.md`
- `docs/developers/plans/2026-04-26-numerical-algorithm-contract-audit.md`
- `docs/developers/plans/active/2026-04-27-issue-burn-down-roadmap.md`
- `gwexpy/timeseries/_spectral_fourier.py`
- `gwexpy/timeseries/matrix_spectral.py`
- `gwexpy/signal/normalization.py`
- `gwexpy/analysis/bruco.py`
- `gwexpy/analysis/stats.py`
- Existing tests under `tests/timeseries/`, `tests/signal/`, and
  `tests/analysis/`.

## Regression Coverage Added

This pass adds focused coverage in:

- `tests/timeseries/test_numerical_contracts.py`
- `tests/analysis/test_numerical_contracts.py`
- `tests/signal/test_normalization.py`

The tests cover:

- `TimeSeries.psd()` density units, `psd(..., scaling="spectrum")` spectrum
  units, `asd()` units, `csd()` units, and `spectrogram()` density units.
- Shared frequency bins for PSD, ASD, CSD, and spectrogram calls using the same
  segment length.
- Current preservation of name/channel metadata for PSD outputs.
- A strict xfail for transient FFT frequency bins when `TimeSeries.dt` is stored
  in milliseconds. This records the current bug without changing the numerical
  axis in a baseline PR.
- `TimeSeriesMatrix._vectorized_fft()` with constructor-provided millisecond
  `dt` normalized to the default second axis, including rows, columns,
  per-element unit, name, and channel metadata.
- A strict xfail for `TimeSeriesMatrix._vectorized_fft()` when an explicit
  non-second matrix axis, such as `xunit="ms"`, leaves `dt.value` in
  non-second units before frequency-bin construction.
- `_vectorized_asd()` values as the square root of `_vectorized_psd()` values on
  the same frequency axis.
- `get_window_normalized()` returning a raw SciPy window array, not an
  `(array, enbw)` tuple.
- `FastCoherenceEngine.compute_coherence()` zero-auxiliary and identical-input
  behavior.
- `SpectralStats.significance()` returning infinite significance at a zero
  sigma bin with the current runtime warning.

## Behavior Changes Deferred

These are intentionally not included in the #273 baseline PR:

- Convert transient FFT `dt` to seconds before calling `np.fft.rfftfreq()`.
- Convert explicit non-second `TimeSeriesMatrix` vectorized spectral axes to
  seconds before frequency-bin and sample-rate construction.
- Change matrix vectorized PSD/ASD/CSD/coherence normalization or unit policy.
- Remove duplicate unreachable code in `_vectorized_asd()` unless paired with a
  no-behavior-change cleanup review.
- Change `get_window_normalized()` to return ENBW or a tuple.
- Clip Bruco coherence to `[0, 1]` or change its one-sided DC/Nyquist scaling.
- Validate transfer-function epsilon values or alter zero-denominator behavior.
- Reject, clip, or regularize zero sigma in `SpectralStats.significance()`.
- Change fitting sigma/covariance validation or GLS numerical conditioning.
- Change whitening epsilon validation or statistical defaults.

## Suggested Follow-Up PR Tracks

1. #285-A: fix or document transient FFT non-second `dt` frequency-axis behavior
   and extend axis tests across special transforms.
2. #286-A: decide `get_window_normalized()` return contract and low-level
   normalization helper semantics.
3. #284-A: add Bruco and coupling reference tests before any normalization or
   excess-power behavior changes.
4. #277-A: add fitting sigma/covariance edge tests and split validation changes.
5. #288-A: add whitening epsilon validation tests before behavior changes.

## Verification

Suggested focused command:

```bash
rtk pytest -q \
  tests/timeseries/test_numerical_contracts.py \
  tests/analysis/test_numerical_contracts.py \
  tests/signal/test_normalization.py
```

Suggested hygiene checks:

```bash
rtk ruff check tests/timeseries/test_numerical_contracts.py \
  tests/analysis/test_numerical_contracts.py \
  tests/signal/test_normalization.py
rtk git diff --check
```

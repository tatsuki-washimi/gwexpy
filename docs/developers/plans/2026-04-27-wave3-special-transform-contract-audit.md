# Wave 3 Special Transform Contract Audit

Date: 2026-04-27
Issue: #285, "Audit time-frequency and space-transform contracts"
Mode: audit-first; docs and regression tests only.

## Scope For This Slice

This slice extends the #285 transform-axis baseline to gwexpy-owned special
transform surfaces that are stable enough to record without changing runtime
behavior:

- `TimeSeries.dct()` axis, unit, metadata, and transform attributes;
- `TimeSeries.cepstrum()` quefrency-axis, dimensionless-unit, metadata, and
  transform attributes;
- `TimeSeries.laplace()` explicit-frequency axis, sigma metadata, and
  normalization-dependent units;
- `TimeSeries.stlt()` `LaplaceGram` shape, time/sigma/frequency axes, scaling
  units, and metadata;
- `TimeSeries.cwt()` optional PyWavelets dependency behavior plus installed
  output shape/frequency-axis behavior when `pywt` is available;
- `TimeSeries.emd()` and `TimeSeries.hht()` optional PyEMD dependency behavior.

This slice does not change transform algorithms, normalization, optional
dependency loading, non-finite input policy, or physics-facing defaults.

## Contracts Recorded

### DCT

- `dct(type=2, norm="ortho")` returns a `FrequencySeries` with one coefficient
  per input sample.
- The DCT axis is represented as a frequency axis in Hz with bins
  `k / (2 * N * dt)`.
- Unit, name suffix, channel, epoch, original sample count, `dt`, DCT type, and
  normalization metadata are preserved on the output.

### Cepstrum

- `cepstrum(kind="real", eps=...)` returns a `FrequencySeries` over a quefrency
  axis stored in `frequencies`.
- The quefrency axis is in seconds with bins `k * dt`.
- Output values are dimensionless, and name suffix, channel, epoch, `dt`, input
  length, transform kind, and cepstrum kind are recorded.

### Laplace

- Explicit frequency inputs are normalized to Hz and preserved on the returned
  `FrequencySeries`.
- `sigma` is stored as a scalar value in `1/s`.
- `normalize="integral"` multiplies the physical unit by seconds.
- `normalize="mean"` preserves the original physical unit.
- Name suffix, channel, and epoch are propagated.

### STLT

- `stlt()` returns a `LaplaceGram` with shape `(time, sigma, frequency)`.
- Axis names are `time`, `sigma`, and `frequency`.
- Time coordinates are window centers in seconds; sigma coordinates are in
  `1/s`; frequency coordinates are in Hz.
- `scaling="dt"` multiplies the source unit by seconds; `scaling="none"`
  preserves the source unit.
- Window, stride, overlap, and source name are recorded in `meta`.
- `.at_sigma()` returns a two-dimensional plane with the same physical unit as
  the parent `LaplaceGram`.

### Optional Dependencies

- `cwt()` reports an analysis-extra install hint when `pywt` is unavailable.
- When `pywt` is available, `cwt(widths=..., output="ndarray")` returns
  coefficients shaped `(n_widths, n_times)` and a Hz frequency axis.
- `emd()` and `hht()` report an analysis-extra install hint when `PyEMD` is
  unavailable.

## Deferred Behavior Changes

These are intentionally not included in this docs/test-only slice:

- Change DCT or cepstrum axis class naming beyond the current `FrequencySeries`
  representation.
- Change Laplace or STLT normalization and scaling conventions.
- Change CWT, EMD, or HHT optional dependency handling.
- Add PyEMD-dependent HHT numerical/physics contracts to the default test gate.
- Decide Q-transform passthrough versus gwexpy-owned behavior.
- Audit spectrogram cleaning and field/space transform workflows.

## Follow-Up Slices For #285

1. Q-transform: document GWpy passthrough versus gwexpy-owned contract.
2. Spectrogram cleaning: shape, mask, units, and metadata preservation.
3. Field/space transforms: identify first-class contracts versus tutorial-only
   workflows.
4. HHT numerical behavior: only after optional dependency availability and
   physics-review expectations are explicit.

## Verification

Suggested focused command:

```bash
rtk proxy pytest tests/timeseries/test_special_transform_contracts.py -q
```

Suggested compatibility command:

```bash
rtk proxy pytest \
  tests/timeseries/test_special_transform_contracts.py \
  tests/timeseries/test_new_transforms.py \
  tests/timeseries/test_laplace.py \
  tests/timeseries/test_p1_delivery.py -q
```

Suggested hygiene checks:

```bash
rtk ruff check tests/timeseries/test_special_transform_contracts.py
rtk ruff format --check tests/timeseries/test_special_transform_contracts.py
rtk git diff --check
```

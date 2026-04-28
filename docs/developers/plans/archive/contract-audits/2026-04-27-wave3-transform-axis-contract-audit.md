# Wave 3 Transform Axis Contract Audit

Date: 2026-04-27
Issue: #285, "Audit time-frequency and space-transform contracts"
Mode: audit-first; docs and regression tests only.

## Scope For This Slice

This slice follows the #273 numerical baseline and focuses on the stable
time-frequency axis surface that can be tested without changing runtime
behavior:

- transient FFT one-sided frequency-bin construction for second-based axes;
- regular-sampling failure behavior for FFT, PSD, and ASD transforms;
- public `TimeSeriesMatrix.fft()`, `psd()`, and `asd()` axis and metadata
  propagation;
- public matrix `csd()` and `coherence()` frequency-axis, unit, and coherence
  bound behavior;
- public matrix `spectrogram()` time/frequency axes, shape, unit, and metadata
  propagation.

This slice does not change runtime algorithms. It also does not attempt to close
all of #285. HHT, CWT, STLT, Laplace, DCT, cepstrum, Q-transform, spectrogram
cleaning, and field/space transform workflows remain follow-up slices.

## Contracts Recorded

### TimeSeries Frequency-Domain Transforms

- `TimeSeries.fft(mode="transient")` with second-based sampling returns
  one-sided `np.fft.rfftfreq()` bins in Hz.
- Even-length transient FFT outputs include a Nyquist bin at
  `sample_rate / 2`; odd-length outputs do not.
- `fft()`, `psd()`, and `asd()` require regular sampling and raise `ValueError`
  before delegating to GWpy on irregular axes.

The known non-second `dt` transient FFT gap remains recorded in the #273
baseline as a strict xfail.

### TimeSeriesMatrix Frequency-Domain Transforms

- Public matrix `fft()`, `psd()`, and `asd()` preserve row and column keys.
- Public matrix `fft()` preserves per-element units, names, and channels.
- Public matrix `psd()` and `asd()` preserve per-element physical units as
  density and amplitude-density units respectively.
- Public matrix `csd()` and `coherence()` share a common frequency axis.
- Self-coherence remains finite and within numerical tolerance of `[0, 1]`.
- Public matrix `spectrogram()` returns a `SpectrogramMatrix` with matrix shape
  `(rows, cols, times, frequencies)`, regular time/frequency axes, density
  units, and per-element metadata.

## Deferred Behavior Changes

These are intentionally not included in this docs/test-only slice:

- Fix non-second `TimeSeries.fft(mode="transient")` axes.
- Fix explicit non-second `TimeSeriesMatrix` vectorized spectral axes.
- Change PSD, ASD, CSD, coherence, or spectrogram normalization.
- Change public matrix spectral return classes or metadata policy.
- Change GWpy-passthrough behavior for Q-transform or special transforms.

## Follow-Up Slices For #285

1. Special transforms: HHT, CWT, STLT, Laplace, DCT, and cepstrum output types,
   axes, units, optional dependencies, and non-finite input policy.
2. Q-transform: document GWpy passthrough versus gwexpy-owned contract.
3. Spectrogram cleaning: shape, mask, units, and metadata preservation.
4. Field/space transforms: identify first-class contracts versus tutorial-only
   workflows.

## Verification

Suggested focused command:

```bash
rtk pytest -q \
  tests/timeseries/test_transform_axis_contracts.py \
  tests/timeseries/test_numerical_contracts.py \
  tests/timeseries/test_fft.py \
  tests/timeseries/test_matrix_spectral.py
```

Suggested hygiene checks:

```bash
rtk ruff check tests/timeseries/test_transform_axis_contracts.py
rtk git diff --check
```

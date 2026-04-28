# Wave 3 Noise Contract Audit

Date: 2026-04-28
Issue: #278, "Audit noise model PSD ASD and optional backend contracts"
Mode: audit-first; docs and regression tests only.

## Scope For This Slice

This first #278 slice records current `gwexpy.noise` behavior that can be
baselined without changing physical, unit, stochastic, or optional-backend
runtime semantics:

- ASD helper conventions for colored spectra, generic peaks, Schumann magnetic
  lines, pyGWINC detector models, and ObsPy noise models;
- PSD-vs-ASD relationships currently implied by helper formulas;
- frequency-axis generation, interpolation, and return-type expectations;
- generated object units, names, channels, and time-series metadata where
  currently preserved;
- deterministic controls for seed- and `numpy.random.Generator`-driven noise;
- optional `gwinc`/`obspy` unavailable behavior and install-hint messages;
- the current `non_gaussian.transient_gaussian_noise(psd=...)` placeholder.

The tests are concentrated in `tests/noise/test_noise_contracts.py` so later
metadata or physics-policy PRs can update one focused baseline.

## Contracts Recorded

### ASD and PSD Conventions

- `gwexpy.noise.colored.power_law()` and the white/pink/red wrappers return
  ASD `FrequencySeries` values, not PSD values.
- Pink-noise ASD follows `amplitude * (f / f_ref)^-0.5`; squaring the output
  gives the corresponding PSD power-law ratio.
- Float amplitudes are interpreted directly in the explicit `unit=` passed to
  the helper. There is no implicit physical rescaling when the amplitude is not
  an `astropy.units.Quantity`.
- Peak helpers are ASD peak-height normalized:
  - `lorentzian_line()` uses `A * gamma / sqrt((f - f0)^2 + gamma^2)`;
  - `gaussian_line()` uses `A * exp(-(f - f0)^2 / (2 sigma^2))`;
  - `voigt_line()` is normalized so the sampled center reaches `amplitude`.
- `schumann_resonance()` combines mode amplitudes incoherently in PSD space and
  returns the square-root ASD.

### Return Types, Axes, Units, and Metadata

- Colored and peak helpers return `FrequencySeries` objects on the caller's
  frequency axis.
- Quantity amplitudes preserve their units. Float amplitudes use the provided
  `unit=` value when present.
- `schumann_resonance()` preserves explicit `unit`, `name`, and `channel`
  keyword metadata in the returned `FrequencySeries`.
- `wave.from_asd()` returns a `TimeSeries` with the requested sample rate and
  `t0`. Its default time-domain unit is the ASD unit multiplied by `sqrt(Hz)`;
  for `m/sqrt(Hz)` this is equivalent to `m`.
- `wave.from_asd()` preserves `name` and `channel` metadata from the ASD unless
  overridden by explicit keyword arguments.

### Optional Backends

- `gwinc_.from_pygwinc()` raises an `ImportError` with an install hint when
  `gwinc` is unavailable.
- `obspy_.from_obspy()` raises an `ImportError` with an install hint when
  `obspy` is unavailable.
- pyGWINC default frequency generation uses
  `np.arange(fmin, fmax + df, df)` and includes the `fmax` sample for the
  covered exact-step case.
- pyGWINC `"A+"` is normalized to `"Aplus"` in the returned model name.
- pyGWINC `"darm"` output is current strain ASD multiplied by
  `ifo.Infrastructure.Length`, with unit `m/sqrt(Hz)`.
- ObsPy seismic models return acceleration ASD by default. Velocity and
  displacement conversions divide by `(2*pi*f)` and `(2*pi*f)^2`,
  respectively, and set `f=0` converted values to `nan`.
- ObsPy infrasound models return pressure ASD and ignore the seismic quantity
  conversion path.

### Stochastic Reproducibility

- `wave.gaussian()`, `wave.uniform()`, `wave.colored()`, and `wave.from_asd()`
  accept either a `seed` or a caller-provided `numpy.random.Generator`.
- Reusing the same `seed` reproduces the same sampled output for the covered
  deterministic cases.
- Reusing the same `Generator` instance advances its state; constructing a new
  generator with the same seed reproduces the first draw.

### Non-Gaussian Placeholder

- `non_gaussian.transient_gaussian_noise()` currently accepts a `psd` argument
  but ignores it. The new contract test resets NumPy's global RNG and confirms
  identical output with and without `psd` for the covered case.
- Treat PSD coloring for `transient_gaussian_noise()` as planned but currently
  unsupported until a physics/statistics review defines normalization,
  frequency-axis, unit, metadata, and reproducibility contracts.

## Deferred Behavior Changes

These are intentionally not included in this docs/test-only slice:

- Change any ASD, PSD, FFT-normalization, interpolation, or stochastic sampling
  behavior.
- Standardize metadata propagation across all noise helpers, including ObsPy
  conversion paths and `non_gaussian.py`.
- Add or change physical validation for float amplitudes, implicit target
  units, `f=0` behavior, negative frequencies, non-finite frequencies, or sharp
  interpolated peaks.
- Implement `non_gaussian.transient_gaussian_noise(psd=...)` coloring.
- Replace global `np.random` usage in `non_gaussian.py` with explicit seed/RNG
  controls.
- Clean up duplicate argument handling or diagnostic printing in `gwinc_.py`.
- Change optional dependency packaging, extras, import guards, or skip policy.
- Alter channel/name defaults for generated `FrequencySeries` or `TimeSeries`
  objects.

Any of these changes should be reviewed as a separate runtime PR with explicit
physics/statistics sign-off where normalization or units are affected.

## Follow-Up Slices For #278

1. Metadata policy: standardize `name`, `channel`, axis metadata, and custom
   attribute preservation across colored, peak, magnetic, wave, pyGWINC, ObsPy,
   and non-Gaussian helpers.
2. Unit policy: decide how float amplitudes should be interpreted when no
   `Quantity` is provided, and document PSD/ASD unit expectations for every
   public helper.
3. Optional backend policy: audit extras, skip markers, unavailable behavior,
   pyGWINC argument cleanup, and ObsPy interpolation accuracy for sharp peaks.
4. Stochastic policy: add explicit reproducibility controls and statistical
   tolerance guidance for `non_gaussian.py` and any generated time-domain noise.
5. Non-Gaussian PSD coloring: decide whether to implement, reject, or document
   as unsupported with a clear warning/error.

## Verification

Focused contract check:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q tests/noise/test_noise_contracts.py
```

Related noise regression checks:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q tests/noise
```

Optional-backend interop checks:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q tests/interop/test_interop_obspy.py tests/interop/test_interop_gwinc.py \
  tests/frequencyseries/test_interop_obspy.py tests/spectrogram/test_interop_obspy_spec.py
```

Changed-file hygiene:

```bash
rtk ruff check tests/noise/test_noise_contracts.py
rtk ruff format --check tests/noise/test_noise_contracts.py
rtk python -c "import yaml; yaml.safe_load(open('docs/developers/plans/audit-manifest-278-noise-contracts.yaml'))"
rtk git diff --check
```

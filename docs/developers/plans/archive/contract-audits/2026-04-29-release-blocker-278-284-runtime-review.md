# Release-Blocker Runtime Review For #278 And #284

Date: 2026-04-29
Issues: #278, #284
Mode: runtime correction with physics/statistics review required before release.

## Decision Update

Maintainer direction changed the release policy for #278 and #284:

- #278 and #284 are release-blocking for 0.1.1.
- They are no longer accepted as known limitations for the first release.
- Runtime corrections may be merged only after focused verification and
  human physics/statistics review.
- #293 remains blocked until #278 and #284 are corrected and accepted.

## Runtime Corrections In This Slice

### #278 Noise PSD/ASD Contracts

Implemented:

- `transient_gaussian_noise(psd=...)` now colors both Gaussian components from
  the supplied one-sided PSD instead of silently ignoring the argument.
- PSD input is converted to ASD via `sqrt(PSD)` before time-domain synthesis.
- PSD and ASD synthesis inputs now reject non-finite values, negative spectral
  values, negative frequencies, empty spectra, and non-strictly-increasing
  frequency axes.
- `transient_gaussian_noise()` gained explicit `rng` and `seed` controls for
  the PSD-colored path while preserving legacy global-NumPy behavior when no
  PSD and no seed/generator are supplied.

Review-sensitive points:

- The colored transient model applies the same PSD to `n0(t)` and `n1(t)` in
  `x(t) = n0(t) + A1 * B(t) * n1(t)`.
- FFT synthesis still follows the existing one-sided ASD-to-time-series
  normalization in `wave.from_asd()`; this PR validates inputs but does not
  redefine the existing normalization convention.
- Human review should confirm whether the model should expose an independent
  PSD for the transient component in a later API.

### #284 Bruco Welch Scaling

Implemented:

- `FastCoherenceEngine` now uses a per-bin one-sided Welch density scale.
- Interior RFFT bins are doubled; DC and Nyquist bins are not doubled.
- A regression test compares the cached target PSD against
  `scipy.signal.welch(..., scaling="density", detrend=False)` using the same
  Hann window array, segment length, and overlap.

Review-sensitive points:

- Coherence values are expected to remain numerically equivalent for ordinary
  bins because PSD and CSD scaling cancels in
  `abs(Pxy)**2 / (Pxx * Pyy)`.
- The cached PSD and CSD now have the same density scaling convention as SciPy
  at DC and Nyquist.
- Existing Bruco display behavior still distinguishes squared coherence from
  amplitude coherence when `asd=True`; this remains an API policy decision, not
  a normalization bug in this slice.

## Primary References Consulted

- SciPy `welch` documentation: Welch estimates PSD by averaging modified
  periodograms; `scaling="density"` returns power per Hz; one-sided spectra add
  negative-frequency energy to the corresponding positive bins.
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html>
- SciPy `coherence` documentation: magnitude-squared coherence is
  `abs(Pxy)**2 / (Pxx * Pyy)`, with Pxx/Pyy PSD estimates and Pxy CSD estimate.
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html>
- ObsPy `get_nlnm` documentation: Peterson noise models return periods and PSD
  values; current ObsPy behavior was reviewed for #278 but not changed in this
  runtime slice.
  <https://docs.obspy.org/archive/1.4.0/packages/autogen/obspy.signal.spectral_estimation.get_nlnm.html>

## Verification Performed

Focused checks run locally:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib pytest -q \
  tests/noise/test_noise_contracts.py::test_from_asd_rejects_invalid_physical_spectra \
  tests/noise/test_noise_contracts.py::test_non_gaussian_transient_psd_colors_both_gaussian_components \
  tests/noise/test_noise_contracts.py::test_non_gaussian_transient_rejects_invalid_psd \
  -p no:cacheprovider

rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib pytest -q \
  tests/analysis/test_numerical_contracts.py::test_fast_bruco_target_psd_matches_scipy_welch_density_scaling \
  -p no:cacheprovider

rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib pytest -q \
  tests/noise/test_noise_contracts.py tests/noise/test_wave.py \
  -p no:cacheprovider

rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib pytest -q \
  tests/analysis/test_numerical_contracts.py \
  tests/analysis/test_analysis_workflow_contracts.py \
  tests/analysis/test_bruco_result.py \
  -p no:cacheprovider

rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib pytest -q \
  tests/noise \
  -p no:cacheprovider

rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib pytest -q \
  tests/analysis/test_numerical_contracts.py \
  tests/analysis/test_analysis_workflow_contracts.py \
  tests/analysis/test_bruco_result.py \
  tests/analysis/test_bruco_logic.py \
  tests/analysis/test_coupling.py \
  tests/analysis/test_coupling_analysis.py \
  tests/analysis/test_response.py \
  tests/analysis/test_response_compat.py \
  tests/analysis/test_stats_and_export.py \
  tests/analysis/test_integration_phase04.py \
  -p no:cacheprovider
```

Results:

- Focused #278 checks: 3 passed.
- Focused #284 check: 1 passed.
- Noise slice: 23 passed, 8 existing Astropy unit-format warnings.
- Analysis slice: 76 passed.
- Related noise suite: 147 passed, 46 existing Astropy unit-format warnings.
- Related analysis suite: 166 passed, 14 existing warnings.

## Required Human Review Before Release

Human physics/statistics review should explicitly sign off on:

- PSD-to-ASD conversion and unit handling for `transient_gaussian_noise(psd=...)`.
- Whether applying the same PSD to both Gaussian components is acceptable for
  the first release.
- Bruco one-sided density scaling at DC and Nyquist relative to SciPy/Welch.
- Whether remaining Bruco amplitude-coherence display and threshold semantics
  are acceptable as documented behavior for 0.1.1.
- Whether #278 and #284 can close after CI passes and review signs off, or
  whether follow-up runtime PRs are still release-blocking.

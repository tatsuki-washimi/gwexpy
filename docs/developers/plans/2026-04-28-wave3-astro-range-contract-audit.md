# Wave 3 Astro Range Contract Audit

Date: 2026-04-28
Issue: #282, "Audit astro range units and compatibility contracts"
Mode: audit-first; docs and regression tests only.

## Scope For This Slice

This first #282 slice records deterministic astro range contracts that can be
baselined without network access, data downloads, or physics-facing runtime
changes:

- `gwexpy.astro.range` currently re-exports representative objects from
  `gwpy.astro.range`;
- `gwexpy.astro` package-level dynamic access mirrors `gwpy.astro` for
  representative range functions and exposes them through `dir()` and
  `__all__`;
- `sensemon_range()` accepts a single PSD-like `FrequencySeries` and returns an
  `astropy.units.Quantity` equivalent to length for a sane deterministic PSD;
- `inspiral_range()` has the same broad length-output contract when the optional
  `inspiral_range` backend is installed;
- gwexpy does not add vectorized/list/matrix range wrappers in this slice.

This slice does not change range algorithms, astrophysical constants,
cosmology assumptions, optional dependency errors, input validation, unit
guards, vectorization behavior, metadata propagation, or GWpy compatibility
semantics.

## Contracts Recorded

### GWpy Proxy Boundary

- `gwexpy.astro.range` is a re-export boundary for the covered range API. The
  representative names `burst_range`, `burst_range_spectrum`,
  `inspiral_range`, `inspiral_range_psd`, `range_spectrogram`,
  `range_timeseries`, `sensemon_range`, `sensemon_range_psd`, and `PSD_UNIT`
  are the same objects as `gwpy.astro.range`.
- `gwexpy.astro` uses dynamic package-level access to `gwpy.astro`. The covered
  names `burst_range`, `inspiral_range`, `range_spectrogram`,
  `range_timeseries`, and `sensemon_range` resolve to the same objects as
  `gwpy.astro` and are exposed through `gwexpy.astro.__all__` and
  `dir(gwexpy.astro)`.
- This boundary means GWexpy currently inherits GWpy range behavior, including
  backend-specific exceptions and numerical details, rather than wrapping or
  normalizing them.

### Unit And Output Assumptions

- The covered deterministic PSD input is a `FrequencySeries` with strain PSD
  units equivalent to `1 / Hz`.
- `sensemon_range()` returns an `astropy.units.Quantity` with length-equivalent
  units and a positive finite value for the covered sane PSD.
- `inspiral_range()` is guarded by
  `importlib.util.find_spec("inspiral_range")`. When that optional backend is
  present, the same broad length-equivalent, positive finite output contract is
  asserted.
- The tests intentionally avoid exact astrophysical values. Exact ranges depend
  on GWpy and optional backend implementation details and should not be locked
  by this audit slice.

### Scalar Versus Vector Behavior

- The covered scalar contract is a single PSD-like `FrequencySeries` input.
- No GWexpy-specific `sensemon_range_many` or `inspiral_range_many` vectorized
  wrapper is provided in the current `gwexpy.astro.range` surface.
- Matrix/list/vectorized handling is deferred because any wrapper or stricter
  unsupported-input policy would be a public behavior change.

## Stable Versus Deferred Surfaces

For this audit, the stable surface is the GWpy compatibility boundary and the
broad unit shape of successful single-`FrequencySeries` range calls. This is
the safest baseline because current astro behavior is essentially a GWpy
proxy/re-export layer.

The following surfaces are intentionally left as follow-up decisions:

- whether GWexpy should add explicit PSD/ASD unit guards or clearer invalid-unit
  errors;
- whether non-finite, zero, negative, empty, or boundary-frequency inputs should
  be prevalidated before calling GWpy;
- whether vector/list/matrix range wrappers should exist and how they should
  preserve metadata;
- whether optional dependency errors should be standardized beyond GWpy's
  current behavior;
- whether astrophysical parameters, cosmology defaults, and mass/unit
  conversions need GWexpy-specific documentation or validation.

## Deferred Behavior Changes

These are intentionally not included in this docs/test-only slice:

- Add GWexpy wrappers around `sensemon_range()`, `inspiral_range()`, or related
  PSD/spectrum/time-series helpers.
- Change units accepted for PSD or ASD inputs.
- Add strict validation for invalid units, non-finite PSD values, empty
  frequency axes, negative PSD values, or frequency boundary conditions.
- Normalize optional dependency errors for `inspiral_range`.
- Add vectorized list/matrix contracts or metadata restoration for collections.
- Change any astrophysical constants, source parameter defaults, cosmology
  assumptions, integration thresholds, or GWpy numerical behavior.

## Follow-Up Slices For #282

1. Unit validation decision: define accepted PSD/ASD units, error types, and
   whether guards live in GWexpy wrappers or remain GWpy-owned.
2. Edge-case contract slice: non-finite values, zero/negative PSDs, empty or
   unsorted frequency axes, and boundary-frequency behavior across available
   GWpy/backend versions.
3. Optional dependency policy: standardize or document `inspiral_range` backend
   discovery and error messages without hiding GWpy compatibility details.
4. Vectorization and metadata slice: decide whether list/matrix range helpers
   should exist and how output units, labels, and source metadata propagate.
5. Astrophysical assumptions slice: document mass defaults, redshift/cosmology
   assumptions, orientation conventions, and acceptable tolerance strategy for
   physics-facing regression tests.

## Verification

Focused astro range contract check:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q tests/astro/test_astro_range_contracts.py -p no:cacheprovider
```

Related astro regression checks:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q \
  tests/astro/test_astro_range_contracts.py \
  tests/astro/test_astro_semantics.py \
  tests/astro/test_range.py \
  -p no:cacheprovider
```

Changed-file hygiene:

```bash
rtk ruff check tests/astro/test_astro_range_contracts.py
rtk ruff format --check tests/astro/test_astro_range_contracts.py
rtk python -c "import yaml; from pathlib import Path; yaml.safe_load(Path('docs/developers/plans/audit-manifest-282-astro-range-contracts.yaml').read_text())"
rtk git diff --check
```

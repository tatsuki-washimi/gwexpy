# Wave 3 Time GPS And Leap-Second Contract Audit

Date: 2026-04-28
Issue: #279, "Audit time GPS and leap-second conversion contracts"
Mode: audit-first; docs and regression tests only.

## Scope For This Slice

This first #279 slice records deterministic time conversion contracts that can
be baselined without changing runtime behavior:

- the `gwexpy.time` package boundary between GWExPy vectorized wrappers and
  proxied `gwpy.time` names;
- scalar and vector `to_gps()`, `from_gps()`, and `tconvert()` return types;
- string, numeric, numpy array, and UTC `datetime` vector handling;
- current timezone policy in `gwexpy.interop._time`;
- stable non-leap GPS/UTC round trips through the interop helpers.

This slice does not change GPS, UTC, TAI, IERS, leap-second, timezone,
precision, or GWpy compatibility behavior. Any future change to time
conversion semantics requires human review.

## Contracts Recorded

### Package Boundary

- `gwexpy.time.to_gps`, `gwexpy.time.from_gps`, and
  `gwexpy.time.tconvert` are the vectorized wrappers from
  `gwexpy.time.core`.
- Other public GWpy time names remain dynamically proxied through
  `gwexpy.time.__getattr__`. The contract test records `LIGOTimeGPS` as a
  representative proxied name that appears in both `gwexpy.time.__all__` and
  `dir(gwexpy.time)`.

### `to_gps()` Scalar And Vector Inputs

- A numeric scalar is delegated to GWpy and returns a GPS-equivalent scalar
  object. For the covered case, the value is a `gwpy.time.LIGOTimeGPS` with
  the same numeric GPS seconds.
- Numeric lists and numeric numpy arrays are treated as vector inputs and
  return `numpy.ndarray` values with `float64` dtype.
- ISO UTC string lists and UTC-aware `datetime` lists return numpy float arrays
  matching `astropy.time.Time(..., scale="utc").gps` for the covered
  ordinary, non-leap date.

### `from_gps()` Scalar And Vector Inputs

- A numeric scalar is delegated to GWpy and returns the same timezone-aware UTC
  `datetime` as `gwpy.time.from_gps()` for the covered non-leap GPS second.
- Numeric lists and arrays take the current vectorized path through
  `astropy.time.Time(..., format="gps").to_datetime()`.
- The current vector path returns a numpy object array of Python `datetime`
  objects. Those datetimes are timezone-naive in the covered case. This is
  recorded as current behavior and not as a desired semantic change.
- The scalar and vector paths therefore currently differ in timezone
  preservation. This should be reviewed before any runtime harmonization.

### `tconvert()` Dispatch

- `tconvert()` without an argument remains outside these deterministic
  contract tests because it depends on the current time.
- Numeric arrays dispatch to the current vector `from_gps()` behavior and
  therefore inherit its object-array and timezone-loss contracts.
- String and `datetime` vectors dispatch to the current vector `to_gps()`
  behavior and return numeric GPS arrays.

### Interop Timezone Policy

- `datetime_utc_to_gps()` treats naive datetimes as UTC under the current
  implementation.
- UTC-aware datetimes and timezone-aware non-UTC datetimes that represent the
  same UTC instant convert to the same GPS value in the covered ordinary date.
- `gps_to_datetime_utc()` returns a timezone-aware UTC `datetime` for a stable
  non-leap GPS input.
- A stable non-leap GPS value round-trips through
  `gps_to_datetime_utc()` and `datetime_utc_to_gps()` within a small floating
  tolerance.

### Leap-Second Caveats

- Existing mocked coverage in `tests/interop/test_time.py` records the
  `gps_to_datetime_utc(..., leap=...)` branches for `raise`, `floor`, `ceil`,
  unknown policy, and unrelated `ValueError` propagation.
- This slice does not add real leap-second boundary tests. Real boundary tests
  can be brittle if they depend on local IERS/leap-second table freshness or
  network updates.
- Python `datetime` cannot represent a `:60` second. The current interop layer
  exposes `LeapSecondConversionError` for unsupported leap-second conversion
  into standard datetime when the underlying `astropy` conversion raises a
  leap-related error.

## Stable Versus Deferred Surfaces

Stable for this audit:

- GWExPy's public package boundary for the three vectorized wrappers and GWpy
  dynamic proxy names.
- Deterministic scalar/vector return-type contracts for the covered ordinary
  GPS and UTC examples.
- Current interop timezone policy for naive, UTC-aware, and non-UTC-aware
  datetimes representing the same instant.

Deferred for behavior-changing follow-up:

- Whether vector `from_gps()` should match scalar GWpy timezone-aware UTC
  behavior instead of the current naive astropy `to_datetime()` result.
- Whether vector `from_gps()` should use `Time(..., format="gps").utc` before
  converting to datetimes.
- Whether naive input should continue to mean UTC everywhere, or whether
  future APIs should reject naive datetimes at stricter boundaries.
- IERS freshness checks, stale table warnings, and offline behavior.
- Explicit leap-second boundary fixtures that remain deterministic without
  network access.
- Stricter accepted-type errors for heterogeneous arrays, object arrays, pandas
  containers, `astropy.time.Time`, and ObsPy `UTCDateTime`.
- Precision contracts for fractional GPS seconds, sub-microsecond loss through
  Python `datetime`, and round trips involving `LIGOTimeGPS` nanoseconds.
- Extended epoch support and dates outside the ordinary Astropy/GWpy ranges
  covered by current tests.

## Follow-Up Slices For #279

1. Scalar/vector harmonization review: decide whether vector `from_gps()`
   should preserve UTC timezone and match GWpy scalar semantics.
2. Leap-second boundary review: add deterministic real-boundary tests only
   after deciding local IERS/leap-second table freshness policy.
3. Accepted input matrix: document and test `astropy.time.Time`, pandas
   `Timestamp`/`Series`/`DatetimeIndex`, numpy `datetime64`, tuples,
   heterogeneous object arrays, and ObsPy `UTCDateTime` when optional
   dependencies are present.
4. Precision and fractional GPS: record fractional second behavior, Python
   `datetime` microsecond truncation/rounding, and `LIGOTimeGPS` nanosecond
   preservation or loss.
5. Error and warning policy: define unsupported types, stale IERS warnings,
   invalid timezone inputs, and leap-second conversion errors.

## Verification

Focused contract check:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q tests/time/test_time_contracts.py -p no:cacheprovider
```

Related time regression checks:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q \
  tests/time/test_time_contracts.py \
  tests/time/test_time_core.py \
  tests/time/test_time.py \
  tests/interop/test_time.py \
  -p no:cacheprovider
```

Changed-file hygiene:

```bash
rtk ruff check tests/time/test_time_contracts.py
rtk ruff format --check tests/time/test_time_contracts.py
rtk python -c "import yaml; from pathlib import Path; yaml.safe_load(Path('docs/developers/plans/audit-manifest-279-time-contracts.yaml').read_text())"
rtk git diff --check
```

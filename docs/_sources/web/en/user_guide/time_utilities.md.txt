---
myst:
  html_meta:
    description: "Use GWexpy time helpers safely for GPS and UTC conversion, including to_gps, from_gps, tconvert, LIGOTimeGPS, vectorized inputs, and timezone edge cases."
---

# GPS Time Utility Functions (`gwexpy.time`)

`gwexpy` extends GWpy's time utilities to support vector operations on pandas Series, NumPy ndarrays, and Astropy Time objects, in addition to standard string/datetime scalars.

## At a Glance

| Item | Details |
| --- | --- |
| **Page Role** | Guide |
| **Audience** | Users who need safe GPS/UTC conversion and contributors looking for the public `gwexpy.time` entry points |
| **Prerequisites** | Basic familiarity with Python `datetime`, timezones, and the use of GPS time in GW workflows |
| **Use Cases** | Choose between `to_gps`, `from_gps`, and `tconvert`, or avoid leap-second and timezone pitfalls |
| **Search Keywords** | GPS time, `to_gps`, `from_gps`, `tconvert`, `LIGOTimeGPS`, leap second, timezone |

**Search hints:** GPS time, `to_gps`, `from_gps`, `tconvert`, `LIGOTimeGPS`, leap second, timezone

## On This Page

- [Quick Guide: Important Considerations](#quick-guide-important-considerations-faq)
- [Function Selection Quick Reference](#function-selection-quick-reference)
- [Basic Examples](#basic-examples)
- [`to_gps`](#to_gps--datetime--gps-seconds)
- [`from_gps`](#from_gps--gps-seconds--datetime)
- [`tconvert`](#tconvert--automatic-conversion)
- [`LIGOTimeGPS`](#ligotimegps--high-precision-gps-time)
- [TimeSeries Integration](#timeseries-integration)

```python
from gwexpy.time import to_gps, from_gps, tconvert, LIGOTimeGPS
```

(time-utils-en-faq)=
## Quick Guide: Important Considerations (FAQ)

It is important to understand these fundamental behaviors before performing conversions.

### Handling Leap Seconds
- **GPS Time**: Continuous seconds starting from 1980-01-06 00:00:00 UTC. It does not include leap seconds.
- **Conversion**: When converting to/from UTC, `astropy.time`'s IERS leap second tables are automatically applied.
- **Note**: For future dates or very recent leap seconds, you may need to update `astropy` data or use the latest IERS table.

### Strings Without Timezones
- Strings like `"2015-09-14 09:50:45"` without a timezone are treated as **UTC by default** in `to_gps` and `tconvert`.
- For reliable local time conversion, always include the timezone name (e.g., `"Asia/Tokyo"`) or provide a timezone-aware `datetime` object.

### Common Failure Modes
- Strings that cannot be parsed as datetimes, such as `to_gps("not-a-time")` or `tconvert("not-a-time")`, fail with `ValueError`.
- Inputs that cannot be converted to numeric GPS seconds, such as `from_gps("abc")`, fail with `ValueError`.
- Unsupported keyword arguments such as `to_gps(..., timezone="Asia/Tokyo")` raise `TypeError` in the current implementation. Pass the timezone in the string itself or use a timezone-aware `datetime`.
- Naive `datetime` inputs are treated as UTC. If you mean a local civil time, use a timezone-aware `datetime`.
- For DST boundaries or other ambiguous local civil times, this page does not guarantee automatic resolution. Prefer explicit UTC offsets or timezone-aware `datetime` objects for boundary cases.

## Function Selection Quick Reference

Choose the most appropriate function for your goal.

| Goal | Use | Input types | Output | Key args |
| :--- | :--- | :--- | :--- | :--- |
| **DateTime → GPS** | `to_gps` | `str`, `datetime`, `Time`, `Series` | `LIGOTimeGPS` / `f8 ndarray` | — |
| **GPS → DateTime** | `from_gps` | `int`, `float`, `LIGOTimeGPS`, `ndarray` | `datetime` / `astropy.time.Time` | — |
| **Convenience (Auto)** | `tconvert` | All above + `"now"` | Context-dependent (Scalar pref.) | — |
| **High-Precision Object** | `LIGOTimeGPS` | `seconds`, `nanoseconds` | `LIGOTimeGPS` (integer s+ns) | — |

---

## Basic Examples

### 1. Simple Conversion

- Purpose: verify round-trip conversion between a datetime string and GPS seconds
- Input: one UTC string and one GPS scalar
- Output: `LIGOTimeGPS`, `datetime`, and `tconvert` return values

```python
from gwexpy.time import to_gps, from_gps, tconvert

# Convert date string to GPS seconds (default UTC)
gps = to_gps("2015-09-14 09:50:45")
# → 1126259462.391

# Convert GPS seconds back to datetime object
dt = from_gps(1126259462.391)
# → datetime.datetime(2015, 9, 14, 9, 50, 45, ...)

# Automatic type detection via tconvert
tconvert("now")     # Current GPS time
tconvert(1126259462) # Formatted string ("September 14 2015, ...")
```

### 2. Vectorized Operations

Passing lists or NumPy arrays will invoke optimized batch conversions.

- Purpose: convert multiple timestamps in one call
- Input: a list of datetime strings or a NumPy array of GPS seconds
- Output: `numpy.ndarray` or `astropy.time.Time` array

```python
import numpy as np

# Convert lists or arrays in bulk
gps_list = to_gps(["2015-09-14 09:50:00", "2015-09-14 09:51:00"])
# → array([1126259417., 1126259477.])

# Convert numeric arrays back to Astropy Time objects
times = from_gps(np.arange(1126259400, 1126259410))
# → <Time object: scale='utc' format='gps' value=[1.1262594e+09 ...]>
```

### 3. Timezones & Leap Seconds

- Purpose: show the difference between timezone-free input and timezone-aware input
- Input: strings or `datetime` objects
- Output: GPS-second conversions with explicit interpretation

```python
# Use explicit timezone strings (recommended)
to_gps("2024-01-01 09:00:00 JST") 

# Be careful: inputs without timezones are treated as UTC
to_gps("2024-01-01 09:00:00") # UTC 09:00:00
```

```python
# Naive datetime is treated as UTC
from datetime import datetime
to_gps(datetime(2024, 1, 1, 9, 0, 0))

# For local civil time, use a timezone-aware datetime
from zoneinfo import ZoneInfo
to_gps(datetime(2024, 1, 1, 9, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo")))
```

### 4. Invalid Input Examples

- Purpose: identify representative `ValueError` and `TypeError` cases before wiring this into a workflow
- Input: unparsable strings, non-numeric GPS input, unsupported keyword arguments
- Output: failure-mode examples

```python
# String cannot be parsed as a datetime
to_gps("not-a-time")  # -> ValueError

# GPS input is not numeric
from_gps("abc")       # -> ValueError

# Unsupported keyword argument
to_gps("2024-01-01 09:00:00", timezone="Asia/Tokyo")  # -> TypeError
```

(time-utils-en-to-gps)=
## `to_gps` — DateTime → GPS Seconds

**Signature**: `to_gps(t, *args, **kwargs)`

Converts various time representations into GPS seconds. It performs efficient vectorized operations on lists, arrays, and Series in addition to individual scalars.

### Working with Strings and DateTime Objects

- Purpose: convert a scalar datetime-like value into GPS seconds
- Input: a string or `datetime`
- Output: `LIGOTimeGPS` or equivalent scalar GPS representation

```python
from gwexpy.time import to_gps

# ISO 8601 string (assumed UTC if no timezone is specified)
to_gps("2015-09-14 09:50:45 UTC")
# → LIGOTimeGPS(1126259462, 391000000)

# Python datetime (timezone-aware recommended)
from datetime import datetime, timezone
to_gps(datetime(2015, 9, 14, 9, 50, 45, tzinfo=timezone.utc))
```

### Working with pandas (Vectorized)

- Purpose: convert multiple timestamps in one call
- Input: a pandas `Series`
- Output: a NumPy array of GPS seconds

```python
import pandas as pd
from gwexpy.time import to_gps

dates = pd.Series(pd.to_datetime(["2015-09-14", "2015-09-15", "2015-09-16"]))
gps_array = to_gps(dates)
# → numpy array([1126224017., 1126310417., 1126396817.])
```

---

(time-utils-en-from-gps)=
## `from_gps` — GPS Seconds → DateTime

**Signature**: `from_gps(t, *args, **kwargs)`

Converts GPS seconds into human-readable formats. Scalar inputs return `datetime` objects, while array inputs return `astropy.time.Time` objects.

- Scalar input such as `int`, `float`, or `LIGOTimeGPS`: returns `datetime.datetime`
- Vector input such as a list, NumPy array, or pandas object: returns `astropy.time.Time`

```python
from_gps(1126259462)
# → datetime.datetime(2015, 9, 14, 9, 50, 45, tzinfo=...)
```

---

(time-utils-en-tconvert)=
## `tconvert` — Automatic Conversion

**Signature**: `tconvert(t=None, *args, **kwargs)`

Automatically detects the input type and dispatches to `to_gps` or `from_gps`.

- Purpose: use one convenience function when the input type may vary
- Input: a datetime-like value, a GPS value, or `"now"`
- Output: GPS seconds or a formatted UTC-side representation depending on input type

```python
from gwexpy.time import tconvert

tconvert("2015-09-14 09:50:45 UTC") # → 1126259462
tconvert("now")                     # Current time as GPS
```

---

(time-utils-en-ligotimegps)=
## `LIGOTimeGPS` — High-Precision GPS Time

`LIGOTimeGPS` stores GPS time with nanosecond precision (integer seconds + integer nanoseconds).

- Purpose: preserve integer-second and nanosecond precision explicitly
- Input: integer seconds and optional nanoseconds
- Output: a `LIGOTimeGPS` object

```python
from gwexpy.time import LIGOTimeGPS

t = LIGOTimeGPS(1126259462, 391000000)
```

---

(time-utils-en-timeseries)=
## TimeSeries Integration

Many methods in `gwexpy` accept time specifications in any format supported by `to_gps()`.

- Purpose: show where the time helpers surface in everyday `TimeSeries` workflows
- Input: time strings, datetimes, or GPS-style values accepted by `to_gps()`
- Output: fetched or cropped `TimeSeries` objects

```python
import gwexpy

ts = gwexpy.TimeSeries.fetch("H1:GDS-CALIB_STRAIN",
                             "2015-09-14 09:50:40",
                             "2015-09-14 09:51:00")

# .crop() accepts strings, datetimes, etc.
segment = ts.crop("2015-09-14 09:50:44", "2015-09-14 09:50:50")
```

---

## Next to Read

- [API Reference](../reference/api/time.rst) — Complete API reference for `gwexpy.time`
- [Migration from GWpy](gwexpy_for_gwpy_users_en.md) — Overview of all GWexpy extensions
- [Prerequisites and Conventions](prerequisites_and_conventions.md) — Shared assumptions for GPS time, timezones, and FFT behavior

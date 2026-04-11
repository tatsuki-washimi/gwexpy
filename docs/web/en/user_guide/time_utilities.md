# GPS Time Utility Functions (`gwexpy.time`)

`gwexpy` extends GWpy's time utilities to support vector operations on pandas Series, NumPy ndarrays, and Astropy Time objects, in addition to standard string/datetime scalars.

```python
from gwexpy.time import to_gps, from_gps, tconvert, LIGOTimeGPS
```

## Quick Guide: Important Considerations (FAQ)

It is important to understand these fundamental behaviors before performing conversions.

### Handling Leap Seconds
- **GPS Time**: Continuous seconds starting from 1980-01-06 00:00:00 UTC. It does not include leap seconds.
- **Conversion**: When converting to/from UTC, `astropy.time`'s IERS leap second tables are automatically applied.
- **Note**: For future dates or very recent leap seconds, you may need to update `astropy` data or use the latest IERS table.

### Strings Without Timezones
- Strings like `"2015-09-14 09:50:45"` without a timezone are treated as **UTC by default** in `to_gps` and `tconvert`.
- For reliable local time conversion, always include the timezone name (e.g., `"Asia/Tokyo"`) or provide a timezone-aware `datetime` object.

## Function Selection Quick Reference

Choose the most appropriate function for your goal.

| Goal | Recommended Function | Primary Input Types | Output Type (Scalar / Vector) | Key Args |
| :--- | :--- | :--- | :--- | :--- |
| **DateTime → GPS** | `to_gps` | `str`, `datetime`, `Time`, `Series` | `LIGOTimeGPS` / `f8 ndarray` | `timezone` |
| **GPS → DateTime** | `from_gps` | `int`, `float`, `LIGOTimeGPS`, `ndarray` | `datetime` / `Time (Quantity)` | — |
| **Convenience (Auto)** | `tconvert` | All above + `"now"` | Context-dependent (Scalar pref.) | — |
| **High-Precision Object** | `LIGOTimeGPS` | `seconds`, `nanoseconds` | `LIGOTimeGPS` (integer s+ns) | — |

---

## Basic Examples

### 1. Simple Conversion

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

```python
# Use explicit timezone strings (recommended)
to_gps("2024-01-01 09:00:00 JST") 

# Be careful: inputs without timezones are treated as UTC
to_gps("2024-01-01 09:00:00") # UTC 09:00:00
```

## `to_gps` — DateTime → GPS Seconds

**Signature**: `to_gps(t, *args, **kwargs)`

Converts various time representations into GPS- **Incompatible API Shims**: Modifying standard library functions or third-party API signatures without an explicit user opt-in.

## Documentation
s.

### Working with Strings and DateTime Objects

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

```python
import pandas as pd
from gwexpy.time import to_gps

dates = pd.Series(pd.to_datetime(["2015-09-14", "2015-09-15", "2015-09-16"]))
gps_array = to_gps(dates)
# → numpy array([1126224017., 1126310417., 1126396817.])
```

---

## `from_gps` — GPS Seconds → DateTime

**Signature**: `from_gps(t, *args, **kwargs)`

Converts GPS seconds into human-readable formats. Scalar inputs return `datetime` objects, while array inputs return `astropy.time.Time` objects.

```python
from_gps(1126259462)
# → datetime.datetime(2015, 9, 14, 9, 50, 45, tzinfo=...)
```

---

## `tconvert` — Automatic Conversion

**Signature**: `tconvert(t=None, *args, **kwargs)`

Automatically detects the input type and dispatches to `to_gps` or `from_gps`.

```python
from gwexpy.time import tconvert

tconvert("2015-09-14 09:50:45 UTC") # → 1126259462
tconvert("now")                     # Current time as GPS
```

---

## `LIGOTimeGPS` — High-Precision GPS Time

`LIGOTimeGPS` stores GPS time with nanosecond precision (integer seconds + integer nanoseconds).

```python
from gwexpy.time import LIGOTimeGPS

t = LIGOTimeGPS(1126259462, 391000000)
```

---

## TimeSeries Integration

Many methods in `gwexpy` accept time specifications in any format supported by `to_gps()`.

```python
import gwexpy

ts = gwexpy.TimeSeries.fetch("H1:GDS-CALIB_STRAIN",
                             "2015-09-14 09:50:40",
                             "2015-09-14 09:51:00")

# .crop() accepts strings, datetimes, etc.
segment = ts.crop("2015-09-14 09:50:44", "2015-09-14 09:50:50")
```

---

## Related Documents

- {doc}`../reference/api/time` — Complete API reference for `gwexpy.time`
- {doc}`gwexpy_for_gwpy_users_en` — Overview of all GWexpy extensions

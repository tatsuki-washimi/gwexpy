# GPS Time Utility Functions (`gwexpy.time`)

`gwexpy` extends GWpy's time utilities to support vector operations on pandas Series, NumPy ndarrays, and Astropy Time objects, in addition to standard string/datetime scalars.

```python
from gwexpy.time import to_gps, from_gps, tconvert, LIGOTimeGPS
```

## Function Selection Quick Reference

Choose the most appropriate function for your goal.

| Goal | Recommended Function | Input Type | Output Type (Scalar / Vector) | Key Args |
| :--- | :--- | :--- | :--- | :--- |
| **DateTime → GPS** | `to_gps` | `str`, `datetime`, `Time`, `Series` | `LIGOTimeGPS` / `f8 ndarray` | `timezone` |
| **GPS → DateTime** | `from_gps` | `int`, `float`, `LIGOTimeGPS`, `ndarray` | `datetime` / `Time (Quantity)` | — |
| **Convenience** | `tconvert` | All above + `"now"` | Context-dependent | — |
| **High Precision** | `LIGOTimeGPS` | `seconds`, `nanoseconds` | `LIGOTimeGPS` (Sec + NanoSec) | — |

---

## `to_gps` — DateTime → GPS Seconds

**Signature**: `to_gps(t, *args, **kwargs)`

Converts various time representations into GPS seconds. Supports both scalars and collections (lists, arrays) with efficient vector operations.

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

## FAQ

### How are leap seconds handled?

`gwexpy` follows the conversion rules provided by `astropy.time` and `gwpy` (using IERS tables).

- **GPS Time**: Continuous seconds from 1980-01-06 00:00:00 UTC.
- **Leap Seconds**: Automatically applied during UTC conversion using the latest available tables.

### How are strings without timezones treated?

A string like `"2015-09-14 09:50:45"` is treated as **UTC by default**.

## Related Documents

- {doc}`../reference/api/time` — Complete API reference for `gwexpy.time`
- {doc}`architecture` — Internal data flow and logic

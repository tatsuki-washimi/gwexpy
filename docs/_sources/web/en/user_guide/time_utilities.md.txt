# Time Utilities (`gwexpy.time`)

GWexpy extends GWpy's time utilities with support for vectorized operations and
a wider range of input types — including pandas, NumPy, and Astropy objects —
in addition to the scalar string/datetime inputs supported by GWpy.

```python
from gwexpy.time import to_gps, from_gps, tconvert, LIGOTimeGPS
```

---

## `to_gps` — Convert to GPS seconds

Converts a wide variety of time representations to GPS seconds.

### Strings and datetime objects

```python
from gwexpy.time import to_gps

# ISO 8601 string (UTC assumed if no timezone given)
to_gps("2015-09-14 09:50:45 UTC")
# → LIGOTimeGPS(1126259462, 391000000)

# Python datetime (timezone-aware recommended)
from datetime import datetime, timezone
to_gps(datetime(2015, 9, 14, 9, 50, 45, tzinfo=timezone.utc))
```

### Astropy Time

```python
from astropy.time import Time

t = Time("2015-09-14T09:50:45", format="isot", scale="utc")
to_gps(t)
# → 1126259462.391
```

### pandas Series / DatetimeIndex (vectorized)

```python
import pandas as pd

dates = pd.Series(pd.to_datetime(["2015-09-14", "2015-09-15", "2015-09-16"]))
gps_array = to_gps(dates)
# → numpy array([1126224017., 1126310417., 1126396817.])
```

### NumPy datetime64 arrays (vectorized)

```python
import numpy as np

dt64 = np.array(["2015-09-14T09:50:45", "2015-09-14T09:51:00"], dtype="datetime64[ns]")
to_gps(dt64)
```

---

## `from_gps` — Convert GPS seconds to datetime

Converts GPS seconds back to human-readable time. Scalar inputs return a
`datetime`, while array inputs return an `astropy.time.Time` array.

```python
from gwexpy.time import from_gps

# Scalar → datetime
from_gps(1126259462)
# → datetime.datetime(2015, 9, 14, 9, 50, 45, tzinfo=...)

# List → astropy Time array
from_gps([1126259462, 1126259472, 1126259482])
```

---

## `tconvert` — Auto-detecting conversion

`tconvert` inspects the input type and dispatches to `to_gps` or `from_gps`
automatically. This mirrors the behavior of the GWpy `tconvert` but adds
array support.

```python
from gwexpy.time import tconvert

# String / datetime → GPS seconds
tconvert("2015-09-14 09:50:45 UTC")
# → 1126259462

# GPS seconds → formatted UTC string
tconvert(1126259462)
# → "September 14 2015, 09:50:45 UTC"

# Current time as GPS seconds
tconvert("now")
```

---

## `LIGOTimeGPS` — High-precision GPS time

`LIGOTimeGPS` stores GPS time with nanosecond precision (integer seconds +
integer nanoseconds). It is the standard representation used by LIGO data
access libraries.

```python
from gwexpy.time import LIGOTimeGPS

t = LIGOTimeGPS(1126259462, 391000000)   # GW150914 merger time

print(t.gpsSeconds)      # 1126259462
print(t.gpsNanoSeconds)  # 391000000
print(float(t))          # 1126259462.391

# Arithmetic
print(t + 10)    # LIGOTimeGPS(1126259472, 391000000)
print(t - 5)     # LIGOTimeGPS(1126259457, 391000000)
```

---

## Integration with TimeSeries

Many GWexpy methods accept time specifications directly as strings, datetime
objects, or GPS numbers — `gwexpy.time.to_gps` is called internally.

```python
import gwexpy

ts = gwexpy.TimeSeries.fetch("H1:GDS-CALIB_STRAIN",
                             "2015-09-14 09:50:40",
                             "2015-09-14 09:51:00")

# .crop() accepts any format supported by to_gps()
segment = ts.crop("2015-09-14 09:50:44", "2015-09-14 09:50:50")
```

---

## See Also

- {doc}`../reference/api/time` — Full API reference for `gwexpy.time`
- {doc}`gwexpy_for_gwpy_users_en` — Overview of all GWexpy extensions

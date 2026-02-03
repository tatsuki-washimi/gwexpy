# Quickstart

Create and analyze a simple time series.

:::{note}
For a more detailed learning path, see [getting_started](getting_started.md).
:::

```python
import numpy as np
from gwexpy.timeseries import TimeSeries

ts = TimeSeries(np.random.randn(4096), t0=0, dt=1/1024, name="demo")
bandpassed = ts.bandpass(30, 300)
spectrum = bandpassed.fft()

print(spectrum.frequencies[:5])
```

Working with collections:

```python
from gwexpy.timeseries import TimeSeriesDict

tsd = TimeSeriesDict()
tsd["H1:TEST"] = ts
tsd["L1:TEST"] = ts * 0.5

matrix = tsd.to_matrix()
print(matrix.shape)
```

## Time utilities and auto series

```python
import numpy as np
from astropy import units as u
import pandas as pd
from gwexpy import as_series
from gwexpy.time import to_gps, from_gps

times = pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:00:01"])
gps = to_gps(times)
iso = from_gps(gps)

ts_axis = as_series((1419724818 + np.arange(10)) * u.s, unit="h")
fs_axis = as_series(np.arange(5) * u.Hz, unit="mHz")
```

## File I/O notes

gwexpy adds/extends several readers beyond GWpy's defaults. For WIN (NIED) files,
gwexpy includes decoding fixes for 0.5-byte (4-bit) and 3-byte (24-bit) compressed
deltas based on the discussion in:

- Shigeki Nakagawa and Aitaro Kato, "New Module for Reading WIN Format Data in ObsPy",
  Technical Research Report, Earthquake Research Institute, the University of Tokyo,
  No. 26, pp. 31-36, 2020.

See `gwexpy/timeseries/io/win.py` for implementation details.

## Pickle / shelve notes

:::{warning}
Never unpickle data from untrusted sources. `pickle`/`shelve` can execute
arbitrary code during loading.
:::

For portability, gwexpy pickling is designed to return **GWpy types** on load
(i.e. unpickling does not require gwexpy to be installed).

```python
import pickle
import numpy as np
from gwexpy.timeseries import TimeSeries

ts = TimeSeries(np.arange(10.0), sample_rate=1.0, t0=0, unit="m")
obj = pickle.loads(pickle.dumps(ts))
# obj is a gwpy.timeseries.TimeSeries
```

Compatibility notes:

- `TimeSeries`, `FrequencySeries`, `Spectrogram` -> unpickle as GWpy objects.
- `TimeSeriesDict`, `TimeSeriesList` -> unpickle as GWpy collections.
- `FrequencySeriesDict/List`, `SpectrogramDict/List` -> unpickle as built-in `dict`/`list` containing GWpy objects.
- gwexpy-only types (e.g. matrix/field classes) are not covered by this portability contract.

What is preserved (best-effort):

- numeric data (`.value`)
- axis information (`times` / `frequencies`)
- metadata commonly supported by GWpy (`unit`, `name`, `channel`, `epoch`)

What is not preserved:

- gwexpy-only internal attributes (e.g. `_gwex_*`) and any behavior added only by gwexpy

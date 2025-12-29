
from ._optional import require_optional
import numpy as np

def to_polars_series(ts, name=None):
    """
    Convert TimeSeries or FrequencySeries to polars.Series.
    This only contains the data values, not the index.
    """
    pl = require_optional("polars")
    from .base import to_plain_array
    data = to_plain_array(ts)
    return pl.Series(name=name or str(ts.name or "series"), values=data)

def to_polars_dataframe(ts, index_column="time", time_unit="datetime"):
    """
    Convert TimeSeries to polars.DataFrame with a time column.
    """
    pl = require_optional("polars")
    from .base import to_plain_array
    data = to_plain_array(ts)
    times_gps = to_plain_array(ts.times)

    if time_unit == "datetime":
        from astropy.time import Time
        t_vals = Time(times_gps, format="gps").utc.to_datetime()
    elif time_unit == "gps":
        t_vals = times_gps
    elif time_unit == "unix":
        from astropy.time import Time
        t_vals = Time(times_gps, format="gps").unix
    else:
        raise ValueError(f"Unknown time_unit: {time_unit}")

    return pl.DataFrame({
        index_column: t_vals,
        ts.name or "value": data
    })

def from_polars_series(cls, series, unit=None, t0=0, dt=1):
    """Create TimeSeries or FrequencySeries from polars.Series."""
    # polars Series is basically just an array.
    return cls(series.to_numpy(), unit=unit, t0=t0, dt=dt)

def to_polars_frequencyseries(fs, index_column="frequency", index_unit="Hz"):
    """Convert FrequencySeries to polars.DataFrame."""
    pl = require_optional("polars")
    from .base import to_plain_array
    data = to_plain_array(fs)
    freqs = to_plain_array(fs.frequencies)

    # Normally frequencies are in Hz.
    return pl.DataFrame({
        index_column: freqs,
        fs.name or "value": data
    })

def from_polars_dataframe(cls, df, index_column="time", unit=None):
    """
    Create TimeSeries from polars.DataFrame.
    Attempts to infer t0 and dt from the time_column.
    """
    require_optional("polars")

    # Extract data column (everything except index_column)
    cols = [c for c in df.columns if c != index_column]
    if not cols:
         raise ValueError("DataFrame must have at least one data column")

    # polars -> numpy
    data = df[cols[0]].to_numpy()
    times = df[index_column]

    # Infer t0, dt
    t0 = 0
    dt = 1

    if len(times) > 0:
        t0_val = times[0]
        # Handle datetime
        if isinstance(t0_val, (np.datetime64,)):
             # Convert to GPS
             from astropy.time import Time
             t0 = Time(t0_val).gps
        elif hasattr(t0_val, "timestamp"): # datetime.datetime
             from ._time import datetime_utc_to_gps
             t0 = datetime_utc_to_gps(t0_val)
        else:
             t0 = float(t0_val)

        if len(times) > 1:
             t1_val = times[1]
             if isinstance(t1_val, (np.datetime64,)):
                 from astropy.time import Time
                 t1 = Time(t1_val).gps
                 dt = t1 - t0
             elif hasattr(t1_val, "timestamp"):
                 from ._time import datetime_utc_to_gps
                 t1 = datetime_utc_to_gps(t1_val)
                 dt = t1 - t0
             else:
                 dt = float(t1_val) - float(t0)

    # If it is a regular grid, use x0, dx for efficiency.
    # Otherwise, pass the full index.
    from gwexpy.time import to_gps
    times_gps_arr = np.asarray(to_gps(times.to_list()), dtype=float)
    diffs = np.diff(times_gps_arr)
    is_regular = len(diffs) < 1 or np.allclose(diffs, diffs[0], atol=1e-12, rtol=1e-10)

    if is_regular:
         t0_final = times_gps_arr[0] if len(times_gps_arr) > 0 else t0
         dt_final = diffs[0] if len(diffs) > 0 else dt
         return cls(data, x0=float(t0_final), dx=float(dt_final), unit=unit, name=cols[0])
    else:
         # Non-regular grid
         if "Frequency" in cls.__name__:
             return cls(data, frequencies=times_gps_arr, unit=unit, name=cols[0])
         else:
             return cls(data, times=times_gps_arr, unit=unit, name=cols[0])

def to_polars_dict(tsd, index_column="time", time_unit="datetime"):
    """TimeSeriesDict -> polars.DataFrame"""
    pl = require_optional("polars")
    from .base import to_plain_array

    keys = list(tsd.keys())
    if not keys:
        return pl.DataFrame()

    s0 = tsd[keys[0]]
    times_gps = to_plain_array(s0.times)

    if time_unit == "datetime":
        from astropy.time import Time
        t_vals = Time(times_gps, format="gps").to_datetime()
    elif time_unit == "gps":
        t_vals = times_gps
    elif time_unit == "unix":
        from astropy.time import Time
        t_vals = Time(times_gps, format="gps").unix
    else:
        raise ValueError(f"Unknown time_unit: {time_unit}")

    data_dict = {index_column: t_vals}
    for k in keys:
        data_dict[k] = to_plain_array(tsd[k])

    return pl.DataFrame(data_dict)

def from_polars_dict(cls, df, index_column="time", unit_map=None):
    """polars.DataFrame -> TimeSeriesDict"""
    tsd = cls()
    # Logic similar to from_pandas_dataframe but for polars
    # We create one TimeSeries for each column (except time_column)
    # the time_column defines GPS start and dt.

    # We can use from_polars_dataframe for each column for simplicity
    # but it's more efficient to calculate t0/dt once.

    times = df[index_column]
    t0 = 0
    dt = 1
    if len(times) > 0:
        t0_val = times[0]
        if isinstance(t0_val, (np.datetime64,)):
             from astropy.time import Time
             t0 = Time(t0_val).gps
        elif hasattr(t0_val, "timestamp"):
             from ._time import datetime_utc_to_gps
             t0 = datetime_utc_to_gps(t0_val)
        else:
             t0 = float(t0_val)

        if len(times) > 1:
             t1_val = times[1]
             if isinstance(t1_val, (np.datetime64,)):
                 from astropy.time import Time
                 t1 = Time(t1_val).gps
                 dt = t1 - t0
             elif hasattr(t1_val, "timestamp"):
                 from ._time import datetime_utc_to_gps
                 t1 = datetime_utc_to_gps(t1_val)
                 dt = t1 - t0
             else:
                 dt = float(t1_val) - float(t0)

    for col in df.columns:
        if col == index_column:
            continue
        unit = unit_map.get(col) if unit_map else None
        data = df[col].to_numpy()
        tsd[str(col)] = tsd.EntryClass(data, x0=float(t0), dx=float(dt), unit=unit, name=str(col))

    return tsd

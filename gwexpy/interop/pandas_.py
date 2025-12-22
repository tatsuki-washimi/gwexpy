
from ._optional import require_optional
from ._time import gps_to_datetime_utc, datetime_utc_to_gps, gps_to_unix, unix_to_gps
from gwpy.time import LIGOTimeGPS
import numpy as np

def to_pandas_series(ts, index="datetime", name=None, copy=False):
    """
    Convert TimeSeries to pandas.Series.
    
    Parameters
    ----------
    index : str, default "datetime"
        "datetime" (UTC aware), "seconds" (unix), or "gps".
    copy : bool
        Whether to copy data.
        
    Returns
    -------
    pandas.Series
    """
    pd = require_optional("pandas")
    from .base import to_plain_array
    data = to_plain_array(ts, copy=copy)
        
    # Build index
    times_gps = to_plain_array(ts.times)
    
    if index == "gps":
        idx = pd.Index(times_gps, name="gps_time")
    elif index == "seconds":
        # Convert GPS array to Unix array? 
        # For array conversion, using astropy Time on array is slow.
        # Vectorized simple offset is t_unix = t_gps + (gps_to_unix(0) - 0)?
        # NO. GPS-UTC offset changes with leap seconds.
        # However, for short durations (most TimeSeries), offset is constant.
        # For exact correctness across leap seconds within a Series, we need vector conversion.
        # Here we assume constant offset for simplicity if len is small, or use Time(array).
        
        # Safe route: Use Time(times_gps, format='gps').unix
        # But this requires astropy constraint.
        # Optimization: check span. If no leap second in span, linear shift.
        # For now, minimal impl -> vector offset.
        from astropy.time import Time
        idx = pd.Index(Time(times_gps, format='gps').unix, name="time_unix")
    elif index == "datetime":
        from astropy.time import Time
        # Time(..).to_datetime() returns array of datetimes (naive or aware?)
        # astropy usually returns naive. We need aware.
        dts = Time(times_gps, format='gps').to_datetime()
        # Localize to UTC
        idx = pd.DatetimeIndex(dts).tz_localize("UTC")
        idx.name = "time_utc"
    else:
        raise ValueError(f"Unknown index type: {index}")
        
    return pd.Series(data, index=idx, name=name or ts.name)

def from_pandas_series(cls, series, *, unit=None, t0=None, dt=None):
    """
    Create TimeSeries from pandas.Series.
    """
    pd = require_optional("pandas")
    from .base import to_plain_array
    values = to_plain_array(series) # Use to_plain_array on the series values
    index = series.index

    # Infer t0, dt if not provided
    inferred_t0 = None
    inferred_dt = None
    
    if t0 is None or dt is None:
        if isinstance(index, pd.DatetimeIndex):
            # Convert first and second to GPS
            t0_dt = index[0]
            # Handle naive as UTC
            if t0_dt.tzinfo is None:
                 t0_dt = t0_dt.replace(tzinfo=pd.Timestamp.utcnow().tz)
            
            inferred_t0 = datetime_utc_to_gps(t0_dt)
            
            if len(index) > 1:
                t1_dt = index[1]
                if t1_dt.tzinfo is None:
                    t1_dt = t1_dt.replace(tzinfo=pd.Timestamp.utcnow().tz)
                t1_gps = datetime_utc_to_gps(t1_dt)
                inferred_dt = t1_gps - inferred_t0
        elif isinstance(index, (pd.Index, pd.RangeIndex)) and np.issubdtype(index.dtype, np.number):
            inferred_t0 = float(index[0])
            if len(index) > 1:
                inferred_dt = float(index[1] - index[0])

    final_t0 = t0 if t0 is not None else (inferred_t0 if inferred_t0 is not None else 0)
    final_dt = dt if dt is not None else (inferred_dt if inferred_dt is not None else 1)
    
    # Ensure dt is not LIGOTimeGPS (convert to float seconds)
    if isinstance(final_dt, LIGOTimeGPS):
        final_dt = float(final_dt)
        
    return cls(values, t0=final_t0, dt=final_dt, unit=unit, name=series.name)

def to_pandas_dataframe(tsd, index="datetime", copy=False):
    """TimeSeriesDict -> DataFrame"""
    pd = require_optional("pandas")
    
    # Check alignment
    # If aligned, share index. If not, error or outer join?
    # Requirement says "round-trip". Usually Dict implies aligned for DF conversion.
    # We take the first series to define index.
    
    keys = list(tsd.keys())
    if not keys:
        return pd.DataFrame()
        
    s0 = tsd[keys[0]]
    # Check consistency
    # (Simplified)
    
    # Build Series
    series_dict = {}
    for k, v in tsd.items():
        s_pd = to_pandas_series(v, index=index, name=k, copy=copy)
        series_dict[k] = s_pd
        
    # Concat
    # axis=1 maps keys to columns
    df = pd.concat(series_dict.values(), axis=1, keys=keys)
    return df

def from_pandas_dataframe(cls, df, *, unit_map=None, t0=None, dt=None):
    """DataFrame -> TimeSeriesDict"""
    tsd = cls()
    for col in df.columns:
        s = df[col]
        unit = unit_map.get(col) if unit_map else None
        tsd[str(col)] = from_pandas_series(tsd.EntryClass, s, unit=unit, t0=t0, dt=dt)
    return tsd

def to_pandas_frequencyseries(fs, index="frequency", name=None, copy=False):
    """
    Convert FrequencySeries to pandas.Series.
    
    Parameters
    ----------
    index : str, default "frequency"
        "frequency" or "index" (integer sample index).
    """
    pd = require_optional("pandas")
    from .base import to_plain_array
    data = to_plain_array(fs, copy=copy)
        
    if index == "frequency":
        freqs = to_plain_array(fs.frequencies)
        if hasattr(fs.frequencies, "unit") and fs.frequencies.unit:
             # Try to normalize to Hz if possible for consistency, or keep raw
             # Usually frequencies is Quantity(Hz).
             pass
        idx = pd.Index(freqs, name="frequency")
    else:
        idx = pd.RangeIndex(len(data), name="index")
        
    return pd.Series(data, index=idx, name=name or fs.name)

def from_pandas_frequencyseries(cls, series, **kwargs):
    """
    Create FrequencySeries from pandas.Series.
    """
    pd = require_optional("pandas")
    values = series.values
    
    # Try to extract frequencies from index
    index = series.index
    frequencies = None
    
    if isinstance(index, pd.Index) and np.issubdtype(index.dtype, np.number) and index.name == "frequency":
        frequencies = index.values
        
    # kwargs might contain frequencies override
    if "frequencies" in kwargs:
         frequencies = kwargs.pop("frequencies")
         
    return cls(values, frequencies=frequencies, name=series.name, **kwargs)

# Re-export or adapt df functions if needed, but FrequencySeriesDict is minimal.

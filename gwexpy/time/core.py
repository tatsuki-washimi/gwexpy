import numpy as np
import pandas as pd
from astropy.time import Time
from gwpy.time import to_gps as _gwpy_to_gps
from gwpy.time import tconvert as _gwpy_tconvert
from gwpy.time import from_gps as _gwpy_from_gps

__all__ = ["to_gps", "from_gps", "tconvert"]

def _is_array(obj):
    """Check if the object is an array-like (list, numpy, pandas), but not a string."""
    return (
        isinstance(obj, (list, tuple, np.ndarray, pd.Series, pd.Index)) 
        and not isinstance(obj, (str, bytes))
    )

def _normalize_time_input(t):
    """Normalize input to standard types (datetime, numpy array, etc.) for interoperability."""
    # Pandas types
    if isinstance(t, (pd.Timestamp, pd.DatetimeIndex, pd.Series)):
        if hasattr(t, "values"):
            return t.values
        return t.to_pydatetime()
    
    # ObsPy UTCDateTime (avoid hard dependency check by name)
    # We check for 'datetime' attribute which ObsPy objects usually have
    if type(t).__name__ == "UTCDateTime" and hasattr(t, "datetime"):
        return t.datetime
    
    # List/Array containing ObsPy UTCDateTime objects
    # Efficiently check only the first element if iterable
    if isinstance(t, (list, tuple, np.ndarray)) and len(t) > 0:
        first = t[0]
        if type(first).__name__ == "UTCDateTime" and hasattr(first, "datetime"):
            # Convert all to datetimes
            return [x.datetime for x in t]

    return t

def to_gps(t, *args, **kwargs):
    """
    Convert a time input (scalar or array) to GPS seconds.
    
    Supports Pandas, ObsPy, and NumPy types seamlessly.
    Returns float array for vector inputs, or LIGOTimeGPS/float for scalar inputs.
    """
    t_norm = _normalize_time_input(t)
    
    if not _is_array(t_norm):
        # Handle numpy scalar (e.g. numpy.datetime64)
        # gwpy might expect standard python types or strings
        if isinstance(t_norm, np.datetime64):
             t_norm = t_norm.item()
        return _gwpy_to_gps(t_norm, *args, **kwargs)

    # Vectorized conversion
    try:
        # Check if already numeric (assuming GPS numbers)
        t_arr = np.asarray(t_norm)
        if np.issubdtype(t_arr.dtype, np.number):
            return t_arr.astype(float)
        
        # Astropy vectorized conversion
        return Time(t_norm, *args, **kwargs).gps

    except Exception:
        # Fallback to loop
        return np.array([_gwpy_to_gps(x, *args, **kwargs) for x in t_norm])

def from_gps(gps, *args, **kwargs):
    """
    Convert GPS time(s) to datetime object(s).
    """
    gps_norm = _normalize_time_input(gps)

    if not _is_array(gps_norm):
        return _gwpy_from_gps(gps_norm, *args, **kwargs)
    
    try:
        return Time(gps_norm, format='gps', *args, **kwargs).to_datetime()
    except Exception:
        return np.array([_gwpy_from_gps(x, *args, **kwargs) for x in gps_norm])

def tconvert(t="now", *args, **kwargs):
    """
    Convert time to GPS, or GPS to datetime, supporting arrays and interops.
    """
    t_norm = _normalize_time_input(t)

    # Scalar keyword handling (e.g. "now")
    if isinstance(t_norm, str) and not _is_array(t_norm):
        return _gwpy_tconvert(t_norm, *args, **kwargs)

    if _is_array(t_norm):
        arr = np.asarray(t_norm)
        # If numeric, assumes GPS -> datetime
        if np.issubdtype(arr.dtype, np.number):
            return from_gps(t_norm, *args, **kwargs)
        # Otherwise, assumes time input -> GPS
        else:
            return to_gps(t_norm, *args, **kwargs)
            
    # Scalar fallback
    return _gwpy_tconvert(t_norm, *args, **kwargs)

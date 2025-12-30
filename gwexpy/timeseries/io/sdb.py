"""
SDB (Davis Vantage Pro2 Weather Station / WeeWX SQLite) format reader.
"""

from __future__ import annotations

import sqlite3
import pandas as pd
import numpy as np
from astropy.time import Time

from gwpy.io import registry as io_registry
from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

# Unit extraction factors (Imperial to Metric)
UNIT_CONVERSION = {
    'barometer': ('hPa', 33.8639),      # inHg -> hPa
    'pressure': ('hPa', 33.8639),       # inHg -> hPa
    'altimeter': ('hPa', 33.8639),      # inHg -> hPa
    'inTemp': ('deg_C', lambda x: (x - 32) / 1.8),
    'outTemp': ('deg_C', lambda x: (x - 32) / 1.8),
    'dewpoint': ('deg_C', lambda x: (x - 32) / 1.8),
    'windchill': ('deg_C', lambda x: (x - 32) / 1.8),
    'heatindex': ('deg_C', lambda x: (x - 32) / 1.8),
    'extraTemp1': ('deg_C', lambda x: (x - 32) / 1.8),
    'extraTemp2': ('deg_C', lambda x: (x - 32) / 1.8),
    'extraTemp3': ('deg_C', lambda x: (x - 32) / 1.8),
    'soilTemp1': ('deg_C', lambda x: (x - 32) / 1.8),
    'rain': ('mm', 25.4),               # inch -> mm
    'rainRate': ('mm/h', 25.4),         # inch/h -> mm/h
    'windSpeed': ('m/s', 0.44704),      # mph -> m/s
    'windGust': ('m/s', 0.44704),       # mph -> m/s
    'inHumidity': ('%', 1.0),
    'outHumidity': ('%', 1.0),
    'radiation': ('W/m^2', 1.0),
    'UV': ('', 1.0),
}

def read_timeseriesdict_sdb(source, table='archive', columns=None, **kwargs):
    """
    Read SDB (SQLite) file into TimeSeriesDict.
    
    Parameters
    ----------
    source : str
        Path to SQLite database file.
    table : str, optional
        Table name to read from, default 'archive'.
    columns : list, optional
        List of column names to read. If None, reads all columns found in UNIT_CONVERSION + dateTime.
    """
    filename = str(source)
    # Open SQLite connection
    conn = sqlite3.connect(filename)
    
    try:
        # Determine columns to query
        if columns is None:
            # Check available columns in the table using PRAGMA
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table})")
            table_cols = [info[1] for info in cursor.fetchall()]
            
            # Filter columns that we know how to convert + dynamic others?
            # For now, stick to known weather columns
            target_cols = [c for c in table_cols if c in UNIT_CONVERSION or c == 'dateTime']
        else:
            target_cols = columns
            if 'dateTime' not in target_cols:
                target_cols.append('dateTime')
        
        col_str = ", ".join(target_cols)
        query = f"SELECT {col_str} FROM {table} ORDER BY dateTime"
        
        # Use pandas for easy reading using the connection context
        df = pd.read_sql_query(query, conn)
        
    finally:
        conn.close()

    if df.empty:
        return TimeSeriesDict()
    
    # Coerce numeric columns to handle potential NULLs -> NaN
    # We expect all columns except potentially descriptive ones (none here) to be numeric
    for col in df.columns:
        if col != 'dateTime':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check if dateTime is present
    if 'dateTime' not in df.columns:
        raise ValueError("Table must contain 'dateTime' column for time series conversion.")

    # Convert dateTime to GPS time (it's usually UNIX timestamp)
    # TimeSeries expects t0 in GPS. Unix to GPS is roughly +18s (leap seconds).
    # gwpy.time.to_gps handles datetime objects.
    # Convert first timestamp
    t_start_unix = df['dateTime'].values[0]
    
    # Calculate sampling rate/dt
    # Assume regular?
    if len(df) > 1:
        # Check median delta
        dt_vals = np.diff(df['dateTime'].values)
        dt_median = np.median(dt_vals)
        if dt_median == 0:
            dt_median = 1.0 # fallback
        sample_rate = 1.0 / dt_median
    else:
        sample_rate = 1.0 # default
        
    # Convert to TimeSeriesDict
    tsd = TimeSeriesDict()
    
    # Use astropy Time for accurate conversion if needed, but simple offset is faster
    # Unix 0 = 1970-01-01 00:00:00 UTC = GPS 315964819 (wait, 315964819 is with leap seconds)
    # Correct way: Time(unix_val, format='unix').gps
    t0_gps = Time(t_start_unix, format='unix').gps
    
    for col in df.columns:
        if col == 'dateTime':
            continue
            
        data = df[col].values
        unit = ''
        
        # Apply conversion
        if col in UNIT_CONVERSION:
            u_name, factor = UNIT_CONVERSION[col]
            unit = u_name
            if callable(factor):
                data = factor(data)
            else:
                data = data * factor
                
        ts = TimeSeries(
            data,
            t0=t0_gps,
            sample_rate=sample_rate,
            name=col,
            unit=unit
        )
        tsd[col] = ts
        
    return tsd


def read_timeseries_sdb(source, **kwargs):
    """
    Read SDB file as TimeSeries.
    If multiple columns, returns the first one (excluding dateTime).
    """
    tsd = read_timeseriesdict_sdb(source, **kwargs)
    if not tsd:
        raise ValueError("No time series data found in sdb file")
    return tsd[next(iter(tsd.keys()))]


# -- Registration

for fmt in ["sdb", "sqlite", "sqlite3"]:
    io_registry.register_reader(fmt, TimeSeriesDict, read_timeseriesdict_sdb, force=True)
    io_registry.register_reader(fmt, TimeSeries, read_timeseries_sdb, force=True)
    io_registry.register_reader(fmt, TimeSeriesMatrix, lambda *a, **k: read_timeseriesdict_sdb(*a, **k).to_matrix(), force=True)
    
    io_registry.register_identifier(
        fmt, TimeSeriesDict,
        lambda *args, **kwargs: str(args[1]).lower().endswith(f".{fmt}"))
    io_registry.register_identifier(
        fmt, TimeSeries,
        lambda *args, **kwargs: str(args[1]).lower().endswith(f".{fmt}"))

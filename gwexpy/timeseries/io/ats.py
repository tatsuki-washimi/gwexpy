"""
Metronix ATS format reader for gwexpy.
Based on binary header parsing logic provided by user (ats2gwf.py).
"""

from __future__ import annotations

import datetime

import numpy as np
from gwpy.time import to_gps

from gwpy.io import registry as io_registry
from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


def read_timeseriesdict_ats(source, **kwargs):
    """
    Read a Metronix ATS file into a TimeSeriesDict.
    """
    ts = read_timeseries_ats(source, **kwargs)
    return TimeSeriesDict({ts.name: ts})


def read_timeseries_ats(source, **kwargs):
    """
    Read a Metronix ATS file into a TimeSeries.
    """
    filename = str(source)
    with open(filename, mode='rb') as f:
        # Read entire file content buffer? 
        # For large files, memory mapping or chunk reading is better, 
        # but following original script's logic:
        # f_data = f.read() 
        # Header is fixed size usually. Let's read header first.
        
        # Header length is at offset 0 (int16)
        header_len_bytes = f.read(2)
        header_length = np.frombuffer(header_len_bytes, dtype=np.int16)[0]
        
        # Read the rest of the header (assuming 1024 bytes based on previous checks)
        f.seek(0)
        header_bytes = f.read(header_length)
        
        # Parse Header
        # Offsets based on ats2gwf.py
        # SampleFreq: 8-12 (float32)
        sample_freq = np.frombuffer(header_bytes[8:12], dtype=np.float32)[0]
        
        # StartTime: 12-16 (int32 - Reference Unix Timestamp?)
        start_time_int = np.frombuffer(header_bytes[12:16], dtype=np.int32)[0]
        
        # LSB Value: 16-24 (float64)
        lsb_val = np.frombuffer(header_bytes[16:24], dtype=np.float64)[0]
        
        # System Number: 32-34 (int16)
        system_number = np.frombuffer(header_bytes[32:34], dtype=np.int16)[0]
        
        # Strings are fixed width, null padded?
        # Channel Type: 38-40
        channel_type = header_bytes[38:40].decode('utf-8', errors='ignore').strip()
        
        # Sensor Type: 40-46
        sensor_type = header_bytes[40:46].decode('utf-8', errors='ignore').strip()
        
        # Serial Number: 46-48 (int16)
        serial_number = np.frombuffer(header_bytes[46:48], dtype=np.int16)[0]
        
        # System Type: 132-144
        system_type = header_bytes[132:144].strip(b'\x00').decode('utf-8', errors='ignore').strip()

        # Construct Channel Name
        chname = 'Metronix_{}_{:03}_{}_{}_{:04}'.format(
            system_type,
            system_number,
            channel_type,
            sensor_type,
            serial_number
        )
        
        # Calculate t0
        # StartTime is Unix timestamp
        dt_obj = datetime.datetime.fromtimestamp(start_time_int, tz=datetime.timezone.utc)
        t0 = to_gps(dt_obj)
        
        # Read Data
        # Data starts after HeaderLength
        # Using memmap for data to be efficient if needed, or fromfile
        f.seek(header_length)
        data_raw = np.fromfile(f, dtype=np.int32)
        
        # Scale Data
        # unit='Volt', value = raw * LSB / 1000 ? (LSB is likely nV or similar?)
        # User script: * LSBval / 1000
        # If LSBval is in uV/count? 
        # Assuming result unit is Volts? Or mV?
        # ats2gwf says unit='Volt'.
        
        data_scaled = data_raw * lsb_val / 1000.0
        
        ts = TimeSeries(
            data_scaled,
            sample_rate=sample_freq,
            t0=t0,
            name=chname,
            channel=chname,
            unit='V' # 'Volt'
        )
        
        return ts


def read_timeseries_ats_mth5(source, **kwargs):
    """
    Read Metronix ATS/ATSS file using mth5 library.
    
    This function uses `mth5.io.metronix.metronix_atss.read_atss`.
    Note that mth5 enforces strict filename conventions (e.g. extension .atss, specific underscores)
    to parse metadata. If your file does not adhere to these conventions, use the default
    `read_timeseries_ats` which parses the binary header directly.
    """
    try:
        from mth5.io.metronix import metronix_atss
    except ImportError:
        raise ImportError("mth5 library is required to use this reader. Install it via pip.")
    
    # mth5 requires .atss extension?
    # Based on investigation, it checks .suffix in _get_file_type.
    # If source does not end in .atss, we might fail or need to symlink.
    # We will pass it as is and let mth5 handle (or fail).

    from gwexpy.timeseries import TimeSeries

    # mth5 read_atss returns a ChannelTS object (wrapping xarray)
    # or sometimes directly ChannelTS?
    # read_atss(fn) -> atss_obj.to_channel_ts()

    channel_ts = metronix_atss.read_atss(str(source))

    # Convert ChannelTS (xarray) to TimeSeries
    # channel_ts.ts is the xarray DataArray
    data_array = channel_ts.ts
    
    # Extract data
    data = data_array.data
    
    # Extract metadata
    # t0: start time. Obspy/MTH5 usually uses comparison-safe types.
    # xarray time index:
    # data_array.time[0]
    
    from gwpy.time import to_gps
    import pandas as pd
    
    # Attempt to get t0 from time index
    if len(data_array.time) > 0:
        # Convert numpy.datetime64/pandas timestamp to GPS
        t0_timestamp = pd.Timestamp(data_array.time.values[0])
        t0 = to_gps(t0_timestamp)
        dt = 1.0 / channel_ts.sample_rate
    else:
        # Fallback or error
        t0 = 0
        dt = 1.0
        
    name = channel_ts.channel_metadata.id or "mth5_channel"
    unit = channel_ts.channel_metadata.units
    
    return TimeSeries(
        data,
        t0=t0,
        dt=dt,
        name=name,
        unit=unit
    )


# -- Registration

for fmt in ["ats"]:
    io_registry.register_reader(fmt, TimeSeriesDict, read_timeseriesdict_ats, force=True)
    io_registry.register_reader(fmt, TimeSeries, read_timeseries_ats, force=True)
    io_registry.register_reader(fmt, TimeSeriesMatrix, lambda *a, **k: read_timeseriesdict_ats(*a, **k).to_matrix(), force=True)
    
    io_registry.register_identifier(
        fmt, TimeSeriesDict,
        lambda *args, **kwargs: str(args[1]).lower().endswith(f".{fmt}"))
    io_registry.register_identifier(
        fmt, TimeSeries,
        lambda *args, **kwargs: str(args[1]).lower().endswith(f".{fmt}"))

# Register mth5 variant
io_registry.register_reader("ats.mth5", TimeSeries, read_timeseries_ats_mth5)

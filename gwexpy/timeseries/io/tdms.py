"""
TDMS (National Instruments) reader for gwexpy.
"""

from __future__ import annotations

import datetime
import numpy as np
from gwpy.io import registry as io_registry

from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
from gwexpy.io.utils import apply_unit, set_provenance


def _import_nptdms():
    try:
        from nptdms import TdmsFile
    except ImportError as exc:
        raise ImportError(
            "npTDMS is required for reading TDMS files. "
            "Install with `pip install nptdms`."
        ) from exc
    return TdmsFile


def read_timeseriesdict_tdms(
    source,
    *,
    channels=None,
    unit=None,
    **kwargs,
) -> TimeSeriesDict:
    """
    Read a TDMS file into a TimeSeriesDict.
    """
    TdmsFile = _import_nptdms()
    tdms_file = TdmsFile.read(source)

    # Fallback timing from root
    if 'DateTime' in tdms_file.properties:
        dt_root = tdms_file.properties['DateTime']
        if isinstance(dt_root, (np.datetime64, datetime.datetime)):
             # We will convert it below in the loop if needed
             pass

    tsd = TimeSeriesDict()

    for group in tdms_file.groups():
        for channel in group.channels():
            full_name = f"{group.name}/{channel.name}"

            if channels and full_name not in channels and channel.name not in channels:
                continue

            data = channel.read_data()
            props = channel.properties

            # Timing
            dt = props.get('wf_increment', 1.0)
            if dt == 0 or np.isinf(dt) or np.isnan(dt):
                dt = 1.0 # fallback

            t0 = props.get('wf_start_time', 0.0)
            if (t0 == 0.0 or (isinstance(t0, np.datetime64) and np.isnat(t0))) and 'DateTime' in tdms_file.properties:
                t0 = tdms_file.properties['DateTime']

            # Convert numpy.datetime64 or datetime.datetime to GPS
            if isinstance(t0, (np.datetime64, datetime.datetime)):
                from gwexpy.io.utils import datetime_to_gps
                if isinstance(t0, np.datetime64):
                    if np.isnat(t0):
                        t0 = 0.0
                    else:
                        # Convert to python datetime
                        unix_epoch = np.datetime64('1970-01-01T00:00:00Z')
                        seconds = (t0 - unix_epoch) / np.timedelta64(1, 's')
                        dt_obj = datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc)
                        t0 = datetime_to_gps(dt_obj)
                else: # datetime.datetime
                    if t0.tzinfo is None:
                        t0 = t0.replace(tzinfo=datetime.timezone.utc)
                    t0 = datetime_to_gps(t0)

            ts = TimeSeries(
                data,
                dt=dt,
                t0=t0,
                name=full_name,
                channel=full_name,
            )
            ts = apply_unit(ts, unit)
            tsd[full_name] = ts

    set_provenance(
        tsd,
        {
            "format": "tdms",
            "channels": list(tsd.keys()),
        },
    )
    return tsd


def read_timeseries_tdms(*args, channels=None, **kwargs) -> TimeSeries:
    tsd = read_timeseriesdict_tdms(*args, channels=channels, **kwargs)
    if not tsd:
        raise ValueError("No channels found in TDMS file")
    return tsd[next(iter(tsd.keys()))]


def read_timeseriesmatrix_tdms(*args, channels=None, **kwargs) -> TimeSeriesMatrix:
    tsd = read_timeseriesdict_tdms(*args, channels=channels, **kwargs)
    return tsd.to_matrix()


# -- Registration
io_registry.register_reader("tdms", TimeSeriesDict, read_timeseriesdict_tdms)
io_registry.register_reader("tdms", TimeSeries, read_timeseries_tdms)
io_registry.register_reader("tdms", TimeSeriesMatrix, read_timeseriesmatrix_tdms)

io_registry.register_identifier("tdms", TimeSeriesDict, lambda *args, **kwargs: str(args[1]).lower().endswith(".tdms"))
io_registry.register_identifier("tdms", TimeSeries, lambda *args, **kwargs: str(args[1]).lower().endswith(".tdms"))

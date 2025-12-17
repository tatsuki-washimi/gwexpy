"""
Readers for ObsPy-supported seismic formats (miniSEED, SAC).
"""

from __future__ import annotations

import numpy as np

from gwpy.io import registry as io_registry

from gwexpy.io.utils import (
    apply_unit,
    datetime_to_gps,
    ensure_datetime,
    parse_timezone,
    set_provenance,
)
from ..timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


def _import_obspy():
    try:
        import obspy  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "ObsPy is required for reading miniSEED/SAC files. "
            "Install with `pip install obspy`."
        ) from exc
    return obspy


def _trace_to_timeseries(trace, *, unit, timezone, epoch_override):
    tzinfo = parse_timezone(timezone) if timezone else None
    default_tz = tzinfo or parse_timezone("UTC")
    if epoch_override is not None:
        gps_start = (
            float(epoch_override)
            if isinstance(epoch_override, (int, float, np.floating))
            else datetime_to_gps(ensure_datetime(epoch_override, tzinfo=tzinfo))
        )
    else:
        start_dt = trace.stats.starttime.datetime.replace(tzinfo=tzinfo or None)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=default_tz)
        gps_start = datetime_to_gps(start_dt)
    ts = TimeSeries(
        trace.data.astype(float),
        t0=gps_start,
        dt=trace.stats.delta,
        channel=str(trace.id),
        name=str(trace.id),
    )
    return apply_unit(ts, unit)


def _read_obspy_stream(format_name, source, *, pad=np.nan, gap="pad", **kwargs):
    obspy = _import_obspy()
    stream = obspy.read(source, format=format_name, **kwargs)
    gaps = stream.get_gaps()
    if gaps and gap == "raise":
        raise ValueError(f"Gaps detected in {format_name} data: {gaps}")
    stream.merge(method=1, fill_value=pad)
    return stream


def _build_dict(stream, *, channels, unit, timezone, epoch):
    traces = stream
    if channels:
        selected = []
        wanted = set(channels)
        for tr in traces:
            if tr.id in wanted or tr.stats.channel in wanted:
                selected.append(tr)
        traces = selected
    tsd = TimeSeriesDict()
    for tr in traces:
        ts = _trace_to_timeseries(tr, unit=unit, timezone=timezone, epoch_override=epoch)
        tsd[ts.channel] = ts
    return tsd


def _read_timeseriesdict(format_name, source, *, channels=None, unit=None, epoch=None, timezone=None, pad=np.nan, gap="pad", **kwargs):
    stream = _read_obspy_stream(format_name, source, pad=pad, gap=gap, **kwargs)
    tsd = _build_dict(stream, channels=channels, unit=unit, timezone=timezone, epoch=epoch)
    set_provenance(
        tsd,
        {
            "format": format_name,
            "gap": "pad",
            "pad_value": pad,
            "timezone": timezone,
            "unit_source": "override" if unit else "file",
            "channels": list(channels) if channels else [tr.id for tr in stream],
        },
    )
    return tsd


def read_miniseed_timeseriesdict(*args, **kwargs):
    return _read_timeseriesdict("MSEED", *args, **kwargs)


def read_sac_timeseriesdict(*args, **kwargs):
    return _read_timeseriesdict("SAC", *args, **kwargs)


def _adapt_timeseries(reader, *args, channels=None, **kwargs):
    tsd = reader(*args, channels=channels, **kwargs)
    if not tsd:
        raise ValueError("No channels found in input file")
    return tsd[next(iter(tsd.keys()))]


def _adapt_matrix(reader, *args, channels=None, **kwargs):
    tsd = reader(*args, channels=channels, **kwargs)
    return tsd.to_matrix()


# -- registration
io_registry.register_reader("miniseed", TimeSeriesDict, read_miniseed_timeseriesdict)
io_registry.register_reader(
    "miniseed", TimeSeries, lambda *a, **k: _adapt_timeseries(read_miniseed_timeseriesdict, *a, **k)
)
io_registry.register_reader(
    "miniseed", TimeSeriesMatrix, lambda *a, **k: _adapt_matrix(read_miniseed_timeseriesdict, *a, **k)
)

io_registry.register_reader("sac", TimeSeriesDict, read_sac_timeseriesdict)
io_registry.register_reader(
    "sac", TimeSeries, lambda *a, **k: _adapt_timeseries(read_sac_timeseriesdict, *a, **k)
)
io_registry.register_reader(
    "sac", TimeSeriesMatrix, lambda *a, **k: _adapt_matrix(read_sac_timeseriesdict, *a, **k)
)

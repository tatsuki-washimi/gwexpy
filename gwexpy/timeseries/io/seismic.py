"""
Readers/Writers for ObsPy-supported seismic formats (miniSEED, SAC, GSE2, KNET).
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
from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


def _import_obspy():
    try:
        import obspy  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "ObsPy is required for reading seismic files. "
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
        # ObsPy UTCDateTime to python datetime
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
    try:
        stream = obspy.read(source, format=format_name, **kwargs)
    except (OSError, TypeError, ValueError) as exc:
        # Fallback: try reading without format specifier if specifically WIN/KNET fails obscurely
        # but usually we want to respect the format_name.
        raise ValueError(f"Failed to read {format_name} file: {exc}") from exc

    gaps = stream.get_gaps()
    if gaps and gap == "raise":
        raise ValueError(f"Gaps detected in {format_name} data: {gaps}")

    # Merge traces if necessary (handle gaps/overlaps)
    if pad is not None:
        try:
            if np.isnan(pad):
                for tr in stream:
                    if not np.issubdtype(tr.data.dtype, np.floating):
                        tr.data = tr.data.astype(float)
        except (TypeError, ValueError):
            pass

    stream.merge(method=1, fill_value=pad)
    return stream


def _build_dict(stream, *, channels, unit, timezone, epoch):
    traces = stream
    if channels:
        selected = []
        wanted = set(channels)
        for tr in traces:
            # Check against trace ID (Network.Station.Location.Channel) or just Channel
            if tr.id in wanted or tr.stats.channel in wanted:
                selected.append(tr)
        traces = selected

    tsd = TimeSeriesDict()
    for tr in traces:
        ts = _trace_to_timeseries(tr, unit=unit, timezone=timezone, epoch_override=epoch)
        # Handle duplicate channels if necessary, currently overwrites
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


# -- Specific Readers
def read_miniseed_timeseriesdict(*args, **kwargs):
    return _read_timeseriesdict("MSEED", *args, **kwargs)

def read_sac_timeseriesdict(*args, **kwargs):
    return _read_timeseriesdict("SAC", *args, **kwargs)

def read_gse2_timeseriesdict(*args, **kwargs):
    return _read_timeseriesdict("GSE2", *args, **kwargs)

def read_knet_timeseriesdict(*args, **kwargs):
    return _read_timeseriesdict("KNET", *args, **kwargs)


# -- Adaptors for Single/Matrix
def _adapt_timeseries(reader, *args, channels=None, **kwargs):
    tsd = reader(*args, channels=channels, **kwargs)
    if not tsd:
        raise ValueError("No channels found in input file")
    # Return the first channel found
    return tsd[next(iter(tsd.keys()))]

def _adapt_matrix(reader, *args, channels=None, **kwargs):
    tsd = reader(*args, channels=channels, **kwargs)
    return tsd.to_matrix()


# -- Writers (via ObsPy)
def _write_obspy(tsd, target, format_name, **kwargs):
    """Write TimeSeriesDict to file using ObsPy"""
    from ..interop.obspy_ import to_obspy_trace
    obspy = _import_obspy()

    stream = obspy.Stream()
    for key, ts in tsd.items():
        tr = to_obspy_trace(ts)
        stream.append(tr)

    stream.write(target, format=format_name, **kwargs)

def write_miniseed(tsd, target, **kwargs):
    _write_obspy(tsd, target, "MSEED", **kwargs)

def write_sac(tsd, target, **kwargs):
    # SAC does not support multi-trace in one file standardly like MSEED,
    # but ObsPy handles writing multiple files if target is likely a pattern or directory,
    # or raises error. We wrap simply here.
    _write_obspy(tsd, target, "SAC", **kwargs)

def write_gse2(tsd, target, **kwargs):
    _write_obspy(tsd, target, "GSE2", **kwargs)


# -- Registration

# MINISEED
io_registry.register_reader("miniseed", TimeSeriesDict, read_miniseed_timeseriesdict)
io_registry.register_reader("miniseed", TimeSeries, lambda *a, **k: _adapt_timeseries(read_miniseed_timeseriesdict, *a, **k))
io_registry.register_reader("miniseed", TimeSeriesMatrix, lambda *a, **k: _adapt_matrix(read_miniseed_timeseriesdict, *a, **k))
io_registry.register_writer("miniseed", TimeSeriesDict, write_miniseed)
io_registry.register_writer("miniseed", TimeSeries, lambda ts, f, **k: write_miniseed({ts.name: ts}, f, **k))

# SAC
io_registry.register_reader("sac", TimeSeriesDict, read_sac_timeseriesdict)
io_registry.register_reader("sac", TimeSeries, lambda *a, **k: _adapt_timeseries(read_sac_timeseriesdict, *a, **k))
io_registry.register_reader("sac", TimeSeriesMatrix, lambda *a, **k: _adapt_matrix(read_sac_timeseriesdict, *a, **k))
io_registry.register_writer("sac", TimeSeriesDict, write_sac)
io_registry.register_writer("sac", TimeSeries, lambda ts, f, **k: write_sac({ts.name: ts}, f, **k))

# GSE2
io_registry.register_reader("gse2", TimeSeriesDict, read_gse2_timeseriesdict)
io_registry.register_reader("gse2", TimeSeries, lambda *a, **k: _adapt_timeseries(read_gse2_timeseriesdict, *a, **k))
io_registry.register_reader("gse2", TimeSeriesMatrix, lambda *a, **k: _adapt_matrix(read_gse2_timeseriesdict, *a, **k))
io_registry.register_writer("gse2", TimeSeriesDict, write_gse2)
io_registry.register_writer("gse2", TimeSeries, lambda ts, f, **k: write_gse2({ts.name: ts}, f, **k))


# KNET (Read only typically)
io_registry.register_reader("knet", TimeSeriesDict, read_knet_timeseriesdict)
io_registry.register_reader("knet", TimeSeries, lambda *a, **k: _adapt_timeseries(read_knet_timeseriesdict, *a, **k))
io_registry.register_reader("knet", TimeSeriesMatrix, lambda *a, **k: _adapt_matrix(read_knet_timeseriesdict, *a, **k))



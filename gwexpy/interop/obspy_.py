from collections.abc import Mapping

import numpy as np

from ._optional import require_optional
from ._time import datetime_utc_to_gps, gps_to_datetime_utc

# -----------------------------------------------------------------------------
# TimeSeries
# -----------------------------------------------------------------------------


def _ts_to_obspy_trace(ts, stats_extra=None, dtype=None):
    obspy = require_optional("obspy")
    from .base import to_plain_array

    data = to_plain_array(ts)
    if dtype:
        data = data.astype(dtype)

    stats = {
        "starttime": gps_to_datetime_utc(ts.t0, leap="raise"),
        "sampling_rate": ts.sample_rate.value,
        "npts": len(data),
    }

    # Map name/channel
    if ts.channel:
        stats["channel"] = (
            str(ts.channel.name) if hasattr(ts.channel, "name") else str(ts.channel)
        )

    # Use name as station if available, strictly speaking Obspy station is short code
    if ts.name:
        stats["station"] = str(ts.name)

    if stats_extra:
        stats.update(stats_extra)

    return obspy.Trace(data=data, header=stats)


def to_obspy_trace(ts, stats_extra=None, dtype=None):
    """Legacy alias for internal use, prefers using new generic to_obspy."""
    return _ts_to_obspy_trace(ts, stats_extra, dtype)


def _from_obspy_trace_to_ts(cls, tr, unit=None, name_policy="id"):
    data = tr.data
    stats = tr.stats

    t0 = datetime_utc_to_gps(stats.starttime.datetime)
    dt = 1.0 / stats.sampling_rate

    name = tr.id if name_policy == "id" else stats.channel

    return cls(data, t0=t0, dt=dt, unit=unit, name=name)


def from_obspy_trace(cls, tr, unit=None, name_policy="id"):
    """Legacy alias."""
    require_optional("obspy")
    return _from_obspy_trace_to_ts(cls, tr, unit, name_policy)


# -----------------------------------------------------------------------------
# FrequencySeries
# -----------------------------------------------------------------------------


def _fs_to_obspy_trace(fs, stats_extra=None, **kwargs):
    """
    Convert FrequencySeries to Obspy Trace.
    Treats frequency axis as time axis.
    """
    obspy = require_optional("obspy")
    from .base import to_plain_array

    data = to_plain_array(fs)
    stats = {
        "delta": fs.df.value,
        "starttime": fs.f0.value if hasattr(fs, "f0") else 0.0,
        "npts": len(data),
    }

    if fs.name:
        stats["station"] = str(fs.name)

    if stats_extra:
        stats.update(stats_extra)

    # Mark strictly as frequency data if possible?
    # Obspy doesn't have a standard field for this.

    return obspy.Trace(data=data, header=stats)


def _trace_to_fs(cls, tr, unit=None, name_policy="id", **kwargs):
    """
    Convert Obspy Trace to FrequencySeries.
    Assumes Trace contains spectral data where delta = df.
    """
    data = tr.data
    stats = tr.stats

    df = stats.delta
    # starttime is UTCDateTime, we might want float if it was stored as float (0.0)
    # But Obspy always converts to UTCDateTime.
    # If we stored f0 as starttime, we retrieve timestamp.
    # If using default 1970-01-01, checking usage...
    # We will assume starttime timestamp represents f0 value.
    f0 = stats.starttime.timestamp

    name = tr.id if name_policy == "id" else stats.channel

    return cls(data, df=df, f0=f0, unit=unit, name=name)


# -----------------------------------------------------------------------------
# Spectrogram
# -----------------------------------------------------------------------------


def _spec_to_obspy_stream(spec, **kwargs):
    """
    Convert Spectrogram to Obspy Stream (Filter Bank).
    Each frequency bin becomes a Trace.
    """
    obspy = require_optional("obspy")
    from .base import to_plain_array

    st = obspy.Stream()

    # spec data is (freq, time) usually in gwexpy?
    # Check Spectrogram implementation. Spectrogram is usually (frequencies, times) or (times, frequencies)?
    # gwexpy Spectrogram is typically (frequency, time) like FrequencySeries array ??
    # Actually Spectrogram inherits from SeriesMatrix? or Array2D?
    # Let's check Spectrogram.value.shape vs axes.
    # Usually: axis 0 = frequencies, axis 1 = times.

    data = to_plain_array(spec)
    # If data is (freqs, times)

    freqs = spec.frequencies.value
    t0 = spec.t0
    dt = spec.dt

    station = str(spec.name) if spec.name else "SPEC"

    # Ensure t0 is properly formatted for Obspy
    starttime = gps_to_datetime_utc(t0, leap="raise")

    for i, freq in enumerate(freqs):
        # Time series for this frequency bin
        # data[:, i] because gwexpy Spectrogram is (n_times, n_freqs)

        tr_data = data[:, i].astype(np.float64)  # Obspy likes float64 or 32

        stats = {
            "network": "",
            "station": station,
            "location": "",
            "channel": f"F{i:04d}",  # Fake channel name
            "starttime": starttime,
            "sampling_rate": 1.0 / dt.value,
            "npts": len(tr_data),
        }

        # Store metadata
        # stats['frequency'] = freq  # standard obspy stats doesn't support custom keys in init??
        # It allows assignment after creation or via AttribDict in header

        tr = obspy.Trace(data=tr_data, header=stats)
        tr.stats.frequency = freq  # Custom header
        st.append(tr)

    return st


def _stream_to_spec(cls, st, unit=None, name_policy="id", **kwargs):
    # Reconstruct Spectrogram from Stream
    # Assumes all traces have same starttime, sampling_rate, npts
    if len(st) == 0:
        raise ValueError("Empty stream")

    st.sort(keys=["channel"])  # Sort by channel (F0000, F0001...) or ideally frequency

    # Extract common stats
    tr0 = st[0]
    t0 = datetime_utc_to_gps(tr0.stats.starttime.datetime)
    dt = tr0.stats.delta

    # Collect data and freqs
    data_list = []
    freqs = []

    for tr in st:
        if hasattr(tr.stats, "frequency"):
            f = tr.stats.frequency
        else:
            # Try to guess from channel name or index?
            # If we don't have frequency info, we can't reconstruct properly without guessing.
            # Assuming linear spacing if missing?
            f = 0.0  # Placeholder

        data_list.append(tr.data)
        freqs.append(f)

    data = np.stack(data_list).T  # (n_times, n_freqs)

    # If frequencies were not stored, create dummy
    if all(f == 0.0 for f in freqs):
        freqs = np.arange(len(freqs), dtype=float)

    name = tr0.stats.station

    return cls(data, t0=t0, dt=dt, frequencies=freqs, unit=unit, name=name)


# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------


def to_obspy(data, **kwargs):
    """
    Convert gwexpy object to Obspy object.

    FrequencySeries -> Trace (Spectrum mode)
    Spectrogram     -> Stream (Filter Bank mode)
    TimeSeries      -> Trace
    """
    # Import locally to identify types if necessary, or check attributes

    # Check for Spectrogram first (has frequencies and times)
    is_spec = hasattr(data, "frequencies") and hasattr(data, "times")
    if is_spec:
        return _spec_to_obspy_stream(data, **kwargs)

    # Check for FrequencySeries (has frequencies)
    is_fs = hasattr(data, "frequencies")
    if is_fs:
        return _fs_to_obspy_trace(data, **kwargs)

    # Check for TimeSeries (has t0, dt, but NO frequencies)
    is_ts = hasattr(data, "t0") and hasattr(data, "dt")
    if is_ts:
        return _ts_to_obspy_trace(data, **kwargs)

    # If dict map
    if isinstance(data, Mapping):
        # Recurse or handle dict?
        # Obspy Stream can hold multiple traces.
        # If dict of TimeSeries -> Stream
        obspy = require_optional("obspy")
        st = obspy.Stream()
        for k, v in data.items():
            # If v is TimeSeries, to_obspy returns Trace.
            res = to_obspy(v, **kwargs)
            if isinstance(res, obspy.Trace):
                # Update ID from key if needed?
                st.append(res)
            elif isinstance(res, obspy.Stream):
                st += res
        return st

    raise TypeError(f"Unsupported type for to_obspy: {type(data)}")


def from_obspy(cls, data, **kwargs):
    """
    Convert Obspy object to gwexpy object of type cls.
    """
    obspy = require_optional("obspy")

    if isinstance(data, obspy.Trace):
        # Target cls determines interpretation
        if hasattr(cls, "frequencies"):  # FrequencySeries?
            # How to check if cls is FrequencySeries class without importing?
            # Check name or check known classes if imported
            if "FrequencySeries" in cls.__name__:
                return _trace_to_fs(cls, data, **kwargs)
            elif "Spectrogram" in cls.__name__:
                raise ValueError("Cannot convert single Trace to Spectrogram directly.")
            else:
                # Default to TimeSeries
                return _from_obspy_trace_to_ts(cls, data, **kwargs)

    if isinstance(data, obspy.Stream):
        if "Spectrogram" in cls.__name__:
            return _stream_to_spec(cls, data, **kwargs)
        # If Stream and target is TimeSeriesDict or similar?
        # Implementing basic Spectrogram support first.

    raise TypeError(f"Unsupported conversion: {type(data)} -> {cls}")

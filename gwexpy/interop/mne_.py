from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from ._optional import require_optional


def _infer_sfreq_hz(ts: Any) -> float:
    sample_rate = getattr(ts, "sample_rate", None)
    if sample_rate is not None:
        try:
            return float(sample_rate.to("Hz").value)
        except (AttributeError, TypeError):
            return float(getattr(sample_rate, "value", sample_rate))

    dt = getattr(ts, "dt", None)
    if dt is not None:
        try:
            dt_s = float(dt.to("s").value)
        except (AttributeError, TypeError):
            dt_s = float(getattr(dt, "value", dt))
        if dt_s == 0:
            raise ValueError("Cannot infer sampling frequency from dt=0")
        return 1.0 / dt_s

    raise ValueError("Cannot infer sampling frequency (missing sample_rate/dt)")


def _default_ch_name(ts: Any, *, fallback: str) -> str:
    name = getattr(ts, "name", None)
    if isinstance(name, str) and name:
        return name
    channel = getattr(ts, "channel", None)
    if channel is not None:
        return str(channel)
    return fallback


def _select_items(items: list[tuple[Any, Any]], picks: Optional[Any]) -> list[tuple[Any, Any]]:
    if picks is None:
        return items
    if isinstance(picks, (str, int)):
        picks = [picks]

    if not isinstance(picks, Sequence):
        raise TypeError("picks must be a sequence of channel names or indices")

    if all(isinstance(p, str) for p in picks):
        pick_set = set(picks)
        return [(k, v) for (k, v) in items if str(k) in pick_set]

    indices = [int(p) for p in picks]
    return [items[i] for i in indices]


def to_mne_rawarray(tsd, info=None, picks=None):
    """
    Convert a TimeSeries-like object to an ``mne.io.RawArray``.

    Parameters
    ----------
    tsd
        ``TimeSeriesDict``-like mapping (multi-channel) or a single ``TimeSeries``.
    info
        Optional MNE ``Info``. If omitted, a minimal ``Info`` is created.
    picks
        Optional channel selection (names or indices). Only supported for mapping inputs.
    """
    mne = require_optional("mne")

    # Single-channel input
    if not isinstance(tsd, Mapping):
        if picks is not None:
            raise TypeError("picks is only supported for mapping inputs")

        from .base import to_plain_array
        data_1d = to_plain_array(tsd)
        if data_1d.ndim != 1:
            raise ValueError("Single-channel input must be 1D")

        ch_name = _default_ch_name(tsd, fallback="ch0")
        sfreq = _infer_sfreq_hz(tsd)

        if info is None:
            info = mne.create_info(ch_names=[ch_name], sfreq=sfreq, ch_types=["misc"])
        elif int(info["nchan"]) != 1:
            raise ValueError(f"info expects nchan=1, got {info['nchan']}")

        return mne.io.RawArray(data_1d[None, :], info)

    # Multi-channel mapping input
    items = _select_items(list(tsd.items()), picks)
    if not items:
        raise ValueError("No channels selected")

    ch_names = [str(k) for (k, _) in items]
    series = [v for (_, v) in items]

    sfreq = _infer_sfreq_hz(series[0])
    for ts in series[1:]:
        if not np.isclose(_infer_sfreq_hz(ts), sfreq):
            raise ValueError("All channels must share the same sampling frequency")

    lengths = {len(ts) for ts in series}
    if len(lengths) == 1:
        data = np.stack([np.asarray(ts.value) for ts in series], axis=0)
    elif hasattr(tsd, "to_matrix"):
        try:
            tsd_sel = tsd.__class__()  # TimeSeriesDict-like
            for k, ts in items:
                 tsd_sel[k] = ts
            from .base import to_plain_array
            mat = tsd_sel.to_matrix()
            data = to_plain_array(mat)
            if data.ndim == 3:
                data = data[:, 0, :]
            if data.shape[0] != len(ch_names):
                raise ValueError("Unexpected channel dimension after alignment")
            ch_names = list(getattr(mat, "channel_names", ch_names))
        except (ValueError, TypeError, AttributeError, IndexError, KeyError) as e:
            raise ValueError(
                "Channels have mismatched lengths and could not be aligned via to_matrix()"
            ) from e
    else:
        raise ValueError(
            "All channels must have the same length (or provide a TimeSeriesDict with to_matrix() for alignment)"
        )

    if info is None:
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["misc"] * len(ch_names))
    elif int(info["nchan"]) != len(ch_names):
        raise ValueError(f"info expects nchan={len(ch_names)}, got {info['nchan']}")

    return mne.io.RawArray(data, info)
    
def from_mne_raw(cls, raw, unit_map=None):
    """
    raw: mne.io.Raw
    """
    data, times = raw.get_data(return_times=True)
    # data: (n_ch, n_times)
    # times: (n_times,) relative to first sample usually 0
    
    ch_names = raw.ch_names
    sfreq = raw.info['sfreq']
    dt = 1.0 / sfreq
    
    # meas_date check for t0?
    t0 = 0
    if raw.info['meas_date']:
        # datetime to gps
        from ._time import datetime_utc_to_gps
        # meas_date is datetime (aware UTC usually)
        t0 = datetime_utc_to_gps(raw.info['meas_date'])
        
    tsd = cls()
    for i, name in enumerate(ch_names):
        tsd[name] = tsd.EntryClass(data[i], t0=t0, dt=dt, name=name)
        
    return tsd

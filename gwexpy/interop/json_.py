from __future__ import annotations

import json

import numpy as np


def to_json(ts):
    """Convert TimeSeries to a JSON string.

    Includes data and basic metadata.
    """
    data_dict = to_dict(ts)
    return json.dumps(data_dict, indent=2, default=str)


def to_dict(ts):
    """Convert a `TimeSeries` to a dictionary."""
    from .base import to_plain_array

    data = to_plain_array(ts)

    meta = {
        "t0": float(ts.t0.value) if hasattr(ts.t0, "value") else float(ts.t0),
        "dt": float(ts.dt.value) if hasattr(ts.dt, "value") else float(ts.dt),
        "unit": str(ts.unit),
        "name": str(ts.name) if ts.name else None,
        # Persist ``channel`` so it survives a to_dict/from_dict round-trip
        # (guard against the falsy/None channel to avoid writing the literal
        # string "None", which would round-trip back as Channel("None")).
        "channel": str(ts.channel) if ts.channel else None,
        "data": data.tolist(),
    }
    return meta


def from_json(cls, json_str, **kwargs):
    """Create a `TimeSeries` from a JSON string.

    Extra keyword arguments are forwarded to :func:`from_dict` so callers can
    override metadata (``unit``/``channel``/``name``/``t0``/``dt``) absent from
    the JSON payload.
    """
    data_dict = json.loads(json_str)
    return from_dict(cls, data_dict, **kwargs)


def from_dict(cls, data_dict, *, unit=None, channel=None, name=None, t0=None, dt=None):
    """Create a `TimeSeries` from a dictionary.

    Metadata not present in ``data_dict`` can be supplied explicitly via the
    keyword arguments; an explicit argument always takes priority over a value
    stored in the dictionary (``user > source``). Missing ``t0``/``dt`` fall
    back to ``0``/``1`` with a :class:`UserWarning` rather than silently.
    """
    from .base import resolve_meta, resolve_timing

    data = np.array(data_dict["data"])

    final_t0, final_dt = resolve_timing(
        t0,
        dt,
        source="dict",
        inferred_t0=data_dict.get("t0"),
        inferred_dt=data_dict.get("dt"),
    )
    final_unit = resolve_meta(unit, data_dict.get("unit"))
    final_name = resolve_meta(name, data_dict.get("name"))
    final_channel = resolve_meta(channel, data_dict.get("channel"))

    return cls(
        data,
        t0=final_t0,
        dt=final_dt,
        unit=final_unit,
        name=final_name,
        channel=final_channel,
    )

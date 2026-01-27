from __future__ import annotations

import json

import numpy as np


def to_json(ts):
    """
    Convert TimeSeries to a JSON string.
    Includes data and basic metadata.
    """
    data_dict = to_dict(ts)
    return json.dumps(data_dict, indent=2, default=str)


def to_dict(ts):
    """
    Convert TimeSeries to a dictionary.
    """
    from .base import to_plain_array

    data = to_plain_array(ts)

    meta = {
        "t0": float(ts.t0.value) if hasattr(ts.t0, "value") else float(ts.t0),
        "dt": float(ts.dt.value) if hasattr(ts.dt, "value") else float(ts.dt),
        "unit": str(ts.unit),
        "name": str(ts.name) if ts.name else None,
        "data": data.tolist(),
    }
    return meta


def from_json(cls, json_str):
    """
    Create TimeSeries from a JSON string.
    """
    data_dict = json.loads(json_str)
    return from_dict(cls, data_dict)


def from_dict(cls, data_dict):
    """
    Create TimeSeries from a dictionary.
    """
    data = np.array(data_dict["data"])
    t0 = data_dict.get("t0", 0)
    dt = data_dict.get("dt", 1)
    unit = data_dict.get("unit")
    name = data_dict.get("name")

    return cls(data, t0=t0, dt=dt, unit=unit, name=name)

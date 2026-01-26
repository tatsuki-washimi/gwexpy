from __future__ import annotations

import numpy as np


def normalize_series(val) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Normalize various data types (TimeSeries, FrequencySeries, dict) into (x, y) ndarrays.
    """
    if val is None:
        return None

    data = None
    x_axis = None

    # Check for GWEXPY/GWpy objects
    if hasattr(val, "value"):
        data = val.value
        if hasattr(val, "times"):  # TimeSeries
            x_axis = val.times.value
        elif hasattr(val, "frequencies"):  # FrequencySeries
            x_axis = val.frequencies.value
        elif hasattr(val, "xindex"):  # Generic Series
            x_axis = val.xindex.value

    elif isinstance(val, dict):  # Fallback Dictionary
        data = val.get("data") or val.get("value")
        if data is None:
            return None

        freqs = val.get("frequencies")
        dt = val.get("dt")

        if dt is not None:
            n = len(data)
            t0 = val.get("epoch") or 0
            x_axis = np.arange(n) * dt + t0
        elif freqs is not None and len(freqs) > 0:
            x_axis = freqs
        elif val.get("df") is not None:
            n = len(data)
            x_axis = np.arange(n) * val["df"]
        else:
            x_axis = np.arange(len(data))

    if data is not None and x_axis is not None:
        return np.asarray(x_axis), np.asarray(data)

    return None

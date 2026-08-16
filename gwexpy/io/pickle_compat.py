from __future__ import annotations

"""
Pickle / shelve compatibility helpers.

Design goal
-----------
Enable "Level2" portability: objects pickled from gwexpy should be unpicklable
in an environment that has **GWpy** installed but does **not** have gwexpy.

Security
--------
Never unpickle data from untrusted sources. ``pickle`` / ``shelve`` can execute
arbitrary code during loading.
"""

from collections.abc import Callable
from typing import Any

import numpy as np


def _build_gwpy_timeseries(data, kwargs: dict[str, Any]):
    from gwpy.timeseries import TimeSeries

    from gwexpy.timeseries.utils import _validate_t0_gps_ns

    t0_gps_ns = kwargs.pop("_gwex_t0_gps_ns", None)
    t0_gps_precision = kwargs.pop("_gwex_t0_gps_precision", None)
    if t0_gps_ns is None:
        if t0_gps_precision is not None:
            raise ValueError("GPS precision state requires a GPS nanosecond state")
    else:
        t0_gps_ns = _validate_t0_gps_ns(t0_gps_ns)
        if t0_gps_precision not in ("exact", "quantized"):
            raise ValueError("invalid GPS nanosecond precision state")
    series = TimeSeries(data, **kwargs)
    if t0_gps_ns is not None:
        series._gwex_t0_gps_ns = t0_gps_ns
        series._gwex_t0_gps_precision = t0_gps_precision
    return series


def _build_gwpy_frequencyseries(data, kwargs: dict[str, Any]):
    from gwpy.frequencyseries import FrequencySeries

    return FrequencySeries(data, **kwargs)


def _build_gwpy_spectrogram(data, kwargs: dict[str, Any]):
    from gwpy.spectrogram import Spectrogram

    from gwexpy.provenance import copy_provenance

    provenance = kwargs.pop("_gwex_provenance", None)
    spectrogram = Spectrogram(data, **kwargs)
    if provenance is not None:
        spectrogram.provenance = copy_provenance(provenance)
    return spectrogram


def _series_kwargs(series) -> dict[str, Any]:
    return {
        "unit": getattr(series, "unit", None),
        "name": getattr(series, "name", None),
        "channel": getattr(series, "channel", None),
        "epoch": getattr(series, "epoch", None),
    }


def timeseries_reduce_args(
    ts,
) -> tuple[Callable[[Any, dict[str, Any]], Any], tuple[Any, dict[str, Any]]]:
    """Return pickle reduce arguments for a GWpy-compatible time series."""
    from gwexpy.timeseries.utils import _validate_t0_gps_ns

    t0_gps_ns = getattr(ts, "_gwex_t0_gps_ns", None)
    t0_gps_precision = getattr(ts, "_gwex_t0_gps_precision", None)
    validated_t0_gps_ns = None
    if t0_gps_ns is not None:
        validated_t0_gps_ns = _validate_t0_gps_ns(t0_gps_ns)
        if t0_gps_precision not in ("exact", "quantized"):
            raise ValueError("invalid GPS nanosecond precision state")
    elif t0_gps_precision is not None:
        raise ValueError("GPS precision state requires a GPS nanosecond state")

    kwargs = _series_kwargs(ts)
    times = getattr(ts, "times", None)
    if times is not None:
        kwargs["times"] = times
    else:
        kwargs["t0"] = getattr(ts, "t0", None)
        kwargs["dt"] = getattr(ts, "dt", None)
    data = np.asarray(ts.value)
    if validated_t0_gps_ns is not None:
        kwargs["_gwex_t0_gps_ns"] = validated_t0_gps_ns
        kwargs["_gwex_t0_gps_precision"] = t0_gps_precision
    return _build_gwpy_timeseries, (data, kwargs)


def frequencyseries_reduce_args(
    fs,
) -> tuple[Callable[[Any, dict[str, Any]], Any], tuple[Any, dict[str, Any]]]:
    """Return pickle reduce arguments for a GWpy-compatible frequency series."""
    kwargs = _series_kwargs(fs)
    freqs = getattr(fs, "frequencies", None)
    if freqs is not None:
        kwargs["frequencies"] = freqs
    else:
        kwargs["f0"] = getattr(fs, "f0", None)
        kwargs["df"] = getattr(fs, "df", None)
    data = np.asarray(fs.value)
    return _build_gwpy_frequencyseries, (data, kwargs)


def spectrogram_reduce_args(
    sg,
) -> tuple[Callable[[Any, dict[str, Any]], Any], tuple[Any, dict[str, Any]]]:
    """Return pickle reduce arguments for a GWpy-compatible spectrogram."""
    kwargs = _series_kwargs(sg)
    kwargs["times"] = getattr(sg, "times", None)
    kwargs["frequencies"] = getattr(sg, "frequencies", None)
    data = np.asarray(sg.value)
    provenance = getattr(sg, "provenance", None)
    if provenance is not None:
        from gwexpy.provenance import copy_provenance

        kwargs["_gwex_provenance"] = copy_provenance(provenance)
    return _build_gwpy_spectrogram, (data, kwargs)

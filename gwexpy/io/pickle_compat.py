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

from typing import Any, Callable

import numpy as np


def _build_gwpy_timeseries(data, kwargs: dict[str, Any]):
    from gwpy.timeseries import TimeSeries

    return TimeSeries(data, **kwargs)


def _build_gwpy_frequencyseries(data, kwargs: dict[str, Any]):
    from gwpy.frequencyseries import FrequencySeries

    return FrequencySeries(data, **kwargs)


def _build_gwpy_spectrogram(data, kwargs: dict[str, Any]):
    from gwpy.spectrogram import Spectrogram

    return Spectrogram(data, **kwargs)


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
    kwargs = _series_kwargs(ts)
    times = getattr(ts, "times", None)
    if times is not None:
        kwargs["times"] = times
    else:
        kwargs["t0"] = getattr(ts, "t0", None)
        kwargs["dt"] = getattr(ts, "dt", None)
    data = np.asarray(ts.value)
    return _build_gwpy_timeseries, (data, kwargs)


def frequencyseries_reduce_args(
    fs,
) -> tuple[Callable[[Any, dict[str, Any]], Any], tuple[Any, dict[str, Any]]]:
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
    kwargs = _series_kwargs(sg)
    kwargs["times"] = getattr(sg, "times", None)
    kwargs["frequencies"] = getattr(sg, "frequencies", None)
    data = np.asarray(sg.value)
    return _build_gwpy_spectrogram, (data, kwargs)

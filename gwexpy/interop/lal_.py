"""
gwexpy.interop.lal_
-------------------

Interoperability with LALSuite (LIGO Algorithmic Library).

Provides bidirectional conversion between LALSuite time/frequency series
types and GWexpy TimeSeries / FrequencySeries.

Notes
-----
GWpy already inherits ``from_lal`` / ``to_lal`` classmethods for TimeSeries
and FrequencySeries via GWpy. This module provides an explicit interop layer
that ensures GWexpy types are returned and adds ``to_lal`` for FrequencySeries.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gwexpy.interop._registry import ConverterRegistry

from ._optional import require_optional

__all__ = [
    "from_lal_timeseries",
    "to_lal_timeseries",
    "from_lal_frequencyseries",
    "to_lal_frequencyseries",
]


def from_lal_timeseries(
    cls: type,
    lalts: Any,
    *,
    copy: bool = True,
) -> Any:
    """Create a GWexpy TimeSeries from a LAL TimeSeries struct.

    Parameters
    ----------
    cls : type
        The ``TimeSeries`` class to instantiate.
    lalts : lal.REAL4TimeSeries or lal.REAL8TimeSeries or lal.COMPLEX8TimeSeries or lal.COMPLEX16TimeSeries
        LAL time series struct.
    copy : bool, default True
        Whether to copy the underlying data array.

    Returns
    -------
    TimeSeries
        GWexpy TimeSeries with epoch, sample rate and unit from the LAL struct.

    Examples
    --------
    >>> import lal
    >>> from gwexpy.timeseries import TimeSeries
    >>> lalts = lal.CreateREAL8TimeSeries("test", lal.LIGOTimeGPS(0), 0, 1/1024, lal.DimensionlessUnit, 1024)
    >>> ts = TimeSeries.from_lal(lalts)
    """
    require_optional("lal")
    from gwexpy.utils.lal import from_lal_unit

    TimeSeries = ConverterRegistry.get_constructor("TimeSeries")

    data = np.array(lalts.data.data, copy=copy)
    epoch = float(lalts.epoch)
    dt = lalts.deltaT
    unit = from_lal_unit(lalts.sampleUnits)
    name = lalts.name or ""

    return TimeSeries(data, t0=epoch, dt=dt, unit=unit, name=name)


def to_lal_timeseries(
    ts: Any,
    *,
    dtype: str | None = None,
) -> Any:
    """Convert a GWexpy TimeSeries to a LAL TimeSeries struct.

    Parameters
    ----------
    ts : TimeSeries
        GWexpy TimeSeries to convert.
    dtype : str, optional
        LAL type string (e.g., ``"REAL8"``, ``"COMPLEX16"``).
        If *None*, inferred from the array dtype.

    Returns
    -------
    lal.REAL8TimeSeries or similar
        LAL time series struct.

    Examples
    --------
    >>> from gwexpy.timeseries import TimeSeries
    >>> import numpy as np
    >>> ts = TimeSeries(np.zeros(1024), t0=0, dt=1/1024, name="test")
    >>> lalts = ts.to_lal()
    """
    require_optional("lal")
    from gwexpy.utils.lal import (  # noqa: PLC0415
        LAL_TYPE_FROM_NUMPY,
        find_typed_function,
        to_lal_ligotimegps,
        to_lal_unit,
    )

    data = np.asarray(ts.value)
    if dtype is None:
        lal_type = LAL_TYPE_FROM_NUMPY.get(data.dtype.type)
        if lal_type is None:
            data = data.astype(np.float64)
            lal_type = LAL_TYPE_FROM_NUMPY[np.float64]
    else:
        lal_type = dtype.upper()

    create_fn = find_typed_function(lal_type, "Create", "TimeSeries")
    epoch = to_lal_ligotimegps(float(ts.t0.value))
    unit = to_lal_unit(ts.unit)
    lalts = create_fn(ts.name or "", epoch, 0, float(ts.dt.value), unit, len(data))
    lalts.data.data[:] = data
    return lalts


def from_lal_frequencyseries(
    cls: type,
    lalfs: Any,
    *,
    copy: bool = True,
) -> Any:
    """Create a GWexpy FrequencySeries from a LAL FrequencySeries struct.

    Parameters
    ----------
    cls : type
        The ``FrequencySeries`` class to instantiate.
    lalfs : lal.REAL8FrequencySeries or lal.COMPLEX16FrequencySeries
        LAL frequency series struct.
    copy : bool, default True
        Whether to copy the underlying data array.

    Returns
    -------
    FrequencySeries
        GWexpy FrequencySeries with f0, df, epoch and unit from the LAL struct.
    """
    require_optional("lal")
    from gwexpy.utils.lal import from_lal_unit

    FrequencySeries = ConverterRegistry.get_constructor("FrequencySeries")

    data = np.array(lalfs.data.data, copy=copy)
    f0 = float(lalfs.f0)
    df = lalfs.deltaF
    epoch = float(lalfs.epoch)
    unit = from_lal_unit(lalfs.sampleUnits)
    name = lalfs.name or ""

    n = len(data)
    frequencies = f0 + np.arange(n) * df

    return FrequencySeries(data, frequencies=frequencies, unit=unit, name=name, epoch=epoch)


def to_lal_frequencyseries(
    fs: Any,
) -> Any:
    """Convert a GWexpy FrequencySeries to a LAL FrequencySeries struct.

    Parameters
    ----------
    fs : FrequencySeries
        GWexpy FrequencySeries to convert.

    Returns
    -------
    lal.REAL8FrequencySeries or similar
        LAL frequency series struct.
    """
    require_optional("lal")
    from gwexpy.utils.lal import (  # noqa: PLC0415
        LAL_TYPE_FROM_NUMPY,
        find_typed_function,
        to_lal_ligotimegps,
        to_lal_unit,
    )

    data = np.asarray(fs.value)
    lal_type = LAL_TYPE_FROM_NUMPY.get(data.dtype.type)
    if lal_type is None:
        data = data.astype(np.float64)
        lal_type = LAL_TYPE_FROM_NUMPY[np.float64]

    create_fn = find_typed_function(lal_type, "Create", "FrequencySeries")
    freqs = np.asarray(fs.frequencies.value)
    f0 = float(freqs[0]) if len(freqs) > 0 else 0.0
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    epoch_val = getattr(fs, "epoch", None)
    epoch_gps = float(epoch_val.value) if epoch_val is not None else 0.0
    epoch = to_lal_ligotimegps(epoch_gps)
    unit = to_lal_unit(fs.unit)

    lalfs = create_fn(fs.name or "", epoch, f0, df, unit, len(data))
    lalfs.data.data[:] = data
    return lalfs

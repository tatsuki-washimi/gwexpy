"""
gwexpy.interop.pycbc_
---------------------

Interoperability with PyCBC (Python gravitational-wave data analysis library).

Provides bidirectional conversion between PyCBC time/frequency series
types and GWexpy TimeSeries / FrequencySeries.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gwexpy.interop._registry import ConverterRegistry

from ._optional import require_optional

__all__ = [
    "from_pycbc_timeseries",
    "to_pycbc_timeseries",
    "from_pycbc_frequencyseries",
    "to_pycbc_frequencyseries",
]


def from_pycbc_timeseries(
    cls: type,
    pycbc_ts: Any,
    *,
    copy: bool = True,
) -> Any:
    """Create a GWexpy TimeSeries from a PyCBC TimeSeries.

    Parameters
    ----------
    cls : type
        The ``TimeSeries`` class to instantiate.
    pycbc_ts : pycbc.types.TimeSeries
        PyCBC time series object.
    copy : bool, default True
        Whether to copy the underlying data array.

    Returns
    -------
    TimeSeries
        GWexpy TimeSeries with epoch, sample rate and unit from PyCBC.

    Examples
    --------
    >>> from pycbc.types import TimeSeries as PyCBCTimeSeries
    >>> import numpy as np
    >>> pycbc_ts = PyCBCTimeSeries(np.zeros(1024), delta_t=1/1024, epoch=0)
    >>> from gwexpy.timeseries import TimeSeries
    >>> ts = TimeSeries.from_pycbc(pycbc_ts)
    """
    require_optional("pycbc")

    TimeSeries = ConverterRegistry.get_constructor("TimeSeries")

    data = np.array(pycbc_ts.numpy(), copy=copy)
    t0 = float(pycbc_ts.start_time)
    dt = float(pycbc_ts.delta_t)

    # PyCBC carries unit as a string; convert to astropy-compatible form
    unit = getattr(pycbc_ts, "_unit", None) or ""

    return TimeSeries(data, t0=t0, dt=dt, unit=unit)


def to_pycbc_timeseries(
    ts: Any,
) -> Any:
    """Convert a GWexpy TimeSeries to a PyCBC TimeSeries.

    Parameters
    ----------
    ts : TimeSeries
        GWexpy TimeSeries to convert.

    Returns
    -------
    pycbc.types.TimeSeries
        PyCBC time series.

    Examples
    --------
    >>> from gwexpy.timeseries import TimeSeries
    >>> import numpy as np
    >>> ts = TimeSeries(np.zeros(1024), t0=0, dt=1/1024)
    >>> pycbc_ts = ts.to_pycbc()
    """
    require_optional("pycbc")
    from pycbc.types import TimeSeries as PyCBCTimeSeries  # noqa: PLC0415

    data = np.asarray(ts.value)
    dt = float(ts.dt.value)
    t0 = float(ts.t0.value)

    return PyCBCTimeSeries(data, delta_t=dt, epoch=t0)


def from_pycbc_frequencyseries(
    cls: type,
    pycbc_fs: Any,
    *,
    copy: bool = True,
) -> Any:
    """Create a GWexpy FrequencySeries from a PyCBC FrequencySeries.

    Parameters
    ----------
    cls : type
        The ``FrequencySeries`` class to instantiate.
    pycbc_fs : pycbc.types.FrequencySeries
        PyCBC frequency series object.
    copy : bool, default True
        Whether to copy the underlying data array.

    Returns
    -------
    FrequencySeries
        GWexpy FrequencySeries with df, epoch and unit from PyCBC.

    Examples
    --------
    >>> from pycbc.types import FrequencySeries as PyCBCFrequencySeries
    >>> import numpy as np
    >>> pycbc_fs = PyCBCFrequencySeries(np.zeros(512, dtype=complex), delta_f=1.0, epoch=0)
    >>> from gwexpy.frequencyseries import FrequencySeries
    >>> fs = FrequencySeries.from_pycbc(pycbc_fs)
    """
    require_optional("pycbc")

    FrequencySeries = ConverterRegistry.get_constructor("FrequencySeries")

    data = np.array(pycbc_fs.numpy(), copy=copy)
    df = float(pycbc_fs.delta_f)
    epoch = float(pycbc_fs.epoch)
    n = len(data)
    frequencies = np.arange(n) * df

    unit = getattr(pycbc_fs, "_unit", None) or ""

    return FrequencySeries(data, frequencies=frequencies, unit=unit, epoch=epoch)


def to_pycbc_frequencyseries(
    fs: Any,
) -> Any:
    """Convert a GWexpy FrequencySeries to a PyCBC FrequencySeries.

    Parameters
    ----------
    fs : FrequencySeries
        GWexpy FrequencySeries to convert.

    Returns
    -------
    pycbc.types.FrequencySeries
        PyCBC frequency series.

    Examples
    --------
    >>> from gwexpy.frequencyseries import FrequencySeries
    >>> import numpy as np
    >>> fs = FrequencySeries(np.zeros(512, dtype=complex), frequencies=np.arange(512))
    >>> pycbc_fs = fs.to_pycbc()
    """
    require_optional("pycbc")
    from pycbc.types import FrequencySeries as PyCBCFrequencySeries

    data = np.asarray(fs.value)
    freqs = np.asarray(fs.frequencies.value)
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    epoch_val = getattr(fs, "epoch", None)
    epoch = float(epoch_val.value) if epoch_val is not None else 0.0

    return PyCBCFrequencySeries(data, delta_f=df, epoch=epoch)

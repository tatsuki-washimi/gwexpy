"""
gwexpy.interop.control_
------------------------

Interoperability with the python-control library for control systems analysis.

Provides conversion between FrequencySeries and control.FRD (Frequency Response Data).
"""

from __future__ import annotations

import numpy as np

from ._optional import require_optional

__all__ = ["to_control_frd", "from_control_frd", "from_control_response"]


def to_control_frd(fs, frequency_unit: str = "rad/s"):
    """
    Convert FrequencySeries to control.FRD.

    Parameters
    ----------
    fs : FrequencySeries
        Input frequency response data. Frequencies are assumed to be in Hz.
    frequency_unit : str, optional
        Unit for the frequency axis in the output FRD object.
        Either ``'rad/s'`` or ``'Hz'``. Default is ``'rad/s'``
        (standard for ``control.FRD``).

    Returns
    -------
    control.FRD
        Frequency response data object compatible with python-control.

    Examples
    --------
    >>> from gwexpy.frequencyseries import FrequencySeries
    >>> from gwexpy.interop.control_ import to_control_frd
    >>> fs = FrequencySeries([1+0j, 0.5+0.5j], f0=1, df=1)
    >>> frd = to_control_frd(fs)
    """
    ctl = require_optional("control")

    freqs = np.asarray(fs.frequencies.value)

    # Convert Hz to rad/s if needed (control.FRD expects rad/s by default)
    if frequency_unit == "rad/s":
        omega = freqs * 2 * np.pi
    elif frequency_unit == "Hz":
        omega = freqs
    else:
        raise ValueError(
            f"frequency_unit must be 'rad/s' or 'Hz', got '{frequency_unit}'"
        )

    frd = ctl.frd(fs.value, omega)

    # Set system name from FrequencySeries metadata
    name = getattr(fs, "name", None) or "FrequencyResponse"
    try:
        frd.sysname = str(name)
    except AttributeError:
        pass  # Older versions of control may not support sysname

    return frd


def from_control_frd(cls, frd, frequency_unit: str = "Hz"):
    """
    Create FrequencySeries from control.FRD.

    Parameters
    ----------
    cls : type
        The FrequencySeries class to instantiate.
    frd : control.FRD
        Frequency response data from python-control.
    frequency_unit : str, optional
        Unit of the input FRD's omega attribute.
        Either ``'Hz'`` or ``'rad/s'``. Default is ``'Hz'``
        (output will be in Hz).

    Returns
    -------
    FrequencySeries or FrequencySeriesMatrix
        The converted frequency response. Returns FrequencySeriesMatrix
        for MIMO systems.

    Examples
    --------
    >>> from gwexpy.frequencyseries import FrequencySeries
    >>> from gwexpy.interop.control_ import from_control_frd
    >>> # frd = control.frd(data, omega)
    >>> # fs = from_control_frd(FrequencySeries, frd)
    """
    omega = np.asarray(frd.omega)

    # NOTE:
    # - control >= 0.10 provides `frdata` (preferred)
    # - control 0.10 also keeps `fresp` but emits a FutureWarning on access
    # - older versions may not have `frdata`
    if hasattr(frd, "frdata"):
        data = np.asarray(frd.frdata)
    elif hasattr(frd, "_fresp"):
        # Avoid triggering the `fresp` deprecation warning on newer control
        data = np.asarray(frd._fresp)
    else:
        data = np.asarray(frd.fresp)

    # Convert rad/s to Hz (standard for gwexpy)
    if frequency_unit == "rad/s" or frequency_unit == "Hz":
        # FRD.omega is always in rad/s internally
        freqs = omega / (2 * np.pi)
    else:
        raise ValueError(
            f"frequency_unit must be 'Hz' or 'rad/s', got '{frequency_unit}'"
        )

    # Check for regular spacing
    if len(freqs) > 1:
        diffs = np.diff(freqs)
        is_regular = np.allclose(diffs, diffs[0], rtol=1e-6)
    else:
        is_regular = True
        diffs = np.array([1.0])

    # Determine if this is a MIMO system
    is_matrix = data.ndim == 3 and (data.shape[0] > 1 or data.shape[1] > 1)

    if is_regular:
        df = diffs[0] if len(diffs) > 0 else 1.0
        f0 = freqs[0] if len(freqs) > 0 else 0.0

        if is_matrix:
            from gwexpy.frequencyseries import FrequencySeriesMatrix

            return FrequencySeriesMatrix(data, df=df, f0=f0)
        return cls(np.asarray(data).flatten(), df=df, f0=f0)

    # Irregular frequencies
    if is_matrix:
        from gwexpy.frequencyseries import FrequencySeriesMatrix

        return FrequencySeriesMatrix(data, frequencies=freqs)
    return cls(np.asarray(data).flatten(), frequencies=freqs)


def from_control_response(cls, response, **kwargs):
    """
    Create TimeSeries or TimeSeriesDict from control.TimeResponseData.

    Parameters
    ----------
    cls : type
        The TimeSeries (or TimeSeriesDict) class.
    response : control.TimeResponseData
        The simulation result from python-control (e.g., from forced_response).
    **kwargs : dict
        Additional arguments passed to the TimeSeries constructor (e.g., unit).

    Returns
    -------
    TimeSeries or TimeSeriesDict
        The converted time-domain data.
    """
    # response is a TimeResponseData object
    # It has attributes: time, outputs, inputs, states
    # usually we want outputs.
    # outputs shape is (noutputs, nsteps)

    time = np.asarray(response.time)
    outputs = np.asarray(response.outputs)

    if len(time) > 1:
        dt = time[1] - time[0]
    else:
        dt = 1.0  # Default if only one point

    t0 = time[0]

    # MIMO check
    if response.noutputs > 1:
        from gwexpy.timeseries import TimeSeries, TimeSeriesDict

        # If the user called TimeSeries.from_control, we might still want to return
        # a Dict if it's MIMO, or we could raise an error.
        # Following TimeSeriesDict.from_mne pattern, we should be flexible.

        tdict = TimeSeriesDict()
        for i in range(response.noutputs):
            label = None
            if (
                hasattr(response, "output_labels")
                and response.output_labels is not None
            ):
                label = response.output_labels[i]

            name = label or f"output_{i}"
            tdict[name] = TimeSeries(outputs[i], dt=dt, t0=t0, name=name, **kwargs)
        return tdict
    else:
        from gwexpy.timeseries import TimeSeries

        label = None
        if hasattr(response, "output_labels") and response.output_labels is not None:
            label = response.output_labels[0]

        name = label or "output"
        # If it's SISO but forced to be (1, nsteps), flatten it
        data = outputs.flatten()
        return TimeSeries(data, dt=dt, t0=t0, name=name, **kwargs)

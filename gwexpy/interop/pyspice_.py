"""
gwexpy.interop.pyspice_
------------------------

Interoperability with PySpice circuit simulation library.

Provides conversion from PySpice Analysis results (TransientAnalysis,
AcAnalysis, NoiseAnalysis, DistortionAnalysis) to GWexpy TimeSeries
and FrequencySeries types.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gwexpy.interop._registry import ConverterRegistry

from ._optional import require_optional

__all__ = [
    "from_pyspice_transient",
    "from_pyspice_ac",
    "from_pyspice_noise",
    "from_pyspice_distortion",
]


def from_pyspice_transient(
    cls: type,
    analysis: Any,
    *,
    node: str | None = None,
    branch: str | None = None,
    unit: Any | None = None,
) -> Any:
    """
    Create TimeSeries or TimeSeriesDict from a PySpice TransientAnalysis.

    Parameters
    ----------
    cls : type
        The TimeSeries (or TimeSeriesDict) class to instantiate.
    analysis : PySpice.Spice.Simulation.TransientAnalysis
        The transient analysis result from a PySpice simulation.
    node : str, optional
        Node name to extract (e.g. ``"out"``). If *None* and *branch* is also
        *None*, all nodes and branches are returned as a dict.
    branch : str, optional
        Branch name to extract (e.g. ``"vcc"``). Typically the name of a
        voltage source for current measurement. Cannot be combined with *node*.
    unit : str or astropy.units.Unit, optional
        Unit to assign to the result. PySpice waveforms carry unit information
        that is not astropy-compatible, so physical units must be supplied by
        the caller (e.g. ``"V"``, ``"A"``).

    Returns
    -------
    TimeSeries
        When a single *node* or *branch* is specified.
    TimeSeriesDict
        When neither *node* nor *branch* is given; keyed by signal name.

    Examples
    --------
    >>> from gwexpy.timeseries import TimeSeries
    >>> # analysis = simulator.transient(...)
    >>> ts = TimeSeries.from_pyspice_transient(analysis, node="out")
    """
    require_optional("PySpice")

    if node is not None and branch is not None:
        raise ValueError("Specify at most one of 'node' or 'branch', not both.")

    time = np.asarray(analysis.time, dtype=np.float64)

    if node is not None:
        data = np.asarray(analysis[node], dtype=np.float64)
        name = str(node)
        TimeSeries = ConverterRegistry.get_constructor("TimeSeries")
        return _build_timeseries(TimeSeries, data, time, name=name, unit=unit)

    if branch is not None:
        data = np.asarray(analysis[branch], dtype=np.float64)
        name = str(branch)
        TimeSeries = ConverterRegistry.get_constructor("TimeSeries")
        return _build_timeseries(TimeSeries, data, time, name=name, unit=unit)

    # Return all signals as TimeSeriesDict
    TimeSeries = ConverterRegistry.get_constructor("TimeSeries")
    TimeSeriesDict = ConverterRegistry.get_constructor("TimeSeriesDict")
    want_dict = TimeSeriesDict is not None and (
        cls is TimeSeriesDict or _is_subclass_safe(cls, TimeSeriesDict)
    )

    signals: dict[str, Any] = {}
    if hasattr(analysis, "nodes"):
        for name_key, waveform in analysis.nodes.items():
            signals[str(name_key)] = waveform
    if hasattr(analysis, "branches"):
        for name_key, waveform in analysis.branches.items():
            signals[str(name_key)] = waveform

    if len(signals) == 1 and not want_dict:
        name_key, waveform = next(iter(signals.items()))
        data = np.asarray(waveform, dtype=np.float64)
        return _build_timeseries(TimeSeries, data, time, name=name_key, unit=unit)

    result = TimeSeriesDict()
    for name_key, waveform in signals.items():
        data = np.asarray(waveform, dtype=np.float64)
        result[name_key] = _build_timeseries(
            TimeSeries, data, time, name=name_key, unit=unit
        )
    return result


def from_pyspice_ac(
    cls: type,
    analysis: Any,
    *,
    node: str | None = None,
    branch: str | None = None,
    unit: Any | None = None,
) -> Any:
    """
    Create FrequencySeries or FrequencySeriesDict from a PySpice AcAnalysis.

    AC analysis waveforms contain complex-valued frequency responses (transfer
    functions, impedances, etc.) with the frequency axis given by
    ``analysis.frequency``.

    Parameters
    ----------
    cls : type
        The FrequencySeries (or FrequencySeriesDict) class to instantiate.
    analysis : PySpice.Spice.Simulation.AcAnalysis
        The AC analysis result from a PySpice simulation.
    node : str, optional
        Node name to extract. If *None* and *branch* is also *None*, all
        nodes and branches are returned.
    branch : str, optional
        Branch name to extract.
    unit : str or astropy.units.Unit, optional
        Unit to assign to the result (e.g. ``"V"``).

    Returns
    -------
    FrequencySeries
        When a single signal is selected; data is complex.
    FrequencySeriesDict
        When no specific signal is selected; keyed by signal name.

    Examples
    --------
    >>> from gwexpy.frequencyseries import FrequencySeries
    >>> # analysis = simulator.ac(...)
    >>> fs = FrequencySeries.from_pyspice_ac(analysis, node="out")
    """
    require_optional("PySpice")
    return _from_pyspice_frequency(cls, analysis, node=node, branch=branch, unit=unit)


def from_pyspice_noise(
    cls: type,
    analysis: Any,
    *,
    node: str | None = None,
    unit: Any | None = None,
) -> Any:
    """
    Create FrequencySeries or FrequencySeriesDict from a PySpice NoiseAnalysis.

    Noise analysis waveforms contain real-valued noise spectral densities
    (e.g. V²/Hz or A²/Hz) as a function of frequency.

    Parameters
    ----------
    cls : type
        The FrequencySeries (or FrequencySeriesDict) class to instantiate.
    analysis : PySpice.Spice.Simulation.NoiseAnalysis
        The noise analysis result from a PySpice simulation.
    node : str, optional
        Node name to extract (e.g. ``"onoise"`` for output-referred noise).
        If *None*, all signals are returned.
    unit : str or astropy.units.Unit, optional
        Unit to assign to the result (e.g. ``"V**2/Hz"``).

    Returns
    -------
    FrequencySeries
        When a single signal is selected; data is real.
    FrequencySeriesDict
        When no signal is selected; keyed by signal name.

    Examples
    --------
    >>> from gwexpy.frequencyseries import FrequencySeries
    >>> # analysis = simulator.noise(...)
    >>> fs = FrequencySeries.from_pyspice_noise(analysis, node="onoise")
    """
    require_optional("PySpice")
    return _from_pyspice_frequency(cls, analysis, node=node, branch=None, unit=unit)


def from_pyspice_distortion(
    cls: type,
    analysis: Any,
    *,
    node: str | None = None,
    unit: Any | None = None,
) -> Any:
    """
    Create FrequencySeries or FrequencySeriesDict from a PySpice DistortionAnalysis.

    Distortion analysis waveforms contain harmonic/intermodulation distortion
    components as a function of frequency.

    Parameters
    ----------
    cls : type
        The FrequencySeries (or FrequencySeriesDict) class to instantiate.
    analysis : PySpice.Spice.Simulation.DistortionAnalysis
        The distortion analysis result from a PySpice simulation.
    node : str, optional
        Node name to extract. If *None*, all signals are returned.
    unit : str or astropy.units.Unit, optional
        Unit to assign to the result.

    Returns
    -------
    FrequencySeries
        When a single signal is selected.
    FrequencySeriesDict
        When no signal is selected; keyed by signal name.

    Examples
    --------
    >>> from gwexpy.frequencyseries import FrequencySeries
    >>> # analysis = simulator.distortion(...)
    >>> fs = FrequencySeries.from_pyspice_distortion(analysis, node="out")
    """
    require_optional("PySpice")
    return _from_pyspice_frequency(cls, analysis, node=node, branch=None, unit=unit)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _from_pyspice_frequency(
    cls: type,
    analysis: Any,
    *,
    node: str | None,
    branch: str | None,
    unit: Any | None,
) -> Any:
    """Shared implementation for AC / Noise / Distortion analysis conversion."""
    freqs = np.asarray(analysis.frequency, dtype=np.float64)

    FrequencySeries = ConverterRegistry.get_constructor("FrequencySeries")
    FrequencySeriesDict = ConverterRegistry.get_constructor("FrequencySeriesDict")
    want_dict = FrequencySeriesDict is not None and (
        cls is FrequencySeriesDict or _is_subclass_safe(cls, FrequencySeriesDict)
    )

    if node is not None:
        data = np.asarray(analysis[node])
        name = str(node)
        return FrequencySeries(data, frequencies=freqs, name=name, unit=unit)

    if branch is not None:
        data = np.asarray(analysis[branch])
        name = str(branch)
        return FrequencySeries(data, frequencies=freqs, name=name, unit=unit)

    # Collect all signals
    signals: dict[str, Any] = {}
    if hasattr(analysis, "nodes"):
        for name_key, waveform in analysis.nodes.items():
            signals[str(name_key)] = waveform
    if hasattr(analysis, "branches"):
        for name_key, waveform in analysis.branches.items():
            signals[str(name_key)] = waveform

    if len(signals) == 1 and not want_dict:
        name_key, waveform = next(iter(signals.items()))
        data = np.asarray(waveform)
        return FrequencySeries(data, frequencies=freqs, name=name_key, unit=unit)

    result = FrequencySeriesDict()
    for name_key, waveform in signals.items():
        data = np.asarray(waveform)
        result[name_key] = FrequencySeries(
            data, frequencies=freqs, name=name_key, unit=unit
        )
    return result


def _build_timeseries(
    TimeSeries: type,
    data: np.ndarray,
    time: np.ndarray,
    *,
    name: str,
    unit: Any | None,
) -> Any:
    """Construct a TimeSeries from data and a time array."""
    if len(time) > 1:
        diffs = np.diff(time)
        is_regular = np.allclose(diffs, diffs[0], rtol=1e-6, atol=0)
    else:
        is_regular = True

    if is_regular:
        dt = float(diffs[0]) if len(time) > 1 else 1.0
        t0 = float(time[0])
        return TimeSeries(data, dt=dt, t0=t0, name=name, unit=unit)
    else:
        # Irregular time axis — use times= kwarg
        return TimeSeries(data, times=time, name=name, unit=unit)


def _is_subclass_safe(cls: type, parent: type) -> bool:
    """Check issubclass without raising on non-class inputs."""
    try:
        return issubclass(cls, parent)
    except TypeError:
        return False

"""
gwexpy.interop.skrf_
---------------------

Interoperability with scikit-rf (skrf) for RF/microwave network analysis.

Provides bidirectional conversion between scikit-rf ``Network`` objects and
GWexpy FrequencySeries types, plus one-way conversion of time-domain
impulse/step responses to TimeSeries.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gwexpy.interop._registry import ConverterRegistry

from ._optional import require_optional

__all__ = [
    "from_skrf_network",
    "to_skrf_network",
    "from_skrf_impulse_response",
    "from_skrf_step_response",
]

# Supported network parameter names
_PARAMETER_ATTRS = {
    "s": "s",
    "z": "z",
    "y": "y",
    "a": "a",
    "t": "t",
    "h": "h",
}

# Default units per parameter type
_PARAMETER_UNITS: dict[str, str | None] = {
    "s": None,  # dimensionless
    "z": "ohm",
    "y": "S",  # Siemens
    "a": None,  # dimensionless (ABCD)
    "t": None,  # dimensionless (scattering transfer)
    "h": None,  # mixed units (hybrid)
}


def from_skrf_network(
    cls: type,
    ntwk: Any,
    *,
    parameter: str = "s",
    port_pair: tuple[int, int] | None = None,
    unit: Any | None = None,
) -> Any:
    """
    Create FrequencySeries, FrequencySeriesDict, or FrequencySeriesMatrix
    from a scikit-rf Network.

    Parameters
    ----------
    cls : type
        The FrequencySeries (or FrequencySeriesDict) class to instantiate.
    ntwk : skrf.Network
        The scikit-rf Network object to convert.
    parameter : str, default ``"s"``
        Which network parameter to extract. One of ``"s"`` (scattering),
        ``"z"`` (impedance), ``"y"`` (admittance), ``"a"`` (ABCD, 2-port
        only), ``"t"`` (transfer scattering, 2-port only), or ``"h"``
        (hybrid, 2-port only).
    port_pair : tuple[int, int], optional
        Zero-based ``(row, col)`` port indices to extract a single element of
        the parameter matrix, e.g. ``(1, 0)`` for S₂₁.  If *None*:

        * 1-port networks return a :class:`FrequencySeries` directly.
        * Multi-port networks return a :class:`FrequencySeriesMatrix` when
          *cls* is ``FrequencySeries``, or a :class:`FrequencySeriesDict`
          when *cls* is ``FrequencySeriesDict``.
    unit : str or astropy.units.Unit, optional
        Unit to assign to the result. If *None*, a default unit is inferred
        from *parameter*: ``None`` (dimensionless) for ``"s"``, ``"t"``,
        ``"a"``; ``"ohm"`` for ``"z"``; ``"S"`` for ``"y"``.

    Returns
    -------
    FrequencySeries
        For a single port pair or a 1-port network.
    FrequencySeriesMatrix
        For a multi-port network when *cls* is ``FrequencySeries``.
    FrequencySeriesDict
        When *cls* is ``FrequencySeriesDict``; keys are ``"P{row+1}{col+1}"``
        or ``"S{row+1}{col+1}"`` etc.

    Examples
    --------
    >>> from gwexpy.frequencyseries import FrequencySeries
    >>> import skrf
    >>> ntwk = skrf.Network("myfilter.s2p")
    >>> fs_s21 = FrequencySeries.from_skrf_network(ntwk, port_pair=(1, 0))
    >>> matrix = FrequencySeries.from_skrf_network(ntwk)
    """
    require_optional("skrf")

    if parameter not in _PARAMETER_ATTRS:
        raise ValueError(
            f"parameter must be one of {list(_PARAMETER_ATTRS)}, got '{parameter}'"
        )

    freqs = np.asarray(ntwk.f, dtype=np.float64)
    param_data = np.asarray(getattr(ntwk, _PARAMETER_ATTRS[parameter]))
    # param_data shape: (nfreq, nports, nports)

    # Resolve unit
    if unit is None:
        unit = _PARAMETER_UNITS.get(parameter)

    FrequencySeries = ConverterRegistry.get_constructor("FrequencySeries")
    FrequencySeriesDict = ConverterRegistry.get_constructor("FrequencySeriesDict")
    want_dict = FrequencySeriesDict is not None and (
        cls is FrequencySeriesDict or _is_subclass_safe(cls, FrequencySeriesDict)
    )

    nfreq, nports_row, nports_col = param_data.shape
    param_upper = parameter.upper()

    # Port names (for readable keys)
    port_names: list[str] | None = getattr(ntwk, "port_names", None)
    network_name: str = getattr(ntwk, "name", None) or ""

    # Single port pair
    if port_pair is not None:
        i, j = port_pair
        data = param_data[:, i, j]
        name = _port_pair_name(
            param_upper, i, j, port_names=port_names, network_name=network_name
        )
        return FrequencySeries(data, frequencies=freqs, name=name, unit=unit)

    # 1×1 network → single FrequencySeries
    if nports_row == 1 and nports_col == 1 and not want_dict:
        data = param_data[:, 0, 0]
        name = _port_pair_name(
            param_upper, 0, 0, port_names=port_names, network_name=network_name
        )
        return FrequencySeries(data, frequencies=freqs, name=name, unit=unit)

    # Multi-port: dict mode
    if want_dict:
        result = FrequencySeriesDict()
        for i in range(nports_row):
            for j in range(nports_col):
                data = param_data[:, i, j]
                key = _port_pair_name(
                    param_upper, i, j, port_names=port_names, network_name=network_name
                )
                result[key] = FrequencySeries(
                    data, frequencies=freqs, name=key, unit=unit
                )
        return result

    # Multi-port: matrix mode
    FrequencySeriesMatrix = ConverterRegistry.get_constructor("FrequencySeriesMatrix")
    # param_data shape is (nfreq, nports, nports); matrix expects (nrows, ncols, nfreq)
    matrix_data = np.moveaxis(param_data, 0, -1)  # → (nports, nports, nfreq)

    channel_names = np.empty((nports_row, nports_col), dtype=object)
    for i in range(nports_row):
        for j in range(nports_col):
            channel_names[i, j] = _port_pair_name(
                param_upper, i, j, port_names=port_names, network_name=network_name
            )

    return FrequencySeriesMatrix(
        matrix_data,
        frequencies=freqs,
        channel_names=channel_names,
        unit=unit,
        name=str(network_name) if network_name else None,
    )


def to_skrf_network(
    fs: Any,
    *,
    parameter: str = "s",
    z0: float = 50.0,
    port_names: list[str] | None = None,
    name: str | None = None,
) -> Any:
    """
    Create a scikit-rf Network from a FrequencySeries or FrequencySeriesMatrix.

    Parameters
    ----------
    fs : FrequencySeries or FrequencySeriesMatrix
        Source data. A :class:`FrequencySeries` is treated as a 1-port
        network (scalar parameter). A :class:`FrequencySeriesMatrix` of
        shape ``(N, N, nfreq)`` is treated as an N-port network.
    parameter : str, default ``"s"``
        Network parameter represented by the data. Currently only ``"s"``
        is supported for direct construction; other parameters require
        manual conversion.
    z0 : float, default 50.0
        Reference impedance in Ohms applied to all ports at all frequencies.
    port_names : list[str], optional
        Names for each port. Defaults to ``["1", "2", ...]``.
    name : str, optional
        Name for the resulting Network. If *None*, uses ``fs.name``.

    Returns
    -------
    skrf.Network
        The constructed Network object.

    Examples
    --------
    >>> from gwexpy.frequencyseries import FrequencySeries
    >>> import numpy as np
    >>> fs = FrequencySeries(np.array([0.1+0j, 0.2+0j]), frequencies=[1e9, 2e9])
    >>> ntwk = fs.to_skrf_network()
    """
    skrf = require_optional("skrf")

    freqs = np.asarray(fs.frequencies.value, dtype=np.float64)
    data = np.asarray(fs.value)

    nfreq = len(freqs)

    # Determine shape of s-matrix
    if data.ndim == 1:
        # 1-port: (nfreq,) → (nfreq, 1, 1)
        s = data.reshape(nfreq, 1, 1)
        nports = 1
    elif data.ndim == 3:
        # FrequencySeriesMatrix: (nrows, ncols, nfreq) → (nfreq, nrows, ncols)
        s = np.moveaxis(data, -1, 0)
        nports = s.shape[1]
    else:
        raise ValueError(
            f"Cannot convert data with {data.ndim} dimensions to a Network. "
            "Expected 1D (FrequencySeries) or 3D (FrequencySeriesMatrix)."
        )

    freq_obj = skrf.Frequency.from_f(freqs, unit="hz")

    net_name = name or getattr(fs, "name", None) or ""
    p_names = port_names or [str(i + 1) for i in range(nports)]

    ntwk = skrf.Network(
        frequency=freq_obj,
        s=s,
        z0=z0,
        name=net_name,
    )
    ntwk.port_names = p_names
    return ntwk


def from_skrf_impulse_response(
    cls: type,
    ntwk: Any,
    *,
    port_pair: tuple[int, int] | None = None,
    n: int | None = None,
    pad: int = 0,
    unit: Any | None = None,
) -> Any:
    """
    Create TimeSeries or TimeSeriesDict from a scikit-rf Network impulse response.

    Computes the time-domain impulse response via IFFT on the S-parameters
    (or the selected port pair) and wraps the result in a :class:`TimeSeries`.

    Parameters
    ----------
    cls : type
        The TimeSeries (or TimeSeriesDict) class to instantiate.
    ntwk : skrf.Network
        The scikit-rf Network object.
    port_pair : tuple[int, int], optional
        Zero-based ``(row, col)`` port indices to compute the impulse response
        for. If *None*:

        * 1-port networks compute the single impulse response.
        * Multi-port networks compute all port pairs and return a
          :class:`TimeSeriesDict`.
    n : int, optional
        Number of points for the IFFT. See ``Network.impulse_response``.
    pad : int, default 0
        Number of zero-padding points before IFFT.
    unit : str or astropy.units.Unit, optional
        Unit to assign to the result.

    Returns
    -------
    TimeSeries
        For a single port pair or a 1-port network.
    TimeSeriesDict
        For a multi-port network when no *port_pair* is specified.

    Examples
    --------
    >>> from gwexpy.timeseries import TimeSeries
    >>> import skrf
    >>> ntwk = skrf.Network("myfilter.s2p")
    >>> ts = TimeSeries.from_skrf_impulse_response(ntwk, port_pair=(1, 0))
    """
    require_optional("skrf")
    return _from_skrf_time_response(
        cls,
        ntwk,
        response_type="impulse",
        port_pair=port_pair,
        n=n,
        pad=pad,
        unit=unit,
    )


def from_skrf_step_response(
    cls: type,
    ntwk: Any,
    *,
    port_pair: tuple[int, int] | None = None,
    n: int | None = None,
    pad: int = 0,
    unit: Any | None = None,
) -> Any:
    """
    Create TimeSeries or TimeSeriesDict from a scikit-rf Network step response.

    Computes the time-domain step response via IFFT on the S-parameters
    (or the selected port pair) and wraps the result in a :class:`TimeSeries`.

    Parameters
    ----------
    cls : type
        The TimeSeries (or TimeSeriesDict) class to instantiate.
    ntwk : skrf.Network
        The scikit-rf Network object.
    port_pair : tuple[int, int], optional
        Zero-based ``(row, col)`` port indices to compute the step response
        for. If *None*:

        * 1-port networks compute the single step response.
        * Multi-port networks compute all port pairs and return a
          :class:`TimeSeriesDict`.
    n : int, optional
        Number of points for the IFFT. See ``Network.step_response``.
    pad : int, default 0
        Number of zero-padding points before IFFT.
    unit : str or astropy.units.Unit, optional
        Unit to assign to the result.

    Returns
    -------
    TimeSeries
        For a single port pair or a 1-port network.
    TimeSeriesDict
        For a multi-port network when no *port_pair* is specified.

    Examples
    --------
    >>> from gwexpy.timeseries import TimeSeries
    >>> import skrf
    >>> ntwk = skrf.Network("myfilter.s2p")
    >>> ts = TimeSeries.from_skrf_step_response(ntwk, port_pair=(1, 0))
    """
    require_optional("skrf")
    return _from_skrf_time_response(
        cls,
        ntwk,
        response_type="step",
        port_pair=port_pair,
        n=n,
        pad=pad,
        unit=unit,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _from_skrf_time_response(
    cls: type,
    ntwk: Any,
    *,
    response_type: str,
    port_pair: tuple[int, int] | None,
    n: int | None,
    pad: int,
    unit: Any | None,
) -> Any:
    """Shared implementation for impulse/step response conversion."""
    TimeSeries = ConverterRegistry.get_constructor("TimeSeries")
    TimeSeriesDict = ConverterRegistry.get_constructor("TimeSeriesDict")
    want_dict = TimeSeriesDict is not None and (
        cls is TimeSeriesDict or _is_subclass_safe(cls, TimeSeriesDict)
    )

    nports = ntwk.s.shape[1]
    network_name: str = getattr(ntwk, "name", None) or ""
    port_names: list[str] | None = getattr(ntwk, "port_names", None)
    param_upper = "S"  # impulse/step responses always use S-parameters

    kwargs: dict[str, Any] = {}
    if n is not None:
        kwargs["n"] = n
    if pad:
        kwargs["pad"] = pad

    method_name = "impulse_response" if response_type == "impulse" else "step_response"

    def _compute(i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        """Extract single-port sub-network and compute response."""
        # Build a 1-port sub-network from the selected s-parameter slice
        import skrf  # noqa: PLC0415

        sub_ntwk = skrf.Network(
            frequency=skrf.Frequency.from_f(ntwk.f, unit="hz"),
            s=ntwk.s[:, i, j].reshape(-1, 1, 1),
        )
        t_arr, h_arr = getattr(sub_ntwk, method_name)(**kwargs)
        return np.asarray(t_arr, dtype=np.float64), np.asarray(h_arr)

    if port_pair is not None:
        i, j = port_pair
        t_arr, h_arr = _compute(i, j)
        name = _port_pair_name(
            param_upper, i, j, port_names=port_names, network_name=network_name
        )
        return _build_timeseries_from_time_array(
            TimeSeries, h_arr, t_arr, name=name, unit=unit
        )

    # 1-port
    if nports == 1 and not want_dict:
        t_arr, h_arr = _compute(0, 0)
        name = _port_pair_name(
            param_upper, 0, 0, port_names=port_names, network_name=network_name
        )
        return _build_timeseries_from_time_array(
            TimeSeries, h_arr, t_arr, name=name, unit=unit
        )

    # Multi-port → TimeSeriesDict
    result = TimeSeriesDict()
    for i in range(nports):
        for j in range(nports):
            t_arr, h_arr = _compute(i, j)
            key = _port_pair_name(
                param_upper, i, j, port_names=port_names, network_name=network_name
            )
            result[key] = _build_timeseries_from_time_array(
                TimeSeries, h_arr, t_arr, name=key, unit=unit
            )
    return result


def _build_timeseries_from_time_array(
    TimeSeries: type,
    data: np.ndarray,
    time: np.ndarray,
    *,
    name: str,
    unit: Any | None,
) -> Any:
    """Construct a TimeSeries from data and time arrays."""
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
        return TimeSeries(data, times=time, name=name, unit=unit)


def _port_pair_name(
    param: str,
    i: int,
    j: int,
    *,
    port_names: list[str] | None,
    network_name: str,
) -> str:
    """Generate a readable name for a port-pair element.

    Uses ``{param}{i+1}{j+1}`` (e.g. ``S21``) when no meaningful port names
    are supplied, and ``{param}({port_i},{port_j})`` (e.g. ``S(in,out)``)
    when named ports are provided.  The default port names generated by
    scikit-rf (``["1", "2", ...]``) are treated as unnamed.
    """
    default_port_names = [str(k + 1) for k in range(max(i, j) + 1)]
    has_named_ports = (
        port_names is not None
        and len(port_names) > max(i, j)
        and port_names[: max(i, j) + 1] != default_port_names[: max(i, j) + 1]
    )

    if has_named_ports:
        assert port_names is not None
        pi = port_names[i]
        pj = port_names[j]
        label = f"{param}({pi},{pj})"
    else:
        label = f"{param}{i + 1}{j + 1}"

    if network_name:
        return f"{network_name}: {label}"
    return label


def _is_subclass_safe(cls: type, parent: type) -> bool:
    """Check issubclass without raising on non-class inputs."""
    try:
        return issubclass(cls, parent)
    except TypeError:
        return False

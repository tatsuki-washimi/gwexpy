"""
gwexpy.interop.pyroomacoustics_
-------------------------------

Interoperability with pyroomacoustics room acoustics simulation library.

Provides conversion from pyroomacoustics Room simulation results (RIR,
microphone signals, source signals, STFT) and ScalarField grid data
to GWexpy TimeSeries, Spectrogram, and ScalarField types.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from astropy import units as u

from gwexpy.interop._registry import ConverterRegistry

from ._optional import require_optional

__all__ = [
    "from_pyroomacoustics_rir",
    "from_pyroomacoustics_mic_signals",
    "from_pyroomacoustics_source",
    "from_pyroomacoustics_stft",
    "from_pyroomacoustics_field",
    "to_pyroomacoustics_source",
    "to_pyroomacoustics_stft",
]


# ---------------------------------------------------------------------------
# From pyroomacoustics → GWexpy
# ---------------------------------------------------------------------------


def from_pyroomacoustics_rir(
    cls: type,
    room: Any,
    *,
    source: int | None = None,
    mic: int | None = None,
    unit: Any | None = None,
) -> Any:
    """
    Create TimeSeries or TimeSeriesDict from pyroomacoustics room impulse responses.

    Parameters
    ----------
    cls : type
        The TimeSeries (or TimeSeriesDict) class to instantiate.
    room : pyroomacoustics.Room
        The room object after ``compute_rir()`` or ``simulate()`` has been called.
        ``room.rir`` must follow the pyroomacoustics convention
        ``rir[mic_index][source_index]`` (outer index = microphone).
    source : int, optional
        Source index to extract. If *None* and *mic* is also *None*, all
        source-mic pairs are returned.
    mic : int, optional
        Microphone index to extract. If *None* and *source* is also *None*,
        all source-mic pairs are returned.
    unit : str or astropy.units.Unit, optional
        Unit to assign to the result.

    Returns
    -------
    TimeSeries
        When a single source-mic pair is selected.
    TimeSeriesDict
        When multiple pairs are returned.

    Raises
    ------
    ValueError
        If RIR has not been computed yet.
    """
    require_optional("pyroomacoustics")

    rir = room.rir
    if rir is None:
        raise ValueError(
            "RIR not computed. Call room.compute_rir() or room.simulate() first."
        )

    fs = float(room.fs)
    dt = 1.0 / fs

    TimeSeries = ConverterRegistry.get_constructor("TimeSeries")
    TimeSeriesDict = ConverterRegistry.get_constructor("TimeSeriesDict")
    want_dict = TimeSeriesDict is not None and (
        cls is TimeSeriesDict or _is_subclass_safe(cls, TimeSeriesDict)
    )

    # Real pyroomacoustics convention: rir[mic_index][source_index]
    n_mics = len(rir)
    n_sources = len(rir[0]) if n_mics > 0 else 0

    # Single pair
    if source is not None and mic is not None:
        data = np.asarray(rir[mic][source], dtype=np.float64)
        name = f"rir_src{source}_mic{mic}"
        return TimeSeries(data, dt=dt, t0=0.0, name=name, unit=unit)

    # Single source, all mics
    if source is not None:
        result = TimeSeriesDict()
        for m in range(n_mics):
            data = np.asarray(rir[m][source], dtype=np.float64)
            name = f"mic_{m}"
            result[name] = TimeSeries(data, dt=dt, t0=0.0, name=name, unit=unit)
        return result

    # Single mic, all sources
    if mic is not None:
        result = TimeSeriesDict()
        for s in range(n_sources):
            data = np.asarray(rir[mic][s], dtype=np.float64)
            name = f"src_{s}"
            result[name] = TimeSeries(data, dt=dt, t0=0.0, name=name, unit=unit)
        return result

    # All pairs
    if n_mics == 1 and n_sources == 1 and not want_dict:
        data = np.asarray(rir[0][0], dtype=np.float64)
        return TimeSeries(data, dt=dt, t0=0.0, name="rir", unit=unit)

    result = TimeSeriesDict()
    for s in range(n_sources):
        for m in range(n_mics):
            data = np.asarray(rir[m][s], dtype=np.float64)
            name = f"src{s}_mic{m}"
            result[name] = TimeSeries(data, dt=dt, t0=0.0, name=name, unit=unit)
    return result


def from_pyroomacoustics_mic_signals(
    cls: type,
    room: Any,
    *,
    mic: int | None = None,
    unit: Any | None = None,
) -> Any:
    """
    Create TimeSeries or TimeSeriesDict from pyroomacoustics microphone signals.

    Parameters
    ----------
    cls : type
        The TimeSeries (or TimeSeriesDict) class to instantiate.
    room : pyroomacoustics.Room
        The room object after ``simulate()`` has been called.
    mic : int, optional
        Microphone index to extract. If *None*, all microphones are returned.
    unit : str or astropy.units.Unit, optional
        Unit to assign to the result.

    Returns
    -------
    TimeSeries
        When a single microphone is selected.
    TimeSeriesDict
        When multiple microphones are returned.

    Raises
    ------
    ValueError
        If signals have not been computed yet.
    """
    require_optional("pyroomacoustics")

    signals = room.mic_array.signals
    if signals is None:
        raise ValueError("Signals not computed. Call room.simulate() first.")

    signals = np.asarray(signals, dtype=np.float64)
    fs = float(room.fs)
    dt = 1.0 / fs

    TimeSeries = ConverterRegistry.get_constructor("TimeSeries")
    TimeSeriesDict = ConverterRegistry.get_constructor("TimeSeriesDict")
    want_dict = TimeSeriesDict is not None and (
        cls is TimeSeriesDict or _is_subclass_safe(cls, TimeSeriesDict)
    )

    n_mics = signals.shape[0]

    # Single mic selection
    if mic is not None:
        data = signals[mic]
        name = f"mic_{mic}"
        return TimeSeries(data, dt=dt, t0=0.0, name=name, unit=unit)

    # Single mic, auto-unwrap
    if n_mics == 1 and not want_dict:
        data = signals[0]
        return TimeSeries(data, dt=dt, t0=0.0, name="mic_0", unit=unit)

    # Multiple mics
    result = TimeSeriesDict()
    for m in range(n_mics):
        name = f"mic_{m}"
        result[name] = TimeSeries(signals[m], dt=dt, t0=0.0, name=name, unit=unit)
    return result


def from_pyroomacoustics_source(
    cls: type,
    room: Any,
    *,
    source: int = 0,
    unit: Any | None = None,
) -> Any:
    """
    Create TimeSeries from a pyroomacoustics sound source signal.

    Parameters
    ----------
    cls : type
        The TimeSeries class to instantiate.
    room : pyroomacoustics.Room
        The room object with at least one source added.
    source : int, default 0
        Source index to extract.
    unit : str or astropy.units.Unit, optional
        Unit to assign to the result.

    Returns
    -------
    TimeSeries
    """
    require_optional("pyroomacoustics")

    src = room.sources[source]
    data = np.asarray(src.signal, dtype=np.float64)
    fs = float(room.fs)
    dt = 1.0 / fs
    t0 = float(getattr(src, "delay", 0.0))
    name = f"source_{source}"

    TimeSeries = ConverterRegistry.get_constructor("TimeSeries")
    return TimeSeries(data, dt=dt, t0=t0, name=name, unit=unit)


def from_pyroomacoustics_stft(
    cls: type,
    stft_obj: Any,
    *,
    channel: int | None = None,
    fs: float | None = None,
    unit: Any | None = None,
) -> Any:
    """
    Create Spectrogram or SpectrogramDict from a pyroomacoustics STFT object.

    Parameters
    ----------
    cls : type
        The Spectrogram (or SpectrogramDict) class to instantiate.
    stft_obj : pyroomacoustics.stft.STFT or similar
        Object with ``.X`` (complex STFT data), ``.hop``, and ``.N`` attributes.
    channel : int, optional
        Channel index to extract from multi-channel STFT. If *None*, all
        channels are returned.
    fs : float, optional
        Sample rate in Hz. Required if ``stft_obj`` does not have an ``fs``
        attribute.
    unit : str or astropy.units.Unit, optional
        Unit to assign to the result.

    Returns
    -------
    Spectrogram
        When a single channel is selected or the STFT is single-channel.
    SpectrogramDict
        When multiple channels are present and no channel is selected.
    """
    require_optional("pyroomacoustics")

    X = np.asarray(stft_obj.X)
    hop = int(stft_obj.hop)
    N = int(stft_obj.N)

    # Resolve sample rate
    if fs is None:
        fs_attr = getattr(stft_obj, "fs", None)
        if fs_attr is None:
            _raise_fs_required()
            raise AssertionError("unreachable")
        fs = float(fs_attr)

    dt = hop / fs
    df = fs / N

    Spectrogram = ConverterRegistry.get_constructor("Spectrogram")
    SpectrogramDict = ConverterRegistry.get_constructor("SpectrogramDict")
    want_dict = SpectrogramDict is not None and (
        cls is SpectrogramDict or _is_subclass_safe(cls, SpectrogramDict)
    )

    # Determine dimensionality
    if X.ndim == 2:
        # Single channel: (n_frames, n_freq_bins)
        return Spectrogram(
            X, dt=dt * u.s, f0=0.0 * u.Hz, df=df * u.Hz, unit=unit, name="stft"
        )
    elif X.ndim == 3:
        # Multi-channel: (n_channels, n_frames, n_freq_bins)
        n_channels = X.shape[0]

        if channel is not None:
            data = X[channel]
            name = f"ch_{channel}"
            return Spectrogram(
                data, dt=dt * u.s, f0=0.0 * u.Hz, df=df * u.Hz, unit=unit, name=name
            )

        if n_channels == 1 and not want_dict:
            return Spectrogram(
                X[0],
                dt=dt * u.s,
                f0=0.0 * u.Hz,
                df=df * u.Hz,
                unit=unit,
                name="ch_0",
            )

        result = SpectrogramDict()
        for ch in range(n_channels):
            name = f"ch_{ch}"
            result[name] = Spectrogram(
                X[ch], dt=dt * u.s, f0=0.0 * u.Hz, df=df * u.Hz, unit=unit, name=name
            )
        return result
    else:
        raise ValueError(
            f"Expected STFT.X to be 2D or 3D, got {X.ndim}D with shape {X.shape}"
        )


def from_pyroomacoustics_field(
    cls: type,
    room: Any,
    *,
    grid_shape: tuple[int, ...],
    source: int = 0,
    mode: str = "rir",
    unit: Any | None = None,
) -> Any:
    """
    Create ScalarField from pyroomacoustics room with grid-placed microphones.

    When microphones are placed on a regular spatial grid, this function
    converts the RIR or microphone signals into a 4D ScalarField with
    shape ``(nt, nx, ny, nz)``.

    Parameters
    ----------
    cls : type
        The ScalarField class to instantiate.
    room : pyroomacoustics.Room
        Room object with microphones placed on a regular grid.
    grid_shape : tuple of int
        Spatial grid shape ``(nx, ny, nz)`` for 3D rooms or ``(nx, ny)``
        for 2D rooms. Must satisfy ``prod(grid_shape) == n_mics``.
    source : int, default 0
        Source index (used only when ``mode='rir'``).
    mode : {'rir', 'signals'}
        Which data to extract:

        - ``'rir'``: Room impulse responses for the given source.
        - ``'signals'``: Simulated microphone signals.
    unit : str or astropy.units.Unit, optional
        Unit to assign to the data.

    Returns
    -------
    ScalarField
        4D field with shape ``(nt, nx, ny, nz)``.

    Raises
    ------
    ValueError
        If ``prod(grid_shape)`` does not match the number of microphones,
        or if the requested data has not been computed.
    """
    require_optional("pyroomacoustics")

    n_mics_expected = math.prod(grid_shape)
    R = np.asarray(room.mic_array.R)  # shape (dim, n_mics)
    dim, n_mics = R.shape

    if n_mics_expected != n_mics:
        raise ValueError(
            f"grid_shape {grid_shape} implies {n_mics_expected} microphones, "
            f"but room has {n_mics} microphones."
        )

    fs = float(room.fs)

    # Collect data: shape (n_mics, n_samples)
    if mode == "rir":
        rir = room.rir
        if rir is None:
            raise ValueError(
                "RIR not computed. Call room.compute_rir() or room.simulate() first."
            )
        rir_list = [
            rir[m][source] for m in range(n_mics)
        ]  # gather across mics for one source
        # Pad to max length
        max_len = max(len(r) for r in rir_list)
        data_2d = np.zeros((n_mics, max_len), dtype=np.float64)
        for i, r in enumerate(rir_list):
            arr = np.asarray(r, dtype=np.float64)
            data_2d[i, : len(arr)] = arr
    elif mode == "signals":
        signals = room.mic_array.signals
        if signals is None:
            raise ValueError("Signals not computed. Call room.simulate() first.")
        data_2d = np.asarray(signals, dtype=np.float64)
    else:
        raise ValueError(f"mode must be 'rir' or 'signals', got '{mode}'")

    nt = data_2d.shape[1]

    # Reshape: (n_mics, nt) → (*grid_shape, nt) → (nt, *grid_shape)
    # Pad grid_shape to 3D if 2D
    if len(grid_shape) == 2:
        grid_3d = grid_shape + (1,)
        # For 2D rooms, add a dummy z-axis position
        R_3d = np.vstack([R, np.zeros((1, n_mics))])
    elif len(grid_shape) == 3:
        grid_3d = grid_shape
        R_3d = R
    else:
        raise ValueError(
            f"grid_shape must be 2D or 3D, got {len(grid_shape)}D: {grid_shape}"
        )

    nx, ny, nz = grid_3d
    # (n_mics, nt) → (nx, ny, nz, nt)
    data_4d = data_2d.reshape(nx, ny, nz, nt)
    # → (nt, nx, ny, nz)
    data_4d = np.moveaxis(data_4d, -1, 0)

    # Extract spatial axis coordinates from mic positions
    # R_3d shape: (3, n_mics) → reshape to (3, nx, ny, nz)
    R_grid = R_3d.reshape(3, nx, ny, nz)
    # x varies along axis 0 of the grid
    x_coords = R_grid[0, :, 0, 0]  # (nx,)
    y_coords = R_grid[1, 0, :, 0]  # (ny,)
    z_coords = R_grid[2, 0, 0, :]  # (nz,)

    times = np.arange(nt) / fs * u.s
    axis1 = x_coords * u.m
    axis2 = y_coords * u.m
    axis3 = z_coords * u.m

    if mode == "rir":
        name = f"rir_field_src{source}"
    else:
        name = "signal_field"

    return cls(
        data_4d,
        axis0=times,
        axis1=axis1,
        axis2=axis2,
        axis3=axis3,
        axis0_domain="time",
        space_domain="real",
        unit=unit,
        name=name,
    )


# ---------------------------------------------------------------------------
# From GWexpy → pyroomacoustics
# ---------------------------------------------------------------------------


def to_pyroomacoustics_source(ts: Any) -> tuple[np.ndarray, int]:
    """
    Export TimeSeries as a signal and sample rate tuple for pyroomacoustics.

    The returned tuple can be used to add a source to a pyroomacoustics room::

        signal, fs = ts.to_pyroomacoustics_source()
        room = pra.ShoeBox([5, 4, 3], fs=fs)
        room.add_source([1, 2, 1.5], signal=signal)

    Parameters
    ----------
    ts : TimeSeries
        The time series to export.

    Returns
    -------
    signal : numpy.ndarray
        1D float64 array of the signal samples.
    fs : int
        Sample rate in Hz.
    """
    signal = ts.value.astype(np.float64)
    fs = int(ts.sample_rate.value)
    return signal, fs


def to_pyroomacoustics_stft(
    spec: Any,
    *,
    hop: int | None = None,
    analysis_window: np.ndarray | None = None,
) -> Any:
    """
    Export Spectrogram as a pyroomacoustics STFT object.

    Parameters
    ----------
    spec : Spectrogram
        The spectrogram to export. Shape ``(n_frames, n_freq_bins)``.
    hop : int, optional
        Hop size in samples. If *None*, estimated from the spectrogram's
        time resolution and frequency resolution.
    analysis_window : numpy.ndarray, optional
        Analysis window to set on the STFT object.

    Returns
    -------
    pyroomacoustics.stft.STFT
        STFT object with ``.X`` set to the spectrogram data.
    """
    pra = require_optional("pyroomacoustics")

    data = np.asarray(spec)
    n_frames, n_freq_bins = data.shape

    N = 2 * (n_freq_bins - 1)

    if hop is None:
        # Estimate hop from dt and df
        dt_val = (
            float(spec.dt.to("s").value) if hasattr(spec.dt, "to") else float(spec.dt)
        )
        df_val = (
            float(spec.df.to("Hz").value) if hasattr(spec.df, "to") else float(spec.df)
        )
        fs = df_val * N
        hop = int(round(dt_val * fs))

    stft_obj = pra.transform.stft.STFT(N, hop=hop, analysis_window=analysis_window)
    stft_obj.X = data

    return stft_obj


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_subclass_safe(cls: type, parent: type) -> bool:
    """Check issubclass without raising on non-class inputs."""
    try:
        return issubclass(cls, parent)
    except TypeError:
        return False


def _raise_fs_required() -> None:
    """Raise ValueError when fs is needed but not available."""
    raise ValueError(
        "Sample rate 'fs' must be provided either as an argument or as "
        "an attribute of the STFT object."
    )

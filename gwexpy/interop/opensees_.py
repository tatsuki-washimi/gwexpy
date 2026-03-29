"""
gwexpy.interop.opensees_
-------------------------

Interoperability with OpenSeesPy recorder output.

OpenSeesPy writes node/element recorder data to text files with columns
``[time, n1_d1, n1_d2, ..., n2_d1, ...]``.  This module reads such files
into ``TimeSeriesMatrix`` or ``TimeSeriesDict``.

References
----------
https://opensees.berkeley.edu/
https://openseespydoc.readthedocs.io/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from ._modal_helpers import infer_unit_from_response_type
from ._registry import ConverterRegistry

__all__ = ["from_opensees_recorder"]


def from_opensees_recorder(
    cls: type,
    filepath: str | Path,
    *,
    nodes: list[int],
    dofs: list[int],
    response_type: str = "disp",
    dt: float | None = None,
    has_time_column: bool = True,
) -> Any:
    """Read an OpenSeesPy recorder text file into a ``TimeSeriesMatrix``.

    Parameters
    ----------
    cls : type
        ``TimeSeriesMatrix`` or ``TimeSeriesDict``.
    filepath : str or Path
        Path to the recorder output file (space-delimited text).
    nodes : list[int]
        Node numbers recorded (in order).
    dofs : list[int]
        DOF numbers recorded per node (1-based, e.g. ``[1, 2, 3]``
        for X, Y, Z).
    response_type : str
        Response quantity for unit inference (``"disp"``, ``"vel"``,
        ``"accel"``, ``"force"``).
    dt : float, optional
        Time step.  Required if *has_time_column* is ``False``.
    has_time_column : bool
        Whether the first column is a time vector (default ``True``).

    Returns
    -------
    TimeSeriesMatrix or TimeSeriesDict
    """
    raw = np.loadtxt(filepath)
    if raw.ndim == 1:
        raw = cast(np.ndarray[Any, np.dtype[np.float64]], raw.reshape(1, -1))

    if has_time_column:
        time = raw[:, 0]
        data = raw[:, 1:]
    else:
        if dt is None:
            raise ValueError("dt is required when has_time_column=False")
        n_samples = raw.shape[0]
        time = np.arange(n_samples, dtype=np.float64) * float(dt)
        data = raw

    dt_val = float(time[1] - time[0]) if len(time) > 1 else (dt or 1.0)
    t0 = float(time[0])

    # Build channel names from nodes × dofs
    _dof_names = {1: "X", 2: "Y", 3: "Z", 4: "RX", 5: "RY", 6: "RZ"}
    ch_names = []
    for node in nodes:
        for dof in dofs:
            dof_label = _dof_names.get(dof, str(dof))
            ch_names.append(f"N{node}_{dof_label}")

    n_channels = len(ch_names)
    n_samples = data.shape[0]

    # Validate column count
    if data.shape[1] < n_channels:
        raise ValueError(
            f"Expected at least {n_channels} data columns "
            f"(nodes={nodes} × dofs={dofs}), got {data.shape[1]}"
        )
    data = data[:, :n_channels]

    unit = infer_unit_from_response_type(response_type)

    # TimeSeriesDict mode
    TimeSeriesDict = ConverterRegistry.get_constructor("TimeSeriesDict")
    TimeSeries = ConverterRegistry.get_constructor("TimeSeries")

    if cls is TimeSeriesDict or (
        isinstance(cls, type) and issubclass(cls, TimeSeriesDict)
    ):
        result = TimeSeriesDict()
        for i, name in enumerate(ch_names):
            result[name] = TimeSeries(
                data[:, i], dt=dt_val, t0=t0, unit=unit, name=name
            )
        return result

    # TimeSeriesMatrix mode: (n_channels, 1, n_samples)
    matrix_data = data.T.reshape(n_channels, 1, n_samples)

    channel_names_arr = np.empty((n_channels, 1), dtype=object)
    for i, name in enumerate(ch_names):
        channel_names_arr[i, 0] = name

    cls_any = cast(Any, cls)
    return cls_any(
        matrix_data,
        dt=dt_val,
        t0=t0,
        channel_names=channel_names_arr,
        unit=unit,
        name=f"OpenSees {response_type}",
    )

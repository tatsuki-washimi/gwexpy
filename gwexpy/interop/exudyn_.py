"""
gwexpy.interop.exudyn_
-----------------------

Interoperability with Exudyn sensor output.

Exudyn's ``mbs.GetSensorStoredData()`` returns an ndarray where column 0
is time and the remaining columns are sensor values.  Alternatively,
sensor data can be exported to text files via ``exu.SensorExport``.

References
----------
https://github.com/jgerstmayr/EXUDYN
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ._modal_helpers import infer_unit_from_response_type
from ._registry import ConverterRegistry

__all__ = ["from_exudyn_sensor"]


def from_exudyn_sensor(
    cls: type,
    data: np.ndarray | str | Path,
    *,
    output_variable: str = "Displacement",
    column_names: list[str] | None = None,
) -> Any:
    """Convert Exudyn sensor data to ``TimeSeries`` or ``TimeSeriesMatrix``.

    Parameters
    ----------
    cls : type
        ``TimeSeries`` or ``TimeSeriesMatrix``.
    data : ndarray or str or Path
        Either the array returned by ``mbs.GetSensorStoredData()``
        (column 0 = time), or a path to a space-delimited text file
        with the same layout.
    output_variable : str
        Exudyn output variable name (e.g. ``"Displacement"``,
        ``"Velocity"``, ``"Force"``).  Used for unit inference.
    column_names : list[str], optional
        Names for non-time columns.  If not given, default names like
        ``"col_0"``, ``"col_1"`` etc. are used.

    Returns
    -------
    TimeSeries or TimeSeriesMatrix
    """
    if isinstance(data, (str, Path)):
        arr = np.loadtxt(data)
    else:
        arr = np.asarray(data, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # Column 0 = time
    time = arr[:, 0]
    values = arr[:, 1:]
    n_samples = values.shape[0]
    n_cols = values.shape[1]

    dt = float(time[1] - time[0]) if len(time) > 1 else 1.0
    t0 = float(time[0])

    # Map output variable to response type
    _var_map = {
        "displacement": "disp",
        "velocity": "vel",
        "acceleration": "accel",
        "force": "force",
        "stress": "stress",
        "strain": "strain",
        "rotation": "rotation",
        "angularvelocity": "angular_velocity",
    }
    resp_key = _var_map.get(output_variable.lower().strip(), None)
    unit = infer_unit_from_response_type(resp_key) if resp_key else None

    # Generate column names
    if column_names is None:
        column_names = [f"col_{i}" for i in range(n_cols)]

    TimeSeries = ConverterRegistry.get_constructor("TimeSeries")

    # Single column → TimeSeries
    if n_cols == 1 and (
        cls is TimeSeries or (isinstance(cls, type) and issubclass(cls, TimeSeries))
    ):
        return TimeSeries(
            values[:, 0],
            dt=dt,
            t0=t0,
            unit=unit,
            name=column_names[0] if column_names else output_variable,
        )

    # Multiple columns → TimeSeriesMatrix
    matrix_data = values.T.reshape(n_cols, 1, n_samples)

    channel_names_arr = np.empty((n_cols, 1), dtype=object)
    for i in range(n_cols):
        channel_names_arr[i, 0] = (
            column_names[i] if i < len(column_names) else f"col_{i}"
        )

    return cls(
        matrix_data,
        dt=dt,
        t0=t0,
        channel_names=channel_names_arr,
        unit=unit,
        name=f"Exudyn {output_variable}",
    )

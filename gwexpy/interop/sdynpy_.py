"""
gwexpy.interop.sdynpy_
-----------------------

Interoperability with SDynPy (Structural Dynamics in Python).

SDynPy represents modal data via NumPy structured arrays:
- ``ShapeArray``: mode shapes with coordinate/frequency/damping fields
- ``TransferFunctionArray``: FRF data with abscissa/ordinate
- ``TimeHistoryArray``: time-domain data with abscissa/ordinate

References
----------
https://github.com/sandialabs/sdynpy
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._modal_helpers import (
    build_frf_matrix,
    build_mode_dataframe,
    infer_unit_from_response_type,
)
from ._optional import require_optional
from ._registry import ConverterRegistry

__all__ = [
    "from_sdynpy_shape",
    "from_sdynpy_frf",
    "from_sdynpy_timehistory",
]


def from_sdynpy_shape(shape_array: Any) -> Any:
    """Convert an SDynPy ``ShapeArray`` to a ``pandas.DataFrame``.

    Parameters
    ----------
    shape_array : sdynpy.core.sdynpy_shape.ShapeArray
        SDynPy shape array.  Must expose ``.frequency``, ``.damping``,
        ``.shape_matrix``, and ``.coordinate`` attributes.

    Returns
    -------
    pandas.DataFrame
        Columns include DOF labels, optional node coordinates, and
        ``mode_1 ... mode_N`` columns.  ``frequency_Hz`` and
        ``damping_ratio`` are stored in ``df.attrs``.
    """
    frequencies = np.asarray(shape_array.frequency, dtype=float)
    damping = np.asarray(shape_array.damping, dtype=float)

    # shape_matrix: (n_modes, n_dof) → transpose to (n_dof, n_modes)
    shape_mat = np.asarray(shape_array.shape_matrix)
    if shape_mat.ndim == 1:
        shape_mat = shape_mat.reshape(1, -1)
    if shape_mat.shape[0] == len(frequencies):
        shape_mat = shape_mat.T

    # coordinate: SDynPy stores as structured array with .node, .direction
    coords = shape_array.coordinate
    dof_labels = None
    node_ids = None
    node_coords = None

    if hasattr(coords, "node") and hasattr(coords, "direction"):
        nodes = np.asarray(coords.node).ravel()
        dirs = np.asarray(coords.direction).ravel()
        _dir_map = {1: "+X", 2: "+Y", 3: "+Z", 4: "+RX", 5: "+RY", 6: "+RZ"}
        dof_labels = np.array(
            [f"{n}:{_dir_map.get(int(d), str(d))}" for n, d in zip(nodes, dirs)]
        )
        node_ids = np.unique(nodes)

        # Extract coordinates if available
        if hasattr(coords, "coordinate") and coords.coordinate is not None:
            raw = np.asarray(coords.coordinate)
            if raw.ndim == 2 and raw.shape[1] >= 3:
                # one row per DOF; collapse to unique nodes
                seen: dict[int, np.ndarray] = {}
                for nid, row in zip(nodes, raw):
                    if int(nid) not in seen:
                        seen[int(nid)] = row[:3]
                node_coords = np.array([seen[int(n)] for n in node_ids])

    return build_mode_dataframe(
        frequencies,
        damping,
        mode_shapes=shape_mat,
        node_ids=node_ids,
        dof_labels=dof_labels,
        coordinates=node_coords,
    )


def from_sdynpy_frf(
    cls: type,
    tfa: Any,
    *,
    response_type: str | None = None,
) -> Any:
    """Convert an SDynPy ``TransferFunctionArray`` to a ``FrequencySeriesMatrix``.

    Parameters
    ----------
    cls : type
        ``FrequencySeriesMatrix``.
    tfa : sdynpy.core.sdynpy_data.TransferFunctionArray
        SDynPy transfer function array.
    response_type : str, optional
        Response type for unit inference (e.g. ``"accel"``).

    Returns
    -------
    FrequencySeriesMatrix
    """
    # TransferFunctionArray stores ordinate (complex) and abscissa (freq)
    ordinate = np.asarray(tfa.ordinate)  # shape varies
    abscissa = np.asarray(tfa.abscissa)

    # Frequency vector (take first row if 2-D)
    if abscissa.ndim > 1:
        freq = abscissa[0] if abscissa.ndim == 2 else abscissa.ravel()
    else:
        freq = abscissa

    # Reshape ordinate to (n_resp, n_ref, n_freq) if needed
    if ordinate.ndim == 1:
        frf_data = ordinate.reshape(1, 1, -1)
    elif ordinate.ndim == 2:
        frf_data = ordinate.reshape(ordinate.shape[0], 1, ordinate.shape[1])
    elif ordinate.ndim == 3:
        frf_data = ordinate
    else:
        raise ValueError(
            f"Unexpected ordinate shape {ordinate.shape}; expected 1-D, 2-D, or 3-D"
        )

    # Channel names from coordinate attribute
    response_names = None
    reference_names = None
    if hasattr(tfa, "response_coordinate") and hasattr(tfa, "reference_coordinate"):
        resp_c = tfa.response_coordinate
        ref_c = tfa.reference_coordinate
        if resp_c is not None:
            response_names = [str(c) for c in np.atleast_1d(resp_c)]
        if ref_c is not None:
            reference_names = [str(c) for c in np.atleast_1d(ref_c)]

    unit = infer_unit_from_response_type(response_type) if response_type else None

    return build_frf_matrix(
        cls,
        freq,
        frf_data,
        response_names=response_names,
        reference_names=reference_names,
        unit=unit,
    )


def from_sdynpy_timehistory(
    cls: type,
    tha: Any,
    *,
    response_type: str | None = None,
) -> Any:
    """Convert an SDynPy ``TimeHistoryArray`` to a ``TimeSeriesMatrix``.

    Parameters
    ----------
    cls : type
        ``TimeSeriesMatrix`` or ``TimeSeriesDict``.
    tha : sdynpy.core.sdynpy_data.TimeHistoryArray
        SDynPy time-history array.
    response_type : str, optional
        Response type for unit inference.

    Returns
    -------
    TimeSeriesMatrix or TimeSeriesDict
    """
    ordinate = np.asarray(tha.ordinate)
    abscissa = np.asarray(tha.abscissa)

    # Time vector
    if abscissa.ndim > 1:
        time = abscissa[0]
    else:
        time = abscissa

    dt = float(time[1] - time[0]) if len(time) > 1 else 1.0
    t0 = float(time[0])

    unit = infer_unit_from_response_type(response_type) if response_type else None

    # Check if caller wants TimeSeriesDict
    TimeSeriesDict = ConverterRegistry.get_constructor("TimeSeriesDict")
    TimeSeries = ConverterRegistry.get_constructor("TimeSeries")

    if cls is TimeSeriesDict or (
        isinstance(cls, type) and issubclass(cls, TimeSeriesDict)
    ):
        result = TimeSeriesDict()
        if ordinate.ndim == 1:
            ordinate = ordinate.reshape(1, -1)
        for i in range(ordinate.shape[0]):
            name = f"ch_{i}"
            if hasattr(tha, "coordinate"):
                coords = np.atleast_1d(tha.coordinate)
                if i < len(coords):
                    name = str(coords[i])
            result[name] = TimeSeries(
                ordinate[i], dt=dt, t0=t0, unit=unit, name=name
            )
        return result

    # TimeSeriesMatrix mode
    if ordinate.ndim == 1:
        ordinate = ordinate.reshape(1, 1, -1)
    elif ordinate.ndim == 2:
        ordinate = ordinate.reshape(ordinate.shape[0], 1, ordinate.shape[1])

    channel_names = None
    if hasattr(tha, "coordinate"):
        coords = np.atleast_1d(tha.coordinate)
        n_ch = ordinate.shape[0]
        channel_names = np.empty((n_ch, 1), dtype=object)
        for i in range(n_ch):
            channel_names[i, 0] = str(coords[i]) if i < len(coords) else f"ch_{i}"

    return cls(
        ordinate,
        dt=dt,
        t0=t0,
        channel_names=channel_names,
        unit=unit,
    )

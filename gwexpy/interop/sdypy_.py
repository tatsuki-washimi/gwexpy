"""
gwexpy.interop.sdypy_
-----------------------

Interoperability with SDyPy / pyuff (Universal File Format).

Reads UFF dataset type 58 (function at nodal DOF) and type 55
(modal data) into GWexpy types.

References
----------
https://github.com/ladisk/pyuff
https://sdypy.readthedocs.io/
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._modal_helpers import build_mode_dataframe, infer_unit_from_response_type
from ._optional import require_optional
from ._registry import ConverterRegistry

__all__ = [
    "from_uff_dataset58",
    "from_uff_dataset55",
]

# UFF type 58 function types:
# 0 = General/Unknown, 1 = Time Response, 2 = Auto Spectrum,
# 3 = Cross Spectrum, 4 = Frequency Response Function, ...
_TIME_DOMAIN_TYPES = {0, 1, 7, 8, 9, 11, 12, 13}
_FREQ_DOMAIN_TYPES = {2, 3, 4, 5, 6, 10}


def from_uff_dataset58(
    cls: type,
    uff_data: dict,
    *,
    response_type: str | None = None,
) -> Any:
    """Convert a pyuff dataset-58 dict to ``TimeSeries`` or ``FrequencySeries``.

    Parameters
    ----------
    cls : type
        ``TimeSeries`` or ``FrequencySeries``.  If *None* or ambiguous, the
        function type field in the UFF data is used to auto-select.
    uff_data : dict
        A single dataset-58 record as returned by ``pyuff.UFF().read_sets()``.
        Expected keys: ``"x"``, ``"data"``, ``"func_type"``, ``"id1"``
        (description), ``"rsp_dir"``, ``"ref_dir"``, etc.
    response_type : str, optional
        Override for unit inference (e.g. ``"accel"``).

    Returns
    -------
    TimeSeries or FrequencySeries
    """
    x = np.asarray(uff_data["x"], dtype=float)
    data = np.asarray(uff_data["data"])

    func_type = int(uff_data.get("func_type", 0))
    name = str(uff_data.get("id1", "")).strip() or None

    # Infer unit
    unit = None
    if response_type:
        unit = infer_unit_from_response_type(response_type)

    # Determine domain
    TimeSeries = ConverterRegistry.get_constructor("TimeSeries")
    FrequencySeries = ConverterRegistry.get_constructor("FrequencySeries")

    is_time = func_type in _TIME_DOMAIN_TYPES
    is_freq = func_type in _FREQ_DOMAIN_TYPES

    # If cls is explicitly given, respect it
    if cls is TimeSeries or (isinstance(cls, type) and issubclass(cls, TimeSeries)):
        is_time = True
    elif cls is FrequencySeries or (
        isinstance(cls, type) and issubclass(cls, FrequencySeries)
    ):
        is_freq = True

    if is_time and not is_freq:
        dt = float(x[1] - x[0]) if len(x) > 1 else 1.0
        t0 = float(x[0])
        return TimeSeries(data.real, dt=dt, t0=t0, unit=unit, name=name)

    # Frequency domain (default fallback)
    df = float(x[1] - x[0]) if len(x) > 1 else 1.0
    f0 = float(x[0])
    return FrequencySeries(data, frequencies=x, unit=unit, name=name)


def from_uff_dataset55(uff_data: dict) -> Any:
    """Convert a pyuff dataset-55 dict to a ``pandas.DataFrame``.

    Dataset type 55 contains modal model data: natural frequencies,
    modal damping, and mode shapes.

    Parameters
    ----------
    uff_data : dict
        A single dataset-55 record from ``pyuff.UFF().read_sets()``.
        Expected keys: ``"modal_m"`` (mode number), ``"modal_freq"``,
        ``"modal_damp"``, ``"modal_viscous_damp"``, ``"r1"``–``"r6"``
        (DOF responses), ``"node_nums"``, etc.

    Returns
    -------
    pandas.DataFrame
    """
    frequencies = np.atleast_1d(uff_data.get("modal_freq", uff_data.get("freq", [])))
    damping = np.atleast_1d(uff_data.get("modal_damp", uff_data.get("damping", [])))

    n_modes = len(frequencies)

    # Mode shapes: r1..r6 contain DOF responses per mode
    # Shape varies by implementation; handle common layouts
    mode_shapes = None
    node_ids = None

    if "r1" in uff_data:
        # r1..r6: each is (n_nodes,) for a single mode, or (n_nodes, n_modes)
        components = []
        for key in ("r1", "r2", "r3", "r4", "r5", "r6"):
            if key in uff_data and uff_data[key] is not None:
                arr = np.atleast_1d(uff_data[key])
                if arr.size > 0:
                    components.append(arr)
        if components:
            # Stack components → interleave DOFs per node
            # Each component shape: (n_nodes,) or (n_modes, n_nodes)
            stacked = np.array(components)  # (n_dirs, ...)
            if stacked.ndim == 2:
                # (n_dirs, n_nodes) → single mode
                n_dirs, n_nodes = stacked.shape
                mode_shapes = stacked.T.reshape(n_nodes * n_dirs, 1)
            elif stacked.ndim == 3:
                # (n_dirs, n_modes, n_nodes) → full
                n_dirs, nm, n_nodes = stacked.shape
                mode_shapes = stacked.transpose(2, 0, 1).reshape(
                    n_nodes * n_dirs, nm
                )

    if "node_nums" in uff_data:
        node_ids = np.atleast_1d(uff_data["node_nums"])

    return build_mode_dataframe(
        frequencies,
        damping,
        mode_shapes=mode_shapes,
        node_ids=node_ids,
    )

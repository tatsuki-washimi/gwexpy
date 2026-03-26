"""
gwexpy.interop._modal_helpers
------------------------------

Shared helper functions for modal analysis interop modules
(SDynPy, SDyPy/pyuff, pyOMA, OpenSeesPy, Exudyn).

Provides:
- ``build_mode_dataframe``: assemble modal parameters into a pandas DataFrame
- ``build_frf_matrix``: assemble FRF data into a FrequencySeriesMatrix
- ``infer_unit_from_response_type``: map response-type strings to astropy units
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._registry import ConverterRegistry

__all__ = [
    "build_mode_dataframe",
    "build_frf_matrix",
    "infer_unit_from_response_type",
]


# ── unit inference ──────────────────────────────────────────────────

_RESPONSE_UNIT_MAP: dict[str, str] = {
    "disp": "m",
    "displacement": "m",
    "vel": "m/s",
    "velocity": "m/s",
    "accel": "m/s2",
    "acceleration": "m/s2",
    "force": "N",
    "stress": "Pa",
    "strain": "",
    "rotation": "rad",
    "angular_velocity": "rad/s",
}


def infer_unit_from_response_type(response_type: str) -> Any:
    """Return an ``astropy.units.Unit`` for a response-type string.

    Parameters
    ----------
    response_type : str
        One of ``"disp"``, ``"vel"``, ``"accel"``, ``"force"``,
        ``"stress"``, ``"strain"``, ``"rotation"``, ``"angular_velocity"``,
        or their long-form equivalents.

    Returns
    -------
    astropy.units.Unit or None
        The inferred unit, or *None* if the response type is unrecognised.
    """
    from astropy import units as u

    key = response_type.lower().strip()
    unit_str = _RESPONSE_UNIT_MAP.get(key)
    if unit_str is None:
        return None
    if unit_str == "":
        return u.dimensionless_unscaled
    return u.Unit(unit_str)


# ── mode DataFrame ──────────────────────────────────────────────────


def build_mode_dataframe(
    frequencies: np.ndarray,
    damping_ratios: np.ndarray,
    *,
    mode_shapes: np.ndarray | None = None,
    node_ids: np.ndarray | None = None,
    dof_labels: np.ndarray | None = None,
    coordinates: np.ndarray | None = None,
) -> Any:
    """Assemble modal parameters into a ``pandas.DataFrame``.

    Parameters
    ----------
    frequencies : ndarray, shape (n_modes,)
        Natural frequencies in Hz.
    damping_ratios : ndarray, shape (n_modes,)
        Modal damping ratios (dimensionless, e.g. 0.02 = 2 %).
    mode_shapes : ndarray, shape (n_dof, n_modes), optional
        Mode-shape matrix (real or complex).  When provided the resulting
        DataFrame has one row per DOF and columns ``mode_1 ... mode_N``.
    node_ids : ndarray, shape (n_nodes,), optional
        Node identifiers.  If *mode_shapes* is provided and *dof_labels*
        is not, node IDs are repeated for each DOF direction.
    dof_labels : ndarray, shape (n_dof,), optional
        DOF labels such as ``["1:+X", "1:+Y", "1:+Z", "2:+X", ...]``.
    coordinates : ndarray, shape (n_nodes, 3), optional
        (x, y, z) coordinates per node.

    Returns
    -------
    pandas.DataFrame
        Columns vary depending on supplied arguments.  Always includes
        ``frequency_Hz`` and ``damping_ratio``.
    """
    from ._optional import require_optional

    pd = require_optional("pandas")

    n_modes = len(frequencies)

    # ── summary-only (no mode shapes) ───────────────────────────────
    if mode_shapes is None:
        df = pd.DataFrame(
            {
                "mode": np.arange(1, n_modes + 1),
                "frequency_Hz": np.asarray(frequencies, dtype=float),
                "damping_ratio": np.asarray(damping_ratios, dtype=float),
            }
        )
        return df

    # ── full mode-shape table ───────────────────────────────────────
    mode_shapes = np.asarray(mode_shapes)
    n_dof = mode_shapes.shape[0]

    records: dict[str, Any] = {}

    # DOF labels
    if dof_labels is not None:
        records["dof"] = np.asarray(dof_labels)
    elif node_ids is not None:
        n_nodes = len(node_ids)
        n_dirs = n_dof // n_nodes if n_nodes > 0 else 1
        dir_names = ["+X", "+Y", "+Z", "+RX", "+RY", "+RZ"][:n_dirs]
        labels = []
        for nid in node_ids:
            for d in dir_names:
                labels.append(f"{nid}:{d}")
        records["dof"] = np.array(labels[:n_dof])

    # Node IDs per DOF
    if node_ids is not None:
        n_nodes = len(node_ids)
        n_dirs = n_dof // n_nodes if n_nodes > 0 else 1
        records["node_id"] = np.repeat(node_ids, n_dirs)[:n_dof]

    # Coordinates per DOF (repeated for each direction)
    if coordinates is not None:
        coords = np.asarray(coordinates)
        n_nodes_c = coords.shape[0]
        n_dirs = n_dof // n_nodes_c if n_nodes_c > 0 else 1
        for ax_i, ax_name in enumerate(("x", "y", "z")):
            records[ax_name] = np.repeat(coords[:, ax_i], n_dirs)[:n_dof]

    # Mode shape columns
    for m in range(n_modes):
        col = f"mode_{m + 1}"
        records[col] = mode_shapes[:, m]

    df = pd.DataFrame(records)

    # Attach frequency / damping as DataFrame attrs (metadata)
    df.attrs["frequency_Hz"] = np.asarray(frequencies, dtype=float).tolist()
    df.attrs["damping_ratio"] = np.asarray(damping_ratios, dtype=float).tolist()

    return df


# ── FRF matrix ──────────────────────────────────────────────────────


def build_frf_matrix(
    cls: type,
    frequencies: np.ndarray,
    frf_data: np.ndarray,
    *,
    response_names: list[str] | None = None,
    reference_names: list[str] | None = None,
    unit: Any | None = None,
    name: str | None = None,
) -> Any:
    """Assemble FRF data into a ``FrequencySeriesMatrix``.

    Parameters
    ----------
    cls : type
        ``FrequencySeriesMatrix`` (or its subclass).
    frequencies : ndarray, shape (n_freq,)
        Frequency vector in Hz.
    frf_data : ndarray, shape (n_resp, n_ref, n_freq)
        Complex FRF values.
    response_names, reference_names : list[str], optional
        Names for response and reference DOFs.
    unit : astropy.units.Unit, optional
        Physical unit of the FRF values.
    name : str, optional
        Name for the matrix.

    Returns
    -------
    FrequencySeriesMatrix
    """
    frf_data = np.asarray(frf_data)
    frequencies = np.asarray(frequencies, dtype=float)

    n_resp, n_ref = frf_data.shape[0], frf_data.shape[1]

    # Build channel_names matrix
    channel_names = np.empty((n_resp, n_ref), dtype=object)
    for i in range(n_resp):
        for j in range(n_ref):
            r_name = response_names[i] if response_names else f"resp_{i}"
            ref_name = reference_names[j] if reference_names else f"ref_{j}"
            channel_names[i, j] = f"{r_name} / {ref_name}"

    return cls(
        frf_data,
        frequencies=frequencies,
        channel_names=channel_names,
        unit=unit,
        name=name,
    )

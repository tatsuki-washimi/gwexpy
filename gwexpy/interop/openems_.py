"""
gwexpy.interop.openems_
-----------------------

Interoperability with openEMS field dump HDF5 files.

openEMS writes electromagnetic field data via ``CSPropDumpBox``.
The HDF5 layout is::

    /Mesh/
        x   # 1-D node coordinates along x
        y
        z
    /FieldData/
        TD/          # time-domain dumps
            <step_0> # 4-D array (Nx, Ny, Nz, 3)  — 3 = x/y/z components
            <step_1>
            ...
        FD/          # frequency-domain dumps
            f0_real  # 4-D (Nx, Ny, Nz, 3)
            f0_imag
            f1_real
            ...

The DumpType value (set in the CSXCAD ``AddDump`` call) controls which physical
quantity is stored and in which domain.  Only ``h5py`` is required; the openEMS
Python bindings are not needed.

HDF5 dataset attributes
-----------------------
If time-domain datasets carry a ``"Time"`` attribute (float, seconds), the
returned axis0 uses those physical time values.  If frequency-domain datasets
carry a ``"frequency"`` attribute (float, Hz), axis0 uses those physical
frequency values.  Both fall back to integer indices when the attribute is
absent.

References
----------
https://openems.de/index.php/HDF5_Field_Dumps.html
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import numpy as np

from gwexpy.fields import ScalarField, VectorField

from ._optional import require_optional

__all__ = ["from_openems_hdf5", "DUMP_TYPE_MAP"]

# ---------------------------------------------------------------------------
# DumpType → (quantity_name, unit_string, domain)
# ---------------------------------------------------------------------------

DUMP_TYPE_MAP: dict[int, tuple[str, str, str]] = {
    0: ("E-field", "V/m", "time"),
    1: ("H-field", "A/m", "time"),
    2: ("current density J", "A/m2", "time"),
    3: ("current density rot(H)", "A/m2", "time"),
    10: ("E-field", "V/m", "frequency"),
    11: ("H-field", "A/m", "frequency"),
    20: ("SAR local", "W/kg", "frequency"),
    21: ("SAR 1g", "W/kg", "frequency"),
    22: ("SAR 10g", "W/kg", "frequency"),
}

# SAR types produce scalar (no spatial component axis)
_SAR_DUMP_TYPES = {20, 21, 22}

# Component axis → label
_COMP_LABELS = {0: "x", 1: "y", 2: "z"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_openems_mesh(
    h5file: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read mesh node coordinates from an openEMS HDF5 file.

    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file.

    Returns
    -------
    tuple of ndarray
        ``(x, y, z)`` coordinate arrays (1-D each).

    Raises
    ------
    KeyError
        If the ``/Mesh`` group or any coordinate dataset is absent.
    """
    mesh = h5file["Mesh"]
    x = np.asarray(mesh["x"], dtype=np.float64)
    y = np.asarray(mesh["y"], dtype=np.float64)
    z = np.asarray(mesh["z"], dtype=np.float64)
    return x, y, z


def _read_openems_td(
    h5file: Any,
    timestep: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load time-domain field data from ``/FieldData/TD``.

    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file.
    timestep : int or None
        If an integer, load that single time step.
        If ``None``, concatenate all steps along a new leading axis.

    Returns
    -------
    data : ndarray
        Shape ``(n_time, Nx, Ny, Nz, 3)`` or ``(1, Nx, Ny, Nz, 3)``.
    times : ndarray
        Time values (1-D, length ``n_time``), taken from dataset names or
        sequential integers when attributes are unavailable.
    """
    td = h5file["FieldData/TD"]
    # Sort keys numerically
    keys = sorted(td.keys(), key=lambda k: int(k) if k.isdigit() else 0)

    if not keys:
        raise ValueError("No time steps found in /FieldData/TD.")

    if timestep is not None:
        key = str(timestep)
        if key not in td:
            raise ValueError(
                f"Timestep {timestep!r} not found in /FieldData/TD. "
                f"Available: {keys}"
            )
        arr = np.asarray(td[key], dtype=np.float64)[np.newaxis, ...]  # (1, Nx, Ny, Nz, 3)
        t_attr = td[key].attrs.get("Time", None)
        times = np.array(
            [float(t_attr) if t_attr is not None else timestep], dtype=np.float64
        )
    else:
        slices = [np.asarray(td[k], dtype=np.float64) for k in keys]
        arr = np.stack(slices, axis=0)  # (n_time, Nx, Ny, Nz, 3)
        # Try to read physical time values from dataset attributes
        time_vals = [td[k].attrs.get("Time", None) for k in keys]
        if all(v is not None for v in time_vals):
            times = np.array([float(v) for v in time_vals], dtype=np.float64)
        else:
            times = np.arange(len(keys), dtype=np.float64)

    return arr, times


def _read_openems_fd(
    h5file: Any,
    freq_idx: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load frequency-domain field data from ``/FieldData/FD``.

    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file.
    freq_idx : int or None
        If an integer, load that single frequency index.
        If ``None``, concatenate all frequencies.

    Returns
    -------
    data : ndarray
        Complex array of shape ``(n_freq, Nx, Ny, Nz, 3)`` or
        ``(1, Nx, Ny, Nz, 3)``.
    freqs : ndarray
        Frequency index values (1-D).
    """
    fd = h5file["FieldData/FD"]
    # Collect frequency indices from dataset names like "f0_real", "f1_real"
    real_keys = {k for k in fd.keys() if re.match(r"f\d+_real$", k)}
    indices = sorted({int(re.search(r"f(\d+)_real", k).group(1)) for k in real_keys})

    if not indices:
        raise ValueError("No frequency datasets found in /FieldData/FD.")

    if freq_idx is not None:
        if freq_idx not in indices:
            raise ValueError(
                f"Frequency index {freq_idx} not found. Available: {indices}"
            )
        rk = f"f{freq_idx}_real"
        ik = f"f{freq_idx}_imag"
        real = np.asarray(fd[rk], dtype=np.float64)
        imag = np.asarray(fd[ik], dtype=np.float64)
        arr = (real + 1j * imag)[np.newaxis, ...]  # (1, Nx, Ny, Nz, 3)
        f_attr = fd[rk].attrs.get("frequency", None)
        freqs = np.array(
            [float(f_attr) if f_attr is not None else freq_idx], dtype=np.float64
        )
    else:
        slices = []
        freq_vals = []
        for idx in indices:
            rk = f"f{idx}_real"
            real = np.asarray(fd[rk], dtype=np.float64)
            imag = np.asarray(fd[f"f{idx}_imag"], dtype=np.float64)
            slices.append(real + 1j * imag)
            f_attr = fd[rk].attrs.get("frequency", None)
            freq_vals.append(float(f_attr) if f_attr is not None else None)
        arr = np.stack(slices, axis=0)  # (n_freq, Nx, Ny, Nz, 3)
        if all(v is not None for v in freq_vals):
            freqs = np.array(freq_vals, dtype=np.float64)
        else:
            freqs = np.array(indices, dtype=np.float64)

    return arr, freqs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def from_openems_hdf5(
    cls: type,
    filepath: str | Path,
    *,
    dump_type: int = 0,
    timestep: int | None = None,
    frequency_index: int | None = None,
    component: Literal["x", "y", "z"] | None = None,
    unit: Any | None = None,
) -> ScalarField | VectorField:
    """Read an openEMS field dump HDF5 file into a ScalarField or VectorField.

    openEMS dumps field data as HDF5 files containing mesh coordinates under
    ``/Mesh/x``, ``/Mesh/y``, ``/Mesh/z`` and field values under
    ``/FieldData/TD`` (time domain) or ``/FieldData/FD`` (frequency domain).
    Each time step or frequency bin stores a 4-D array of shape
    ``(Nx, Ny, Nz, 3)`` where the last axis holds the x, y, z components.

    Parameters
    ----------
    cls : type
        ``ScalarField`` or ``VectorField`` class.
    filepath : str or Path
        Path to the HDF5 dump file.
    dump_type : int, default 0
        openEMS ``DumpType`` value used when the dump was configured.
        Determines the physical quantity and domain:

        - 0  → E-field, time domain
        - 1  → H-field, time domain
        - 10 → E-field, frequency domain
        - 11 → H-field, frequency domain
        - 20-22 → SAR (scalar, frequency domain)
    timestep : int, optional
        For time-domain dumps: which step index to extract.
        ``None`` loads all steps along axis 0.
    frequency_index : int, optional
        For frequency-domain dumps: which frequency index to extract.
        ``None`` loads all frequencies along axis 0.
    component : {"x", "y", "z"}, optional
        Return a single vector component as ``ScalarField``.
        ``None`` returns all three components as ``VectorField``.
    unit : str or astropy.units.Unit, optional
        Override the physical unit.  Defaults to the unit implied by
        *dump_type* (see ``DUMP_TYPE_MAP``).

    Returns
    -------
    ScalarField
        When *component* is given, dump type is SAR, or *cls* is
        ``ScalarField``.
    VectorField
        When *component* is ``None`` and *cls* is ``VectorField``.

    Raises
    ------
    ValueError
        If the required field group is absent, or if a requested time step /
        frequency index does not exist.
    ImportError
        If ``h5py`` is not installed.

    Examples
    --------
    Read all components of a time-domain E-field dump:

    >>> from gwexpy.fields import VectorField
    >>> vf = VectorField.from_openems_hdf5("e_dump.h5", dump_type=0)

    Read only the z-component:

    >>> from gwexpy.fields import ScalarField
    >>> sf = ScalarField.from_openems_hdf5("e_dump.h5", component="z")
    """
    h5py = require_optional("h5py")

    filepath = Path(filepath)
    type_info = DUMP_TYPE_MAP.get(dump_type, (f"DumpType{dump_type}", "", "time"))
    _quantity, auto_unit_str, domain = type_info
    axis0_domain: Literal["time", "frequency"] = (
        "frequency" if domain == "frequency" else "time"
    )
    effective_unit = unit if unit is not None else (auto_unit_str or None)

    with h5py.File(str(filepath), "r") as f:
        # Validate required groups
        if "Mesh" not in f:
            raise ValueError(f"'Mesh' group not found in '{filepath}'.")

        x_coords, y_coords, z_coords = _read_openems_mesh(f)

        is_sar = dump_type in _SAR_DUMP_TYPES

        if axis0_domain == "time":
            if "FieldData/TD" not in f:
                raise ValueError(
                    f"'/FieldData/TD' group not found in '{filepath}'. "
                    "Check that dump_type is correct (time-domain types: 0,1,2,3)."
                )
            data, axis0_vals = _read_openems_td(f, timestep)
        else:
            if "FieldData/FD" not in f:
                raise ValueError(
                    f"'/FieldData/FD' group not found in '{filepath}'. "
                    "Check that dump_type is correct (freq-domain types: 10,11,20-22)."
                )
            data, axis0_vals = _read_openems_fd(f, frequency_index)

    # data shape: (n_axis0, Nx, Ny, Nz, 3) for vector, or (n_axis0, Nx, Ny, Nz) for SAR
    # SAR data has no component axis — treat as scalar
    if is_sar or data.ndim == 4:
        # Already (n_axis0, Nx, Ny, Nz)
        return ScalarField(
            data,
            unit=effective_unit,
            axis0=axis0_vals,
            axis0_domain=axis0_domain,
            axis1=x_coords,
            axis2=y_coords,
            axis3=z_coords,
        )

    # Vector data: (n_axis0, Nx, Ny, Nz, 3)
    comp_idx = {"x": 0, "y": 1, "z": 2}

    if component is not None:
        if component not in comp_idx:
            raise ValueError(
                f"Invalid component {component!r}. Expected 'x', 'y', or 'z'."
            )
        c = comp_idx[component]
        arr = data[..., c]  # (n_axis0, Nx, Ny, Nz)
        return ScalarField(
            arr,
            unit=effective_unit,
            axis0=axis0_vals,
            axis0_domain=axis0_domain,
            axis1=x_coords,
            axis2=y_coords,
            axis3=z_coords,
        )

    # Build one ScalarField per component
    scalar_fields: dict[str, ScalarField] = {}
    for ci, label in _COMP_LABELS.items():
        arr = data[..., ci]  # (n_axis0, Nx, Ny, Nz)
        scalar_fields[label] = ScalarField(
            arr,
            unit=effective_unit,
            axis0=axis0_vals,
            axis0_domain=axis0_domain,
            axis1=x_coords,
            axis2=y_coords,
            axis3=z_coords,
        )

    if issubclass(cls, VectorField):
        return VectorField(scalar_fields)

    # Fallback: return x component as ScalarField
    return scalar_fields["x"]

"""
gwexpy.interop.meep_
--------------------

Interoperability with Meep FDTD simulation output files.

Meep's ``output_field_function`` writes HDF5 files where each field component
is stored as a pair of real/imaginary datasets named ``<field>.r`` / ``<field>.i``
(complex fields) or as a single dataset named ``<field>`` (real-only fields).

This module reads those HDF5 files and reconstructs GWexpy ``ScalarField`` or
``VectorField`` objects without requiring the Meep Python library itself — only
``h5py`` is needed.

References
----------
https://meep.readthedocs.io/en/latest/Python_User_Interface/#output-functions
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from gwexpy.fields import ScalarField, VectorField

from ._optional import require_optional

if TYPE_CHECKING:
    pass

__all__ = ["from_meep_hdf5"]

# Field component names recognised as Cartesian vector components
_VECTOR_COMPONENTS = {
    "ex": "x",
    "ey": "y",
    "ez": "z",
    "hx": "x",
    "hy": "y",
    "hz": "z",
    "dx": "x",
    "dy": "y",
    "dz": "z",
    "bx": "x",
    "by": "y",
    "bz": "z",
}


def _parse_meep_datasets(h5file: Any) -> dict[str, dict[str, str | None]]:
    """Parse HDF5 dataset names into field component groups.

    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file.

    Returns
    -------
    dict
        Mapping ``{field_base: {"real": key, "imag": key | None}}``.
        For real-only fields, ``"imag"`` is ``None``.

    Examples
    --------
    File with ``ex.r``, ``ex.i``, ``ey.r``, ``ey.i`` →
    ``{"ex": {"real": "ex.r", "imag": "ex.i"}, "ey": ...}``

    File with ``ez`` →
    ``{"ez": {"real": "ez", "imag": None}}``
    """
    keys = list(h5file.keys())
    result: dict[str, dict[str, str | None]] = {}

    real_keys = {k for k in keys if k.endswith(".r")}
    imag_keys = {k for k in keys if k.endswith(".i")}

    # Collect complex pairs
    for rk in real_keys:
        base = rk[:-2]  # strip ".r"
        ik = f"{base}.i"
        if ik in imag_keys:
            result[base] = {"real": rk, "imag": ik}

    # Collect real-only (no .r/.i suffix, not already processed)
    processed_bases = set(result.keys())
    for k in keys:
        if k.endswith(".r") or k.endswith(".i"):
            continue
        if k not in processed_bases:
            result[k] = {"real": k, "imag": None}

    return result


def _build_complex_array(
    h5file: Any, real_key: str, imag_key: str | None
) -> np.ndarray:
    """Load datasets and return a complex (or real) numpy array.

    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file.
    real_key : str
        Dataset name for the real part.
    imag_key : str or None
        Dataset name for the imaginary part, or *None* for real-only data.

    Returns
    -------
    ndarray
        Complex array if *imag_key* is given, otherwise real float64 array.
    """
    real = np.asarray(h5file[real_key], dtype=np.float64)
    if imag_key is None:
        return real
    imag = np.asarray(h5file[imag_key], dtype=np.float64)
    if real.shape != imag.shape:
        raise ValueError(
            f"Shape mismatch between real '{real_key}' {real.shape} "
            f"and imaginary '{imag_key}' {imag.shape}."
        )
    return real + 1j * imag


def _build_spatial_coords(
    shape: tuple[int, ...],
    resolution: float,
    origin: tuple[float, ...],
) -> tuple[np.ndarray, ...]:
    """Build spatial coordinate arrays from grid shape and resolution.

    Parameters
    ----------
    shape : tuple[int, ...]
        Spatial dimensions (length 1, 2, or 3).
    resolution : float
        Grid points per unit length.
    origin : tuple[float, ...]
        Grid origin coordinates (same length as *shape*).

    Returns
    -------
    tuple of ndarray
        One 1-D coordinate array per spatial dimension.
    """
    coords = []
    for n, orig in zip(shape, origin):
        dx = 1.0 / resolution if resolution > 0 else 1.0
        coords.append(orig + np.arange(n) * dx)
    return tuple(coords)


def _pad_to_4d(data: np.ndarray) -> np.ndarray:
    """Ensure *data* is 4-dimensional by prepending singleton axes."""
    while data.ndim < 4:
        data = data[np.newaxis, ...]
    return data


def from_meep_hdf5(
    cls: type,
    filepath: str | Path,
    *,
    field_name: str | None = None,
    component: str | None = None,
    resolution: float | None = None,
    origin: tuple[float, ...] | None = None,
    axis0_domain: Literal["time", "frequency"] = "frequency",
    unit: Any | None = None,
) -> Any:
    """Read a Meep HDF5 field output into a ScalarField or VectorField.

    Meep writes electromagnetic field data as HDF5 files with dataset naming
    conventions:

    - Complex fields: ``<name>.r`` (real part) and ``<name>.i`` (imaginary part).
    - Real-only fields: ``<name>`` (no suffix).

    Common field names are ``ex``, ``ey``, ``ez``, ``hx``, ``hy``, ``hz``.

    Parameters
    ----------
    cls : type
        ``ScalarField`` or ``VectorField`` class to instantiate.
    filepath : str or Path
        Path to the HDF5 file produced by Meep.
    field_name : str, optional
        Base name of the field dataset(s) to read.  If *None*, all datasets
        are auto-detected:

        - ``ScalarField`` cls → first dataset found.
        - ``VectorField`` cls → all recognised vector components.
    component : str, optional
        Specific component to extract (e.g., ``"ex"``). When given, a single
        ``ScalarField`` is always returned, regardless of *cls*.
    resolution : float, optional
        Meep resolution (grid points per unit length).  Used to build
        spatial coordinate axes.  Defaults to ``1.0`` when not set.
    origin : tuple of float, optional
        Grid origin coordinates (x, y, z) in Meep length units.
        Defaults to ``(0.0, 0.0, 0.0)``.
    axis0_domain : {"time", "frequency"}, default "frequency"
        Physical domain of axis 0 in the resulting field.  Meep frequency-domain
        outputs should use ``"frequency"``; time snapshots should use ``"time"``.
    unit : str or astropy.units.Unit, optional
        Physical unit to assign to the field data.

    Returns
    -------
    ScalarField
        When ``component`` is given, or only one component is found, or
        *cls* is ``ScalarField``.
    VectorField
        When multiple vector components are found and *cls* is ``VectorField``.

    Raises
    ------
    ValueError
        If no matching datasets are found in the file, or if real/imaginary
        dataset shapes do not match.
    ImportError
        If ``h5py`` is not installed.

    Examples
    --------
    Read the Ez component from a frequency-domain output:

    >>> from gwexpy.fields import ScalarField
    >>> sf = ScalarField.from_meep_hdf5("ez.h5")

    Read all E-field components as a VectorField:

    >>> from gwexpy.fields import VectorField
    >>> vf = VectorField.from_meep_hdf5("fields.h5")
    """
    h5py = require_optional("h5py")

    filepath = Path(filepath)
    res = resolution if resolution is not None else 1.0
    orig = origin if origin is not None else (0.0, 0.0, 0.0)

    with h5py.File(str(filepath), "r") as f:
        dataset_map = _parse_meep_datasets(f)

        if not dataset_map:
            raise ValueError(
                f"No field datasets found in '{filepath}'. "
                "Expected datasets named '<field>.r' / '<field>.i' or '<field>'."
            )

        # Determine which components to load
        if component is not None:
            # Single component requested explicitly
            if component not in dataset_map:
                raise ValueError(
                    f"Component '{component}' not found in '{filepath}'. "
                    f"Available: {sorted(dataset_map.keys())}"
                )
            targets = {component: dataset_map[component]}
        elif field_name is not None:
            if field_name not in dataset_map:
                raise ValueError(
                    f"Field '{field_name}' not found in '{filepath}'. "
                    f"Available: {sorted(dataset_map.keys())}"
                )
            targets = {field_name: dataset_map[field_name]}
        else:
            targets = dataset_map

        # Load all targeted arrays
        arrays: dict[str, np.ndarray] = {}
        for name, keys in targets.items():
            real_key = keys["real"]
            if real_key is None:
                continue
            arrays[name] = _build_complex_array(f, real_key, keys["imag"])

    if not arrays:
        raise ValueError(f"No data could be loaded from '{filepath}'.")

    def _to_scalar_field(name: str, data: np.ndarray) -> Any:
        data4d = _pad_to_4d(data)
        n_spatial = min(data4d.ndim - 1, 3)
        spatial_shape = data4d.shape[1 : 1 + n_spatial]
        coords = _build_spatial_coords(spatial_shape, res, orig[:n_spatial])
        ax_kwargs: dict[str, Any] = {}
        for i, coord in enumerate(coords, start=1):
            ax_kwargs[f"axis{i}"] = coord
        return ScalarField(
            data4d,
            unit=unit,
            axis0_domain=axis0_domain,
            **ax_kwargs,
        )

    # Return single ScalarField if component specified or only one found
    if component is not None or len(arrays) == 1:
        name, data = next(iter(arrays.items()))
        return _to_scalar_field(name, data)

    # Try to build VectorField if cls is VectorField and we have vector components
    _is_vector_cls = issubclass(cls, VectorField)
    if _is_vector_cls:
        components: dict[str, Any] = {}
        for name, data in arrays.items():
            axis_label = _VECTOR_COMPONENTS.get(name.lower(), name)
            components[axis_label] = _to_scalar_field(name, data)
        if components:
            return VectorField(components)

    # Fallback: return first as ScalarField
    name, data = next(iter(arrays.items()))
    return _to_scalar_field(name, data)

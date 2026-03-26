"""
gwexpy.interop.meshio_
----------------------

Interoperability with meshio for unstructured-mesh field data.

meshio reads 40+ mesh formats (XDMF, VTK/VTU, Gmsh, Abaqus, COMSOL, …) into
a uniform ``meshio.Mesh`` object with ``points``, ``cells``, ``point_data``
and ``cell_data``.

Since GWexpy's ``ScalarField`` requires a regular grid, this module
interpolates unstructured mesh data to a regular grid using
``scipy.interpolate.griddata``.

References
----------
https://github.com/nschloe/meshio
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from gwexpy.fields import ScalarField, VectorField

from ._optional import require_optional

__all__ = ["from_meshio", "from_fenics_xdmf", "from_fenics_vtk"]

# Known vector-component naming patterns
_VECTOR_PATTERNS: dict[str, str] = {
    "ex": "x", "ey": "y", "ez": "z",
    "ux": "x", "uy": "y", "uz": "z",
    "vx": "x", "vy": "y", "vz": "z",
    "fx": "x", "fy": "y", "fz": "z",
    "displacement_x": "x", "displacement_y": "y", "displacement_z": "z",
    "velocity_x": "x", "velocity_y": "y", "velocity_z": "z",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_vector_components(data_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Detect vector components from field data key names.

    Returns a mapping ``{"x": array, "y": array, ...}`` if vector components
    are found, otherwise an empty dict.
    """
    components: dict[str, np.ndarray] = {}
    for key, arr in data_dict.items():
        low = key.lower()
        if low in _VECTOR_PATTERNS:
            label = _VECTOR_PATTERNS[low]
            if label not in components:
                components[label] = arr
    return components


def _build_regular_grid(
    points: np.ndarray,
    values: np.ndarray,
    resolution: float,
    method: str = "linear",
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Interpolate scattered data to a regular grid.

    Parameters
    ----------
    points : ndarray, shape (N, 2) or (N, 3)
        Spatial coordinates of the mesh nodes.
    values : ndarray, shape (N,)
        Field values at each node.
    resolution : float
        Grid spacing for the regular output grid.
    method : str
        Interpolation method: ``"linear"`` or ``"nearest"``.

    Returns
    -------
    data : ndarray
        Interpolated data on the regular grid.  Shape depends on dimensionality:
        ``(nx, ny)`` for 2D or ``(nx, ny, nz)`` for 3D.
    axes : tuple of ndarray
        Coordinate arrays for each spatial axis.
    """
    griddata = require_optional("scipy").interpolate.griddata

    ndim = points.shape[1]  # 2 or 3

    # Build axis coordinate arrays from bounding box
    axes = []
    for d in range(ndim):
        lo, hi = points[:, d].min(), points[:, d].max()
        ax = np.arange(lo, hi + resolution * 0.5, resolution)
        axes.append(ax)

    # Create meshgrid
    grids = np.meshgrid(*axes, indexing="ij")
    grid_points = np.column_stack([g.ravel() for g in grids])

    # Interpolate
    interp_values = griddata(points, values, grid_points, method=method)

    # Reshape
    shape = tuple(len(ax) for ax in axes)
    data = interp_values.reshape(shape)

    return data, tuple(axes)


def _get_field_data(
    mesh: Any,
    field_name: str | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract points, values, and all data from a meshio Mesh.

    Returns
    -------
    points : ndarray, shape (N, 2 or 3)
    values : ndarray, shape (N,) or (N, ncomp)
    all_data : dict  (all point_data or cell_data entries)
    """
    points = np.asarray(mesh.points, dtype=np.float64)

    # Try point_data first, then cell_data
    if mesh.point_data:
        data_dict = dict(mesh.point_data)
    elif mesh.cell_data:
        # Cell-data values are associated with cell centroids, not mesh nodes.
        # Interpolating using mesh.points coordinates with cell-data values is
        # incorrect because len(cell_data) != len(points) in general meshes.
        raise ValueError(
            "meshio.Mesh has cell_data but no point_data. "
            "Cell-data interpolation is not supported because cell values "
            "are associated with cell centroids, not mesh nodes. "
            "Convert cell_data to point_data before calling from_meshio() "
            "(e.g. use meshio's cell_data_to_point_data utility)."
        )
    else:
        raise ValueError("meshio.Mesh has neither point_data nor cell_data.")

    if field_name is not None:
        if field_name not in data_dict:
            raise ValueError(
                f"Field {field_name!r} not found. "
                f"Available: {list(data_dict.keys())}"
            )
        values = np.asarray(data_dict[field_name], dtype=np.float64)
    else:
        # Use first available field
        first_key = next(iter(data_dict))
        values = np.asarray(data_dict[first_key], dtype=np.float64)

    # Trim points to 2D if z is constant (all zeros or identical)
    if points.shape[1] == 3:
        z_vals = points[:, 2]
        if np.allclose(z_vals, z_vals[0]):
            points = points[:, :2]

    return points, values, data_dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def from_meshio(
    cls: type,
    mesh: Any,
    *,
    field_name: str | None = None,
    grid_resolution: float,
    method: str = "linear",
    axis0: np.ndarray | None = None,
    axis0_domain: Literal["time", "frequency"] = "time",
    unit: Any | None = None,
) -> ScalarField | VectorField:
    """Convert a ``meshio.Mesh`` to a GWexpy ``ScalarField`` or ``VectorField``.

    The unstructured mesh data is interpolated onto a regular grid using
    ``scipy.interpolate.griddata``.

    Parameters
    ----------
    cls : type
        ``ScalarField`` or ``VectorField``.
    mesh : meshio.Mesh
        Input mesh object.
    field_name : str, optional
        Key in ``point_data`` or ``cell_data`` to use.  Defaults to the first
        available field.
    grid_resolution : float
        Spacing of the output regular grid (mandatory).
    method : {"linear", "nearest"}, default "linear"
        Interpolation method.
    axis0 : ndarray, optional
        Values for axis0 (e.g. time steps).  Default: singleton ``[0]``.
    axis0_domain : {"time", "frequency"}, default "time"
        Physical domain of axis0.
    unit : str or astropy unit, optional
        Unit of the field values.

    Returns
    -------
    ScalarField or VectorField

    Notes
    -----
    Only ``point_data`` is supported for interpolation.  If the mesh contains
    only ``cell_data`` and no ``point_data``, a ``ValueError`` is raised.
    To use cell data, first convert it to point data
    (e.g. meshio's ``cell_data_to_point_data`` utility).
    """
    require_optional("scipy")

    points, values, data_dict = _get_field_data(mesh, field_name)
    ndim = points.shape[1]  # 2 or 3

    # Check for vector components
    vec_comps = _detect_vector_components(data_dict) if issubclass(cls, VectorField) else {}

    if vec_comps and issubclass(cls, VectorField):
        # Build VectorField from detected components
        scalar_fields: dict[str, ScalarField] = {}
        for comp_label, comp_values in sorted(vec_comps.items()):
            data, axes = _build_regular_grid(points, comp_values, grid_resolution, method)
            sf = _pack_as_scalar_field(data, axes, ndim, axis0, axis0_domain, unit)
            scalar_fields[comp_label] = sf
        return VectorField(scalar_fields)

    # Scalar field path
    # Handle multi-component values (e.g., displacement [N, 3])
    if values.ndim == 2 and values.shape[1] in (2, 3) and issubclass(cls, VectorField):
        labels = ["x", "y", "z"][:values.shape[1]]
        scalar_fields = {}
        for i, label in enumerate(labels):
            data, axes = _build_regular_grid(points, values[:, i], grid_resolution, method)
            sf = _pack_as_scalar_field(data, axes, ndim, axis0, axis0_domain, unit)
            scalar_fields[label] = sf
        return VectorField(scalar_fields)

    if values.ndim == 2:
        values = values[:, 0]  # take first component for ScalarField

    data, axes = _build_regular_grid(points, values, grid_resolution, method)
    return _pack_as_scalar_field(data, axes, ndim, axis0, axis0_domain, unit)


def _pack_as_scalar_field(
    data: np.ndarray,
    axes: tuple[np.ndarray, ...],
    ndim: int,
    axis0: np.ndarray | None,
    axis0_domain: str,
    unit: Any | None,
) -> ScalarField:
    """Wrap interpolated data into a 4D ScalarField."""
    if axis0 is None:
        axis0 = np.array([0.0])

    # data is (nx, ny) for 2D or (nx, ny, nz) for 3D
    # Prepend axis0 dimension
    data_4d = data[np.newaxis, ...]  # (1, nx, ny) or (1, nx, ny, nz)

    # Pad to 4D
    while data_4d.ndim < 4:
        data_4d = data_4d[..., np.newaxis]

    ax1 = axes[0] if len(axes) > 0 else None
    ax2 = axes[1] if len(axes) > 1 else None
    ax3 = axes[2] if len(axes) > 2 else None

    return ScalarField(
        data_4d,
        unit=unit,
        axis0=axis0,
        axis0_domain=axis0_domain,
        axis1=ax1,
        axis2=ax2,
        axis3=ax3,
    )


def from_fenics_xdmf(
    cls: type,
    filepath: str | Path,
    *,
    field_name: str | None = None,
    grid_resolution: float,
    method: str = "linear",
    unit: Any | None = None,
) -> ScalarField | VectorField:
    """Read a dolfinx XDMF file via meshio and convert to ScalarField.

    Parameters
    ----------
    cls : type
        ``ScalarField`` or ``VectorField``.
    filepath : str or Path
        Path to the ``.xdmf`` file.
    field_name : str, optional
        Field name in the XDMF data.
    grid_resolution : float
        Regular grid spacing (mandatory).
    method : {"linear", "nearest"}, default "linear"
        Interpolation method.
    unit : str or astropy unit, optional
        Physical unit of the field values.

    Returns
    -------
    ScalarField or VectorField
    """
    meshio = require_optional("meshio")
    mesh = meshio.read(str(filepath))
    return from_meshio(
        cls, mesh, field_name=field_name, grid_resolution=grid_resolution,
        method=method, unit=unit,
    )


def from_fenics_vtk(
    cls: type,
    filepath: str | Path,
    *,
    field_name: str | None = None,
    grid_resolution: float,
    method: str = "linear",
    unit: Any | None = None,
) -> ScalarField | VectorField:
    """Read a VTK/VTU file via meshio and convert to ScalarField.

    Parameters
    ----------
    cls : type
        ``ScalarField`` or ``VectorField``.
    filepath : str or Path
        Path to a ``.vtk`` or ``.vtu`` file.
    field_name : str, optional
        Field name in the VTK point/cell data.
    grid_resolution : float
        Regular grid spacing (mandatory).
    method : {"linear", "nearest"}, default "linear"
        Interpolation method.
    unit : str or astropy unit, optional
        Physical unit of the field values.

    Returns
    -------
    ScalarField or VectorField
    """
    meshio = require_optional("meshio")
    mesh = meshio.read(str(filepath))
    return from_meshio(
        cls, mesh, field_name=field_name, grid_resolution=grid_resolution,
        method=method, unit=unit,
    )

"""
gwexpy.interop.emg3d_
---------------------

Interoperability with emg3d electromagnetic modelling fields.

``emg3d.fields.Field`` stores electric or magnetic field components on a
staggered grid (Yee grid).  Each Cartesian component has a slightly different
shape because it lives on edges (E-field) or faces (H-field).

GWexpy's ``ScalarField`` requires all spatial components to share the same
shape, so this module interpolates each component to the *cell centres* of the
``TensorMesh`` before packing them into a ``VectorField``.  The original
staggered layout is recorded in the ``ScalarField`` metadata.

Only the ``emg3d`` Python package is required; no external C dependencies
beyond those already installed with emg3d.

References
----------
https://emg3d.emsig.xyz/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gwexpy.fields import ScalarField, VectorField

from ._optional import require_optional

__all__ = ["from_emg3d_field", "from_emg3d_h5", "to_emg3d_field"]


# ---------------------------------------------------------------------------
# Internal helpers — staggered → cell-centre interpolation
# ---------------------------------------------------------------------------


def _interp_edge_to_cell(arr: np.ndarray, axis: int) -> np.ndarray:
    """Interpolate edge-centred values to cell centres along *axis*.

    For the E-field, component *i* lives on edges parallel to axis *i*.  Along
    axis *i* the values are at cell centres already (``nC`` dimension), but the
    two transverse axes have ``nN = nC + 1`` nodes.  This function averages
    adjacent node pairs to produce a cell-centre array.

    Parameters
    ----------
    arr : ndarray
        Input array with node spacing along *axis*.  Shape must have
        ``arr.shape[axis] == nN``.
    axis : int
        Axis to interpolate (0, 1, or 2).

    Returns
    -------
    ndarray
        Array with ``nN - 1`` values along *axis*.
    """
    slc_lo = [slice(None)] * arr.ndim
    slc_hi = [slice(None)] * arr.ndim
    slc_lo[axis] = slice(None, -1)
    slc_hi[axis] = slice(1, None)
    return 0.5 * (arr[tuple(slc_lo)] + arr[tuple(slc_hi)])


def _interp_face_to_cell(arr: np.ndarray, axis: int) -> np.ndarray:
    """Interpolate face-centred values to cell centres along *axis*.

    For the H-field, component *i* lives on faces perpendicular to axis *i*.
    Along axis *i* the component spans ``nN`` faces.  This function averages
    adjacent face pairs to give ``nC = nN - 1`` cell-centre values.

    Parameters
    ----------
    arr : ndarray
        Face-centred array. Shape has ``arr.shape[axis] == nN``.
    axis : int
        Axis to interpolate.

    Returns
    -------
    ndarray
        Array with ``nN - 1`` values along *axis*.
    """
    return _interp_edge_to_cell(arr, axis)  # same averaging formula


def _build_cell_center_coords(
    mesh: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, z) cell-centre coordinate arrays from a TensorMesh.

    Parameters
    ----------
    mesh : emg3d.meshes.TensorMesh
        emg3d mesh object.

    Returns
    -------
    tuple of ndarray
        ``(cell_centers_x, cell_centers_y, cell_centers_z)``
    """
    return (
        np.asarray(mesh.cell_centers_x, dtype=np.float64),
        np.asarray(mesh.cell_centers_y, dtype=np.float64),
        np.asarray(mesh.cell_centers_z, dtype=np.float64),
    )


def _build_node_coords(
    mesh: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, z) node coordinate arrays from a TensorMesh."""
    return (
        np.asarray(mesh.nodes_x, dtype=np.float64),
        np.asarray(mesh.nodes_y, dtype=np.float64),
        np.asarray(mesh.nodes_z, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def from_emg3d_field(
    cls: type,
    field: Any,
    *,
    component: str | None = None,
    interpolate_to_cell_center: bool = True,
) -> ScalarField | VectorField:
    """Convert an ``emg3d.fields.Field`` to a GWexpy ``VectorField``.

    The emg3d ``Field`` stores each Cartesian component (fx, fy, fz) on a
    staggered (Yee) grid where component shapes may differ.  When
    *interpolate_to_cell_center* is ``True`` (default), each component is
    averaged to the common cell-centre grid so that all three ``ScalarField``
    objects share the same spatial shape.

    Parameters
    ----------
    cls : type
        ``VectorField`` (default) or ``ScalarField`` class.
    field : emg3d.fields.Field
        emg3d field object.  Must have attributes ``fx``, ``fy``, ``fz``,
        ``grid``, and ``electric`` (bool).
    component : {"x", "y", "z"}, optional
        Extract a single Cartesian component and return a ``ScalarField``.
        ``None`` returns all three components as a ``VectorField``.
    interpolate_to_cell_center : bool, default True
        Interpolate staggered-grid components to cell centres.  Setting this
        to ``False`` raises ``ValueError`` if the shapes differ.

    Returns
    -------
    ScalarField
        When *component* is specified or *cls* is ``ScalarField``.
    VectorField
        When *component* is ``None`` and *cls* is ``VectorField``.

    Raises
    ------
    ValueError
        If ``interpolate_to_cell_center=False`` and component shapes differ.

    Examples
    --------
    >>> from gwexpy.fields import VectorField
    >>> vf = VectorField.from_emg3d_field(field)

    >>> from gwexpy.fields import ScalarField
    >>> sf = ScalarField.from_emg3d_field(field, component="z")
    """
    mesh = field.grid
    electric: bool = bool(field.electric)

    # Physical unit
    unit_str = "V/m" if electric else "A/m"
    interp_label = "edge" if electric else "face"

    # axis0 from frequency (singleton if available, else None)
    freq = getattr(field, "frequency", None)
    if freq is not None:
        axis0_vals = np.array([float(freq)])
        axis0_domain = "frequency"
    else:
        axis0_vals = None
        axis0_domain = "time"

    # Raw components: view arrays from the 1-D flat buffer
    fx = np.asarray(
        field.fx, dtype=np.complex128 if np.iscomplexobj(field.fx) else np.float64
    )
    fy = np.asarray(
        field.fy, dtype=np.complex128 if np.iscomplexobj(field.fy) else np.float64
    )
    fz = np.asarray(
        field.fz, dtype=np.complex128 if np.iscomplexobj(field.fz) else np.float64
    )

    if interpolate_to_cell_center:
        # For E-field (edge): fx is (nCx, nNy, nNz) → interp axes 1,2
        #                      fy is (nNx, nCy, nNz) → interp axes 0,2
        #                      fz is (nNx, nNy, nCz) → interp axes 0,1
        # For H-field (face): fx is (nNx, nCy, nCz) → interp axis 0
        #                      fy is (nCx, nNy, nCz) → interp axis 1
        #                      fz is (nCx, nCy, nNz) → interp axis 2
        if electric:
            # E-field on edges
            # fx: (nCx, nNy, nNz) → (nCx, nCy, nCz) interpolate axes 1 and 2
            fx_c = _interp_edge_to_cell(_interp_edge_to_cell(fx, axis=1), axis=2)
            # fy: (nNx, nCy, nNz) → interpolate axes 0 and 2
            fy_c = _interp_edge_to_cell(_interp_edge_to_cell(fy, axis=0), axis=2)
            # fz: (nNx, nNy, nCz) → interpolate axes 0 and 1
            fz_c = _interp_edge_to_cell(_interp_edge_to_cell(fz, axis=0), axis=1)
        else:
            # H-field on faces
            # fx: (nNx, nCy, nCz) → interpolate axis 0
            fx_c = _interp_face_to_cell(fx, axis=0)
            # fy: (nCx, nNy, nCz) → interpolate axis 1
            fy_c = _interp_face_to_cell(fy, axis=1)
            # fz: (nCx, nCy, nNz) → interpolate axis 2
            fz_c = _interp_face_to_cell(fz, axis=2)

        cx, cy, cz = _build_cell_center_coords(mesh)
        metadata = {"interpolated_from": interp_label}
    else:
        # No interpolation — raise if shapes differ
        if not (fx.shape == fy.shape == fz.shape):
            raise ValueError(
                f"Component shapes differ ({fx.shape}, {fy.shape}, {fz.shape}). "
                "Set interpolate_to_cell_center=True to align them."
            )
        fx_c, fy_c, fz_c = fx, fy, fz
        cx, cy, cz = _build_node_coords(mesh)
        metadata = {}

    # Add singleton axis0 dimension
    def _add_axis0(arr: np.ndarray) -> np.ndarray:
        return arr[np.newaxis, ...]  # (1, Nx, Ny, Nz)

    def _make_sf(arr: np.ndarray) -> ScalarField:
        sf = ScalarField(
            _add_axis0(arr),
            unit=unit_str,
            axis0=axis0_vals,
            axis0_domain=axis0_domain,
            axis1=cx,
            axis2=cy,
            axis3=cz,
        )
        sf.metadata = metadata
        return sf

    comp_map = {"x": fx_c, "y": fy_c, "z": fz_c}

    if component is not None:
        if component not in comp_map:
            raise ValueError(
                f"Invalid component {component!r}. Expected 'x', 'y', or 'z'."
            )
        return _make_sf(comp_map[component])

    if issubclass(cls, ScalarField) and not issubclass(cls, VectorField):
        return _make_sf(fx_c)

    return VectorField({"x": _make_sf(fx_c), "y": _make_sf(fy_c), "z": _make_sf(fz_c)})


def to_emg3d_field(
    vf: VectorField | ScalarField,
    *,
    frequency: float | None = None,
    electric: bool = True,
) -> Any:
    """Convert a GWexpy VectorField back to an ``emg3d.fields.Field``.

    Parameters
    ----------
    vf : VectorField or ScalarField
        Source field.  For ``VectorField``, components ``"x"``, ``"y"``,
        ``"z"`` are expected.  For ``ScalarField`` the single component is
        broadcast as the x-component.
    frequency : float, optional
        Field frequency in Hz.  Overrides ``vf``'s axis0 value when given.
    electric : bool, default True
        Whether the field is an E-field (``True``) or H-field (``False``).

    Returns
    -------
    emg3d.fields.Field
        Reconstructed field object.

    Raises
    ------
    ImportError
        If ``emg3d`` is not installed.
    ValueError
        If the VectorField does not contain exactly 3 components.
    """
    emg3d = require_optional("emg3d")

    if isinstance(vf, VectorField):
        keys = list(vf.keys())
        if len(keys) != 3:
            raise ValueError(f"VectorField must have exactly 3 components; got {keys}.")
        fx = np.asarray(vf["x"].value).ravel()
        fy = np.asarray(vf["y"].value).ravel()
        fz = np.asarray(vf["z"].value).ravel()
        flat = np.concatenate([fx, fy, fz])

        # Reconstruct a minimal TensorMesh from the ScalarField axes
        sf_x = vf["x"]
        hx = np.diff(
            np.asarray(
                sf_x._axis1_index.value
                if hasattr(sf_x._axis1_index, "value")
                else sf_x._axis1_index
            )
        )
        hy = np.diff(
            np.asarray(
                sf_x._axis2_index.value
                if hasattr(sf_x._axis2_index, "value")
                else sf_x._axis2_index
            )
        )
        hz = np.diff(
            np.asarray(
                sf_x._axis3_index.value
                if hasattr(sf_x._axis3_index, "value")
                else sf_x._axis3_index
            )
        )
        origin_x = float(
            np.asarray(
                sf_x._axis1_index.value
                if hasattr(sf_x._axis1_index, "value")
                else sf_x._axis1_index
            )[0]
        )
        origin_y = float(
            np.asarray(
                sf_x._axis2_index.value
                if hasattr(sf_x._axis2_index, "value")
                else sf_x._axis2_index
            )[0]
        )
        origin_z = float(
            np.asarray(
                sf_x._axis3_index.value
                if hasattr(sf_x._axis3_index, "value")
                else sf_x._axis3_index
            )[0]
        )
    else:
        arr = np.asarray(vf.value)
        flat = arr.ravel()
        hx = np.diff(
            np.asarray(
                vf._axis1_index.value
                if hasattr(vf._axis1_index, "value")
                else vf._axis1_index
            )
        )
        hy = np.diff(
            np.asarray(
                vf._axis2_index.value
                if hasattr(vf._axis2_index, "value")
                else vf._axis2_index
            )
        )
        hz = np.diff(
            np.asarray(
                vf._axis3_index.value
                if hasattr(vf._axis3_index, "value")
                else vf._axis3_index
            )
        )
        origin_x = float(
            np.asarray(
                vf._axis1_index.value
                if hasattr(vf._axis1_index, "value")
                else vf._axis1_index
            )[0]
        )
        origin_y = float(
            np.asarray(
                vf._axis2_index.value
                if hasattr(vf._axis2_index, "value")
                else vf._axis2_index
            )[0]
        )
        origin_z = float(
            np.asarray(
                vf._axis3_index.value
                if hasattr(vf._axis3_index, "value")
                else vf._axis3_index
            )[0]
        )

    # Resolve frequency
    if frequency is None:
        if isinstance(vf, VectorField):
            sf_ref = next(iter(vf.values()))
        else:
            sf_ref = vf
        ax0 = sf_ref._axis0_index
        ax0_arr = np.asarray(ax0.value if hasattr(ax0, "value") else ax0)
        frequency = float(ax0_arr[0]) if ax0_arr.size > 0 else None

    grid = emg3d.TensorMesh(
        [hx, hy, hz],
        origin=(origin_x, origin_y, origin_z),
    )
    return emg3d.Field(grid, data=flat, frequency=frequency, electric=electric)


def from_emg3d_h5(
    cls: type,
    filepath: str | Path,
    *,
    name: str = "field",
    component: str | None = None,
    interpolate_to_cell_center: bool = True,
) -> ScalarField | VectorField:
    """Load an emg3d field saved via ``emg3d.save()`` and convert it.

    Parameters
    ----------
    cls : type
        ``VectorField`` or ``ScalarField``.
    filepath : str or Path
        Path to the HDF5 / npz / json file written by ``emg3d.save()``.
    name : str, default "field"
        Key under which the field was saved (kwarg name passed to
        ``emg3d.save(**{name: field})``).
    component : {"x", "y", "z"}, optional
        Extract a single component.
    interpolate_to_cell_center : bool, default True
        See :func:`from_emg3d_field`.

    Returns
    -------
    ScalarField or VectorField
    """
    emg3d = require_optional("emg3d")

    data = emg3d.load(str(filepath))
    if name not in data:
        available = list(data.keys())
        raise ValueError(
            f"Key {name!r} not found in '{filepath}'. Available: {available}"
        )
    field = data[name]
    return from_emg3d_field(
        cls,
        field,
        component=component,
        interpolate_to_cell_center=interpolate_to_cell_center,
    )

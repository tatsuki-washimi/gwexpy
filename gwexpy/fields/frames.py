r"""Local Cartesian coordinate frames, detector frames, and spatial rotations (#556).

This module provides the shared spatial-geometry and frame-transformation layer
for :class:`~gwexpy.fields.ScalarField`, :class:`~gwexpy.fields.VectorField`,
and :class:`~gwexpy.fields.TensorField`:

- :class:`GridGeometry` — origin + 3x3 basis matrix + parent frame metadata
  attached via ``field._gwex_frame`` (exposed as ``field.frame``) so grid
  orientation is preserved without mutating the 1D axis model on passive basis
  changes.
- :class:`DetectorFrame` — detector-anchored frame constructed from basis axes
  (:meth:`DetectorFrame.from_axes`), interferometer arm vectors
  (:meth:`DetectorFrame.from_arms`, distinguishing ``orthogonalize=True``
  orthonormal basis construction from ``orthogonalize=False`` raw-arm direction
  preservation), or a detector code (:meth:`DetectorFrame.from_detector`),
  which explicitly distinguishes:
  - ``frame_kind="local_enu"`` — site-centered local East-North-Up reference
    frame constructed directly from the detector site position ``(lon, lat)``
    in :data:`gwexpy.plot.geomap.DETECTORS` (which stores only site coordinates
    and display metadata, not arm orientations).
  - ``frame_kind="arms"`` (default) — arm-aligned detector frame constructed by
    combining the site position with physical arm orientations from either
    :data:`gwexpy.utils.lal.LAL_DETECTORS` (full 3D ``xArmAzimuthRadians``,
    ``yArmAzimuthRadians``, ``xArmAltitudeRadians``, ``yArmAltitudeRadians``
    when LALSuite is available and ``arm_source="lal"``) or the horizontal
    arm-azimuth table :data:`DETECTOR_ARM_AZIMUTHS_DEG` (``arm_source="nominal"``).
- Explicit separation of operations that "rotate" conflates:
  1. ``to_frame(frame)`` — passive basis/component change with **no spatial
     interpolation**: physical sample positions and geometry are preserved,
     scalar values transform as :math:`\phi' = \phi`, vector components as
     :math:`v' = R v`, and rank-2 tensor components as :math:`T' = R T R^{\mathsf T}`.
  2. ``regrid_to_frame(frame, *, resolution=None, method='bilinear')`` —
     explicit spatial resampling onto a rectilinear grid aligned with the
     target frame's axes (plus :math:`v' = R v` / :math:`T' = R T R^{\mathsf T}`
     component transformation for vectors and tensors).
  3. ``rotate_horizontal(angle, *, center=None, resample=False, method='bilinear')``
     — horizontal rotation shortcut around the vertical (z) axis, defaulting to
     passive basis change (``resample=False``) or explicit regridding
     (``resample=True``).
  4. ``assume_frame(frame)`` — metadata-only frame assignment without
     transforming coordinates or components.

Time-dependent celestial frames (Astropy ITRS/GCRS/ICRS) are intentionally kept
separate from this local/detector frame module and tracked as an independent
unit under #556.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from astropy import units as u

from gwexpy.plot.geomap import DETECTORS

if TYPE_CHECKING:
    from .scalar import ScalarField
    from .tensor import TensorField
    from .vector import VectorField

__all__ = [
    "DETECTOR_ARM_AZIMUTHS_DEG",
    "DetectorFrame",
    "GridGeometry",
    "horizontal_rotation_matrix",
    "transform_Matrix_between_frames",
]

# Standard interferometer X/Y arm azimuths (degrees clockwise from local North,
# i.e. compass bearing in the local ENU tangent plane: East = 90 deg, North = 0 deg).
DETECTOR_ARM_AZIMUTHS_DEG: dict[str, tuple[float, float]] = {
    "K1": (61.69, 151.69),  # KAGRA X-arm ~ N61.69E, Y-arm ~ S28.31E
    "H1": (324.00, 234.00),  # LIGO Hanford X-arm ~ N36W, Y-arm ~ S54W
    "L1": (252.28, 162.28),  # LIGO Livingston X-arm ~ S72.28W, Y-arm ~ S17.72E
    "V1": (19.43, 289.43),  # Virgo North arm ~ N19.43E, West arm ~ N70.57W
    "G1": (25.94, 291.61),  # GEO600 non-orthogonal arms (~94.33 deg opening angle)
}

_ELLIPSOID_PARAMS: dict[str, tuple[float, float]] = {
    "grs80": (6378137.0, 298.257222101),
    "wgs84": (6378137.0, 298.257223563),
    "sphere": (6371008.7714, 0.0),
}

_ORTHOGONALITY_ATOL = 1e-10


def _as_3vec_float(
    vec: Any, *, name: str, expected_unit: u.UnitBase | None = None
) -> np.ndarray:
    if isinstance(vec, u.Quantity):
        if expected_unit is not None:
            arr = np.asarray(vec.to_value(expected_unit), dtype=np.float64)
        else:
            if not vec.unit.is_equivalent(u.dimensionless_unscaled):
                raise ValueError(
                    f"{name} must be dimensionless when no unit is expected, got {vec.unit}."
                )
            arr = np.asarray(vec.to_value(u.dimensionless_unscaled), dtype=np.float64)
    else:
        arr = np.asarray(vec, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3-element vector, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values, got {arr!r}.")
    return arr


def _normalize_vec(v: np.ndarray, *, name: str) -> np.ndarray:
    nrm = float(np.linalg.norm(v))
    if not np.isfinite(nrm) or nrm <= 1e-15:
        raise ValueError(f"{name} must be non-zero, got norm {nrm:g}.")
    return v / nrm


def _validate_basis_matrix(
    basis: Any, *, require_orthonormal: bool = False
) -> tuple[
    tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    bool,
]:
    arr = np.asarray(basis, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"basis must have shape (3, 3), got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("basis matrix must contain only finite numbers.")
    det = float(np.linalg.det(arr))
    if abs(det) <= 1e-12:
        raise ValueError(
            f"basis vectors are linearly dependent (det={det:.3e}); "
            "a valid 3D basis requires 3 independent basis vectors."
        )
    gram = arr @ arr.T
    is_ortho = bool(
        np.allclose(gram, np.eye(3), rtol=0.0, atol=_ORTHOGONALITY_ATOL) and det > 0.0
    )
    if require_orthonormal and not is_ortho:
        raise ValueError(
            "Operation requires an orthonormal right-handed basis "
            f"(B @ B^T == I, det == +1), got det={det:.6f}, B @ B^T=\n{gram!r}. "
            "Construct the frame with orthogonalize=True when an orthonormal "
            "basis transformation is needed."
        )
    frozen = (
        (float(arr[0, 0]), float(arr[0, 1]), float(arr[0, 2])),
        (float(arr[1, 0]), float(arr[1, 1]), float(arr[1, 2])),
        (float(arr[2, 0]), float(arr[2, 1]), float(arr[2, 2])),
    )
    return frozen, is_ortho


def horizontal_rotation_matrix(angle: float | u.Quantity) -> np.ndarray:
    r"""Return the 3x3 passive rotation matrix for a horizontal counter-clockwise angle.

    Rotating the frame axes counter-clockwise by :math:`\theta` around the +z
    axis maps old-frame components :math:`(v_x, v_y, v_z)` to new-frame
    components :math:`v' = R v` where:

    .. math::

        R = \begin{pmatrix}
            \cos\theta & \sin\theta & 0 \\
            -\sin\theta & \cos\theta & 0 \\
            0 & 0 & 1
        \end{pmatrix}.
    """
    if isinstance(angle, u.Quantity):
        theta = float(angle.to_value(u.rad))
    else:
        theta = float(np.deg2rad(float(angle)))
    if not np.isfinite(theta):
        raise ValueError(f"Rotation angle must be finite, got {angle!r}.")
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _tangent_offset_metres(
    lon0_deg: float,
    lat0_deg: float,
    lon1_deg: float,
    lat1_deg: float,
    *,
    ellipsoid: str = "GRS80",
) -> tuple[float, float]:
    """Return local tangent-plane ``(dx_east_m, dy_north_m)`` from ``(lon0, lat0)`` to ``(lon1, lat1)``."""
    key = ellipsoid.lower()
    if key not in _ELLIPSOID_PARAMS:
        raise ValueError(f"Unknown ellipsoid {ellipsoid!r}.")
    a_ellip, inv_f = _ELLIPSOID_PARAMS[key]
    if inv_f == 0.0:
        e2 = 0.0
    else:
        f_flat = 1.0 / inv_f
        e2 = f_flat * (2.0 - f_flat)
    phi0 = math.radians(lat0_deg)
    sin_phi0 = math.sin(phi0)
    cos_phi0 = math.cos(phi0)
    w = math.sqrt(1.0 - e2 * sin_phi0 * sin_phi0)
    n_phi0 = a_ellip / w
    m_phi0 = a_ellip * (1.0 - e2) / (w**3)
    deg_to_rad = math.pi / 180.0
    dx = n_phi0 * cos_phi0 * (lon1_deg - lon0_deg) * deg_to_rad
    dy = m_phi0 * (lat1_deg - lat0_deg) * deg_to_rad
    return float(dx), float(dy)


@dataclass(frozen=True)
class GridGeometry:
    r"""Spatial frame and orientation metadata for a 3D rectilinear grid (#556).

    ``origin`` and the rows of ``basis`` (:math:`\mathbf{e}_1, \mathbf{e}_2,
    \mathbf{e}_3`) are expressed in the reference coordinate system named by
    ``parent_frame`` (default ``"local_enu"``, where axes are East, North, Up
    in metres).

    A grid point with local frame coordinates :math:`\mathbf{x} = (x_1, x_2, x_3)`
    corresponds to parent-frame position:

    .. math::

        \mathbf{r}_{\mathrm{parent}} = \mathbf{o} + B^{\mathsf T} \mathbf{x}.
    """

    name: str = "local_enu"
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    basis: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    parent_frame: str = "local_enu"
    local_origin_lonlat: tuple[float, float] | None = None
    vertical_datum: str = "geoid"
    is_orthonormal: bool = True
    raw_arms: tuple[tuple[float, float, float], tuple[float, float, float]] | None = (
        None
    )
    grid_origin: tuple[float, float, float] | None = None
    grid_basis: (
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
        | None
    ) = None
    history: tuple[str, ...] = ()

    @classmethod
    def standard_enu(
        cls,
        *,
        local_origin_lonlat: tuple[float, float] | None = None,
        vertical_datum: str = "geoid",
    ) -> GridGeometry:
        """Return the canonical East-North-Up local Cartesian frame."""
        return cls(
            name="local_enu",
            origin=(0.0, 0.0, 0.0),
            basis=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            parent_frame="local_enu",
            local_origin_lonlat=local_origin_lonlat,
            vertical_datum=vertical_datum,
            is_orthonormal=True,
        )

    @property
    def basis_matrix(self) -> np.ndarray:
        """Return the 3x3 basis matrix whose rows are the component/target frame basis vectors."""
        return np.asarray(self.basis, dtype=np.float64)

    @property
    def origin_vector(self) -> np.ndarray:
        """Return the 3-element origin vector in parent-frame metres."""
        return np.asarray(self.origin, dtype=np.float64)

    @property
    def lattice_origin(self) -> tuple[float, float, float]:
        """Return the parent-frame origin of the physical 1D coordinate lattice."""
        return self.grid_origin if self.grid_origin is not None else self.origin

    @property
    def lattice_basis(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """Return the 3x3 basis tuple of the physical 1D coordinate lattice."""
        return self.grid_basis if self.grid_basis is not None else self.basis

    @property
    def lattice_origin_vector(self) -> np.ndarray:
        """Return the 3-element parent-frame origin of the physical sample lattice."""
        return np.asarray(self.lattice_origin, dtype=np.float64)

    @property
    def lattice_basis_matrix(self) -> np.ndarray:
        """Return the 3x3 basis matrix whose rows are the physical 1D grid axis directions."""
        return np.asarray(self.lattice_basis, dtype=np.float64)

    def parent_position(self, local_coords: Any) -> np.ndarray:
        r"""Map local 1D grid coordinate(s) :math:`\mathbf{x}` to parent-frame position(s) :math:`\mathbf{r}`.

        Uses the physical lattice geometry
        :math:`\mathbf{r}_{\mathrm{parent}} = \mathbf{o}_{\mathrm{grid}} + B_{\mathrm{grid}}^{\mathsf T} \mathbf{x}`
        so passive basis changes (:meth:`to_frame`) preserve exact physical
        sample positions.
        """
        arr = np.asarray(
            u.Quantity(local_coords, copy=False).to_value(u.m)
            if isinstance(local_coords, u.Quantity)
            else local_coords,
            dtype=np.float64,
        )
        if arr.shape[-1] != 3:
            raise ValueError(
                f"local_coords must have trailing dimension 3, got shape {arr.shape}."
            )
        return self.lattice_origin_vector + np.einsum(
            "ij,...i->...j", self.lattice_basis_matrix, arr
        )

    def frame_position(self, local_coords: Any) -> np.ndarray:
        r"""Return coordinates of lattice point(s) :math:`\mathbf{x}` in the active frame ``(origin, basis)``.

        Maps :math:`\mathbf{x}` to parent-frame position
        :math:`\mathbf{r}_{\mathrm{parent}}` via :meth:`parent_position` and
        projects into the active frame via
        :math:`\mathbf{x}_{\mathrm{frame}} = B (\mathbf{r}_{\mathrm{parent}} - \mathbf{o})`
        (or :math:`B^{-\mathsf T} (\mathbf{r}_{\mathrm{parent}} - \mathbf{o})`
        when ``basis`` is non-orthonormal).
        """
        r_parent = self.parent_position(local_coords)
        rel = r_parent - self.origin_vector
        b_mat = self.basis_matrix
        if self.is_orthonormal:
            return np.einsum("ij,...j->...i", b_mat, rel)
        b_inv_t = np.linalg.inv(b_mat.T)
        return np.einsum("ij,...j->...i", b_inv_t, rel)

    def with_history(self, entry: str) -> GridGeometry:
        """Return a new :class:`GridGeometry` with *entry* appended to history."""
        return replace(self, history=(*self.history, str(entry)))


@dataclass(frozen=True)
class DetectorFrame(GridGeometry):
    r"""Coordinate frame anchored at a gravitational-wave detector (#556).

    ``DetectorFrame`` subclasses :class:`GridGeometry` so it can be passed
    directly to ``to_frame()``, ``regrid_to_frame()``, and ``assume_frame()``.

    Distinction between ``orthogonalize=True`` and ``orthogonalize=False`` in
    :meth:`from_arms`:

    - ``orthogonalize=True`` (default): constructs a right-handed orthonormal
      basis :math:`(\hat{e}_x, \hat{e}_y, \hat{e}_z)`. When the two arm
      vectors :math:`\hat{u}_x, \hat{u}_y` are not strictly orthogonal (such
      as GEO600 or terrain-tilted arms), ``orthogonalize_mode="x_primary"``
      keeps :math:`\hat{e}_x = \hat{u}_x`, defines
      :math:`\hat{e}_z = \mathrm{normalize}(\hat{u}_x \times \hat{u}_y)`,
      and sets :math:`\hat{e}_y = \hat{e}_z \times \hat{e}_x`;
      ``orthogonalize_mode="symmetric"`` rotates both horizontal arms
      symmetrically around their bisector so both share equal angular offset.
      In both modes the original unit arm vectors are preserved on
      ``frame.raw_arms``. Orthonormal frames are required for passive metric
      basis rotations (:math:`v' = R v`, :math:`T' = R T R^{\mathsf T}` with
      :math:`R \in SO(3)`).
    - ``orthogonalize=False``: keeps the normalized physical arm directions
      :math:`(\hat{u}_x, \hat{u}_y)` verbatim as the first two rows of
      ``basis`` and :math:`\hat{e}_z = \mathrm{normalize}(\hat{u}_x \times
      \hat{u}_y)` as the third row. If the arms are not orthogonal,
      ``is_orthonormal`` is ``False``; directional projections along the
      physical arms remain exact, while operations that require an orthogonal
      Cartesian metric (:meth:`to_frame` with ``require_orthogonal=True`` or
      rectilinear :meth:`regrid_to_frame`) reject non-orthonormal frames with a
      clear ``ValueError``.
    """

    detector: str | None = None
    frame_kind: Literal["arms", "local_enu"] = "arms"
    arm_source: str | None = None

    @classmethod
    def from_axes(
        cls,
        ex: Any,
        ey: Any,
        ez: Any | None = None,
        *,
        origin: Any = (0.0, 0.0, 0.0),
        name: str = "detector",
        detector: str | None = None,
        parent_frame: str = "local_enu",
        local_origin_lonlat: tuple[float, float] | None = None,
        vertical_datum: str = "geoid",
        require_orthonormal: bool = True,
    ) -> DetectorFrame:
        """Construct a :class:`DetectorFrame` from basis vectors ``(ex, ey[, ez])``."""
        vx = _normalize_vec(_as_3vec_float(ex, name="ex"), name="ex")
        vy = _normalize_vec(_as_3vec_float(ey, name="ey"), name="ey")
        if ez is None:
            vz = _normalize_vec(np.cross(vx, vy), name="ex x ey")
        else:
            vz = _normalize_vec(_as_3vec_float(ez, name="ez"), name="ez")

        orig_vec = _as_3vec_float(origin, name="origin", expected_unit=u.m)
        mat = np.vstack([vx, vy, vz])
        frozen_basis, is_ortho = _validate_basis_matrix(
            mat, require_orthonormal=require_orthonormal
        )
        return cls(
            name=name,
            detector=detector,
            frame_kind="arms",
            arm_source="axes",
            origin=(float(orig_vec[0]), float(orig_vec[1]), float(orig_vec[2])),
            basis=frozen_basis,
            parent_frame=parent_frame,
            local_origin_lonlat=local_origin_lonlat,
            vertical_datum=vertical_datum,
            is_orthonormal=is_ortho,
            raw_arms=(
                (float(vx[0]), float(vx[1]), float(vx[2])),
                (float(vy[0]), float(vy[1]), float(vy[2])),
            ),
        )

    @classmethod
    def from_arms(
        cls,
        x_arm: Any,
        y_arm: Any,
        *,
        origin: Any = (0.0, 0.0, 0.0),
        orthogonalize: bool = True,
        orthogonalize_mode: Literal["x_primary", "symmetric"] = "x_primary",
        name: str = "detector",
        detector: str | None = None,
        parent_frame: str = "local_enu",
        local_origin_lonlat: tuple[float, float] | None = None,
        vertical_datum: str = "geoid",
    ) -> DetectorFrame:
        """Construct a :class:`DetectorFrame` from two interferometer arm vectors."""
        ux = _normalize_vec(_as_3vec_float(x_arm, name="x_arm"), name="x_arm")
        uy = _normalize_vec(_as_3vec_float(y_arm, name="y_arm"), name="y_arm")
        cross_xy = np.cross(ux, uy)
        if float(np.linalg.norm(cross_xy)) <= 1e-12:
            raise ValueError(
                "x_arm and y_arm are collinear; cannot define a detector frame plane."
            )
        ez = _normalize_vec(cross_xy, name="x_arm x y_arm")
        orig_vec = _as_3vec_float(origin, name="origin", expected_unit=u.m)
        raw_pair = (
            (float(ux[0]), float(ux[1]), float(ux[2])),
            (float(uy[0]), float(uy[1]), float(uy[2])),
        )

        if orthogonalize:
            if orthogonalize_mode == "x_primary":
                ex = ux
                ey = _normalize_vec(np.cross(ez, ex), name="ez x ex")
            elif orthogonalize_mode == "symmetric":
                bisector = _normalize_vec(ux + uy, name="x_arm + y_arm")
                ortho_bi = _normalize_vec(np.cross(ez, bisector), name="ez x bisector")
                inv_sqrt2 = 1.0 / math.sqrt(2.0)
                ex = inv_sqrt2 * (bisector - ortho_bi)
                ey = inv_sqrt2 * (bisector + ortho_bi)
            else:
                raise ValueError(
                    f"Unknown orthogonalize_mode {orthogonalize_mode!r}; "
                    "expected 'x_primary' or 'symmetric'."
                )
            mat = np.vstack([ex, ey, ez])
            frozen_basis, is_ortho = _validate_basis_matrix(
                mat, require_orthonormal=True
            )
        else:
            mat = np.vstack([ux, uy, ez])
            frozen_basis, is_ortho = _validate_basis_matrix(
                mat, require_orthonormal=False
            )

        return cls(
            name=name,
            detector=detector,
            frame_kind="arms",
            arm_source="explicit_arms",
            origin=(float(orig_vec[0]), float(orig_vec[1]), float(orig_vec[2])),
            basis=frozen_basis,
            parent_frame=parent_frame,
            local_origin_lonlat=local_origin_lonlat,
            vertical_datum=vertical_datum,
            is_orthonormal=is_ortho,
            raw_arms=raw_pair,
        )

    @classmethod
    def from_detector(
        cls,
        detector_name: str,
        *,
        frame_kind: Literal["arms", "local_enu"] = "arms",
        arm_source: Literal["nominal", "lal"] = "nominal",
        origin: Any | None = None,
        elevation: float | u.Quantity = 0.0,
        orthogonalize: bool = True,
        orthogonalize_mode: Literal["x_primary", "symmetric"] = "x_primary",
        vertical_datum: str = "geoid",
    ) -> DetectorFrame:
        """Construct a :class:`DetectorFrame` for a named detector.

        Explicitly distinguishes two detector-anchored coordinate frames and the
        existing repository data sources that construct them:

        1. ``frame_kind="local_enu"`` — **Site-centered local reference frame**
           (East, North, Up identity basis at the detector site). Constructed
           purely from :data:`gwexpy.plot.geomap.DETECTORS` (which stores only
           ``lon``, ``lat``, ``name``, and ``color``—no arm orientations). In
           this mode ``raw_arms`` is ``None`` and ``basis`` is the 3x3 identity.
        2. ``frame_kind="arms"`` (default) — **Arm-aligned detector frame**
           whose axes follow the interferometer X/Y arms. Since
           :data:`gwexpy.plot.geomap.DETECTORS` lacks arm geometry, arm
           directions are sourced from:
           - ``arm_source="nominal"`` (default): the adopted horizontal arm
             azimuth table :data:`DETECTOR_ARM_AZIMUTHS_DEG` combined with the
             adopted ``(lon, lat)`` from :data:`gwexpy.plot.geomap.DETECTORS`.
           - ``arm_source="lal"``: :data:`gwexpy.utils.lal.LAL_DETECTORS`
             (LALSuite ``FrDetector`` ``xArmAzimuthRadians``,
             ``yArmAzimuthRadians``, ``xArmAltitudeRadians``,
             ``yArmAltitudeRadians``), retaining the adopted ``(lon, lat)``
             from ``DETECTORS`` when present or falling back to LAL's vertex
             longitude/latitude for detectors outside ``DETECTORS``.
        """
        key = detector_name.strip().upper()
        if frame_kind not in ("arms", "local_enu"):
            raise ValueError(
                f"Unknown frame_kind {frame_kind!r}; expected 'arms' or 'local_enu'."
            )

        elev_m = (
            float(elevation.to_value(u.m))
            if isinstance(elevation, u.Quantity)
            else float(elevation)
        )
        orig_tuple = (
            (0.0, 0.0, elev_m)
            if origin is None
            else tuple(
                float(x)
                for x in _as_3vec_float(origin, name="origin", expected_unit=u.m)
            )
        )

        if frame_kind == "local_enu":
            if key not in DETECTORS:
                raise ValueError(
                    f"Unknown detector {detector_name!r} in gwexpy.plot.geomap.DETECTORS. "
                    f"Available site detectors: {sorted(DETECTORS)}."
                )
            det_info_enu: dict[str, Any] = DETECTORS[key]
            lon0 = float(det_info_enu["lon"])
            lat0 = float(det_info_enu["lat"])
            return cls(
                name=f"{key}_local_enu",
                detector=key,
                frame_kind="local_enu",
                arm_source="geomap.DETECTORS",
                origin=(orig_tuple[0], orig_tuple[1], orig_tuple[2]),
                basis=(
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                parent_frame="local_enu",
                local_origin_lonlat=(lon0, lat0),
                vertical_datum=vertical_datum,
                is_orthonormal=True,
                raw_arms=None,
            )

        if arm_source == "nominal":
            if key not in DETECTORS:
                raise ValueError(
                    f"Unknown detector {detector_name!r}. Available detectors: "
                    f"{sorted(DETECTORS)}."
                )
            det_info: dict[str, Any] = DETECTORS[key]
            lon0 = float(det_info["lon"])
            lat0 = float(det_info["lat"])
            az_x_deg, az_y_deg = DETECTOR_ARM_AZIMUTHS_DEG[key]
            az_x = math.radians(az_x_deg)
            az_y = math.radians(az_y_deg)
            ux = np.array([math.sin(az_x), math.cos(az_x), 0.0], dtype=np.float64)
            uy = np.array([math.sin(az_y), math.cos(az_y), 0.0], dtype=np.float64)
            resolved_arm_source = "DETECTOR_ARM_AZIMUTHS_DEG"
        elif arm_source == "lal":
            try:
                from gwexpy.utils.lal import LAL_DETECTORS
            except Exception as exc:
                raise ImportError(
                    "arm_source='lal' requires gwexpy.utils.lal (LALSuite)."
                ) from exc
            if key not in LAL_DETECTORS:
                raise ValueError(
                    f"Unknown detector {detector_name!r} in gwexpy.utils.lal.LAL_DETECTORS. "
                    f"Available LAL detectors: {sorted(LAL_DETECTORS)}."
                )
            fr: Any = LAL_DETECTORS[key]
            if key in DETECTORS:
                det_site: dict[str, Any] = DETECTORS[key]
                lon0 = float(det_site["lon"])
                lat0 = float(det_site["lat"])
            else:
                lon0 = math.degrees(float(fr.vertexLongitudeRadians))
                lat0 = math.degrees(float(fr.vertexLatitudeRadians))
            az_x = float(fr.xArmAzimuthRadians)
            az_y = float(fr.yArmAzimuthRadians)
            alt_x = float(getattr(fr, "xArmAltitudeRadians", 0.0))
            alt_y = float(getattr(fr, "yArmAltitudeRadians", 0.0))
            ux = np.array(
                [
                    math.cos(alt_x) * math.sin(az_x),
                    math.cos(alt_x) * math.cos(az_x),
                    math.sin(alt_x),
                ],
                dtype=np.float64,
            )
            uy = np.array(
                [
                    math.cos(alt_y) * math.sin(az_y),
                    math.cos(alt_y) * math.cos(az_y),
                    math.sin(alt_y),
                ],
                dtype=np.float64,
            )
            ux = _normalize_vec(ux, name="lal x_arm")
            uy = _normalize_vec(uy, name="lal y_arm")
            resolved_arm_source = "gwexpy.utils.lal.LAL_DETECTORS"
        else:
            raise ValueError(
                f"Unknown arm_source {arm_source!r}; expected 'nominal' or 'lal'."
            )

        raw_pair = (
            (float(ux[0]), float(ux[1]), float(ux[2])),
            (float(uy[0]), float(uy[1]), float(uy[2])),
        )
        cross_xy = np.cross(ux, uy)
        is_horizontal = abs(float(ux[2])) <= 1e-15 and abs(float(uy[2])) <= 1e-15
        if is_horizontal:
            ez_base = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            ez_base = _normalize_vec(cross_xy, name="x_arm x y_arm")
            if ez_base[2] < 0.0:
                ez_base = -ez_base

        if orthogonalize:
            ez = ez_base
            if orthogonalize_mode == "x_primary":
                ex = _normalize_vec(ux - float(np.dot(ux, ez)) * ez, name="ex")
                ey = _normalize_vec(np.cross(ez, ex), name="ez x ex")
            elif orthogonalize_mode == "symmetric":
                u_x_proj = _normalize_vec(
                    ux - float(np.dot(ux, ez)) * ez, name="ux_proj"
                )
                u_y_proj = _normalize_vec(
                    uy - float(np.dot(uy, ez)) * ez, name="uy_proj"
                )
                cross_z = float(np.dot(np.cross(u_x_proj, u_y_proj), ez))
                bisector = _normalize_vec(u_x_proj + u_y_proj, name="x_arm + y_arm")
                ortho_bi = _normalize_vec(np.cross(ez, bisector), name="ez x bisector")
                inv_sqrt2 = 1.0 / math.sqrt(2.0)
                if cross_z >= 0.0:
                    ex = inv_sqrt2 * (bisector - ortho_bi)
                    ey = inv_sqrt2 * (bisector + ortho_bi)
                else:
                    ex = inv_sqrt2 * (bisector + ortho_bi)
                    ey = _normalize_vec(np.cross(ez, ex), name="ez x ex")
            else:
                raise ValueError(f"Unknown orthogonalize_mode {orthogonalize_mode!r}.")
            mat = np.vstack([ex, ey, ez])
            frozen_basis, is_ortho = _validate_basis_matrix(
                mat, require_orthonormal=True
            )
        else:
            ez = ez_base
            mat = np.vstack([ux, uy, ez])
            frozen_basis, is_ortho = _validate_basis_matrix(
                mat, require_orthonormal=False
            )

        return cls(
            name=key,
            detector=key,
            frame_kind="arms",
            arm_source=resolved_arm_source,
            origin=(orig_tuple[0], orig_tuple[1], orig_tuple[2]),
            basis=frozen_basis,
            parent_frame="local_enu",
            local_origin_lonlat=(lon0, lat0),
            vertical_datum=vertical_datum,
            is_orthonormal=is_ortho,
            raw_arms=raw_pair,
        )


def _coerce_target_frame(
    frame: GridGeometry | DetectorFrame | str,
    *,
    source_field: Any = None,
) -> GridGeometry:
    if isinstance(frame, GridGeometry):
        return frame
    if isinstance(frame, str):
        stripped = frame.strip()
        if stripped.lower() in ("local_enu", "enu", "cartesian"):
            geo = getattr(source_field, "geo", None)
            lo = getattr(geo, "local_origin", None) if geo is not None else None
            vd = getattr(geo, "vertical_datum", "geoid") if geo is not None else "geoid"
            return GridGeometry.standard_enu(
                local_origin_lonlat=lo,
                vertical_datum=vd,
            )
        if stripped.upper() in DETECTORS:
            return DetectorFrame.from_detector(stripped.upper())
        raise ValueError(
            f"Unknown frame identifier {frame!r}. Pass a GridGeometry / "
            f"DetectorFrame instance, 'local_enu', or a known detector code "
            f"({sorted(DETECTORS)})."
        )
    raise TypeError(
        f"Expected GridGeometry, DetectorFrame, or frame name str, got {type(frame).__name__}."
    )


def _resolve_source_frame(field: Any) -> GridGeometry:
    existing = getattr(field, "_gwex_frame", None)
    if isinstance(existing, GridGeometry):
        return existing
    geo = getattr(field, "geo", None)
    lo = getattr(geo, "local_origin", None) if geo is not None else None
    vd = getattr(geo, "vertical_datum", "geoid") if geo is not None else "geoid"
    return GridGeometry.standard_enu(
        local_origin_lonlat=lo,
        vertical_datum=vd,
    )


def _validate_real_metric_axes(field: Any, op_name: str) -> None:
    space_domains = getattr(field, "space_domains", {})
    angle_axes = [k for k, v in space_domains.items() if v == "angle"]
    if angle_axes:
        raise ValueError(
            f"{op_name}() requires metric ('real') spatial axes, but axes "
            f"{angle_axes} are in 'angle' domain. Convert to local Cartesian "
            "coordinates first via to_local_cartesian()."
        )
    k_axes = [k for k, v in space_domains.items() if v == "k"]
    if k_axes:
        raise ValueError(
            f"{op_name}() requires real-space spatial axes, but axes {k_axes} "
            "are in 'k' domain."
        )


def _check_vertical_datum_compatibility(
    src_frame: GridGeometry,
    dst_frame: GridGeometry,
    *,
    strict_datum: bool = True,
) -> None:
    src_vd = (src_frame.vertical_datum or "").strip().lower()
    dst_vd = (dst_frame.vertical_datum or "").strip().lower()
    if (
        src_vd
        and dst_vd
        and src_vd != "unspecified"
        and dst_vd != "unspecified"
        and src_vd != dst_vd
    ):
        # Allow "geoid" to match specific geoid strings like "t.p. (jgd2011 vertical)"
        src_is_geoid = "geoid" in src_vd or "t.p." in src_vd or "egm" in src_vd
        dst_is_geoid = "geoid" in dst_vd or "t.p." in dst_vd or "egm" in dst_vd
        if src_is_geoid != dst_is_geoid:
            msg = (
                f"Vertical datum mismatch between source frame ({src_frame.vertical_datum!r}) "
                f"and target frame ({dst_frame.vertical_datum!r})."
            )
            if strict_datum:
                raise ValueError(msg)
            warnings.warn(msg, UserWarning, stacklevel=3)


def transform_Matrix_between_frames(
    source_frame: GridGeometry,
    target_frame: GridGeometry,
    *,
    require_orthogonal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return ``(R, delta_origin_in_parent)`` mapping source-frame vectors to target-frame.

    For a vector with components :math:`v` in ``source_frame`` (where rows of
    :math:`B_{\mathrm{src}}` are the source basis vectors in the common parent
    frame), the parent-frame vector is :math:`v_{\mathrm{parent}} =
    B_{\mathrm{src}}^{-1} v` (or :math:`B_{\mathrm{src}}^{\mathsf T} v` when
    orthonormal) and its components in ``target_frame`` are
    :math:`v' = R v` where:

    - When both frames are orthonormal: :math:`R = B_{\mathrm{dst}} B_{\mathrm{src}}^{\mathsf T}`.
    - When ``require_orthogonal=False`` and a frame has non-orthogonal raw arms:
      :math:`R = B_{\mathrm{dst}} B_{\mathrm{src}}^{-1}` (contravariant change of basis).

    Also returns ``delta_origin_parent = o_dst - o_src`` in the common parent
    frame, accounting for any ``local_origin_lonlat`` offset between the two
    local tangent planes.
    """
    if source_frame.parent_frame != target_frame.parent_frame:
        raise ValueError(
            f"Cannot transform directly between frames with different parent_frame: "
            f"{source_frame.parent_frame!r} vs {target_frame.parent_frame!r}."
        )
    if require_orthogonal:
        if not source_frame.is_orthonormal:
            raise ValueError(
                f"Source frame {source_frame.name!r} is not orthonormal; "
                "orthonormal metric basis is required (or pass require_orthogonal=False)."
            )
        if not target_frame.is_orthonormal:
            raise ValueError(
                f"Target frame {target_frame.name!r} is not orthonormal; "
                "construct the DetectorFrame with orthogonalize=True or pass "
                "require_orthogonal=False."
            )

    b_src = source_frame.basis_matrix
    b_dst = target_frame.basis_matrix
    if source_frame.is_orthonormal and target_frame.is_orthonormal:
        rot = b_dst @ b_src.T
    else:
        rot = b_dst @ np.linalg.inv(b_src)

    delta_parent = target_frame.origin_vector - source_frame.origin_vector
    if (
        source_frame.local_origin_lonlat is not None
        and target_frame.local_origin_lonlat is not None
        and source_frame.local_origin_lonlat != target_frame.local_origin_lonlat
    ):
        lon_s, lat_s = source_frame.local_origin_lonlat
        lon_t, lat_t = target_frame.local_origin_lonlat
        dx_m, dy_m = _tangent_offset_metres(lon_s, lat_s, lon_t, lat_t)
        delta_parent = delta_parent + np.array([dx_m, dy_m, 0.0], dtype=np.float64)

    return rot, delta_parent


# ---------------------------------------------------------------------------
# ScalarField frame operations
# ---------------------------------------------------------------------------


def scalar_assume_frame(
    field: ScalarField,
    frame: GridGeometry | DetectorFrame | str,
) -> ScalarField:
    """Assign *frame* metadata to *field* without transforming values or axes."""
    _validate_real_metric_axes(field, "assume_frame")
    target = _coerce_target_frame(frame, source_field=field)
    out = field.copy()
    out._gwex_frame = target.with_history(f"assume_frame({target.name!r})")
    return out


def scalar_to_frame(
    field: ScalarField,
    frame: GridGeometry | DetectorFrame | str,
    *,
    require_orthogonal: bool = True,
    strict_datum: bool = True,
) -> ScalarField:
    r"""Passive basis change for a :class:`ScalarField` (no spatial interpolation).

    For a scalar field :math:`\phi' = \phi`: physical sample values and the
    underlying sample points in space are unchanged.

    - When the physical 1D coordinate lattice is aligned with the target
      frame's basis vectors (:math:`B_{\mathrm{grid}} = B_{\mathrm{dst}}`),
      the 1D axis coordinates are shifted directly to the target frame's origin
      without interpolation and ``grid_origin`` / ``grid_basis`` match the
      target frame (`None`).
    - When the target basis is rotated relative to the physical 1D lattice,
      the 1D coordinate arrays remain untouched (no resampling) while
      ``grid_origin`` and ``grid_basis`` on the returned :class:`GridGeometry`
      record the physical sample lattice geometry so that
      :math:`\mathbf{r}_{\mathrm{parent}} = \mathbf{o}_{\mathrm{grid}} +
      B_{\mathrm{grid}}^{\mathsf T} \mathbf{x}` is strictly invariant.
    """
    _validate_real_metric_axes(field, "to_frame")
    src_frame = _resolve_source_frame(field)
    dst_frame = _coerce_target_frame(frame, source_field=field)
    _check_vertical_datum_compatibility(src_frame, dst_frame, strict_datum=strict_datum)
    transform_Matrix_between_frames(
        src_frame, dst_frame, require_orthogonal=require_orthogonal
    )

    out = field.copy()
    lat_basis_tuple = src_frame.lattice_basis
    lat_basis_mat = src_frame.lattice_basis_matrix
    lat_origin_vec = src_frame.lattice_origin_vector.copy()

    # Account for local_origin_lonlat shift if the two frames have different tangent-plane anchors
    if (
        src_frame.local_origin_lonlat is not None
        and dst_frame.local_origin_lonlat is not None
        and src_frame.local_origin_lonlat != dst_frame.local_origin_lonlat
    ):
        lon_s, lat_s = src_frame.local_origin_lonlat
        lon_t, lat_t = dst_frame.local_origin_lonlat
        dx_m, dy_m = _tangent_offset_metres(lon_t, lat_t, lon_s, lat_s)
        lat_origin_vec = lat_origin_vec + np.array([dx_m, dy_m, 0.0], dtype=np.float64)

    dst_basis_mat = dst_frame.basis_matrix
    dst_origin_vec = dst_frame.origin_vector
    lattice_aligned_with_dst = bool(
        np.allclose(lat_basis_mat, dst_basis_mat, rtol=0.0, atol=_ORTHOGONALITY_ATOL)
    )

    if lattice_aligned_with_dst:
        # 1D grid axes are parallel to dst_frame basis: shift 1D coordinates directly
        delta_parent = dst_origin_vec - lat_origin_vec
        if dst_frame.is_orthonormal:
            delta_local = dst_basis_mat @ delta_parent
        else:
            delta_local = np.linalg.inv(dst_basis_mat.T) @ delta_parent
        for ax_idx, shift_m in (
            (1, delta_local[0]),
            (2, delta_local[1]),
            (3, delta_local[2]),
        ):
            if abs(shift_m) > 0.0:
                ax_q = getattr(out, f"_axis{ax_idx}_index")
                setattr(
                    out, f"_axis{ax_idx}_index", ax_q - (shift_m * u.m).to(ax_q.unit)
                )
        new_grid_origin = None
        new_grid_basis = None
    else:
        # Rotated basis without resampling: preserve exact lattice geometry so physical
        # sample positions r_parent = o_grid + B_grid^T x remain invariant.
        new_grid_origin = (
            float(lat_origin_vec[0]),
            float(lat_origin_vec[1]),
            float(lat_origin_vec[2]),
        )
        new_grid_basis = lat_basis_tuple

    new_geom = replace(
        dst_frame,
        grid_origin=new_grid_origin,
        grid_basis=new_grid_basis,
        history=(
            *src_frame.history,
            f"to_frame(from={src_frame.name!r}, to={dst_frame.name!r})",
        ),
    )
    out._gwex_frame = new_geom
    if out.geo is not None:
        out.geo = out.geo.with_history(f"to_frame({dst_frame.name!r})")
    return out


def scalar_regrid_to_frame(
    field: ScalarField,
    frame: GridGeometry | DetectorFrame | str,
    *,
    resolution: float | tuple[float, float] | u.Quantity | None = None,
    method: Literal["nearest", "bilinear", "cubic"] = "bilinear",
    axis_names: tuple[str, str] | None = None,
    strict_datum: bool = True,
) -> ScalarField:
    r"""Explicitly resample *field* onto a horizontal rectilinear grid aligned with *frame*.

    Unlike :meth:`to_frame`, this operation interpolates field values onto a new
    orthogonal grid whose axes ``(x', y')`` are aligned with the target frame's
    horizontal basis vectors :math:`(\hat{e}_1, \hat{e}_2)`.
    """
    from .scalar import ScalarField, _resample_2d_grid

    if np.iscomplexobj(field.value):
        raise TypeError(
            "regrid_to_frame() cannot resample complex field values; "
            "convert to a real-valued quantity explicitly first."
        )
    if method not in ("nearest", "bilinear", "cubic"):
        raise ValueError(
            f"Invalid resampling method {method!r}. "
            "Must be 'nearest', 'bilinear', or 'cubic'."
        )
    _validate_real_metric_axes(field, "regrid_to_frame")
    src_frame = _resolve_source_frame(field)
    dst_frame = _coerce_target_frame(frame, source_field=field)
    _check_vertical_datum_compatibility(src_frame, dst_frame, strict_datum=strict_datum)

    # Spatial regridding maps from the physical 1D coordinate lattice of field to dst_frame
    lattice_geom = replace(
        src_frame,
        origin=src_frame.lattice_origin,
        basis=src_frame.lattice_basis,
        grid_origin=None,
        grid_basis=None,
    )
    rot, delta_parent = transform_Matrix_between_frames(
        lattice_geom, dst_frame, require_orthogonal=True
    )

    # Verify horizontal rotation (z axis parallel) for 2D horizontal grid resampling
    if not (
        np.allclose(rot[2, :2], 0.0, atol=1e-8)
        and np.allclose(rot[:2, 2], 0.0, atol=1e-8)
        and rot[2, 2] > 0.0
    ):
        raise ValueError(
            "regrid_to_frame() on a rectilinear field requires the target frame "
            "vertical axis to align with the source vertical axis (horizontal "
            "rotation + translation). Tilted 3D plane resampling requires a "
            "SurfaceField representation."
        )

    x_axis = field._axis1_index
    y_axis = field._axis2_index
    z_axis = field._axis3_index
    x_m = np.asarray(u.Quantity(x_axis, copy=False).to_value(u.m), dtype=np.float64)
    y_m = np.asarray(u.Quantity(y_axis, copy=False).to_value(u.m), dtype=np.float64)
    z_m = np.asarray(u.Quantity(z_axis, copy=False).to_value(u.m), dtype=np.float64)
    if x_m.size < 2 or y_m.size < 2:
        raise ValueError("regrid_to_frame() requires axes 1 and 2 to have length >= 2.")

    # Shift in source-lattice coordinates from lattice origin to target origin
    delta_src = lattice_geom.basis_matrix @ delta_parent
    rot_2d = rot[:2, :2]  # maps (x_src - delta_x, y_src - delta_y) -> (x_dst, y_dst)

    # Compute bounding box of source cell centers in target frame
    xx_src, yy_src = np.meshgrid(x_m - delta_src[0], y_m - delta_src[1], indexing="ij")
    pts_src = np.stack([xx_src.ravel(), yy_src.ravel()], axis=0)
    pts_dst = rot_2d @ pts_src
    x_dst_min, x_dst_max = float(np.min(pts_dst[0])), float(np.max(pts_dst[0]))
    y_dst_min, y_dst_max = float(np.min(pts_dst[1])), float(np.max(pts_dst[1]))

    if resolution is None:
        dx_out = float(np.median(np.abs(np.diff(x_m))))
        dy_out = float(np.median(np.abs(np.diff(y_m))))
    elif isinstance(resolution, u.Quantity):
        res_v = resolution.to_value(u.m)
        if np.ndim(res_v) == 0:
            dx_out = dy_out = float(res_v)
        else:
            dx_out, dy_out = float(res_v[0]), float(res_v[1])
    elif isinstance(resolution, Sequence):
        dx_out, dy_out = float(resolution[0]), float(resolution[1])
    else:
        dx_out = dy_out = float(resolution)

    if not (np.isfinite(dx_out) and np.isfinite(dy_out) and dx_out > 0 and dy_out > 0):
        raise ValueError(
            f"resolution must be finite and positive, got ({dx_out}, {dy_out})."
        )

    nx_out = max(2, int(round((x_dst_max - x_dst_min) / dx_out)) + 1)
    ny_out = max(2, int(round((y_dst_max - y_dst_min) / dy_out)) + 1)
    x_dst_1d = x_dst_min + np.arange(nx_out, dtype=np.float64) * dx_out
    y_dst_1d = y_dst_min + np.arange(ny_out, dtype=np.float64) * dy_out

    # Inverse transform: target grid (x_dst, y_dst) -> source frame (x_src, y_src)
    xx_dst, yy_dst = np.meshgrid(x_dst_1d, y_dst_1d, indexing="ij")
    back_rel = rot_2d.T @ np.stack([xx_dst.ravel(), yy_dst.ravel()], axis=0)
    dst_x_in_src = back_rel[0].reshape((nx_out, ny_out)) + delta_src[0]
    dst_y_in_src = back_rel[1].reshape((nx_out, ny_out)) + delta_src[1]

    # Sort source axes ascending for _resample_2d_grid
    x_flip = bool(x_m[1] < x_m[0])
    y_flip = bool(y_m[1] < y_m[0])
    src_x_asc = x_m[::-1].copy() if x_flip else x_m.copy()
    src_y_asc = y_m[::-1].copy() if y_flip else y_m.copy()

    nt, _, _, nz = field.shape
    out_data = np.empty((nt, nx_out, ny_out, nz), dtype=np.float64)
    raw_val = np.asarray(field.value, dtype=np.float64)
    for it in range(nt):
        for iz in range(nz):
            sl_2d = raw_val[it, :, :, iz]
            if x_flip:
                sl_2d = sl_2d[::-1, :]
            if y_flip:
                sl_2d = sl_2d[:, ::-1]
            out_data[it, :, :, iz] = _resample_2d_grid(
                src_x_asc,
                src_y_asc,
                sl_2d,
                dst_x_in_src,
                dst_y_in_src,
                method,
            )

    new_names = list(field.axis_names)
    if axis_names is not None:
        if len(axis_names) != 2 or axis_names[0] == axis_names[1]:
            raise ValueError("axis_names must be a 2-tuple of distinct names.")
        new_names[1] = axis_names[0]
        new_names[2] = axis_names[1]

    new_domains = {
        new_names[1]: "real",
        new_names[2]: "real",
        new_names[3]: field.space_domains.get(field._axis3_name, "real"),
    }

    z_unit = getattr(z_axis, "unit", u.m)
    z_out = (z_m - delta_src[2]) * u.m
    result = ScalarField(
        out_data,
        unit=field.unit,
        epoch=field.epoch,
        name=field.name,
        channel=field.channel,
        axis0=field._axis0_index.copy(),
        axis1=x_dst_1d * u.m,
        axis2=y_dst_1d * u.m,
        axis3=z_out.to(z_unit),
        axis_names=new_names,
        axis0_domain=field.axis0_domain,
        space_domain=new_domains,
    )
    field._propagate_gwex_attrs(result)
    hist_msg = (
        f"regrid_to_frame(from={src_frame.name!r}, to={dst_frame.name!r}, "
        f"method={method!r}, dx={dx_out:g}, dy={dy_out:g})"
    )
    result._gwex_frame = replace(
        dst_frame,
        grid_origin=None,
        grid_basis=None,
        history=(*src_frame.history, hist_msg),
    )
    if result.geo is not None:
        result.geo = result.geo.deactivated(
            hist_msg,
            local_origin=dst_frame.local_origin_lonlat or result.geo.local_origin,
        )
    return result


def scalar_rotate_horizontal(
    field: ScalarField,
    angle: float | u.Quantity,
    *,
    center: tuple[float | u.Quantity, float | u.Quantity] | None = None,
    resample: bool = False,
    resolution: float | tuple[float, float] | u.Quantity | None = None,
    method: Literal["nearest", "bilinear", "cubic"] = "bilinear",
) -> ScalarField:
    """Rotate *field* horizontally around the vertical (+z) axis by *angle*.

    Parameters
    ----------
    field : ScalarField
        Input scalar field on a metric spatial grid.
    angle : float or Quantity
        Counter-clockwise angle of the new frame's horizontal axes relative to
        the current frame (in degrees when a bare float is passed, or any
        angular Quantity).
    center : tuple of (cx, cy), optional
        Center of rotation in current frame coordinates (metres). Defaults to
        the current frame origin ``(0, 0)``.
    resample : bool, default False
        When ``False`` (default), performs a passive basis update via
        :meth:`to_frame` without interpolating values. When ``True``, explicitly
        resamples onto a grid aligned with the rotated axes via
        :meth:`regrid_to_frame`.
    resolution : float, tuple, or Quantity, optional
        Output grid spacing when ``resample=True``.
    method : {'nearest', 'bilinear', 'cubic'}, default 'bilinear'
        Interpolation method when ``resample=True``.

    """
    _validate_real_metric_axes(field, "rotate_horizontal")
    src_frame = _resolve_source_frame(field)
    rot = horizontal_rotation_matrix(angle)
    new_basis_mat = rot @ src_frame.basis_matrix

    if center is None:
        new_origin = src_frame.origin
    else:
        if len(center) != 2:
            raise ValueError("center must be a 2-tuple (cx, cy).")
        cx = (
            float(center[0].to_value(u.m))
            if isinstance(center[0], u.Quantity)
            else float(center[0])
        )
        cy = (
            float(center[1].to_value(u.m))
            if isinstance(center[1], u.Quantity)
            else float(center[1])
        )
        offset_parent = src_frame.basis_matrix.T @ np.array(
            [cx, cy, 0.0], dtype=np.float64
        )
        orig_vec = src_frame.origin_vector + offset_parent
        new_origin = (float(orig_vec[0]), float(orig_vec[1]), float(orig_vec[2]))

    deg_val = (
        float(angle.to_value(u.deg)) if isinstance(angle, u.Quantity) else float(angle)
    )
    frozen_basis, is_ortho = _validate_basis_matrix(
        new_basis_mat, require_orthonormal=True
    )
    rotated_frame = replace(
        src_frame,
        name=f"{src_frame.name}_rot({deg_val:g}deg)",
        origin=new_origin,
        basis=frozen_basis,
        is_orthonormal=is_ortho,
    )
    if resample:
        return scalar_regrid_to_frame(
            field,
            rotated_frame,
            resolution=resolution,
            method=method,
        )
    return scalar_to_frame(field, rotated_frame)


# ---------------------------------------------------------------------------
# VectorField frame operations (v' = R v)
# ---------------------------------------------------------------------------


def _extract_vector_xyz(vec_field: VectorField) -> tuple[list[str], list[ScalarField]]:
    keys = list(vec_field.keys())
    if set(keys) == {"x", "y", "z"}:
        ordered = ["x", "y", "z"]
        return ordered, [vec_field["x"], vec_field["y"], vec_field["z"]]
    if set(keys) == {"x", "y"}:
        ordered = ["x", "y"]
        return ordered, [vec_field["x"], vec_field["y"]]
    if len(keys) == 3:
        return keys, [vec_field[k] for k in keys]
    raise ValueError(
        f"VectorField frame rotation requires 2 ('x', 'y') or 3 ('x', 'y', 'z') "
        f"Cartesian components, got keys {keys!r}."
    )


def vector_assume_frame(
    vec_field: VectorField,
    frame: GridGeometry | DetectorFrame | str,
) -> VectorField:
    """Assign *frame* metadata to every component of *vec_field* without rotating."""
    from .vector import VectorField

    if not vec_field:
        raise ValueError("Cannot assume_frame on an empty VectorField.")
    first = next(iter(vec_field.values()))
    target = _coerce_target_frame(frame, source_field=first)
    new_comps = {k: scalar_assume_frame(comp, target) for k, comp in vec_field.items()}
    out = VectorField(new_comps, basis=vec_field.basis, validate=False)
    setattr(out, "_gwex_frame", target.with_history(f"assume_frame({target.name!r})"))
    return out


def vector_to_frame(
    vec_field: VectorField,
    frame: GridGeometry | DetectorFrame | str,
    *,
    require_orthogonal: bool = True,
    strict_datum: bool = True,
) -> VectorField:
    r"""Transform *vec_field* to *frame* without spatial interpolation (:math:`v' = R v`).

    Physical sample positions remain untouched; each spatial point's vector
    components are transformed by the frame rotation matrix :math:`R`:

    .. math::

        v'_i = \sum_j R_{ij} v_j.
    """
    from .vector import VectorField

    if not vec_field:
        raise ValueError("Cannot transform an empty VectorField.")
    keys, comps = _extract_vector_xyz(vec_field)
    first = comps[0]
    _validate_real_metric_axes(first, "VectorField.to_frame")
    src_frame = _resolve_source_frame(first)
    dst_frame = _coerce_target_frame(frame, source_field=first)
    _check_vertical_datum_compatibility(src_frame, dst_frame, strict_datum=strict_datum)
    rot, _ = transform_Matrix_between_frames(
        src_frame, dst_frame, require_orthogonal=require_orthogonal
    )

    # First apply passive coordinate/metadata update to each component
    passive_comps = [
        scalar_to_frame(
            c,
            dst_frame,
            require_orthogonal=require_orthogonal,
            strict_datum=strict_datum,
        )
        for c in comps
    ]
    n_comp = len(keys)
    if n_comp == 2:
        if not (
            np.allclose(rot[2, :2], 0.0, atol=1e-10)
            and np.allclose(rot[:2, 2], 0.0, atol=1e-10)
        ):
            raise ValueError(
                "2-component VectorField ('x', 'y') cannot be rotated into a "
                "3D tilted frame; supply a 3-component ('x', 'y', 'z') VectorField."
            )
        rot_active = rot[:2, :2]
    else:
        rot_active = rot

    ref_unit = passive_comps[0].unit
    vals = np.stack(
        [
            np.asarray(c.to_value(ref_unit), dtype=np.float64)
            if ref_unit is not None
            else np.asarray(c.value, dtype=np.float64)
            for c in passive_comps
        ],
        axis=0,
    )
    rotated_vals = np.einsum("ij,j...->i...", rot_active, vals)

    new_components: dict[str, ScalarField] = {}
    for idx, key in enumerate(keys):
        comp_copy = passive_comps[idx].copy()
        comp_copy[...] = (
            rotated_vals[idx] * ref_unit if ref_unit is not None else rotated_vals[idx]
        )
        new_components[key] = comp_copy

    out = VectorField(new_components, basis=vec_field.basis, validate=False)
    setattr(out, "_gwex_frame", passive_comps[0]._gwex_frame)
    return out


def vector_regrid_to_frame(
    vec_field: VectorField,
    frame: GridGeometry | DetectorFrame | str,
    *,
    resolution: float | tuple[float, float] | u.Quantity | None = None,
    method: Literal["nearest", "bilinear", "cubic"] = "bilinear",
    axis_names: tuple[str, str] | None = None,
    strict_datum: bool = True,
) -> VectorField:
    r"""Resample *vec_field* onto *frame*'s grid and rotate components (:math:`v' = R v`)."""
    from .vector import VectorField

    if not vec_field:
        raise ValueError("Cannot regrid an empty VectorField.")
    # Rotate components first at native grid locations, then regrid each component
    rotated = vector_to_frame(
        vec_field,
        frame,
        require_orthogonal=True,
        strict_datum=strict_datum,
    )
    first_orig = next(iter(vec_field.values()))
    src_frame = _resolve_source_frame(first_orig)
    dst_frame = _coerce_target_frame(frame, source_field=first_orig)

    new_components: dict[str, ScalarField] = {}
    for key, comp in rotated.items():
        # Restore source frame on copy so scalar_regrid_to_frame computes the spatial warp
        comp_src = comp.copy()
        comp_src._gwex_frame = src_frame
        new_components[key] = scalar_regrid_to_frame(
            comp_src,
            dst_frame,
            resolution=resolution,
            method=method,
            axis_names=axis_names,
            strict_datum=strict_datum,
        )
    out = VectorField(new_components, basis=vec_field.basis, validate=False)
    setattr(out, "_gwex_frame", next(iter(new_components.values()))._gwex_frame)
    return out


def vector_rotate_horizontal(
    vec_field: VectorField,
    angle: float | u.Quantity,
    *,
    center: tuple[float | u.Quantity, float | u.Quantity] | None = None,
    resample: bool = False,
    resolution: float | tuple[float, float] | u.Quantity | None = None,
    method: Literal["nearest", "bilinear", "cubic"] = "bilinear",
) -> VectorField:
    r"""Rotate *vec_field* horizontally by *angle* (with :math:`v' = R v`)."""
    if not vec_field:
        raise ValueError("Cannot rotate an empty VectorField.")
    first = next(iter(vec_field.values()))
    dummy_rot = scalar_rotate_horizontal(
        first,
        angle,
        center=center,
        resample=False,
    )
    target_frame = dummy_rot._gwex_frame
    assert isinstance(target_frame, GridGeometry)
    if resample:
        return vector_regrid_to_frame(
            vec_field,
            target_frame,
            resolution=resolution,
            method=method,
        )
    return vector_to_frame(vec_field, target_frame)


# ---------------------------------------------------------------------------
# TensorField frame operations (T' = R T R^T)
# ---------------------------------------------------------------------------


_TENSOR_KEY_MAP_TO_IDX: dict[Any, tuple[int, int]] = {
    (i, j): (i, j) for i in range(3) for j in range(3)
}
for _i, _ci in enumerate(("x", "y", "z")):
    for _j, _cj in enumerate(("x", "y", "z")):
        _TENSOR_KEY_MAP_TO_IDX[f"{_ci}{_cj}"] = (_i, _j)
        _TENSOR_KEY_MAP_TO_IDX[(_ci, _cj)] = (_i, _j)


def _extract_rank2_tensor_grid(
    ten_field: TensorField,
) -> tuple[dict[tuple[int, int], Any], dict[tuple[int, int], ScalarField]]:
    if ten_field.rank not in (2, None):
        raise ValueError(
            f"TensorField frame transformation requires rank 2, got rank {ten_field.rank}."
        )
    key_map: dict[tuple[int, int], Any] = {}
    comp_map: dict[tuple[int, int], ScalarField] = {}
    for k, v in ten_field.items():
        if k not in _TENSOR_KEY_MAP_TO_IDX:
            raise ValueError(
                f"Tensor component key {k!r} cannot be mapped to 3x3 Cartesian "
                "indices (0..2, 0..2) or ('xx'..'zz')."
            )
        ij = _TENSOR_KEY_MAP_TO_IDX[k]
        key_map[ij] = k
        comp_map[ij] = v
    if len(comp_map) != 9:
        missing = [(i, j) for i in range(3) for j in range(3) if (i, j) not in comp_map]
        raise ValueError(
            f"Rank-2 3D TensorField frame rotation requires all 9 components; "
            f"missing indices: {missing}."
        )
    return key_map, comp_map


def tensor_assume_frame(
    ten_field: TensorField,
    frame: GridGeometry | DetectorFrame | str,
) -> TensorField:
    """Assign *frame* metadata to every component of *ten_field* without rotating."""
    from .tensor import TensorField

    if not ten_field:
        raise ValueError("Cannot assume_frame on an empty TensorField.")
    first = next(iter(ten_field.values()))
    target = _coerce_target_frame(frame, source_field=first)
    new_comps = {k: scalar_assume_frame(comp, target) for k, comp in ten_field.items()}
    out = TensorField(new_comps, rank=ten_field.rank, validate=False)
    setattr(out, "_gwex_frame", target.with_history(f"assume_frame({target.name!r})"))
    return out


def tensor_to_frame(
    ten_field: TensorField,
    frame: GridGeometry | DetectorFrame | str,
    *,
    require_orthogonal: bool = True,
    strict_datum: bool = True,
) -> TensorField:
    r"""Transform rank-2 *ten_field* to *frame* without spatial interpolation (:math:`T' = R T R^{\mathsf T}`).

    Physical sample positions remain untouched; each spatial point's 3x3 tensor
    components transform as:

    .. math::

        T'_{ij} = \sum_{m,n} R_{im} T_{mn} R_{jn}.
    """
    from .tensor import TensorField

    if not ten_field:
        raise ValueError("Cannot transform an empty TensorField.")
    key_map, comp_map = _extract_rank2_tensor_grid(ten_field)
    first = comp_map[(0, 0)]
    _validate_real_metric_axes(first, "TensorField.to_frame")
    src_frame = _resolve_source_frame(first)
    dst_frame = _coerce_target_frame(frame, source_field=first)
    _check_vertical_datum_compatibility(src_frame, dst_frame, strict_datum=strict_datum)
    rot, _ = transform_Matrix_between_frames(
        src_frame, dst_frame, require_orthogonal=require_orthogonal
    )

    passive_map: dict[tuple[int, int], ScalarField] = {
        ij: scalar_to_frame(
            comp,
            dst_frame,
            require_orthogonal=require_orthogonal,
            strict_datum=strict_datum,
        )
        for ij, comp in comp_map.items()
    }

    ref_unit = passive_map[(0, 0)].unit
    shape = passive_map[(0, 0)].shape
    t_arr = np.empty((3, 3, *shape), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            c = passive_map[(i, j)]
            t_arr[i, j] = (
                np.asarray(c.to_value(ref_unit), dtype=np.float64)
                if ref_unit is not None
                else np.asarray(c.value, dtype=np.float64)
            )

    rotated_arr = np.einsum("im,mn...,jn->ij...", rot, t_arr, rot)

    new_components: dict[Any, ScalarField] = {}
    for i in range(3):
        for j in range(3):
            orig_key = key_map[(i, j)]
            comp_copy = passive_map[(i, j)].copy()
            comp_copy[...] = (
                rotated_arr[i, j] * ref_unit
                if ref_unit is not None
                else rotated_arr[i, j]
            )
            new_components[orig_key] = comp_copy

    out = TensorField(new_components, rank=2, validate=False)
    setattr(out, "_gwex_frame", passive_map[(0, 0)]._gwex_frame)
    return out


def tensor_regrid_to_frame(
    ten_field: TensorField,
    frame: GridGeometry | DetectorFrame | str,
    *,
    resolution: float | tuple[float, float] | u.Quantity | None = None,
    method: Literal["nearest", "bilinear", "cubic"] = "bilinear",
    axis_names: tuple[str, str] | None = None,
    strict_datum: bool = True,
) -> TensorField:
    r"""Resample *ten_field* onto *frame*'s grid and rotate components (:math:`T' = R T R^{\mathsf T}`)."""
    from .tensor import TensorField

    if not ten_field:
        raise ValueError("Cannot regrid an empty TensorField.")
    rotated = tensor_to_frame(
        ten_field,
        frame,
        require_orthogonal=True,
        strict_datum=strict_datum,
    )
    first_orig = next(iter(ten_field.values()))
    src_frame = _resolve_source_frame(first_orig)
    dst_frame = _coerce_target_frame(frame, source_field=first_orig)

    new_components: dict[Any, ScalarField] = {}
    for key, comp in rotated.items():
        comp_src = comp.copy()
        comp_src._gwex_frame = src_frame
        new_components[key] = scalar_regrid_to_frame(
            comp_src,
            dst_frame,
            resolution=resolution,
            method=method,
            axis_names=axis_names,
            strict_datum=strict_datum,
        )
    out = TensorField(new_components, rank=2, validate=False)
    setattr(out, "_gwex_frame", next(iter(new_components.values()))._gwex_frame)
    return out


def tensor_rotate_horizontal(
    ten_field: TensorField,
    angle: float | u.Quantity,
    *,
    center: tuple[float | u.Quantity, float | u.Quantity] | None = None,
    resample: bool = False,
    resolution: float | tuple[float, float] | u.Quantity | None = None,
    method: Literal["nearest", "bilinear", "cubic"] = "bilinear",
) -> TensorField:
    r"""Rotate rank-2 *ten_field* horizontally by *angle* (with :math:`T' = R T R^{\mathsf T}`)."""
    if not ten_field:
        raise ValueError("Cannot rotate an empty TensorField.")
    first = next(iter(ten_field.values()))
    dummy_rot = scalar_rotate_horizontal(
        first,
        angle,
        center=center,
        resample=False,
    )
    target_frame = dummy_rot._gwex_frame
    assert isinstance(target_frame, GridGeometry)
    if resample:
        return tensor_regrid_to_frame(
            ten_field,
            target_frame,
            resolution=resolution,
            method=method,
        )
    return tensor_to_frame(ten_field, target_frame)

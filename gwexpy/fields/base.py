"""Base class for field objects with domain-aware axis metadata."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from astropy import units as u

from ..time._reference import (
    TIME_REFERENCE_ABSOLUTE,
    TIME_REFERENCE_RELATIVE,
    classify_time_input,
    set_time_reference,
)
from ..types.array4d import Array4D
from .geo import GeoMetadata

__all__ = ["FieldBase"]

# Tolerance for axis coordinate comparison.  This is an absolute magnitude in
# the unscaled, decomposed SI base unit for the axis (for example seconds,
# metres, hertz, or inverse metres).  Keep relative tolerance disabled: large
# absolute coordinates such as GPS seconds must not allow order-second jitter
# through relative scaling.
_AXIS_RTOL = 0.0
_AXIS_ATOL = 1e-12


def _canonical_axis_unit(unit: u.UnitBase) -> u.UnitBase:
    """Return an unscaled decomposed unit for order-invariant comparison."""
    decomposed = unit.decompose()
    canonical = u.dimensionless_unscaled
    for base, power in zip(decomposed.bases, decomposed.powers, strict=True):
        canonical *= base**power
    return canonical


def _axis_coordinates_close(left, right) -> bool:
    """Return True when two coordinate axes describe the same physical grid."""
    if left.shape != right.shape:
        return False

    left_unit = getattr(left, "unit", u.dimensionless_unscaled)
    right_unit = getattr(right, "unit", u.dimensionless_unscaled)
    if not right_unit.is_equivalent(left_unit):
        return False

    comparison_unit = _canonical_axis_unit(left_unit)
    left_values = (
        left.to_value(comparison_unit)
        if hasattr(left, "to_value")
        else np.asarray(left)
    )
    right_values = (
        right.to_value(comparison_unit)
        if hasattr(right, "to_value")
        else np.asarray(right)
    )
    return np.allclose(
        np.asarray(left_values),
        np.asarray(right_values),
        rtol=_AXIS_RTOL,
        atol=_AXIS_ATOL,
    )


class FieldBase(Array4D):
    """Base class providing axis domain metadata handling for field objects.

    Maintains the invariant of four coordinate axes `(axis0, x, y, z)` and
    explicit domain labels for temporal and spatial axes.
    """

    _metadata_slots = Array4D._metadata_slots + (
        "_axis0_domain",
        "_space_domains",
        "_axis0_offset",  # Preserved during fft_time for ifft_time reconstruction
        "_axis0_dt",  # Sampling interval preserved across time/frequency transforms
    )

    # Axis name conventions
    _TIME_AXIS_NAME = "t"
    _FREQ_AXIS_NAME = "f"
    _REAL_AXIS_NAMES = ("x", "y", "z")
    _K_AXIS_NAMES = ("kx", "ky", "kz")

    # Field axis coordinates use `_axis0_index`, not GWpy Series' `_x0`.
    _time_reference_axis_attrs = ("_axis0_index",)
    _is_gwexpy_field = True

    _axis0_index: Any
    _axis1_index: Any
    _axis2_index: Any
    _axis3_index: Any
    _space_domains: dict[str, str]

    def __new__(
        cls,
        value,
        *,
        unit=None,
        name=None,
        epoch=None,
        channel=None,
        dtype=None,
        copy=True,
        subok=True,
        order=None,
        ndmin=0,
        axis0=None,
        axis1=None,
        axis2=None,
        axis3=None,
        axis_names=None,
        axis0_domain: str = "time",
        space_domain: str | dict[str, str] = "real",
    ):
        """Create a field object with domain-aware axis defaults."""
        time_input = classify_time_input(epoch)
        if time_input == "inf":
            raise ValueError(
                "epoch must not be infinite; pass a finite GPS time or omit it "
                "when the absolute time is unknown"
            )
        time_reference = (
            TIME_REFERENCE_ABSOLUTE
            if time_input == "finite"
            else TIME_REFERENCE_RELATIVE
        )
        # NaN means that the Field has no absolute epoch. Do not pass it to
        # GWpy, where it could contaminate metadata or coordinate calculations.
        effective_epoch = None if time_input == "nan" else epoch

        # Set default axis names based on domain
        if axis_names is None:
            time_name = (
                cls._TIME_AXIS_NAME if axis0_domain == "time" else cls._FREQ_AXIS_NAME
            )

            if isinstance(space_domain, dict):
                space_names = ["x", "y", "z"]  # defaults when dict provided
            elif space_domain == "k":
                space_names = list(cls._K_AXIS_NAMES)
            else:
                space_names = list(cls._REAL_AXIS_NAMES)
            axis_names = [time_name] + space_names

        obj = super().__new__(
            cls,
            value,
            unit=unit,
            name=name,
            epoch=effective_epoch,
            channel=channel,
            dtype=dtype,
            copy=copy,
            subok=subok,
            order=order,
            ndmin=ndmin,
            axis0=axis0,
            axis1=axis1,
            axis2=axis2,
            axis3=axis3,
            axis_names=axis_names,
        )

        # Set domain states
        if axis0_domain not in ("time", "frequency"):
            raise ValueError(
                f"axis0_domain must be 'time' or 'frequency', got '{axis0_domain}'"
            )
        obj._axis0_domain = axis0_domain

        # Handle space_domain: str -> all same, dict -> per-axis
        valid_space_domains = ("real", "k", "angle")
        if isinstance(space_domain, str):
            if space_domain not in valid_space_domains:
                raise ValueError(
                    f"space_domain must be 'real', 'k', or 'angle', got '{space_domain}'"
                )
            obj._space_domains = {
                obj._axis1_name: space_domain,
                obj._axis2_name: space_domain,
                obj._axis3_name: space_domain,
            }
        elif isinstance(space_domain, dict):
            for name, dom in space_domain.items():
                if dom not in valid_space_domains:
                    raise ValueError(
                        f"space_domain values must be 'real', 'k', or 'angle', "
                        f"got '{dom}' for '{name}'"
                    )
            obj._space_domains = dict(space_domain)
        else:
            raise TypeError(
                f"space_domain must be str or dict, got {type(space_domain)}"
            )

        obj._validate_domain_units()
        set_time_reference(obj, time_reference)

        return obj

    def with_epoch(self, epoch: Any) -> FieldBase:
        """Return a copy anchored at GPS ``epoch`` while retaining axis offsets.

        The time coordinate on axis 0 is shifted by the GPS anchor. Spatial
        coordinates and their real/k-space domain metadata are left untouched.
        A dimensionless time axis is interpreted as seconds, matching the
        existing SpectrogramMatrix anchoring rule.
        """
        from gwpy.time import to_gps

        if self.time_reference != TIME_REFERENCE_RELATIVE:
            raise ValueError("with_epoch requires a relative time reference")
        if classify_time_input(epoch) != "finite":
            raise ValueError("epoch must be a finite GPS time")

        value = epoch.to_value(u.s) if isinstance(epoch, u.Quantity) else epoch
        anchor = float(to_gps(value))
        if not np.isfinite(anchor):
            raise ValueError("epoch must be a finite GPS time")

        result = self.copy()
        dt = getattr(self, "_axis0_dt", None)
        if self._axis0_domain == "time":
            axis = self._axis0_index
            axis_unit = getattr(axis, "unit", u.dimensionless_unscaled)
            if axis_unit.is_equivalent(u.dimensionless_unscaled):
                offsets = np.asarray(getattr(axis, "value", axis)) * u.s
            elif axis_unit.is_equivalent(u.s):
                offsets = axis.to(u.s)
            else:
                raise ValueError(
                    "axis 0 must have time or dimensionless units to anchor it"
                )

            if len(offsets) >= 2:
                dt = offsets[1] - offsets[0]
            result._axis0_index = offsets + anchor * u.s
        else:
            # A time FFT stores its original time origin and sample spacing
            # separately because axis 0 now contains frequencies.  Anchor that
            # saved origin so a later ifft_time() reconstructs absolute times.
            offset = getattr(self, "_axis0_offset", None)
            if offset is None:
                offsets = 0 * u.s
            else:
                offset_unit = getattr(offset, "unit", u.dimensionless_unscaled)
                if offset_unit.is_equivalent(u.dimensionless_unscaled):
                    offsets = np.asarray(getattr(offset, "value", offset)) * u.s
                elif offset_unit.is_equivalent(u.s):
                    offsets = offset.to(u.s)
                else:
                    raise ValueError(
                        "stored time offset must have time or dimensionless units"
                    )
            result._axis0_offset = offsets + anchor * u.s

            if dt is None and len(self._axis0_index) >= 2:
                df = self._axis0_index[1] - self._axis0_index[0]
                df_unit = getattr(df, "unit", u.dimensionless_unscaled)
                df_value = float(getattr(df, "value", df))
                nout = 2 * (len(self._axis0_index) - 1)
                if np.isfinite(df_value) and df_value != 0 and nout > 0:
                    dt = (1.0 / (nout * df_value)) * df_unit**-1

        if dt is not None:
            result._axis0_dt = dt.copy() if hasattr(dt, "copy") else dt
            dt_unit = getattr(dt, "unit", u.dimensionless_unscaled)
            if dt_unit.is_equivalent(u.dimensionless_unscaled):
                dt_value = float(getattr(dt, "value", dt))
            elif dt_unit.is_equivalent(u.s):
                dt_value = float(dt.to_value(u.s))
            else:
                raise ValueError("stored sampling interval must have time units")
            ulp = abs(float(np.spacing(anchor)))
            if abs(dt_value) < 10 * ulp:
                warnings.warn(
                    f"dt={dt_value:g} s is below 10 ULP of epoch ({ulp:g} s); "
                    "absolute time coordinates may be rounded",
                    UserWarning,
                    stacklevel=2,
                )

        result.epoch = anchor
        set_time_reference(result, TIME_REFERENCE_ABSOLUTE)
        return result

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None:
            return

        # Copy domains from parent if available
        if getattr(self, "_axis0_domain", None) is None:
            self._axis0_domain = getattr(obj, "_axis0_domain", "time")

        parent_domains = getattr(obj, "_space_domains", None)
        if parent_domains is not None:
            # Check if axis names match. If they don't, we need to map keys.
            # But usually if we just copied names in super().__array_finalize__,
            # they should match.
            self._space_domains = dict(parent_domains)
        elif getattr(self, "_space_domains", None) is None:
            self._space_domains = {
                self._axis1_name: "real",
                self._axis2_name: "real",
                self._axis3_name: "real",
            }

        if getattr(self, "_axis0_offset", None) is None:
            self._axis0_offset = getattr(obj, "_axis0_offset", None)
        if getattr(self, "_axis0_dt", None) is None:
            self._axis0_dt = getattr(obj, "_axis0_dt", None)

        # Real-space start and direction of axes currently in k-space, recorded
        # by fft_space so that ifft_space can restore the coordinates.
        if getattr(self, "_space_origins", None) is None:
            parent_origins = getattr(obj, "_space_origins", None)
            self._space_origins = dict(parent_origins) if parent_origins else None

        self._validate_domain_units()

    def _propagate_gwex_attrs(self, other: Any) -> None:
        """Propagate GWexpy and time-FFT metadata to another Field."""
        super()._propagate_gwex_attrs(other)
        if not isinstance(other, FieldBase):
            return

        for name in ("_axis0_offset", "_axis0_dt"):
            value = getattr(self, name, None)
            if value is not None and getattr(other, name, None) is None:
                copied = value.copy() if hasattr(value, "copy") else value
                setattr(other, name, copied)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Validate domain and coordinate consistency before arithmetic."""
        # Identify FieldBase objects among inputs
        fields = [x for x in inputs if isinstance(x, FieldBase)]

        if len(fields) > 1:
            first = fields[0]
            for other in fields[1:]:
                first._validate_field_alignment(other)
                if (
                    first.time_reference == TIME_REFERENCE_ABSOLUTE
                    and first.epoch != other.epoch
                ):
                    raise ValueError(
                        "Field epoch mismatch: absolute fields must share the "
                        "same epoch before operation."
                    )

        # Defer to superclass (Array4D -> Array -> GwpyArray)
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

    def _validate_field_alignment(self, other: FieldBase) -> None:
        """Reject arithmetic between fields on different physical grids."""
        left_geo = self.geo
        right_geo = other.geo
        left_active = left_geo is not None and left_geo.active
        right_active = right_geo is not None and right_geo.active
        if left_active != right_active or (
            left_geo is not None
            and right_geo is not None
            and left_active
            and right_active
            and (left_geo.crs or "").strip().upper()
            != (right_geo.crs or "").strip().upper()
        ):
            raise ValueError(
                "Field CRS mismatch: active georeferencing must agree before "
                "arithmetic. Reproject operands to a common CRS first."
            )

        if self.time_reference != other.time_reference:
            raise ValueError(
                "Field time-reference mismatch: "
                f"{self.time_reference} vs {other.time_reference}. "
                "Convert both fields to the same time reference before operation."
            )

        if self._axis0_domain != other._axis0_domain:
            raise ValueError(
                f"Field domain mismatch: {self._axis0_domain} vs "
                f"{other._axis0_domain}. Perform domain transform "
                "(e.g., fft_time) before operation."
            )

        if self._space_domains != other._space_domains:
            raise ValueError(
                "Field spatial domain mismatch: "
                f"{self._space_domains} vs {other._space_domains}."
            )

        for i in range(4):
            left_index = getattr(self, f"_axis{i}_index")
            right_index = getattr(other, f"_axis{i}_index")
            if not _axis_coordinates_close(left_index, right_index):
                name = self.axis_names[i]
                raise ValueError(
                    f"Field coordinate mismatch on axis {i} ('{name}'). "
                    "Regrid operands to a common grid before operation."
                )

    @property
    def axis0_domain(self):
        """Domain of axis 0: 'time' or 'frequency'."""
        return self._axis0_domain

    @property
    def space_domains(self):
        """Mapping of spatial axis names to domains."""
        return dict(self._space_domains)

    @property
    def geo(self) -> GeoMetadata | None:
        """Georeferencing metadata (``None`` when not georeferenced)."""
        return getattr(self, "_gwex_geo", None)

    @geo.setter
    def geo(self, value: GeoMetadata | None) -> None:
        if value is not None and not isinstance(value, GeoMetadata):
            raise TypeError(
                f"geo must be a GeoMetadata instance or None, got {type(value).__name__}"
            )
        self._gwex_geo = value

    @property
    def frame(self) -> Any:
        """Coordinate frame / grid orientation geometry (:class:`~gwexpy.fields.GridGeometry`)."""
        return getattr(self, "_gwex_frame", None)

    @frame.setter
    def frame(self, value: Any) -> None:
        from .frames import GridGeometry

        if value is not None and not isinstance(value, GridGeometry):
            raise TypeError(
                f"frame must be a GridGeometry instance or None, got {type(value).__name__}"
            )
        self._gwex_frame = value

    def _set_axis_name(self, index: int, name: str) -> None:
        old_name = self.axis_names[index] if index < 4 else None
        super()._set_axis_name(index, name)
        if index in (1, 2, 3) and old_name is not None and old_name != name:
            domains = getattr(self, "_space_domains", None)
            if isinstance(domains, dict) and old_name in domains:
                domains[name] = domains.pop(old_name)

    def copy(self, order: str = "C") -> FieldBase:
        """Return a copy of the field with independent coordinate and domain metadata."""
        result = super().copy(order=order)
        for attr in ("_axis0_index", "_axis1_index", "_axis2_index", "_axis3_index"):
            idx = getattr(self, attr, None)
            if idx is not None and hasattr(idx, "copy"):
                setattr(result, attr, idx.copy())
        result._space_domains = dict(self._space_domains)
        space_origins = getattr(self, "_space_origins", None)
        if space_origins is not None:
            result._space_origins = dict(space_origins)
        for attr in ("_axis0_offset", "_axis0_dt"):
            val = getattr(self, attr, None)
            if val is not None and hasattr(val, "copy"):
                setattr(result, attr, val.copy())
        return result

    # ------------------------------------------------------------------
    # Visualization Helpers
    # ------------------------------------------------------------------
    def get_slice(self, x_axis=None, y_axis=None, **fixed_coords):
        """Extract a 2D slice from the 4D field.

        Parameters
        ----------
        x_axis : str, optional
            Name of the X-axis for the plot. If None, inferred.
        y_axis : str, optional
            Name of the Y-axis for the plot. If None, inferred.
        **fixed_coords
            Coordinates to fix for other axes (e.g., t=0, z=0.5).
            If an axis is not specified in x/y or fixed_coords, a default
            slice (center or index 0) is chosen.

        Returns
        -------
        2D Array/Quantity
            The sliced data.
        x_index, y_index
            Coordinate arrays for the slice axes.

        """
        # 1. Determine which axes are free (x, y) vs fixed
        all_axes = [
            self._axis0_name,
            self._axis1_name,
            self._axis2_name,
            self._axis3_name,
        ]

        # Resolve x/y axes if not provided
        if x_axis is None or y_axis is None:
            # Simple heuristic: try to find spatial axes (x, y) usually axis1, axis2
            # Or if t is involved, t vs x
            candidates = [ax for ax in all_axes if ax not in fixed_coords]
            if len(candidates) < 2:
                # Need at least 2 free axes, unless the data itself is small
                # Fallback to existing if available
                if x_axis is None:
                    x_axis = candidates[0] if candidates else self._axis1_name
                if y_axis is None:
                    y_axis = candidates[1] if len(candidates) > 1 else self._axis2_name
            else:
                if x_axis is None:
                    x_axis = candidates[0]
                if y_axis is None:
                    y_axis = candidates[1]

        # 2. Build selector dictionary for isel/sel
        # We want to perform ONE selection operation.
        # But 'sel' (value based) and 'isel' (index based) are mixed if we just pick default 0.
        # Ideally, we primarily use 'sel' (nearest) for user inputs,
        # and 'isel' for defaults (index 0 or center).

        # Current AxisApiMixin provides sel/isel.
        # Let's map everything to 'sel' with method='nearest' if user provided value,
        # or 'isel' if we pick default.
        # However, mixing them requires two calls or converting everything to one type.
        # Easier to convert defaults to 'sel' via axis index values if possible,
        # OR just iterate and slice.

        # Strategy: Construct a slice object for the 4D array
        slices: list[Any] = [slice(None)] * 4

        for i, ax_name in enumerate(all_axes):
            if ax_name == x_axis or ax_name == y_axis:
                continue

            # This axis needs to be fixed
            if ax_name in fixed_coords:
                val = fixed_coords[ax_name]
                # Check if it's an integer index or a physical quantity
                if isinstance(val, int) and not isinstance(val, u.Quantity):
                    # Treat int as index directly (convenience)
                    slices[i] = val
                else:
                    # Value-based selection: find nearest index
                    idx_arr = getattr(self, f"_axis{i}_index")
                    if isinstance(val, u.Quantity):
                        # find nearest
                        diff = np.abs(idx_arr - val)
                        slices[i] = np.argmin(diff)
                    else:
                        # Assume raw value matches raw quantity value
                        diff = np.abs(idx_arr.value - val)
                        slices[i] = np.argmin(diff)
            else:
                # Default behavior: slice at index 0 (or could be center)
                # Let's pivot to default to 0 for time, center for space?
                # For simplicity in this iteration: index 0
                slices[i] = 0

        # 3. Apply slice
        # Use basic numpy slicing to get data
        # Note: slices list contains ints or slice(None)

        # We need to ensure we return a 2D object, not 4D with size 1 dims
        # The numpy slice will drop dimensions that are ints
        data_slice = self.value[tuple(slices)]

        # 4. Identify x/y data
        # We need the index arrays for the chosen x/y axes
        try:
            x_idx_pos = all_axes.index(x_axis)
            y_idx_pos = all_axes.index(y_axis)
        except ValueError:
            raise ValueError(f"Invalid axis name. Available: {all_axes}")

        x_index = getattr(self, f"_axis{x_idx_pos}_index")
        y_index = getattr(self, f"_axis{y_idx_pos}_index")

        # If x_axis / y_axis were sliced (they shouldn't be, based on logic above),
        # data_slice would be scalar.
        # But `slices` only puts specific indices for non-x, non-y axes.
        # So x and y should remain as full dimensions.

        # Verify shape
        if data_slice.ndim != 2:
            # It's possible we have >2 free axes if user didn't fix enough?
            # Or <2 if user fixed x or y?
            pass

        return data_slice * self.unit, x_index, y_index, x_axis, y_axis

    def plot(
        self,
        x=None,
        y=None,
        slices=None,
        *,
        terrain=None,
        terrain_style: str = "hillshade",
        basemap=None,
        vector=None,
        detectors=None,
        **kwargs,
    ):
        """Plot the field as a 2D slice with optional layered overlays.

        Wrapper around :class:`~gwexpy.plot.field.FieldPlot`.

        Parameters
        ----------
        x : str, optional
            X-axis name.
        y : str, optional
            Y-axis name.
        slices : dict, optional
            Dictionary of axis names to coordinate values for slicing.
            Useful to resolve conflicts when an axis name matches a parameter name.
        terrain : ScalarField, optional
            Optional DEM field rendered as a background terrain layer via
            :meth:`~gwexpy.plot.field.FieldPlot.add_terrain` (``zorder=1.0``).
        terrain_style : str, optional
            Style passed to ``add_terrain`` (default ``'hillshade'``).
        basemap : bool, str, or dict, optional
            Optional XYZ tile basemap layer (``zorder=0.0``). Pass ``True`` for
            the default provider (``'gsi-std'``), a provider name/template string,
            or a dict of keyword arguments for ``add_basemap``.
        vector : VectorField, optional
            Optional vector field overlaid as quiver arrows (``zorder=3.0``).
        detectors : str or sequence of str, optional
            Optional gravitational-wave detector names plotted via
            :meth:`~gwexpy.plot.field.FieldPlot.add_detector` (``zorder=4.0``).
        **kwargs
            Fixed coordinates (e.g. ``z=0``) and plotting keyword arguments
            forwarded to :meth:`~gwexpy.plot.field.FieldPlot.add_scalar`.

        """
        from gwexpy.interop._registry import ConverterRegistry

        FieldPlot = ConverterRegistry.get_constructor("FieldPlot")

        # Initialize empty FieldPlot
        fp = FieldPlot()

        slice_kwargs = {}
        if slices is not None:
            slice_kwargs.update(slices)

        plot_kwargs = {}

        all_axes = [
            self._axis0_name,
            self._axis1_name,
            self._axis2_name,
            self._axis3_name,
        ]

        for k, v in kwargs.items():
            if k in all_axes:
                slice_kwargs[k] = v
            else:
                plot_kwargs[k] = v

        if terrain is not None:
            fp.add_terrain(
                terrain,
                x=x,
                y=y,
                style=terrain_style,
                slice_kwargs=slice_kwargs,
            )

        fp.add_scalar(self, x=x, y=y, slice_kwargs=slice_kwargs, **plot_kwargs)

        if basemap is not None and basemap is not False:
            if basemap is True:
                fp.add_basemap()
            elif isinstance(basemap, str):
                fp.add_basemap(source=basemap)
            elif isinstance(basemap, dict):
                fp.add_basemap(**basemap)
            else:
                raise TypeError(
                    "basemap must be bool, str, or dict of add_basemap kwargs, "
                    f"got {type(basemap).__name__}."
                )

        if vector is not None:
            fp.add_vector(
                vector,
                x=x,
                y=y,
                mode="quiver",
                slice_kwargs=slice_kwargs,
            )

        if detectors is not None:
            det_list = [detectors] if isinstance(detectors, str) else list(detectors)
            for det_name in det_list:
                fp.add_detector(det_name)

        return fp

    def animate(self, x=None, y=None, axis="t", interval=100, **kwargs):
        """Create an animation along a specified axis.

        Delegates to :meth:`~gwexpy.plot.field.FieldPlot.animate_scalar`.

        Parameters
        ----------
        x : str, optional
            X-axis name.
        y : str, optional
            Y-axis name.
        axis : str
            Name of the axis to iterate over (default ``'t'``).
        interval : int
            Delay between frames in milliseconds.
        **kwargs
            Fixed coordinates and plotting arguments.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            The animation object.

        """
        from gwexpy.interop._registry import ConverterRegistry

        FieldPlot = ConverterRegistry.get_constructor("FieldPlot")

        all_axes = [
            self._axis0_name,
            self._axis1_name,
            self._axis2_name,
            self._axis3_name,
        ]
        slice_kwargs = {}
        plot_kwargs = {}

        for k, v in kwargs.items():
            if k in all_axes:
                slice_kwargs[k] = v
            else:
                plot_kwargs[k] = v

        fp = FieldPlot()
        return fp.animate_scalar(
            self,
            x=x,
            y=y,
            axis=axis,
            interval=interval,
            slice_kwargs=slice_kwargs,
            **plot_kwargs,
        )

    # ------------------------------------------------------------------
    # Domain/unit validation
    # ------------------------------------------------------------------
    def _validate_domain_units(self) -> None:
        """Validate that axis units are consistent with declared domains."""
        # Axis 0: time or frequency
        axis0 = getattr(self, "_axis0_index", None)
        if isinstance(axis0, u.Quantity):
            axis0_unit = axis0.unit
            if axis0_unit != u.dimensionless_unscaled:
                expected = u.s if self._axis0_domain == "time" else 1 / u.s
                if not axis0_unit.is_equivalent(expected):
                    raise ValueError(
                        f"Axis0 domain '{self._axis0_domain}' expects units "
                        f"equivalent to {expected}, got {axis0_unit}"
                    )

        # Spatial axes: position, wavenumber, or angle
        spatial_axes = [
            (self._axis1_name, getattr(self, "_axis1_index", None)),
            (self._axis2_name, getattr(self, "_axis2_index", None)),
            (self._axis3_name, getattr(self, "_axis3_index", None)),
        ]
        domain_expected_units = {
            "real": u.m,
            "k": 1 / u.m,
            "angle": u.rad,
        }
        for name, axis in spatial_axes:
            domain = self._space_domains.get(name)
            if domain is None or not isinstance(axis, u.Quantity):
                continue
            axis_unit = axis.unit
            if axis_unit == u.dimensionless_unscaled:
                continue
            if domain not in domain_expected_units:
                raise ValueError(f"Unknown space_domain '{domain}' for axis '{name}'.")
            expected = domain_expected_units[domain]
            if not axis_unit.is_equivalent(expected):
                raise ValueError(
                    f"Spatial axis '{name}' domain '{domain}' expects units "
                    f"equivalent to {expected}, got {axis_unit}"
                )

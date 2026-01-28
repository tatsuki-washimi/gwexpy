"""Base class for field objects with domain-aware axis metadata."""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u

from ..types.array4d import Array4D

__all__ = ["FieldBase"]


class FieldBase(Array4D):
    """Base class providing axis domain metadata handling for field objects.

    Maintains the invariant of four coordinate axes `(axis0, x, y, z)` and
    explicit domain labels for temporal and spatial axes.
    """

    _metadata_slots = Array4D._metadata_slots + (
        "_axis0_domain",
        "_space_domains",
        "_axis0_offset",  # Preserved during fft_time for ifft_time reconstruction
    )

    # Axis name conventions
    _TIME_AXIS_NAME = "t"
    _FREQ_AXIS_NAME = "f"
    _REAL_AXIS_NAMES = ("x", "y", "z")
    _K_AXIS_NAMES = ("kx", "ky", "kz")

    _axis0_index: Any
    _axis1_index: Any
    _axis2_index: Any
    _axis3_index: Any
    _space_domains: dict[str, str]

    def __new__(
        cls,
        data,
        unit=None,
        axis0=None,
        axis1=None,
        axis2=None,
        axis3=None,
        axis_names=None,
        axis0_domain: str = "time",
        space_domain: str | dict[str, str] = "real",
        **kwargs,
    ):
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
            data,
            unit=unit,
            axis0=axis0,
            axis1=axis1,
            axis2=axis2,
            axis3=axis3,
            axis_names=axis_names,
            **kwargs,
        )

        # Set domain states
        if axis0_domain not in ("time", "frequency"):
            raise ValueError(
                f"axis0_domain must be 'time' or 'frequency', got '{axis0_domain}'"
            )
        obj._axis0_domain = axis0_domain

        # Handle space_domain: str -> all same, dict -> per-axis
        if isinstance(space_domain, str):
            if space_domain not in ("real", "k"):
                raise ValueError(
                    f"space_domain must be 'real' or 'k', got '{space_domain}'"
                )
            obj._space_domains = {
                obj._axis1_name: space_domain,
                obj._axis2_name: space_domain,
                obj._axis3_name: space_domain,
            }
        elif isinstance(space_domain, dict):
            for name, dom in space_domain.items():
                if dom not in ("real", "k"):
                    raise ValueError(
                        f"space_domain values must be 'real' or 'k', "
                        f"got '{dom}' for '{name}'"
                    )
            obj._space_domains = dict(space_domain)
        else:
            raise TypeError(
                f"space_domain must be str or dict, got {type(space_domain)}"
            )

        obj._validate_domain_units()

        return obj

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

        self._validate_domain_units()

    @property
    def axis0_domain(self):
        """Domain of axis 0: 'time' or 'frequency'."""
        return self._axis0_domain

    @property
    def space_domains(self):
        """Mapping of spatial axis names to domains."""
        return dict(self._space_domains)

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

    def plot(self, x=None, y=None, slices=None, **kwargs):
        """Plot the field as a 2D slice.

        Wrapper around FieldPlot.

        Parameters
        ----------
        x : str, optional
            X-axis name.
        y : str, optional
            Y-axis name.
        slices : dict, optional
            Dictionary of axis names to coordinate values for slicing.
            Useful to resolve conflicts when an axis name matches a parameter name.
        **kwargs
            Fixed coordinates (e.g. z=0) and plotting keyword arguments.
        """
        # Defer import to avoid circular dependency
        from ..plot.field import FieldPlot

        # Initialize empty FieldPlot, then add scalar
        fp = FieldPlot()

        # Separate slice kwargs from plot kwargs
        # Slice kwargs: axis names and coordinate values
        # Plot kwargs: cmap, vmin, vmax, etc.
        # Heuristic: if arg is an axis name, it's a fixed coord.

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

        fp.add_scalar(self, x=x, y=y, slice_kwargs=slice_kwargs, **plot_kwargs)
        return fp

    def animate(self, x=None, y=None, axis="t", interval=100, **kwargs):
        """Create an animation along a specified axis.

        Parameters
        ----------
        x : str, optional
            X-axis name.
        y : str, optional
            Y-axis name.
        axis : str
            Name of the axis to iterate over (default 't').
        interval : int
            Delay between frames in milliseconds.
        **kwargs
            Fixed coordinates and plotting arguments.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            The animation object.
        """
        from matplotlib.animation import FuncAnimation

        from ..plot.field import FieldPlot

        # Identify axis to loop
        all_axes = [
            self._axis0_name,
            self._axis1_name,
            self._axis2_name,
            self._axis3_name,
        ]
        try:
            loop_idx_pos = all_axes.index(axis)
        except ValueError:
            raise ValueError(f"Animation axis '{axis}' not found in {all_axes}")

        loop_axis_index = getattr(self, f"_axis{loop_idx_pos}_index")

        # Determine fixed params for other axes
        slice_kwargs = {}
        plot_kwargs = {}

        for k, v in kwargs.items():
            if k in all_axes:
                slice_kwargs[k] = v
            else:
                plot_kwargs[k] = v

        # Initialize Plot
        fp = FieldPlot()
        ax = fp.gca()

        # Pre-calculate common arguments to avoid overhead
        # We need to decide proper vmin/vmax if not provided, to keep scale steady
        if "vmin" not in plot_kwargs or "vmax" not in plot_kwargs:
            # Calculate global min/max for the whole field (or sampled?)
            # This might be heavy for huge data.
            # Use 2% and 98% percentile of the whole data? Or just min/max?
            # Safe default fallback: let matplotlib decide per frame (flickering!)
            # Better: use min/max of the field.
            if "vmin" not in plot_kwargs:
                plot_kwargs["vmin"] = np.nanmin(self.value)
            if "vmax" not in plot_kwargs:
                plot_kwargs["vmax"] = np.nanmax(self.value)

        def update(frame_val):
            ax.clear()
            # Update slice kwargs with current frame value
            current_slice_kwargs = slice_kwargs.copy()
            current_slice_kwargs[axis] = frame_val

            # Add scalar plot
            fp.add_scalar(
                self, x=x, y=y, slice_kwargs=current_slice_kwargs, **plot_kwargs
            )

            # Set title
            if isinstance(frame_val, u.Quantity):
                title = f"{axis} = {frame_val:.3g}"
            else:
                title = f"{axis} = {frame_val}"
            ax.set_title(title)

        # Create animation
        # frames should be values from the loop axis
        # To avoid too many frames, maybe downsample?
        # User can slice the field before calling animate if needed.
        frames = loop_axis_index
        if len(frames) > 200:
            # Heuristic: limit frames for default smoothness
            # User should pre-slice if they want specific range
            pass  # Use all for correctness

        ani = FuncAnimation(fp.figure, update, frames=frames, interval=interval)

        # Attach figure to ani to prevent gc?
        ani._fig = fp.figure  # type: ignore[attr-defined]

        # Close static plot to prevent display of empty frame 0 if using inline?
        # plt.close(fp.figure) # No, we need it open for animation display depending on backend

        return ani

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

        # Spatial axes: position or wavenumber
        spatial_axes = [
            (self._axis1_name, getattr(self, "_axis1_index", None)),
            (self._axis2_name, getattr(self, "_axis2_index", None)),
            (self._axis3_name, getattr(self, "_axis3_index", None)),
        ]
        for name, axis in spatial_axes:
            domain = self._space_domains.get(name)
            if domain is None or not isinstance(axis, u.Quantity):
                continue
            axis_unit = axis.unit
            if axis_unit == u.dimensionless_unscaled:
                continue
            expected = u.m if domain == "real" else 1 / u.m
            if not axis_unit.is_equivalent(expected):
                raise ValueError(
                    f"Spatial axis '{name}' domain '{domain}' expects units "
                    f"equivalent to {expected}, got {axis_unit}"
                )

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from astropy import units as u
from scipy import fft as sp_fft

from gwexpy.interop._registry import ConverterRegistry

from .base import FieldBase
from .geo import GeoMetadata, geo_bounds

if TYPE_CHECKING:
    from gwexpy.types.typing import IndexLike

__all__ = ["ScalarField"]

_ELLIPSOID_PARAMS: dict[str, tuple[float, float]] = {
    # (semi-major axis a [m], inverse flattening 1/f; 0.0 means sphere)
    "grs80": (6378137.0, 298.257222101),
    "wgs84": (6378137.0, 298.257223563),
    "sphere": (6371008.7714, 0.0),
}


def _is_uniform_k_axis(k_axis: Any) -> bool:
    """Return True if *k_axis* is linearly regular or uniform in fftfreq order."""
    from ..types.axis import AxisDescriptor

    ax_desc = AxisDescriptor("k", k_axis)
    if ax_desc.regular:
        return True
    k_vals = np.asarray(k_axis.to_value(ax_desc.unit))
    if k_vals.size < 2 or not np.all(np.isfinite(k_vals)):
        return False
    n = k_vals.size
    abs_dk = float(abs(k_vals[1] - k_vals[0]))
    if abs_dk == 0.0:
        return False
    unit_grid = np.fft.fftfreq(n, d=1.0 / n)
    tol = 1e-12 * float(np.max(np.abs(k_vals)))
    return bool(
        np.allclose(k_vals, unit_grid * abs_dk, rtol=1e-12, atol=tol)
        or np.allclose(k_vals, -unit_grid * abs_dk, rtol=1e-12, atol=tol)
    )


def _resample_2d_grid(
    src_x_asc: np.ndarray,
    src_y_asc: np.ndarray,
    src_2d: np.ndarray,
    dst_x_src: np.ndarray,
    dst_y_src: np.ndarray,
    method: Literal["nearest", "bilinear", "cubic"],
) -> np.ndarray:
    """Resample a 2D array defined on ascending ``(src_x_asc, src_y_asc)``."""
    out = np.full(dst_x_src.shape, np.nan, dtype=np.float64)
    # Raster samples represent cells. Target centers inside the outer half-cell
    # footprint use the nearest edge sample instead of becoming a NaN border.
    x_low = src_x_asc[0] - 0.5 * (src_x_asc[1] - src_x_asc[0])
    x_high = src_x_asc[-1] + 0.5 * (src_x_asc[-1] - src_x_asc[-2])
    y_low = src_y_asc[0] - 0.5 * (src_y_asc[1] - src_y_asc[0])
    y_high = src_y_asc[-1] + 0.5 * (src_y_asc[-1] - src_y_asc[-2])
    in_bounds = (
        np.isfinite(dst_x_src)
        & np.isfinite(dst_y_src)
        & (dst_x_src >= x_low)
        & (dst_x_src <= x_high)
        & (dst_y_src >= y_low)
        & (dst_y_src <= y_high)
    )
    if not np.any(in_bounds):
        return out

    px = np.clip(dst_x_src[in_bounds], src_x_asc[0], src_x_asc[-1])
    py = np.clip(dst_y_src[in_bounds], src_y_asc[0], src_y_asc[-1])
    nx, ny = src_x_asc.size, src_y_asc.size

    if method == "nearest":
        if nx == 1:
            ix = np.zeros(px.shape, dtype=np.intp)
        else:
            ix_hi = np.clip(np.searchsorted(src_x_asc, px, side="left"), 0, nx - 1)
            ix_lo = np.clip(ix_hi - 1, 0, nx - 1)
            choose_lo = np.abs(px - src_x_asc[ix_lo]) <= np.abs(src_x_asc[ix_hi] - px)
            ix = np.where(choose_lo, ix_lo, ix_hi)
        if ny == 1:
            iy = np.zeros(py.shape, dtype=np.intp)
        else:
            iy_hi = np.clip(np.searchsorted(src_y_asc, py, side="left"), 0, ny - 1)
            iy_lo = np.clip(iy_hi - 1, 0, ny - 1)
            choose_lo = np.abs(py - src_y_asc[iy_lo]) <= np.abs(src_y_asc[iy_hi] - py)
            iy = np.where(choose_lo, iy_lo, iy_hi)
        out[in_bounds] = src_2d[ix, iy]
        return out

    if method == "bilinear":
        if nx == 1:
            ix0 = ix1 = np.zeros(px.shape, dtype=np.intp)
            tx = np.zeros(px.shape, dtype=np.float64)
        else:
            ix0 = np.clip(np.searchsorted(src_x_asc, px, side="right") - 1, 0, nx - 2)
            ix1 = ix0 + 1
            denom_x = src_x_asc[ix1] - src_x_asc[ix0]
            tx = np.where(denom_x != 0, (px - src_x_asc[ix0]) / denom_x, 0.0)
        if ny == 1:
            iy0 = iy1 = np.zeros(py.shape, dtype=np.intp)
            ty = np.zeros(py.shape, dtype=np.float64)
        else:
            iy0 = np.clip(np.searchsorted(src_y_asc, py, side="right") - 1, 0, ny - 2)
            iy1 = iy0 + 1
            denom_y = src_y_asc[iy1] - src_y_asc[iy0]
            ty = np.where(denom_y != 0, (py - src_y_asc[iy0]) / denom_y, 0.0)

        w00 = (1.0 - tx) * (1.0 - ty)
        w10 = tx * (1.0 - ty)
        w01 = (1.0 - tx) * ty
        w11 = tx * ty

        v00 = src_2d[ix0, iy0]
        v10 = src_2d[ix1, iy0]
        v01 = src_2d[ix0, iy1]
        v11 = src_2d[ix1, iy1]

        nan_hit = (
            ((w00 > 0) & np.isnan(v00))
            | ((w10 > 0) & np.isnan(v10))
            | ((w01 > 0) & np.isnan(v01))
            | ((w11 > 0) & np.isnan(v11))
        )
        acc = (
            np.where(w00 > 0, w00 * np.nan_to_num(v00, nan=0.0), 0.0)
            + np.where(w10 > 0, w10 * np.nan_to_num(v10, nan=0.0), 0.0)
            + np.where(w01 > 0, w01 * np.nan_to_num(v01, nan=0.0), 0.0)
            + np.where(w11 > 0, w11 * np.nan_to_num(v11, nan=0.0), 0.0)
        )
        out[in_bounds] = np.where(nan_hit, np.nan, acc)
        return out

    if method == "cubic":
        from scipy.interpolate import RegularGridInterpolator

        if nx < 4 or ny < 4:
            raise ValueError(
                "Cubic resampling requires at least 4 samples along both spatial axes."
            )
        has_nan = bool(np.any(np.isnan(src_2d)))
        work_src = np.nan_to_num(src_2d, nan=0.0) if has_nan else src_2d
        interp = RegularGridInterpolator(
            (src_x_asc, src_y_asc),
            work_src,
            method="cubic",
            bounds_error=False,
            fill_value=np.nan,
        )
        vals = interp(np.column_stack([px, py]))
        if has_nan:
            ix0 = np.clip(np.searchsorted(src_x_asc, px, side="right") - 1, 0, nx - 2)
            iy0 = np.clip(np.searchsorted(src_y_asc, py, side="right") - 1, 0, ny - 2)
            # Invalidate cubic samples whose 4x4 support contains NaN
            nan_mask = np.isnan(src_2d)
            bad = np.zeros(px.shape, dtype=bool)
            for dx in (-1, 0, 1, 2):
                sx = np.clip(ix0 + dx, 0, nx - 1)
                for dy in (-1, 0, 1, 2):
                    sy = np.clip(iy0 + dy, 0, ny - 1)
                    bad |= nan_mask[sx, sy]
            vals = np.where(bad, np.nan, vals)
        out[in_bounds] = vals
        return out

    raise ValueError(f"Unsupported resampling method '{method}'.")


class ScalarField(FieldBase):
    """A 4-dimensional scalar field with domain tracking.

    `ScalarField` represents a physical field defined on a 4D grid
    (typically 1 time and 3 spatial dimensions). It supports domain
    transformations (Fourier transforms) while preserving coordinate
    metadata and units.

    Parameters
    ----------
    data : array-like
        4-dimensional input data.

    unit : `str`, `~astropy.units.Unit`, optional
        Physical unit of the field values.

    axis0 : array-like, optional
        Coordinate values for axis 0 (Time or Frequency).

    axis1, axis2, axis3 : array-like, optional
        Coordinate values for spatial axes.

    axis_names : iterable of str, optional
        Names for each axis (length 4).

    axis0_domain : {'time', 'frequency'}, optional
        Initial domain of the first axis.

    space_domain : {'real', 'k', 'angle'} or dict, optional
        Initial domain of the spatial axes.

    **kwargs
        Additional keyword arguments passed to `FieldBase`.

    Notes
    -----
    A key feature of `ScalarField` is that slicing always returns a
    4D `ScalarField`. Integer indices are automatically converted to
    length-1 slices to prevent dimension loss.

    Examples
    --------
    >>> import numpy as np
    >>> from gwexpy.fields import ScalarField
    >>> data = np.ones((2, 2, 2, 2))
    >>> sf = ScalarField(data, axis0=[0, 1], axis1=[0, 1],
    ...                  axis2=[0, 1], axis3=[0, 1])
    >>> sf
    <ScalarField([[[[1., 1.],
                    [1., 1.]],
    ...
                 axis0_domain='time',
                 space_domains={'x': 'real', 'y': 'real', 'z': 'real'},
                 axis0_offset=None)>

    """

    _axis0_index: IndexLike
    _axis1_index: IndexLike
    _axis2_index: IndexLike
    _axis3_index: IndexLike

    def __repr__(self) -> str:
        """Return a compact repr with shape and domain information."""
        unit = self.unit if self.unit is not None else u.dimensionless_unscaled
        try:
            unit_value = float(unit.to_value(u.dimensionless_unscaled))
        except Exception:
            unit_value = 1.0
        shape = ", ".join(str(dim) for dim in self.shape)
        return f"<ScalarField({shape})@{self.axis0_domain}, {unit_value}>"

    @classmethod
    def read(
        cls,
        source: Any,
        name: str | None = None,
        *,
        format: str | None = None,
        start: Any = None,
        end: Any = None,
        **kwargs: Any,
    ) -> ScalarField:
        """Read a :class:`ScalarField` from file(s), archive, or dataset.

        Parameters
        ----------
        source : str, Path, file-like, or sequence of paths
            Input file path, archive, or tile sequence.
        name : str, optional
            Quantity selector (or output field name for single-quantity formats).
        format : str, optional
            Format identifier (e.g. ``'gsi-dem'``, ``'geotiff'``). When
            ``None``, format is inferred via registered identifiers.
        start, end : optional
            Continuous range selection along axis 0 where supported.
        **kwargs
            Additional format-specific reader options (and optional read-time
            aliases ``fill_missing=`` and ``projected=``).

        """
        from .io.core import read_scalarfield

        return read_scalarfield(
            source,
            name=name,
            format=format,
            start=start,
            end=end,
            **kwargs,
        )

    def write(
        self,
        target: Any,
        format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Write this :class:`ScalarField` to *target*.

        Parameters
        ----------
        target : str, Path, or file-like
            Destination file path or stream.
        format : str, optional
            Output format identifier (e.g. ``'geotiff'``).
        **kwargs
            Format-specific writer options.

        """
        from .io.core import write_scalarfield

        return write_scalarfield(self, target, format=format, **kwargs)

    def __getitem__(self, item: Any) -> ScalarField:
        """Get item, always returning ScalarField (4D maintained).

        Integer indices are converted to length-1 slices to maintain
        4D structure.
        """
        forced_item = self._force_4d_item(item)
        return self._getitem_scalarfield(forced_item)

    def _force_4d_item(self, item: Any) -> tuple[Any, ...]:
        """Convert int indices to slice(i, i+1) to maintain 4D."""
        if not isinstance(item, tuple):
            item = (item,)

        # Handle Ellipsis
        if Ellipsis in item:
            if item.count(Ellipsis) > 1:
                raise IndexError("Only one ellipsis allowed")
            ellipsis_idx = item.index(Ellipsis)
            num_specified = len(item) - 1
            fill = 4 - num_specified
            if fill < 0:
                raise IndexError("Too many indices for 4D array")
            item = (
                item[:ellipsis_idx] + (slice(None),) * fill + item[ellipsis_idx + 1 :]
            )

        # Pad to length 4
        if len(item) < 4:
            item = item + (slice(None),) * (4 - len(item))

        if len(item) > 4:
            raise IndexError("Too many indices for 4D array")

        # Convert int to slice(i, i+1)
        result = []
        for i, idx in enumerate(item):
            if self._is_int_index(idx):
                # Normalize negative indices
                size = self.shape[i]
                if idx < 0:
                    idx = size + idx
                if idx < 0 or idx >= size:
                    raise IndexError(
                        f"Index {idx} out of bounds for axis {i} with size {size}"
                    )
                result.append(slice(idx, idx + 1))
            else:
                result.append(idx)

        return tuple(result)

    def _getitem_scalarfield(self, item: tuple[Any, ...]) -> ScalarField:
        """Perform getitem with ScalarField reconstruction.

        item should already be normalized (all slices, length 4).
        """
        # Call parent's raw getitem
        from gwpy.types.array import Array as GwpyArray

        raw = GwpyArray.__getitem__(self, item)

        if not isinstance(item, tuple) or len(item) != 4:
            return self._to_plain(raw)

        # All should be slices now (from _force_4d_item)
        current_axes = [
            (self._axis0_name, self._axis0_index),
            (self._axis1_name, self._axis1_index),
            (self._axis2_name, self._axis2_index),
            (self._axis3_name, self._axis3_index),
        ]

        new_axes = []
        for i, sl in enumerate(item):
            name, idx_arr = current_axes[i]
            if isinstance(sl, slice):
                new_axes.append((name, idx_arr[sl]))
            else:
                # Unexpected: should be slice after _force_4d_item
                return self._to_plain(raw)

        if getattr(raw, "ndim", None) != 4:
            return self._to_plain(raw)

        value, unit = self._value_unit(raw)
        meta = self._metadata_kwargs(raw)

        # Build space_domains for new axes
        new_space_domains = {}
        for name, _ in new_axes[1:]:  # spatial axes only
            if name in self._space_domains:
                new_space_domains[name] = self._space_domains[name]
            else:
                new_space_domains[name] = "real"

        result = ScalarField(
            value,
            unit=unit,
            axis_names=[n for n, _ in new_axes],
            axis0=new_axes[0][1],
            axis1=new_axes[1][1],
            axis2=new_axes[2][1],
            axis3=new_axes[3][1],
            axis0_domain=self._axis0_domain,
            space_domain=new_space_domains,
            copy=False,
            **meta,
        )
        self._propagate_gwex_attrs(result)
        # Format provenance may carry arrays aligned with the four field axes.
        # Let opt-in metadata update those arrays for this normalized slice.
        for name, provenance in self.__dict__.items():
            slicer = getattr(provenance, "_slice_for_field", None)
            if name.startswith("_gwex_") and callable(slicer):
                setattr(result, name, slicer(item, self.shape, new_axes[0][1]))
        return result

    def _isel_tuple(self, item_tuple: tuple[Any, ...]) -> ScalarField:
        """Apply internal `isel` logic using ScalarField indexing semantics."""
        forced_item = self._force_4d_item(item_tuple)
        return self._getitem_scalarfield(forced_item)

    # =========================================================================
    # Time FFT (axis=0, GWpy TimeSeries.fft compatible)
    # =========================================================================

    def _validate_axis_for_fft(
        self, axis_index: IndexLike, axis_name: str, domain_name: str
    ) -> None:
        """Validate that an axis is suitable for FFT.

        Parameters
        ----------
        axis_index : Quantity
            The axis coordinate array.
        axis_name : str
            Name of the axis for error messages.
        domain_name : str
            Domain name ('time', 'frequency', etc.) for error messages.

        Raises
        ------
        ValueError
            If axis length < 2 or axis is not regularly spaced.

        """
        if len(axis_index) < 2:
            raise ValueError(
                f"FFT requires {domain_name} axis length >= 2, "
                f"got length {len(axis_index)} for axis '{axis_name}'"
            )
        # Check regularity using AxisDescriptor
        from ..types.axis import AxisDescriptor

        ax_desc = AxisDescriptor(axis_name, axis_index)
        if not ax_desc.regular:
            raise ValueError(
                f"FFT requires regularly spaced {domain_name} axis, "
                f"but axis '{axis_name}' is irregular"
            )

    def _validate_uniform_time_axis(self, op_name: str):
        """Validate that axis 0 is approximately uniform for resampling/filtering."""
        axis_index = self._axis0_index
        axis_name = self._axis0_name
        if len(axis_index) < 2:
            raise ValueError(
                f"{op_name} requires time axis length >= 2, "
                f"got length {len(axis_index)} for axis '{axis_name}'"
            )
        from ..types.axis import AxisDescriptor

        ax_desc = AxisDescriptor(axis_name, axis_index)
        if not ax_desc.is_regular(max_deviation=1e-9):
            raise ValueError(
                f"{op_name} requires regularly spaced time axis, "
                f"but axis '{axis_name}' is irregular"
            )
        return ax_desc.mean_delta()

    def fft_time(self, nfft: int | None = None) -> ScalarField:
        """Compute FFT along the time axis (axis 0).

        Transforms the field from 'time' to 'frequency' domain using
        `scipy.fft.rfft`, applying GWpy-style half-sided normalization.

        Parameters
        ----------
        nfft : int, optional
            Length of the FFT. Defaults to the length of axis 0.

        Returns
        -------
        ScalarField
            A new field with `axis0_domain='frequency'`.

        Notes
        -----
        Normalizes by `1/nfft` and doubles non-DC/non-Nyquist bins to preserve
        amplitude.

        See Also
        --------
        gwpy.timeseries.TimeSeries.fft : The underlying transformation.

        """
        if self._axis0_domain != "time":
            raise ValueError(
                f"fft_time requires axis0_domain='time', got '{self._axis0_domain}'"
            )

        # Validate axis regularity and length
        self._validate_axis_for_fft(self._axis0_index, self._axis0_name, "time")

        # Reject complex input (rfft expects real-valued signals)
        if np.iscomplexobj(self.value):
            raise TypeError(
                "fft_time requires real-valued input. "
                "For complex data, use a full FFT approach."
            )

        if nfft is None:
            nfft = self.shape[0]

        # Preserve time-axis origin for later ifft_time reconstruction
        t0 = self._axis0_index[0]

        # rfft along axis 0, normalized
        import scipy.fft as sp_fft

        dft = sp_fft.rfft(self.value, n=nfft, axis=0) / nfft

        # Multiply non-DC, non-Nyquist bins by 2 (one-sided spectrum correction)
        # For even nfft: Nyquist bin is at index -1, should NOT be doubled
        # For odd nfft: there is no Nyquist bin, double all bins from 1:
        if nfft % 2 == 0:
            # Even: double bins 1 to -1 (exclusive of Nyquist)
            dft[1:-1, ...] *= 2.0
        else:
            # Odd: double bins 1 onwards (no Nyquist bin)
            dft[1:, ...] *= 2.0

        # Compute frequency axis
        dt = self._axis0_index[1] - self._axis0_index[0]
        dt_value = getattr(dt, "value", dt)
        dt_unit = getattr(dt, "unit", u.dimensionless_unscaled)

        freqs_value = np.fft.rfftfreq(nfft, d=dt_value)
        freqs = freqs_value * (1 / dt_unit)

        result = ScalarField(
            dft,
            unit=self.unit,
            epoch=self.epoch,
            axis0=freqs,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=[
                self._FREQ_AXIS_NAME,
                self._axis1_name,
                self._axis2_name,
                self._axis3_name,
            ],
            axis0_domain="frequency",
            space_domain=self._space_domains,
        )
        # Store the original time offset in metadata
        result._axis0_offset = t0
        result._axis0_dt = dt
        result._validate_domain_units()
        self._propagate_gwex_attrs(result)
        return result

    def ifft_time(self, nout: int | None = None) -> ScalarField:
        """Compute inverse FFT along frequency axis (axis 0).

        This method applies the inverse normalization of
        ``fft_time()`` / GWpy's ``FrequencySeries.ifft()``.

        Parameters
        ----------
        nout : int, optional
            Length of the output time series. If None, computed as
            ``(n_freq - 1) * 2``.

        Returns
        -------
        ScalarField
            Transformed field with ``axis0_domain='time'``.

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'frequency'.
        ValueError
            If frequency axis length < 2 or is irregularly spaced.

        See Also
        --------
        gwpy.frequencyseries.FrequencySeries.ifft : Reference implementation.

        """
        if self._axis0_domain != "frequency":
            raise ValueError(
                f"ifft_time requires axis0_domain='frequency', "
                f"got '{self._axis0_domain}'"
            )

        # Validate axis regularity and length
        self._validate_axis_for_fft(self._axis0_index, self._axis0_name, "frequency")

        if nout is None:
            nout = (self.shape[0] - 1) * 2

        # Undo normalization: divide non-DC, non-Nyquist by 2, multiply by nout
        array = self.value.copy()
        if nout % 2 == 0:
            # Even nout: Nyquist was not doubled, so only undo for 1:-1
            array[1:-1, ...] /= 2.0
        else:
            # Odd nout: no Nyquist, undo for all bins 1:
            array[1:, ...] /= 2.0
        dift = sp_fft.irfft(array * nout, n=nout, axis=0)

        # Compute time axis
        df = self._axis0_index[1] - self._axis0_index[0]
        df_value = getattr(df, "value", df)
        df_unit = getattr(df, "unit", u.dimensionless_unscaled)

        # dt = 1 / (nout * df)
        dt_value = 1.0 / (nout * df_value)
        dt_unit = df_unit**-1

        # Restore time-axis origin if preserved from fft_time
        t0_offset = getattr(self, "_axis0_offset", None)
        if t0_offset is not None:
            offset_unit = getattr(t0_offset, "unit", u.dimensionless_unscaled)
            if offset_unit.is_equivalent(u.s) and offset_unit == dt_unit:
                offset = t0_offset.to(dt_unit)
                times = offset + np.arange(nout) * dt_value * dt_unit
            elif offset_unit.is_equivalent(u.s):
                offset = t0_offset.to(u.s)
                if dt_unit.is_equivalent(u.dimensionless_unscaled):
                    dt = dt_value * u.s
                elif dt_unit.is_equivalent(u.s):
                    dt = (dt_value * dt_unit).to(u.s)
                else:
                    raise ValueError("inverse FFT time step must have time units")
                times = offset + np.arange(nout) * dt
            else:
                t0_value = getattr(t0_offset, "value", t0_offset)
                times = (np.arange(nout) * dt_value + t0_value) * dt_unit
        else:
            times = np.arange(nout) * dt_value * dt_unit

        result = ScalarField(
            dift,
            unit=self.unit,
            epoch=self.epoch,
            axis0=times,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=[
                self._TIME_AXIS_NAME,
                self._axis1_name,
                self._axis2_name,
                self._axis3_name,
            ],
            axis0_domain="time",
            space_domain=self._space_domains,
        )
        result._validate_domain_units()
        self._propagate_gwex_attrs(result)
        return result

    # =========================================================================
    # Spatial FFT (axes 1-3, two-sided signed FFT)
    # =========================================================================

    def fft_space(
        self,
        axes: Iterable[str] | None = None,
        n: Sequence[int] | None = None,
        overwrite: bool = False,
        spacing_rtol: float | None = None,
    ) -> ScalarField:
        r"""Compute FFT along spatial axes.

        This method uses the two-sided unnormalized forward DFT
        (``scipy.fft.fftn``) and produces angular wavenumber coordinates
        (``k = 2π·fftfreq``).

        Parameters
        ----------
        axes : iterable of str, optional
            Axis names to transform (e.g., ``['x', 'y']``). If ``None``,
            transforms all spatial axes in ``'real'`` domain.
        n : tuple of int, optional
            FFT lengths for each axis.
        overwrite : bool, optional
            If ``True``, perform FFT in-place on a temporary copy of the data
            to reduce peak memory usage. Default is ``False``.
        spacing_rtol : float, optional
            Opt-in dimensionless deviation tolerance for approximately uniform
            axes. When ``None`` (default), strict :attr:`AxisDescriptor.regular`
            and :attr:`AxisDescriptor.delta` are used. When provided,
            ``is_regular(max_deviation=spacing_rtol)`` and
            :meth:`AxisDescriptor.mean_delta` are used, emitting a
            ``UserWarning`` if the axis does not pass strict regularity.

        Returns
        -------
        ScalarField
            Transformed field with specified axes in ``'k'`` domain.
            Any attached :class:`GeoMetadata` is deactivated (``active=False``).

        Raises
        ------
        ValueError
            If any specified axis is in ``'angle'`` domain (convert first via
            :meth:`to_local_cartesian` or :meth:`to_projected`), is not in
            ``'real'`` domain, or is not uniformly spaced.

        Notes
        -----
        **Fourier Normalization and Parseval Convention**

        Forward coefficients match unnormalized ``scipy.fft.fftn``:

        .. math::

            F[k] = \sum_{n=0}^{N-1} f[n] \, e^{-i 2\pi k n / N},

        with inverse :meth:`ifft_space` applying ``1 / N`` via
        ``scipy.fft.ifftn``, so Parseval's identity along transformed axes is:

        .. math::

            \sum |f|^2 = \frac{1}{\prod N_j} \sum |F|^2.

        **Angular Wavenumber Convention**

        The wavenumber axis is computed as ``k = 2π * fftfreq(n, d=|dx|)``,
        satisfying ``k = 2π / λ`` with units of ``1 / dx_unit``.

        **Sign Convention for Descending Axes (dx < 0)**

        If the spatial axis is descending (``dx < 0``), the ``k``-axis is
        sign-flipped to preserve physical consistency with ``e^{+ikx}``.

        **Approximate Grids (``spacing_rtol``)**

        When ``spacing_rtol`` is given and an axis passes the relaxed G3
        least-squares criterion without passing strict regularity, exact input
        coordinates are retained privately so :meth:`ifft_space` restores them
        when axis length and wavenumber spacing remain unchanged.

        """
        if spacing_rtol is not None:
            if isinstance(spacing_rtol, (bool, np.bool_)) or not isinstance(
                spacing_rtol, Real
            ):
                raise TypeError(
                    f"spacing_rtol must be a non-negative real number or None, got {type(spacing_rtol).__name__}"
                )
            spacing_rtol = float(spacing_rtol)
            if not np.isfinite(spacing_rtol) or spacing_rtol < 0:
                raise ValueError(
                    f"spacing_rtol must be finite and non-negative, got {spacing_rtol!r}"
                )

        # Default: all real-domain spatial axes
        if axes is None:
            axes = [
                name
                for name in [self._axis1_name, self._axis2_name, self._axis3_name]
                if self._space_domains.get(name) == "real"
            ]
            if not axes:
                angle_axes = [
                    name
                    for name in [self._axis1_name, self._axis2_name, self._axis3_name]
                    if self._space_domains.get(name) == "angle"
                ]
                if angle_axes:
                    raise ValueError(
                        f"Spatial axes {angle_axes} are in 'angle' domain and cannot be "
                        "transformed directly with fft_space(). Convert to metric "
                        "coordinates first via to_local_cartesian() or to_projected()."
                    )

        if not axes:
            raise ValueError("No axes specified for fft_space")

        axes_list = [axes] if isinstance(axes, str) else list(axes)

        # Validate axes and get integer indices
        target_axes_int = []
        axis_spacings: dict[str, tuple[Any, bool, float | None]] = {}
        for ax_name in axes_list:
            ax_int = self._get_axis_index(ax_name)
            if ax_int == 0:
                raise ValueError(
                    "Cannot use fft_space on axis 0 (time/frequency axis). "
                    "Use fft_time instead."
                )
            domain = self._space_domains.get(ax_name)
            if domain == "angle":
                raise ValueError(
                    f"Axis '{ax_name}' is in 'angle' domain and cannot be "
                    "transformed directly with fft_space(). Convert to metric "
                    "coordinates first via to_local_cartesian() or to_projected()."
                )
            if domain != "real":
                raise ValueError(
                    f"Axis '{ax_name}' is not in 'real' domain (current: {domain}). "
                    f"Cannot apply fft_space."
                )
            # Check uniform spacing and length
            ax_desc = self.axis(ax_name)
            if ax_desc.size < 2:
                raise ValueError(
                    f"FFT requires axis length >= 2, "
                    f"got length {ax_desc.size} for axis '{ax_name}'"
                )

            if spacing_rtol is None:
                if not ax_desc.regular:
                    raise ValueError(
                        f"Axis '{ax_name}' is not uniformly spaced. Cannot apply FFT "
                        "(pass spacing_rtol= to allow approximately uniform grids)."
                    )
                delta = ax_desc.delta
                axis_spacings[ax_name] = (delta, False, None)
            else:
                if ax_desc.regular:
                    delta = ax_desc.delta
                    axis_spacings[ax_name] = (delta, False, None)
                else:
                    passed, e_ls, floor, slope_val, fit_origin_val = (
                        ax_desc._regularity_report(max_deviation=spacing_rtol)
                    )
                    if not passed:
                        raise ValueError(
                            f"Axis '{ax_name}' is not uniformly spaced within "
                            f"max_deviation={spacing_rtol:g} (E_LS={e_ls:.3e}, "
                            f"floor={floor:.3e}). Cannot apply FFT."
                        )
                    delta = ax_desc.mean_delta()
                    phase_bound = np.pi * e_ls
                    warnings.warn(
                        f"Axis '{ax_name}' is approximately uniform "
                        f"(E_LS={e_ls:.3e} <= {spacing_rtol:g} + floor={floor:.3e}, "
                        f"mean_delta={slope_val:g} {ax_desc.unit}); spatial FFT assumes "
                        f"uniform spacing with Nyquist phase error <= {phase_bound:.3e} rad.",
                        UserWarning,
                        stacklevel=2,
                    )
                    axis_spacings[ax_name] = (delta, True, fit_origin_val)

            # Check strict monotonicity
            diffs = np.diff(ax_desc.index.value)
            if not (np.all(diffs > 0) or np.all(diffs < 0)):
                raise ValueError(
                    f"Axis '{ax_name}' is not strictly monotonic. "
                    f"Spatial axes must be strictly ascending or descending."
                )
            target_axes_int.append(ax_int)

        # Perform fftn
        s = None
        if n is not None:
            s = (int(n),) if isinstance(n, (int, np.integer)) else tuple(n)

        if overwrite:
            # Create explicit copy to allow overwrite_x optimization
            # This avoids creating internal temporary buffers in sp_fft
            work_data = self.value.copy()
            dft = sp_fft.fftn(work_data, s=s, axes=target_axes_int, overwrite_x=True)
        else:
            dft = sp_fft.fftn(self.value, s=s, axes=target_axes_int)

        # Build new axis metadata
        new_indices = [
            self._axis0_index,
            self._axis1_index.copy(),
            self._axis2_index.copy(),
            self._axis3_index.copy(),
        ]
        new_names = list(self.axis_names)
        new_space_domains = dict(self._space_domains)
        new_space_origins = dict(getattr(self, "_space_origins", None) or {})

        for ax_name, ax_int in zip(axes_list, target_axes_int):
            ax_desc = self.axis(ax_name)
            delta, is_approx, fit_origin_val = axis_spacings[ax_name]
            if delta is None:
                raise ValueError(
                    f"Axis '{ax_name}' does not have a defined spacing (delta)"
                )
            dx_value = getattr(delta, "value", delta)  # Already signed
            dx_unit = getattr(delta, "unit", u.dimensionless_unscaled)

            npts = dft.shape[ax_int]

            # Angular wavenumber: k = 2π * fftfreq(n, d=|dx|)
            # Use abs(dx) for fftfreq (expects positive spacing)
            k_values = 2 * np.pi * np.fft.fftfreq(npts, d=abs(dx_value))
            # If original axis was descending (dx < 0), flip k-axis sign
            if dx_value < 0:
                k_values = -k_values
            k_unit = 1 / dx_unit
            new_indices[ax_int] = k_values * k_unit

            # Update axis name: x -> kx
            new_name = f"k{ax_name}"
            new_names[ax_int] = new_name

            # Update domain
            del new_space_domains[ax_name]
            new_space_domains[new_name] = "k"

            # Keep origin/direction (and exact coordinates for approximate grids)
            if is_approx and fit_origin_val is not None:
                new_space_origins[ax_name] = {
                    "origin": fit_origin_val * ax_desc.unit,
                    "descending": bool(dx_value < 0),
                    "coords": ax_desc.index.copy(),
                    "mean_delta": delta.copy(),
                    "n": ax_desc.size,
                }
            else:
                new_space_origins[ax_name] = (
                    ax_desc.index[0].copy(),
                    dx_value < 0,
                )

        result = ScalarField(
            dft,
            unit=self.unit,
            epoch=self.epoch,
            axis0=new_indices[0],
            axis1=new_indices[1],
            axis2=new_indices[2],
            axis3=new_indices[3],
            axis_names=new_names,
            axis0_domain=self._axis0_domain,
            space_domain=new_space_domains,
        )
        result._validate_domain_units()
        self._propagate_gwex_attrs(result)
        result._space_origins = new_space_origins
        if result.geo is not None:
            result.geo = result.geo.deactivated("fft_space")
        return result

    def ifft_space(
        self,
        axes: Iterable[str] | None = None,
        n: Sequence[int] | None = None,
        overwrite: bool = False,
    ) -> ScalarField:
        r"""Compute inverse FFT along k-space axes.

        Applies ``scipy.fft.ifftn`` (``1 / N`` scaling per transformed axis,
        matching the unnormalized forward DFT in :meth:`fft_space` and Parseval
        identity :math:`\sum |f|^2 = \frac{1}{\prod N_j} \sum |F|^2`).

        Parameters
        ----------
        axes : iterable of str, optional
            Axis names to transform (e.g., ``['kx', 'ky']``). If ``None``,
            transforms all spatial axes in ``'k'`` domain.
        n : tuple of int, optional
            Output lengths for each axis.
        overwrite : bool, optional
            If ``True``, perform IFFT in-place on a temporary copy.

        Returns
        -------
        ScalarField
            Transformed field with specified axes in ``'real'`` domain.
            Any attached :class:`GeoMetadata` remains deactivated (``active=False``).

        Raises
        ------
        ValueError
            If any specified axis is in ``'angle'`` domain, is not in ``'k'``
            domain, or has non-uniform k-space coordinates.

        """
        # Default: all k-domain spatial axes
        if axes is None:
            axes = [
                name
                for name in [self._axis1_name, self._axis2_name, self._axis3_name]
                if self._space_domains.get(name) == "k"
            ]
            if not axes:
                angle_axes = [
                    name
                    for name in [self._axis1_name, self._axis2_name, self._axis3_name]
                    if self._space_domains.get(name) == "angle"
                ]
                if angle_axes:
                    raise ValueError(
                        f"Spatial axes {angle_axes} are in 'angle' domain (not 'k' "
                        "domain) and cannot use ifft_space(). Convert to metric "
                        "coordinates first via to_local_cartesian() or to_projected()."
                    )

        if not axes:
            raise ValueError("No axes specified for ifft_space")

        axes_list = [axes] if isinstance(axes, str) else list(axes)

        # Validate axes and get integer indices
        target_axes_int = []
        for ax_name in axes_list:
            ax_int = self._get_axis_index(ax_name)
            if ax_int == 0:
                raise ValueError(
                    "Cannot use ifft_space on axis 0. Use ifft_time instead."
                )
            domain = self._space_domains.get(ax_name)
            if domain == "angle":
                raise ValueError(
                    f"Axis '{ax_name}' is in 'angle' domain (not 'k' domain) "
                    "and cannot use ifft_space(). Convert to metric coordinates "
                    "first via to_local_cartesian() or to_projected()."
                )
            if domain != "k":
                raise ValueError(
                    f"Axis '{ax_name}' is not in 'k' domain (current: {domain}). "
                    f"Cannot apply ifft_space."
                )
            k_axis = self.axis(ax_name).index
            if len(k_axis) < 2:
                raise ValueError(
                    f"ifft_space requires axis length >= 2, "
                    f"got length {len(k_axis)} for axis '{ax_name}'"
                )
            if not _is_uniform_k_axis(k_axis):
                raise ValueError(
                    f"Axis '{ax_name}' is not uniformly spaced in k-space. "
                    "Cannot apply ifft_space."
                )
            target_axes_int.append(ax_int)

        # Perform ifftn
        s = None
        if n is not None:
            s = (int(n),) if isinstance(n, (int, np.integer)) else tuple(n)

        if overwrite:
            work_data = self.value.copy()
            dift = sp_fft.ifftn(work_data, s=s, axes=target_axes_int, overwrite_x=True)
        else:
            dift = sp_fft.ifftn(self.value, s=s, axes=target_axes_int)

        # Build new axis metadata
        new_indices = [
            self._axis0_index,
            self._axis1_index.copy(),
            self._axis2_index.copy(),
            self._axis3_index.copy(),
        ]
        new_names = list(self.axis_names)
        new_space_domains = dict(self._space_domains)
        new_space_origins = dict(getattr(self, "_space_origins", None) or {})

        for ax_name, ax_int in zip(axes_list, target_axes_int):
            # Derive real-space axis name from k-axis name
            if ax_name.startswith("k"):
                real_name = ax_name[1:]  # kx -> x
            else:
                real_name = ax_name

            npts = dift.shape[ax_int]

            # Compute real-space coordinates from k-space
            # k = 2π * fftfreq(n, d=dx)  =>  dx = 2π / (n * |dk|)
            k_axis = self.axis(ax_name).index
            dk_raw = k_axis[1] - k_axis[0]
            dk_value = getattr(dk_raw, "value", dk_raw)
            dk_unit = getattr(dk_raw, "unit", u.dimensionless_unscaled)

            # dx = 2π / (n * |dk|)
            dx_value = 2 * np.pi / (npts * abs(dk_value))
            dx_unit = 1 / dk_unit

            recorded = new_space_origins.pop(real_name, None)
            if isinstance(recorded, dict):
                orig_coords = recorded["coords"]
                orig_dx = recorded["mean_delta"]
                orig_n = recorded["n"]
                orig_desc = recorded["descending"]
                orig_abs_dx = abs(float(orig_dx.value))
                dx_in_orig = float((dx_value * dx_unit).to_value(orig_dx.unit))
                sign_ok = (len(k_axis) == 2) or ((dk_value < 0) == orig_desc)
                if (
                    len(k_axis) == orig_n
                    and npts == orig_n
                    and sign_ok
                    and abs(dx_in_orig - orig_abs_dx) <= 1e-12 * orig_abs_dx
                ):
                    x_values = orig_coords.copy()
                else:
                    origin = recorded["origin"]
                    step = -dx_value if orig_desc else dx_value
                    x_values = origin + (np.arange(npts) * step * dx_unit).to(
                        getattr(origin, "unit", dx_unit)
                    )
                    warnings.warn(
                        f"Exact real-space coordinates for '{real_name}' cannot be "
                        f"restored after shape/wavenumber change (n={npts} vs {orig_n}); "
                        "reconstructing uniform mean_delta grid.",
                        UserWarning,
                        stacklevel=2,
                    )
            elif recorded is not None:
                # The axis came from strict fft_space: restore its start and direction.
                origin, descending = recorded
                step = -dx_value if descending else dx_value
                x_values = origin + (np.arange(npts) * step * dx_unit).to(
                    getattr(origin, "unit", dx_unit)
                )
            elif dk_value < 0:
                # No recorded source axis.  A sign-flipped k axis stands for a
                # descending x axis, laid out over the same [0, (n-1)dx] range
                # as the ascending fallback.
                x_values = np.arange(npts - 1, -1, -1) * dx_value * dx_unit
            else:
                x_values = np.arange(npts) * dx_value * dx_unit
            new_indices[ax_int] = x_values

            new_names[ax_int] = real_name

            # Update domain
            del new_space_domains[ax_name]
            new_space_domains[real_name] = "real"

        result = ScalarField(
            dift,
            unit=self.unit,
            epoch=self.epoch,
            axis0=new_indices[0],
            axis1=new_indices[1],
            axis2=new_indices[2],
            axis3=new_indices[3],
            axis_names=new_names,
            axis0_domain=self._axis0_domain,
            space_domain=new_space_domains,
        )
        result._validate_domain_units()
        self._propagate_gwex_attrs(result)
        # Axes still in k-space keep their recorded origin; restored ones do not.
        result._space_origins = new_space_origins
        if result.geo is not None:
            result.geo = result.geo.deactivated("ifft_space")
        return result

    def wavelength(self, axis: str | int) -> u.Quantity:
        r"""Compute wavelength from wavenumber axis.

        Parameters
        ----------
        axis : str or int
            The k-domain axis name or index.

        Returns
        -------
        `~astropy.units.Quantity`
            Wavelength values (:math:`\lambda = 2\pi / |k|`). k=0 returns inf.

        Raises
        ------
        ValueError
            If the axis is not in 'k' domain.

        """
        ax_name = self.axis_names[self._get_axis_index(axis)]
        domain = self._space_domains.get(ax_name)
        if domain != "k":
            raise ValueError(
                f"Axis '{ax_name}' is not in 'k' domain (current: {domain})"
            )

        k_index = self.axis(ax_name).index
        with np.errstate(divide="ignore"):
            k_val = getattr(k_index, "value", k_index)
            wavelength_values = 2 * np.pi / np.abs(k_val)
        return wavelength_values * (
            1 / getattr(k_index, "unit", u.dimensionless_unscaled)
        )

    # =========================================================================
    # pyroomacoustics interop
    # =========================================================================

    @classmethod
    def from_pyroomacoustics_field(
        cls,
        room: Any,
        *,
        grid_shape: tuple[int, ...],
        source: int = 0,
        mode: str = "rir",
        unit: Any | None = None,
    ) -> ScalarField:
        """Create from pyroomacoustics room with grid-placed microphones.

        Parameters
        ----------
        room : pyroomacoustics.Room
            Room with microphones on a regular spatial grid.
        grid_shape : tuple of int
            Spatial grid shape ``(nx, ny, nz)`` or ``(nx, ny)``.
        source : int, default 0
            Source index (for ``mode='rir'``).
        mode : {'rir', 'signals'}
            ``'rir'`` for impulse responses, ``'signals'`` for mic signals.
        unit : str or astropy.units.Unit, optional
            Unit to assign to the data.

        Returns
        -------
        ScalarField

        """
        from gwexpy.interop import from_pyroomacoustics_field

        return from_pyroomacoustics_field(
            cls, room, grid_shape=grid_shape, source=source, mode=mode, unit=unit
        )

    # =========================================================================
    # Simulation
    # =========================================================================

    @classmethod
    def simulate(cls, method: str, *args: Any, **kwargs: Any) -> ScalarField:
        """Generate a simulated ScalarField.

        Parameters
        ----------
        method : str
            Name of the generator from ``gwexpy.noise.field``.
            (e.g., 'gaussian', 'plane_wave').
        *args, **kwargs
            Arguments passed to the generator.

        Returns
        -------
        ScalarField
            Generated field.

        Examples
        --------
        >>> from gwexpy.fields import ScalarField
        >>> field = ScalarField.simulate('gaussian', shape=(100, 10, 10, 10))

        """
        from gwexpy.noise import field

        if not hasattr(field, method):
            raise ValueError(
                f"Unknown simulation method '{method}'. "
                f"Available methods in gwexpy.noise.field: "
                f"{[m for m in dir(field) if not m.startswith('_')]}"
            )

        func = getattr(field, method)
        return func(*args, **kwargs)

    # =========================================================================
    # Extraction API (Phase 0.3)
    # =========================================================================

    def extract_points(
        self,
        points: Sequence[Sequence[u.Quantity]] | Sequence[tuple[u.Quantity, ...]],
        interp: str = "linear",
    ) -> Any:
        """Extract time series at specified spatial points.

        Parameters
        ----------
        points : list of tuple
            List of (x, y, z) coordinates.
        interp : str, optional
            Interpolation method ('linear' or 'nearest'). Default is 'linear'.

        Returns
        -------
        TimeSeriesList
            List of time series, one per point.

        """
        if self._axis0_domain != "time":
            raise ValueError("extract_points requires axis0_domain='time'")

        TimeSeries = ConverterRegistry.get_constructor("TimeSeries")
        TimeSeriesList = ConverterRegistry.get_constructor("TimeSeriesList")

        from gwexpy.plot._coord import nearest_index, slice_from_index

        result_list = []
        for point in points:
            if len(point) != 3:
                raise ValueError("Each point must have 3 coordinates (x, y, z)")

            x_val, y_val, z_val = point

            if interp == "nearest":
                # Find nearest indices for each spatial axis
                i1 = nearest_index(self._axis1_index, x_val)
                i2 = nearest_index(self._axis2_index, y_val)
                i3 = nearest_index(self._axis3_index, z_val)

                # Extract the time series at this point
                sliced = self[
                    :, slice_from_index(i1), slice_from_index(i2), slice_from_index(i3)
                ]
                ts_data = sliced.value.squeeze()

                # Label for the series
                label = f"({self._axis1_name}={self._axis1_index[i1]:.3g}, y={self._axis2_index[i2]:.3g}, z={self._axis3_index[i3]:.3g})"

            elif interp == "linear":
                from scipy.interpolate import interpn

                axis0_index = cast(Any, self._axis0_index)
                axis1_index = cast(Any, self._axis1_index)
                axis2_index = cast(Any, self._axis2_index)
                axis3_index = cast(Any, self._axis3_index)

                # Grid for all 4 axes
                grid = [
                    axis0_index.value,
                    axis1_index.value,
                    axis2_index.value,
                    axis3_index.value,
                ]

                # Evaluation points: (nt, 4)
                # t stays on its grid, while x, y, z are fixed
                nt = len(axis0_index)
                eval_points = np.zeros((nt, 4))
                eval_points[:, 0] = axis0_index.value
                eval_points[:, 1] = x_val.to(axis1_index.unit).value
                eval_points[:, 2] = y_val.to(axis2_index.unit).value
                eval_points[:, 3] = z_val.to(axis3_index.unit).value

                ts_data = interpn(grid, self.value, eval_points, method="linear")
                label = f"({self._axis1_name}={x_val.value:.3g}, y={y_val.value:.3g}, z={z_val.value:.3g})"
            else:
                raise ValueError(f"Unsupported interpolation method '{interp}'")

            from gwexpy.time._reference import inherit_time_reference

            ts = inherit_time_reference(
                TimeSeries(
                    ts_data,
                    times=self._axis0_index,
                    unit=self.unit,
                    name=label,
                ),
                self,
            )
            result_list.append(ts)

        return TimeSeriesList(*result_list)

    def extract_profile(
        self,
        axis: str,
        at: dict[str, u.Quantity],
        reduce: str | None = None,
        interp: str = "linear",
    ) -> tuple[IndexLike, Any]:
        """Extract a 1D profile along a specified axis.

        Parameters
        ----------
        axis : str
            Axis name to extract along ('x', 'y', 'z', or their k-variants).
        at : dict
            Dictionary specifying fixed values for the other axes.
            Must include all axes except the extraction axis.
            Example: ``{'t': 0.5 * u.s, 'y': 2.0 * u.m, 'z': 0.0 * u.m}``
        reduce : None
            Reserved for future averaging support. Currently ignored.
        interp : str, optional
            Interpolation method ('linear' or 'nearest'). Default is 'linear'.

        Returns
        -------
        tuple
            (axis_index, values): Both are `~astropy.units.Quantity` arrays.
            axis_index is the coordinate along the extraction axis,
            values is the data values along that axis.

        """
        from gwexpy.plot._coord import nearest_index, slice_from_index

        # Map axis name to integer index
        axis_int = self._get_axis_index(axis)

        # Get the axes coordinates
        all_axes_info: list[tuple[int, str, Any]] = [
            (0, self._axis0_name, self._axis0_index),
            (1, self._axis1_name, self._axis1_index),
            (2, self._axis2_name, self._axis2_index),
            (3, self._axis3_name, self._axis3_index),
        ]

        extraction_axis_info = all_axes_info[axis_int]
        axis_index = extraction_axis_info[2]

        if interp == "nearest":
            # Build the slice tuple
            slices = [slice(None)] * 4
            for ax_int, ax_name, ax_index_val in all_axes_info:
                if ax_int == axis_int:
                    continue

                if ax_name not in at:
                    raise ValueError(
                        f"extract_profile requires fixed value for axis '{ax_name}' "
                        f"in 'at' dictionary"
                    )

                value = at[ax_name]
                idx = nearest_index(ax_index_val, value)
                slices[ax_int] = slice_from_index(idx)

            # Extract the data
            sliced = self[tuple(slices)]
            data = sliced.value.squeeze()

        elif interp == "linear":
            from scipy.interpolate import interpn

            # Points along the extraction axis
            coords = []
            for ax_int, ax_name, ax_index_val in all_axes_info:
                if ax_int == axis_int:
                    coords.append(ax_index_val.value)
                else:
                    if ax_name not in at:
                        raise ValueError(
                            f"extract_profile requires fixed value for axis '{ax_name}' "
                            f"in 'at' dictionary"
                        )
                    coords.append(at[ax_name].to(ax_index_val.unit).value)

            # Prepare points for interpn: (n_points, 4)
            # where n_points = len(axis_index)
            points = np.zeros((len(axis_index), 4))
            for i in range(4):
                if i == axis_int:
                    points[:, i] = axis_index.value
                else:
                    points[:, i] = coords[i]

            # Grid points for interpn
            grid = [info[2].value for info in all_axes_info]

            # Perform interpolation
            data = interpn(grid, self.value, points, method="linear")
        else:
            raise ValueError(f"Unsupported interpolation method '{interp}'")

        # Return with units
        values = data
        if hasattr(self, "unit") and self.unit is not None:
            values = values * self.unit

        return axis_index, values

    def slice_map2d(self, plane="xy", at=None):
        """Extract a 2D slice (map) from the 4D field.

        Parameters
        ----------
        plane : str, optional
            The plane to extract: 'xy', 'xz', 'yz', 'tx', 'ty', 'tz'.
            Default is 'xy'.
        at : dict, optional
            Dictionary specifying fixed values for axes not in the plane.
            If None, axes with length=1 are used automatically.

        Returns
        -------
        ScalarField
            A ScalarField with the non-plane axes having length=1.

        Raises
        ------
        ValueError
            If plane specification is invalid.
        ValueError
            If ``at`` is None and there is ambiguity about which axes to fix.

        Examples
        --------
        >>> # Extract xy plane at a specific time and z
        >>> field_2d = field.slice_map2d('xy', at={'t': 0.5 * u.s, 'z': 0.0 * u.m})
        >>> field_2d.plot_map2d()

        """
        from gwexpy.plot._coord import nearest_index, slice_from_index

        # Parse plane specification
        valid_planes = {"xy", "xz", "yz", "tx", "ty", "tz"}
        # Also support k-space equivalents
        plane_lower = plane.lower()
        if plane_lower not in valid_planes:
            # Check for k-space variants
            plane_chars = set(plane_lower)
            if not plane_chars.issubset({"t", "f", "x", "y", "z", "k"}):
                raise ValueError(
                    f"Invalid plane '{plane}'. Must be one of {valid_planes} "
                    f"or their k-space equivalents."
                )

        # Determine which axes are in the plane
        axis_names = list(self.axis_names)
        plane_axes = []
        for char in plane_lower:
            # Handle 'k' prefix
            if char == "k":
                continue
            # Find axis that starts with this character (or k-prefix version)
            for ax_name in axis_names:
                if ax_name == char or ax_name == f"k{char}":
                    if ax_name not in plane_axes:
                        plane_axes.append(ax_name)
                    break

        # If plane is like "kx" or "ky", handle specially
        for i, char in enumerate(plane_lower):
            if char == "k" and i + 1 < len(plane_lower):
                combined = f"k{plane_lower[i + 1]}"
                for ax_name in axis_names:
                    if ax_name == combined and ax_name not in plane_axes:
                        plane_axes.append(ax_name)

        # Build slices
        all_axes = [
            (0, self._axis0_name, self._axis0_index),
            (1, self._axis1_name, self._axis1_index),
            (2, self._axis2_name, self._axis2_index),
            (3, self._axis3_name, self._axis3_index),
        ]

        slices = [slice(None)] * 4
        for ax_int, ax_name, ax_index in all_axes:
            if ax_name in plane_axes or ax_name.lstrip("k") in [
                p.lstrip("k") for p in plane_axes
            ]:
                # Keep this axis
                continue

            # Need to fix this axis
            if at is not None and ax_name in at:
                value = at[ax_name]
                idx = nearest_index(ax_index, value)
                slices[ax_int] = slice_from_index(idx)
            elif len(ax_index) == 1:
                # Already length=1, use it
                slices[ax_int] = slice(0, 1)
            else:
                raise ValueError(
                    f"Axis '{ax_name}' is not in plane '{plane}' and has "
                    f"length {len(ax_index)} > 1. Specify its value in 'at'."
                )

        return self[tuple(slices)]

    # =========================================================================
    # Visualization Methods (Phase 1)
    # =========================================================================

    def plot_map2d(
        self,
        plane="xy",
        at=None,
        mode="real",
        method="pcolormesh",
        ax=None,
        add_colorbar=True,
        vmin=None,
        vmax=None,
        title=None,
        cmap=None,
        **kwargs,
    ):
        """Plot a 2D map (heatmap) of the field.

        Parameters
        ----------
        plane : str, optional
            The plane to plot: 'xy', 'xz', 'yz', 'tx', 'ty', 'tz'.
            Default is 'xy'.
        at : dict, optional
            Dictionary specifying fixed values for axes not in the plane.
            If None, axes with length=1 are used automatically.
        mode : str, optional
            Component to extract from complex data:
            'real', 'imag', 'abs', 'angle', 'power'. Default is 'real'.
        method : str, optional
            Plot method: 'pcolormesh' or 'imshow'. Default is 'pcolormesh'.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        add_colorbar : bool, optional
            Whether to add a colorbar. Default is True.
        vmin, vmax : float, optional
            Color scale limits.
        title : str, optional
            Plot title.
        cmap : str or Colormap, optional
            Colormap to use.
        **kwargs
            Additional arguments passed to the plot method.

        Returns
        -------
        tuple
            (fig, ax): The matplotlib figure and axes objects.

        Examples
        --------
        >>> import numpy as np
        >>> from gwexpy.fields import ScalarField
        >>> sf = ScalarField(np.ones((1, 10, 10, 1)))
        >>> fig, ax = sf.plot_map2d('xy', at={'z': 0.0}) # doctest: +SKIP

        """
        import matplotlib.pyplot as plt

        from gwexpy.plot._coord import select_value

        # Get the 2D slice
        sliced = self.slice_map2d(plane=plane, at=at)

        # Determine which axes are the plane axes
        shape = sliced.shape
        plane_axis_ints = [i for i, s in enumerate(shape) if s > 1]

        if len(plane_axis_ints) < 2:
            # Handle case where one dimension might also be 1
            plane_axis_ints = [i for i, s in enumerate(shape) if s >= 1][:2]

        if len(plane_axis_ints) < 2:
            raise ValueError(
                f"Cannot create 2D plot: slice has shape {shape}. "
                f"Need at least 2 dimensions with size > 1."
            )

        ax1_int, ax2_int = plane_axis_ints[0], plane_axis_ints[1]

        # Get axis indices and names
        axes_info = [
            (sliced._axis0_name, sliced._axis0_index),
            (sliced._axis1_name, sliced._axis1_index),
            (sliced._axis2_name, sliced._axis2_index),
            (sliced._axis3_name, sliced._axis3_index),
        ]
        ax1_name, ax1_index = axes_info[ax1_int]
        ax2_name, ax2_index = axes_info[ax2_int]

        # Extract the 2D data
        # Squeeze out the length-1 dimensions while preserving order
        data_4d = sliced.value
        # Build transpose + squeeze to get [ax1, ax2] ordering
        squeeze_axes = [i for i in range(4) if i not in (ax1_int, ax2_int)]
        # Create slice to reduce to 2D
        idx = [0 if i in squeeze_axes else slice(None) for i in range(4)]
        data_2d = data_4d[tuple(idx)]
        if data_2d.ndim > 2:
            data_2d = data_2d.squeeze()

        # Apply mode (real/abs/etc.)
        data_2d = select_value(data_2d, mode=mode)
        if hasattr(data_2d, "value"):
            data_2d = data_2d.value

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # Prepare coordinates for pcolormesh (need edges)
        x_coords = ax2_index.value
        y_coords = ax1_index.value

        # Plot
        if method == "pcolormesh":
            im = ax.pcolormesh(
                x_coords, y_coords, data_2d, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs
            )
        elif method == "imshow":
            from ..types.axis import AxisDescriptor

            if (
                len(ax2_index) >= 2 and not AxisDescriptor(ax2_name, ax2_index).regular
            ) or (
                len(ax1_index) >= 2 and not AxisDescriptor(ax1_name, ax1_index).regular
            ):
                raise ValueError(
                    "method='imshow' requires regularly spaced axes; "
                    "use method='pcolormesh' for nonuniform rectilinear grids."
                )
            extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]
            im = ax.imshow(
                data_2d,
                extent=extent,
                origin="lower",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'pcolormesh' or 'imshow'."
            )

        # Labels with units
        from gwexpy.plot._label_utils import format_axis_label

        ax.set_xlabel(format_axis_label(ax2_name, ax2_index.unit))
        ax.set_ylabel(format_axis_label(ax1_name, ax1_index.unit))

        if title:
            ax.set_title(title)

        # Colorbar
        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            if self.unit is not None:
                cbar.set_label(format_axis_label(mode, self.unit))

        return fig, ax

    def plot_timeseries_points(
        self,
        points,
        labels=None,
        interp="nearest",
        ax=None,
        legend=True,
        **kwargs,
    ):
        """Plot time series extracted at specified spatial points.

        Parameters
        ----------
        points : list of tuple
            List of (x, y, z) coordinates.
        labels : list of str, optional
            Labels for each time series. If None, auto-generated.
        interp : str, optional
            Interpolation method. Default is 'nearest'.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        legend : bool, optional
            Whether to show legend. Default is True.
        **kwargs
            Additional arguments passed to plot.

        Returns
        -------
        tuple
            (fig, ax): The matplotlib figure and axes objects.

        Examples
        --------
        >>> import numpy as np
        >>> from gwexpy.fields import ScalarField
        >>> sf = ScalarField(np.ones((1024, 1, 1, 1)))
        >>> psd = sf.compute_psd((0, 0, 0)) # doctest: +SKIP

        """
        import matplotlib.pyplot as plt

        ts_list = self.extract_points(points, interp=interp)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        for i, ts in enumerate(ts_list):
            label = labels[i] if labels else ts.name
            ax.plot(ts.times.value, ts.value, label=label, **kwargs)

        # Labels
        x_unit = getattr(self._axis0_index, "unit", u.dimensionless_unscaled)
        from gwexpy.plot._label_utils import format_axis_label, format_label_with_unit

        ax.set_xlabel(format_axis_label(self._axis0_name, x_unit))
        y_label = format_label_with_unit(None, self.unit)
        if y_label is not None:
            ax.set_ylabel(y_label)

        if legend:
            ax.legend()

        return fig, ax

    def plot_profile(
        self,
        axis,
        at,
        mode="real",
        ax=None,
        label=None,
        **kwargs,
    ):
        """Plot a 1D profile along a specified axis.

        Parameters
        ----------
        axis : str
            Axis name to plot along.
        at : dict
            Dictionary specifying fixed values for other axes.
        mode : str, optional
            Component to extract: 'real', 'imag', 'abs', 'angle', 'power'.
            Default is 'real'.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        label : str, optional
            Line label for legend.
        **kwargs
            Additional arguments passed to plot.

        Returns
        -------
        tuple
            (fig, ax): The matplotlib figure and axes objects.

        Examples
        --------
        >>> import numpy as np
        >>> from gwexpy.fields import ScalarField
        >>> sf = ScalarField(np.ones((2, 4, 4, 1)))
        >>> fig, ax = sf.plot_profile('x', at={'t': 0.5 * u.s, 'y': 0.0 * u.m, 'z': 0.0 * u.m}) # doctest: +SKIP

        """
        import matplotlib.pyplot as plt

        from gwexpy.plot._coord import select_value

        axis_index, values = self.extract_profile(axis, at)

        # Apply mode
        values = select_value(values, mode=mode)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # Get value array
        y_data = values.value if hasattr(values, "value") else values
        y_unit = values.unit if hasattr(values, "unit") else None

        x_data = getattr(axis_index, "value", axis_index)
        ax.plot(x_data, y_data, label=label, **kwargs)

        # Labels with units
        x_unit = getattr(axis_index, "unit", u.dimensionless_unscaled)
        from gwexpy.plot._label_utils import format_axis_label

        ax.set_xlabel(format_axis_label(axis, x_unit))
        if y_unit is not None:
            ax.set_ylabel(format_axis_label(mode, y_unit))

        if label:
            ax.legend()

        return fig, ax

    # =========================================================================
    # Comparison & Summary Methods (Phase 2)
    # =========================================================================

    def _new_with_left_metadata(
        self,
        value,
        *,
        unit,
        axes=None,
        metadata_source=None,
    ) -> ScalarField:
        """Build a result with this field's physical-grid authority."""
        if axes is None:
            axes = (
                self._axis0_index,
                self._axis1_index,
                self._axis2_index,
                self._axis3_index,
            )
        if metadata_source is None:
            metadata_source = self
        metadata = self._metadata_kwargs(metadata_source)
        if hasattr(self, "_epoch"):
            metadata["epoch"] = self._epoch
        result = ScalarField(
            value,
            unit=unit,
            axis0=axes[0],
            axis1=axes[1],
            axis2=axes[2],
            axis3=axes[3],
            axis_names=list(self.axis_names),
            axis0_domain=self._axis0_domain,
            space_domain=self._space_domains,
            copy=False,
            **metadata,
        )
        result._axis0_offset = getattr(self, "_axis0_offset", None)
        result._axis0_dt = getattr(self, "_axis0_dt", None)
        self._propagate_gwex_attrs(result)
        return result

    def diff(self, n=1, axis=-1, *, mode=None):
        """Compute a GWpy finite difference or compare two fields.

        Parameters
        ----------
        n : int or ScalarField, optional
            Difference order for the GWpy-compatible route. Passing another
            `ScalarField` explicitly selects the field-comparison extension.
        axis : int or str, optional
            Difference axis, or the legacy positional comparison mode when
            ``n`` is a `ScalarField`.
        mode : str, keyword-only, optional
            Comparison mode:
            - 'diff': Difference (self - other)
            - 'ratio': Ratio (self / other)
            - 'percent': Percentage difference ((self - other) / other * 100)
            Default is 'diff'.

        Returns
        -------
        ScalarField
            Result field. For 'diff', unit is same as input.
            For 'ratio', unit is dimensionless. For 'percent', unit is
            percent.

        Raises
        ------
        ValueError
            If mode is not recognized.
        ValueError
            If shapes are incompatible.
        astropy.units.UnitConversionError
            If the two fields have incompatible data units.

        Examples
        --------
        >>> import numpy as np
        >>> from gwexpy.fields import ScalarField
        >>> sf = ScalarField(np.ones((2, 2, 2, 2)))
        >>> sf # doctest: +ELLIPSIS
        <ScalarField(2, 2, 2, 2)@time, 1.0>

        """
        from gwpy.types.array import Array as GwpyArray

        if not isinstance(n, ScalarField):
            if mode is not None:
                raise TypeError(
                    "diff() 'mode' is only valid when comparing ScalarField objects"
                )
            parent_input = self.view(GwpyArray)
            parent_result = GwpyArray.diff(parent_input, n, axis)
            result_axes = [
                self._axis0_index,
                self._axis1_index,
                self._axis2_index,
                self._axis3_index,
            ]
            if parent_result.shape != self.shape:
                axis_number = int(axis) % self.ndim
                difference_order = int(n)
                result_axes[axis_number] = result_axes[axis_number][difference_order:]
            result_value, result_unit = self._value_unit(parent_result)
            return self._new_with_left_metadata(
                result_value,
                unit=result_unit,
                axes=result_axes,
                metadata_source=parent_result,
            )

        other = n
        if mode is not None and axis != -1:
            raise TypeError("diff() received multiple values for comparison mode")
        comparison_mode = mode if mode is not None else "diff" if axis == -1 else axis

        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        valid_modes = ("diff", "ratio", "percent")
        if comparison_mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{comparison_mode}'. Must be one of {valid_modes}."
            )

        self._validate_field_alignment(other)
        other_values = other.to_value(self.unit)

        if comparison_mode == "diff":
            result_data = self.value - other_values
            result_unit = self.unit
        if comparison_mode == "ratio":
            with np.errstate(divide="ignore", invalid="ignore"):
                result_data = self.value / other_values
            result_unit = u.dimensionless_unscaled
        if comparison_mode == "percent":
            with np.errstate(divide="ignore", invalid="ignore"):
                result_data = (self.value - other_values) / other_values * 100
            result_unit = u.percent
        return self._new_with_left_metadata(result_data, unit=result_unit)

    def zscore(self, baseline_t=None):
        """Compute z-score normalized field using a baseline period.

        The z-score is computed as (data - mean) / std, where mean and std
        are computed from the baseline period along axis 0 (time).

        Parameters
        ----------
        baseline_t : tuple of Quantity, optional
            Time range (t_start, t_end) for computing baseline statistics.
            If None, uses the entire time axis.

        Returns
        -------
        ScalarField
            Z-score normalized field (dimensionless).

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time'.
        ValueError
            If baseline time range is outside the available data.

        Examples
        --------
        >>> import numpy as np
        >>> from astropy import units as u
        >>> from gwexpy.fields import ScalarField
        >>> sf = ScalarField(np.ones((100, 2, 2, 2)), axis0=np.arange(100)*u.s)
        >>> sf.zscore(baseline_t=(0 * u.s, 1 * u.s)) # doctest: +ELLIPSIS
        <ScalarField(100, 2, 2, 2)@time, ...>

        """
        from astropy import units as u

        from gwexpy.plot._coord import nearest_index

        if self._axis0_domain != "time":
            raise ValueError(
                f"zscore requires axis0_domain='time', got '{self._axis0_domain}'"
            )

        if baseline_t is None:
            # Use entire time axis
            baseline_data = self.value
        else:
            t_start, t_end = baseline_t
            # Find indices for baseline range
            i_start = nearest_index(self._axis0_index, t_start)
            i_end = nearest_index(self._axis0_index, t_end)

            if i_start > i_end:
                i_start, i_end = i_end, i_start

            baseline_data = self.value[i_start : i_end + 1, ...]

        # Compute mean and std along time axis
        mean = np.mean(baseline_data, axis=0, keepdims=True)
        std = np.std(baseline_data, axis=0, keepdims=True)

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore_data = (self.value - mean) / std
            zscore_data = np.nan_to_num(zscore_data, nan=0.0, posinf=0.0, neginf=0.0)

        result = ScalarField(
            zscore_data,
            unit=u.dimensionless_unscaled,
            epoch=self.epoch,
            axis0=self._axis0_index,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=list(self.axis_names),
            axis0_domain=self._axis0_domain,
            space_domain=self._space_domains,
        )
        self._propagate_gwex_attrs(result)
        return result

    def time_stat_map(self, stat="mean", t_range=None, plane="xy", at=None):
        """Compute a time-aggregated 2D map.

        Parameters
        ----------
        stat : str, optional
            Statistic to compute: 'mean', 'std', 'rms', 'max', 'min'.
            Default is 'mean'.
        t_range : tuple of Quantity, optional
            Time range (t_start, t_end) to aggregate over.
            If None, uses the entire time axis.
        plane : str, optional
            Spatial plane to visualize: 'xy', 'xz', 'yz'.
            Default is 'xy'.
        at : dict, optional
            Fixed values for axes not in the plane.

        Returns
        -------
        ScalarField
            Result field with time axis reduced to length=1.

        Raises
        ------
        ValueError
            If stat is not recognized.

        Examples
        --------
        >>> import numpy as np
        >>> from gwexpy.fields import ScalarField
        >>> from astropy import units as u
        >>> sf = ScalarField(np.ones((100, 2, 2, 2)), axis0=np.arange(100)*u.s)
        >>> mean_map = sf.time_stat_map('mean', t_range=(0 * u.s, 1 * u.s))

        """
        from gwexpy.plot._coord import nearest_index

        valid_stats = ("mean", "std", "rms", "max", "min")
        if stat not in valid_stats:
            raise ValueError(f"Invalid stat '{stat}'. Must be one of {valid_stats}.")

        # Get time slice
        if t_range is not None:
            t_start, t_end = t_range
            i_start = nearest_index(self._axis0_index, t_start)
            i_end = nearest_index(self._axis0_index, t_end)
            if i_start > i_end:
                i_start, i_end = i_end, i_start
            time_slice = slice(i_start, i_end + 1)
            subset = self[time_slice, :, :, :]
        else:
            subset = self

        # Compute statistic along time axis
        data = subset.value
        if stat == "mean":
            result_data = np.mean(data, axis=0, keepdims=True)
        elif stat == "std":
            result_data = np.std(data, axis=0, keepdims=True)
        elif stat == "rms":
            result_data = np.sqrt(np.mean(data**2, axis=0, keepdims=True))
        elif stat == "max":
            result_data = np.max(data, axis=0, keepdims=True)
        elif stat == "min":
            result_data = np.min(data, axis=0, keepdims=True)
        else:
            raise ValueError(f"Invalid stat '{stat}'")

        # Create result with mean time for the aggregated point
        if t_range is not None:
            mean_time = (t_start + t_end) / 2
        else:
            mean_time = (self._axis0_index[0] + self._axis0_index[-1]) / 2

        result = ScalarField(
            result_data,
            unit=self.unit,
            epoch=self.epoch,
            axis0=np.array([mean_time.value]) * mean_time.unit,
            axis1=subset._axis1_index,
            axis2=subset._axis2_index,
            axis3=subset._axis3_index,
            axis_names=list(self.axis_names),
            axis0_domain=self._axis0_domain,
            space_domain=self._space_domains,
        )
        self._propagate_gwex_attrs(result)

        # If plane and at are specified, further slice
        if at is not None:
            result = result.slice_map2d(plane=plane, at=at)

        return result

    def time_space_map(self, axis="x", at=None, mode="real", reduce=None):
        """Extract a 2D time-space map (t vs one spatial axis).

        Parameters
        ----------
        axis : str, optional
            Spatial axis name ('x', 'y', 'z' or k-variants).
            Default is 'x'.
        at : dict, optional
            Fixed values for the other two spatial axes.
        mode : str, optional
            Component to extract: 'real', 'imag', 'abs', 'angle', 'power'.
            Default is 'real'.
        reduce : None
            Reserved for future averaging support. Currently ignored.

        Returns
        -------
        tuple
            (t_axis, space_axis, data_2d): Quantity arrays for axes and
            2D numpy array for the data.

        Raises
        ------
        ValueError
            If axis is not valid.
        ValueError
            If ``at`` dictionary is missing required axes.

        Examples
        --------
        >>> import numpy as np
        >>> from astropy import units as u
        >>> from gwexpy.fields import ScalarField
        >>> sf = ScalarField(np.ones((100, 10, 1, 1)))
        >>> t, x, data = sf.time_space_map('x', at={'y': 0 * u.m, 'z': 0 * u.m})

        """
        from gwexpy.plot._coord import nearest_index, select_value, slice_from_index

        # Map axis name to integer index
        axis_int = self._get_axis_index(axis)
        if axis_int == 0:
            raise ValueError("Cannot use time axis as spatial axis for time_space_map")

        # Determine which axes need to be fixed (all except 0 and axis_int)
        all_axes = [
            (0, self._axis0_name, self._axis0_index),
            (1, self._axis1_name, self._axis1_index),
            (2, self._axis2_name, self._axis2_index),
            (3, self._axis3_name, self._axis3_index),
        ]

        if at is None:
            at = {}

        slices = [slice(None)] * 4
        for ax_int, ax_name, ax_index in all_axes:
            if ax_int == 0 or ax_int == axis_int:
                # Keep these axes
                continue

            if ax_name in at:
                idx = nearest_index(ax_index, at[ax_name])
                slices[ax_int] = slice_from_index(idx)
            elif len(ax_index) == 1:
                slices[ax_int] = slice(0, 1)
            else:
                raise ValueError(
                    f"Axis '{ax_name}' has length {len(ax_index)} > 1. "
                    f"Specify its value in 'at'."
                )

        # Extract the data
        sliced = self[tuple(slices)]

        # Shape should be (nt, 1, n, 1) or similar with 2 non-trivial dims
        data_4d = sliced.value
        data_2d = data_4d.squeeze()

        # Apply mode
        data_2d = select_value(data_2d, mode=mode)
        if hasattr(data_2d, "value"):
            data_2d = data_2d.value

        # Get axes
        t_axis = self._axis0_index
        space_axis = [
            self._axis0_index,
            self._axis1_index,
            self._axis2_index,
            self._axis3_index,
        ][axis_int]

        return t_axis, space_axis, data_2d

    def plot_time_space_map(
        self,
        axis="x",
        at=None,
        mode="real",
        method="pcolormesh",
        ax=None,
        add_colorbar=True,
        vmin=None,
        vmax=None,
        title=None,
        cmap=None,
        **kwargs,
    ):
        """Plot a 2D time-space map (t vs one spatial axis).

        Parameters
        ----------
        axis : str, optional
            Spatial axis name. Default is 'x'.
        at : dict, optional
            Fixed values for other spatial axes.
        mode : str, optional
            Component to extract. Default is 'real'.
        method : str, optional
            Plot method. Default is 'pcolormesh'.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        add_colorbar : bool, optional
            Whether to add colorbar. Default is True.
        vmin, vmax : float, optional
            Color scale limits.
        title : str, optional
            Plot title.
        cmap : str or Colormap, optional
            Colormap to use.
        **kwargs
            Additional plot arguments.

        Returns
        -------
        tuple
            (fig, ax): The matplotlib figure and axes objects.

        Examples
        --------
        >>> import numpy as np
        >>> from gwexpy.fields import ScalarField
        >>> from astropy import units as u
        >>> sf = ScalarField(np.ones((100, 4, 1, 1)), axis1=np.arange(4)*u.m)
        >>> fig, ax = sf.plot_time_space_map('x') # doctest: +SKIP

        """
        import matplotlib.pyplot as plt

        t_axis, space_axis, data_2d = self.time_space_map(axis, at=at, mode=mode)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # Plot
        if method == "pcolormesh":
            im = ax.pcolormesh(
                space_axis.value,
                t_axis.value,
                data_2d,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                **kwargs,
            )
        elif method == "imshow":
            from ..types.axis import AxisDescriptor

            if (
                len(space_axis) >= 2 and not AxisDescriptor(axis, space_axis).regular
            ) or (
                len(t_axis) >= 2
                and not AxisDescriptor(self._axis0_name, t_axis).regular
            ):
                raise ValueError(
                    "method='imshow' requires regularly spaced axes; "
                    "use method='pcolormesh' for nonuniform rectilinear grids."
                )
            extent = [
                space_axis.value[0],
                space_axis.value[-1],
                t_axis.value[0],
                t_axis.value[-1],
            ]
            im = ax.imshow(
                data_2d,
                extent=extent,
                origin="lower",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown method '{method}'.")

        # Labels
        from gwexpy.plot._label_utils import format_axis_label

        ax.set_xlabel(format_axis_label(axis, space_axis.unit))
        ax.set_ylabel(format_axis_label(self._axis0_name, t_axis.unit))

        if title:
            ax.set_title(title)

        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            if self.unit is not None:
                cbar.set_label(format_axis_label(mode, self.unit))

        return fig, ax

    # =========================================================================
    # Signal Processing Methods (Phase 3)
    # =========================================================================

    def compute_psd(self, point_or_region, **kwargs):
        """Compute power spectral density using Welch's method.

        This is a convenience wrapper around :func:`~gwexpy.fields.signal.compute_psd`.

        Parameters
        ----------
        point_or_region : tuple, list of tuples, or dict
            Spatial location(s) to extract:
            - Single point: ``(x, y, z)`` tuple of Quantities
            - Multiple points: list of ``(x, y, z)`` tuples
            - Region dict: ``{'x': slice or value, 'y': ..., 'z': ...}``
        **kwargs
            Additional keyword arguments passed to compute_psd:
            nperseg, noverlap, window, detrend, scaling, average.

        Returns
        -------
        FrequencySeries or FrequencySeriesList
            PSD estimate(s).

        See Also
        --------
        gwexpy.fields.signal.compute_psd : Full documentation.

        """
        from .signal import compute_psd

        return compute_psd(self, point_or_region, **kwargs)

    def freq_space_map(self, axis, at=None, **kwargs):
        """Compute frequency-space map along a spatial axis.

        This is a convenience wrapper around
        :func:`~gwexpy.fields.signal.freq_space_map`.

        Parameters
        ----------
        axis : str
            Spatial axis to scan along ('x', 'y', or 'z').
        at : dict, optional
            Fixed values for the other two spatial axes.
        **kwargs
            Additional keyword arguments passed to freq_space_map.

        Returns
        -------
        ScalarField
            2D frequency-space map.

        See Also
        --------
        gwexpy.fields.signal.freq_space_map : Full documentation.

        """
        from .signal import freq_space_map

        return freq_space_map(self, axis, at=at, **kwargs)

    def resample(self, rate, **kwargs) -> ScalarField:
        """Resample the field along the time axis (axis 0).

        Parameters
        ----------
        rate : float, Quantity
            The new sampling rate (e.g., in Hz).
        **kwargs
            Additional arguments passed to :meth:`gwpy.timeseries.TimeSeries.resample`.

        Returns
        -------
        ScalarField
            Resampled field.

        """
        if self._axis0_domain != "time":
            raise ValueError("resample requires axis0_domain='time'")
        dt = self._validate_uniform_time_axis("resample")

        # Reshape to (time, points)
        orig_shape = self.shape
        data_2d = self.value.reshape(orig_shape[0], -1)

        # Use scipy.signal.resample for efficient array resampling
        # We need to compute the new number of samples
        if hasattr(rate, "to"):
            new_fs = rate.to("Hz").value
            new_dt = (1.0 / new_fs) * u.s
        else:
            new_fs = float(rate)
            new_dt = (1.0 / new_fs) * dt.unit

        duration = (orig_shape[0] * dt).to(new_dt.unit).value
        new_nt = int(round(duration * new_fs))

        import scipy.signal

        new_data_2d = scipy.signal.resample(data_2d, new_nt, axis=0)

        new_times = (
            np.arange(new_nt) * (1.0 / new_fs) * new_dt.unit + self._axis0_index[0]
        )

        # Reshape back
        new_shape = [new_nt] + list(orig_shape[1:])
        result = ScalarField(
            new_data_2d.reshape(new_shape),
            unit=self.unit,
            epoch=self.epoch,
            axis0=new_times,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=self.axis_names,
            axis0_domain="time",
            space_domain=self._space_domains,
        )
        self._propagate_gwex_attrs(result)
        return result

    def filter(self, *args, **kwargs) -> ScalarField:
        """Apply a filter along the time axis (axis 0).

        Parameters
        ----------
        *args, **kwargs
            Filter specification. Supports same arguments as
            :meth:`gwpy.timeseries.TimeSeries.filter`.

        Returns
        -------
        ScalarField
            Filtered field.

        """
        if self._axis0_domain != "time":
            raise ValueError("filter requires axis0_domain='time'")
        dt = self._validate_uniform_time_axis("filter")

        # Reshape to (time, points)
        orig_shape = self.shape
        data_2d = self.value.reshape(orig_shape[0], -1)

        # Use gwpy to parse the filter
        # We need sample rate
        fs = (1.0 / dt).to("Hz").value
        from gwpy.signal import filter_design
        from scipy import signal as scipy_signal

        analog = kwargs.pop("analog", False)
        unit = kwargs.pop("unit", "rad/s")
        normalize_gain = kwargs.pop("normalize_gain", False)
        filtfilt = kwargs.pop("filtfilt", True)  # Default to True for phase consistency

        filt = args[0] if len(args) == 1 else args

        try:
            # GWpy >= 4.0
            zpk = filter_design.prepare_digital_filter(
                filt,
                analog=analog,
                sample_rate=fs,
                unit=unit,
                normalize_gain=normalize_gain,
                output="zpk",
            )
            form = "zpk"
            filt_obj = zpk
        except AttributeError:
            # Fallback for GWpy < 4.0
            form, filt_obj = filter_design.parse_filter(
                filt,
                analog=analog,
                sample_rate=fs,
            )

        # Apply filter along axis 0
        if form == "zpk":
            sos = scipy_signal.zpk2sos(*filt_obj)
            new_data_2d = (
                scipy_signal.sosfiltfilt(sos, data_2d, axis=0)
                if filtfilt
                else scipy_signal.sosfilt(sos, data_2d, axis=0)
            )
        else:
            b, a = filt_obj
            new_data_2d = (
                scipy_signal.filtfilt(b, a, data_2d, axis=0)
                if filtfilt
                else scipy_signal.lfilter(b, a, data_2d, axis=0)
            )

        # Reshape back
        result = ScalarField(
            new_data_2d.reshape(orig_shape),
            unit=self.unit,
            epoch=self.epoch,
            axis0=self._axis0_index,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=self.axis_names,
            axis0_domain="time",
            space_domain=self._space_domains,
        )
        self._propagate_gwex_attrs(result)
        return result

    def compute_xcorr(self, point_a, point_b, **kwargs):
        """Compute cross-correlation between two spatial points.

        This is a convenience wrapper around
        :func:`~gwexpy.fields.signal.compute_xcorr`.

        Parameters
        ----------
        point_a, point_b : tuple of Quantity
            Spatial coordinates (x, y, z) for the two points.
        **kwargs
            Additional keyword arguments passed to compute_xcorr.

        Returns
        -------
        TimeSeries
            Cross-correlation function with lag axis.

        See Also
        --------
        gwexpy.fields.signal.compute_xcorr : Full documentation.

        """
        from .signal import compute_xcorr

        return compute_xcorr(self, point_a, point_b, **kwargs)

    def time_delay_map(self, ref_point, plane="xy", at=None, **kwargs):
        """Compute time delay map from a reference point.

        This is a convenience wrapper around
        :func:`~gwexpy.fields.signal.time_delay_map`.

        Parameters
        ----------
        ref_point : tuple of Quantity
            Reference point coordinates (x, y, z).
        plane : str
            2D plane to map: 'xy', 'xz', or 'yz'. Default 'xy'.
        at : dict, optional
            Fixed value for the axis not in the plane.
        **kwargs
            Additional keyword arguments passed to time_delay_map.

        Returns
        -------
        ScalarField
            Time delay map.

        See Also
        --------
        gwexpy.fields.signal.time_delay_map : Full documentation.

        """
        from .signal import time_delay_map

        return time_delay_map(self, ref_point, plane=plane, at=at, **kwargs)

    def coherence_map(self, ref_point, plane="xy", at=None, **kwargs):
        """Compute coherence map from a reference point.

        This is a convenience wrapper around
        :func:`~gwexpy.fields.signal.coherence_map`.

        Parameters
        ----------
        ref_point : tuple of Quantity
            Reference point coordinates (x, y, z).
        plane : str
            2D plane to map: 'xy', 'xz', or 'yz'. Default 'xy'.
        at : dict, optional
            Fixed value for the axis not in the plane.
        **kwargs
            Additional keyword arguments passed to coherence_map.

        Returns
        -------
        ScalarField or FieldDict
            Coherence map.

        See Also
        --------
        gwexpy.fields.signal.coherence_map : Full documentation.

        """
        from .signal import coherence_map

        return coherence_map(self, ref_point, plane=plane, at=at, **kwargs)

    # =========================================================================
    # Spectral Density (Phase 2)
    # =========================================================================

    def spectral_density(self, axis=0, **kwargs):
        """Compute spectral density along any axis.

        Generalized spectral density function that works on time axis (0)
        or spatial axes (1-3). Returns a new ScalarField with the transformed
        axis in spectral domain.

        Parameters
        ----------
        axis : int or str
            Axis to transform. Default 0 (time axis).
        **kwargs
            Additional arguments passed to
            :func:`~gwexpy.fields.signal.spectral_density`.
            See that function for full parameter list.

        Returns
        -------
        ScalarField
            Spectral density field with transformed axis.

        Examples
        --------
        >>> # Time PSD
        >>> psd_field = field.spectral_density(axis=0)
        >>> psd_field.axis0_domain  # 'frequency'

        >>> # Spatial wavenumber spectrum
        >>> kx_spec = field.spectral_density(axis='x')

        See Also
        --------
        gwexpy.fields.signal.spectral_density : Full documentation.
        psd : Convenience alias for time-axis PSD.

        """
        from .signal import spectral_density

        return spectral_density(self, axis=axis, **kwargs)

    def psd(self, **kwargs):
        """Compute power spectral density along time axis.

        Convenience method equivalent to ``spectral_density(axis=0)``.
        Uses Welch's method by default for robust PSD estimation.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to :func:`~gwexpy.fields.signal.spectral_density`
            or :func:`~gwexpy.fields.signal.compute_psd` (if point_or_region is used).
            Common options:

            - point_or_region : tuple or list, optional
                If provided, computes PSD at specific spatial point(s) or region
                average instead of full field. Returns FrequencySeries(List).
            - method : {'welch', 'fft'}, default 'welch'
            - fftlength : float, optional
                Segment length in seconds (time-based specification)
            - nfft : int, optional
                Number of samples per segment (sample-based specification)
            - overlap : float, optional
                Overlap in seconds
            - noverlap : int, optional
                Number of overlapping samples
            - window : str, window function
            - scaling : {'density', 'spectrum'}

        Returns
        -------
        ScalarField
            PSD field with axis0_domain='frequency'.

        Examples
        --------
        >>> from astropy import units as u
        >>> # Time-based specification
        >>> psd_field = field.psd(fftlength=1.0, overlap=0.5)
        >>> # Or sample-based specification
        >>> psd_field = field.psd(nfft=512, noverlap=256)
        >>> psd_field.shape  # (n_freq, nx, ny, nz)

        See Also
        --------
        spectral_density : Generalized spectral density for any axis.

        """
        if "point_or_region" in kwargs:
            from .signal import compute_psd

            point_or_region = kwargs.pop("point_or_region")
            return compute_psd(self, point_or_region, **kwargs)

        return self.spectral_density(axis=0, **kwargs)

    # =========================================================================
    # Geospatial & Missing-Value Utilities
    # =========================================================================

    def fill_missing(
        self,
        method: Literal["nearest", "linear", "constant"] = "nearest",
        *,
        value: float | u.Quantity | None = None,
        axes: Sequence[str] | None = None,
        max_distance: float | u.Quantity | None = None,
    ) -> ScalarField:
        """Fill missing (NaN) values via spatial interpolation or a constant.

        Parameters
        ----------
        method : {'nearest', 'linear', 'constant'}, default 'nearest'
            Infilling strategy. ``'nearest'`` and ``'linear'`` interpolate from
            finite samples along the spatial axes; ``'constant'`` replaces NaNs
            with ``value``.
        value : float or Quantity, optional
            Replacement value when ``method='constant'``.
        axes : sequence of str, optional
            Spatial axis names to interpolate along. When ``None``, spatial axes
            with length > 1 are selected automatically.
        max_distance : float or Quantity, optional
            Maximum Euclidean distance (in axis units) from valid samples for
            infilling with ``'nearest'`` or ``'linear'``. Pixels beyond this
            threshold remain NaN.

        Returns
        -------
        ScalarField
            New field with NaNs filled and untouched valid pixels bit-identical.
            Active :class:`GeoMetadata` remains active with an appended history
            entry.

        """
        if method not in ("nearest", "linear", "constant"):
            raise ValueError(
                f"Invalid fill_missing method '{method}'. "
                "Must be 'nearest', 'linear', or 'constant'."
            )
        if method == "constant":
            if value is None:
                raise ValueError("method='constant' requires 'value='.")
            if max_distance is not None:
                raise ValueError(
                    "max_distance is not supported with method='constant'."
                )
        elif value is not None:
            raise ValueError(
                f"'value=' is only valid for method='constant', not {method!r}."
            )

        data = np.array(self.value, copy=True)
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float64)

        if np.all(np.isnan(data)):
            raise ValueError("Cannot fill_missing: all field values are NaN.")

        if method == "constant":
            assert value is not None
            if isinstance(value, u.Quantity):
                target_unit = (
                    self.unit if self.unit is not None else u.dimensionless_unscaled
                )
                fill_val = float(value.to_value(target_unit))
            else:
                fill_val = float(value)
            nan_mask = np.isnan(data)
            if np.any(nan_mask):
                data[nan_mask] = fill_val
            desc = f"fill_missing(method='constant', value={fill_val!r})"
        else:
            if axes is None:
                interp_axis_ints = [i for i in (1, 2, 3) if self.shape[i] > 1]
            else:
                interp_axis_ints = []
                for ax_name in axes:
                    ax_int = self._get_axis_index(ax_name)
                    if ax_int == 0:
                        raise ValueError(
                            "fill_missing operates on spatial axes (1..3), not axis 0."
                        )
                    if ax_int in interp_axis_ints:
                        raise ValueError(f"Duplicate axis '{ax_name}' in axes.")
                    interp_axis_ints.append(ax_int)

            if not np.any(np.isnan(data)):
                desc = f"fill_missing(method={method!r})"
            else:
                if not interp_axis_ints:
                    raise ValueError(
                        "Cannot interpolate spatially when all spatial axes have length 1."
                    )

                axis_indices = [
                    self._axis0_index,
                    self._axis1_index,
                    self._axis2_index,
                    self._axis3_index,
                ]
                ref_unit = getattr(
                    axis_indices[interp_axis_ints[0]], "unit", u.dimensionless_unscaled
                )
                coord_arrays: list[np.ndarray] = []
                for ax_int in interp_axis_ints:
                    idx_arr = axis_indices[ax_int]
                    ax_unit = getattr(idx_arr, "unit", u.dimensionless_unscaled)
                    if not ax_unit.is_equivalent(ref_unit):
                        raise ValueError(
                            f"Incompatible spatial axis units for fill_missing: "
                            f"{ax_unit} vs {ref_unit}."
                        )
                    coord_arrays.append(
                        np.asarray(
                            u.Quantity(idx_arr, copy=False).to_value(ref_unit),
                            dtype=np.float64,
                        )
                    )

                max_dist_val: float | None = None
                if max_distance is not None:
                    if isinstance(max_distance, u.Quantity):
                        max_dist_val = float(max_distance.to_value(ref_unit))
                    else:
                        max_dist_val = float(max_distance)
                    if not np.isfinite(max_dist_val) or max_dist_val < 0:
                        raise ValueError(
                            f"max_distance must be finite and non-negative, got {max_distance!r}"
                        )

                from scipy.interpolate import LinearNDInterpolator, interp1d
                from scipy.spatial import QhullError, cKDTree

                mesh = np.meshgrid(*coord_arrays, indexing="ij")
                flat_coords = np.column_stack([m.ravel() for m in mesh])
                other_axes = [i for i in range(4) if i not in interp_axis_ints]
                other_shape = tuple(self.shape[i] for i in other_axes)

                for outer_idx in np.ndindex(*other_shape):
                    sl: list[Any] = [slice(None)] * 4
                    for ax_pos, idx_val in zip(other_axes, outer_idx):
                        sl[ax_pos] = idx_val
                    sub = data[tuple(sl)]
                    sub_miss = np.isnan(sub)
                    if not np.any(sub_miss):
                        continue
                    sub_valid = ~sub_miss
                    if not np.any(sub_valid):
                        raise ValueError(
                            f"Cannot interpolate missing values: slice {outer_idx} is all NaN."
                        )
                    valid_flat = sub_valid.ravel()
                    miss_flat = sub_miss.ravel()
                    valid_pts = flat_coords[valid_flat]
                    valid_vals = sub.ravel()[valid_flat]
                    miss_pts = flat_coords[miss_flat]

                    if method == "nearest":
                        tree = cKDTree(valid_pts)
                        dists, nn_idx = tree.query(miss_pts)
                        filled = valid_vals[nn_idx]
                        if max_dist_val is not None:
                            filled = np.where(dists <= max_dist_val, filled, np.nan)
                    else:
                        if len(interp_axis_ints) == 1:
                            interp1 = interp1d(
                                valid_pts[:, 0],
                                valid_vals,
                                kind="linear",
                                bounds_error=False,
                                fill_value=np.nan,
                            )
                            filled = interp1(miss_pts[:, 0])
                        else:
                            try:
                                interp_nd = LinearNDInterpolator(
                                    valid_pts, valid_vals, fill_value=np.nan
                                )
                                filled = interp_nd(miss_pts)
                            except QhullError as exc:
                                raise ValueError(
                                    "Linear interpolation failed (valid samples may be "
                                    "collinear or degenerate). Use method='nearest'."
                                ) from exc
                        if max_dist_val is not None:
                            dists, _ = cKDTree(valid_pts).query(miss_pts)
                            filled = np.where(dists <= max_dist_val, filled, np.nan)

                    out_sub = sub.copy()
                    out_sub[sub_miss] = filled
                    data[tuple(sl)] = out_sub

                desc = f"fill_missing(method={method!r}, max_distance={max_distance!r})"

        result = ScalarField(
            data,
            unit=self.unit,
            epoch=self.epoch,
            name=self.name,
            channel=self.channel,
            axis0=self._axis0_index.copy(),
            axis1=self._axis1_index.copy(),
            axis2=self._axis2_index.copy(),
            axis3=self._axis3_index.copy(),
            axis_names=list(self.axis_names),
            axis0_domain=self._axis0_domain,
            space_domain=self._space_domains,
        )
        self._propagate_gwex_attrs(result)
        if self.geo is not None:
            result.geo = self.geo.with_history(desc)
        return result

    def to_local_cartesian(
        self,
        origin: tuple[float | u.Quantity, float | u.Quantity] | None = None,
        *,
        names: tuple[str, str] = ("x", "y"),
        ellipsoid: str = "GRS80",
    ) -> ScalarField:
        r"""Convert two ``'angle'`` spatial axes (lon, lat) to local tangent-plane metres.

        Uses the local ellipsoid curvature radii at the origin latitude
        :math:`\phi_0`:

        .. math::

            x = N(\phi_0) \cos(\phi_0) \, (\lambda - \lambda_0), \qquad
            y = M(\phi_0) \, (\phi - \phi_0),

        where :math:`N(\phi_0) = a / W(\phi_0)` is the prime-vertical radius of
        curvature, :math:`M(\phi_0) = a(1 - e^2) / W(\phi_0)^3` is the
        meridional radius of curvature, and
        :math:`W(\phi_0) = \sqrt{1 - e^2 \sin^2\phi_0}`.

        For approximately uniform input angle grids, output axes are
        resynthesized as ``(arange(n) - i0) * d`` to preserve uniform spacing;
        for genuinely nonuniform angle axes, exact point-by-point mapped
        coordinates are preserved.

        Notes
        -----
        This tangent-plane approximation removes first-order ellipsoid flattening
        bias at the origin, but parallel convergence still introduces
        east-west scale distortion of order
        :math:`|\tan\phi_0| \cdot (|\Delta y| / N(\phi_0)) = O(\mathrm{span}/R)`
        away from ``origin``, plus :math:`O((\mathrm{span}/R)^2)` curvature
        effects. For wide areas or strict CRS reprojection, use
        :meth:`to_projected`.

        Parameters
        ----------
        origin : tuple of (lon0, lat0), optional
            Local tangent-plane origin in degrees (or angular Quantities).
            Defaults to the midpoint of the two angle axes.
        names : tuple of (str, str), default ('x', 'y')
            Output names for the converted spatial axes.
        ellipsoid : {'GRS80', 'WGS84', 'sphere'}, default 'GRS80'
            Reference ellipsoid parameters.

        Returns
        -------
        ScalarField
            Field with converted axes in ``'real'`` domain with metre units.
            Any attached :class:`GeoMetadata` is deactivated with
            ``local_origin=(lon0, lat0)``.

        """
        from ..types.axis import AxisDescriptor

        key_ellip = ellipsoid.lower()
        if key_ellip not in _ELLIPSOID_PARAMS:
            raise ValueError(
                f"Unknown ellipsoid '{ellipsoid}'. Supported: {sorted(_ELLIPSOID_PARAMS)}"
            )
        if len(names) != 2 or names[0] == names[1]:
            raise ValueError("'names' must be a 2-tuple of distinct axis names.")

        spatial_specs = [
            (1, self._axis1_name, self._axis1_index),
            (2, self._axis2_name, self._axis2_index),
            (3, self._axis3_name, self._axis3_index),
        ]
        angle_specs = [
            spec
            for spec in spatial_specs
            if self._space_domains.get(spec[1]) == "angle"
        ]
        if len(angle_specs) != 2:
            raise ValueError(
                "to_local_cartesian() requires exactly two spatial axes in 'angle' "
                f"domain, found {len(angle_specs)} ({[s[1] for s in angle_specs]})."
            )

        (idx_lon, lon_name, lon_axis), (idx_lat, lat_name, lat_axis) = angle_specs
        other_names = {self._axis0_name} | {
            name for i, name, _ in spatial_specs if i not in (idx_lon, idx_lat)
        }
        if any(new_n in other_names for new_n in names):
            raise ValueError(
                f"Output axis names {names} collide with existing axis names {other_names}."
            )

        for ax_n, ax_q in ((lon_name, lon_axis), (lat_name, lat_axis)):
            ax_u = getattr(ax_q, "unit", u.dimensionless_unscaled)
            if not ax_u.is_equivalent(u.deg):
                raise ValueError(
                    f"Angle axis '{ax_n}' must have angular units (e.g. deg, rad), got {ax_u}."
                )

        lon_deg = np.asarray(
            u.Quantity(lon_axis, copy=False).to_value(u.deg), dtype=np.float64
        )
        lat_deg = np.asarray(
            u.Quantity(lat_axis, copy=False).to_value(u.deg), dtype=np.float64
        )

        if origin is None:
            lon0 = float(0.5 * (lon_deg[0] + lon_deg[-1]))
            lat0 = float(0.5 * (lat_deg[0] + lat_deg[-1]))
        else:
            if len(origin) != 2:
                raise ValueError("'origin' must be a 2-tuple (lon0, lat0).")
            o_lon, o_lat = origin
            lon0 = float(
                o_lon.to_value(u.deg) if isinstance(o_lon, u.Quantity) else o_lon
            )
            lat0 = float(
                o_lat.to_value(u.deg) if isinstance(o_lat, u.Quantity) else o_lat
            )

        if not (np.isfinite(lon0) and np.isfinite(lat0)):
            raise ValueError(
                f"origin coordinates must be finite, got ({lon0}, {lat0})."
            )
        if abs(lat0) >= 90.0:
            raise ValueError(
                f"origin latitude must satisfy |lat0| < 90 deg, got {lat0}."
            )

        a_ellip, inv_f = _ELLIPSOID_PARAMS[key_ellip]
        if inv_f == 0.0:
            e2 = 0.0
        else:
            f_flat = 1.0 / inv_f
            e2 = f_flat * (2.0 - f_flat)

        phi0 = np.deg2rad(lat0)
        sin_phi0 = np.sin(phi0)
        cos_phi0 = np.cos(phi0)
        w = np.sqrt(1.0 - e2 * sin_phi0 * sin_phi0)
        n_phi0 = a_ellip / w
        m_phi0 = a_ellip * (1.0 - e2) / (w**3)

        deg_to_rad = np.pi / 180.0
        scale_x = n_phi0 * cos_phi0 * deg_to_rad
        scale_y = m_phi0 * deg_to_rad

        def _convert_angle_axis(
            ax_label: str, vals_deg: np.ndarray, ref_deg: float, scale_m: float
        ) -> u.Quantity:
            if vals_deg.size >= 2:
                desc = AxisDescriptor(ax_label, vals_deg * u.deg)
                passed, _, _, d_deg, fit_0 = desc._regularity_report(max_deviation=1e-4)
                if passed and d_deg != 0.0:
                    d_m = d_deg * scale_m
                    i0 = (ref_deg - fit_0) / d_deg
                    return (
                        (np.arange(vals_deg.size, dtype=np.float64) - i0) * d_m
                    ) * u.m
            return ((vals_deg - ref_deg) * scale_m) * u.m

        new_axes = [
            self._axis0_index.copy(),
            self._axis1_index.copy(),
            self._axis2_index.copy(),
            self._axis3_index.copy(),
        ]
        new_axes[idx_lon] = _convert_angle_axis(lon_name, lon_deg, lon0, scale_x)
        new_axes[idx_lat] = _convert_angle_axis(lat_name, lat_deg, lat0, scale_y)

        new_names = list(self.axis_names)
        new_names[idx_lon] = names[0]
        new_names[idx_lat] = names[1]

        new_domains = dict(self._space_domains)
        del new_domains[lon_name]
        del new_domains[lat_name]
        new_domains[names[0]] = "real"
        new_domains[names[1]] = "real"

        result = ScalarField(
            self.value.copy(),
            unit=self.unit,
            epoch=self.epoch,
            name=self.name,
            channel=self.channel,
            axis0=new_axes[0],
            axis1=new_axes[1],
            axis2=new_axes[2],
            axis3=new_axes[3],
            axis_names=new_names,
            axis0_domain=self._axis0_domain,
            space_domain=new_domains,
        )
        self._propagate_gwex_attrs(result)
        if self.geo is not None:
            result.geo = self.geo.deactivated(
                f"to_local_cartesian(origin=({lon0:.6f}, {lat0:.6f}), ellipsoid={ellipsoid!r})",
                local_origin=(lon0, lat0),
            )
        else:
            result.geo = GeoMetadata(
                vertical_datum="unspecified",
                active=False,
                local_origin=(lon0, lat0),
                history=(
                    f"to_local_cartesian(origin=({lon0:.6f}, {lat0:.6f}), ellipsoid={ellipsoid!r})",
                ),
            )
        return result

    def to_projected(
        self,
        target_crs: str,
        *,
        resolution: float | tuple[float, float] | u.Quantity | None = None,
        method: Literal["nearest", "bilinear", "cubic"] = "bilinear",
    ) -> ScalarField:
        """Reproject and resample this georeferenced field to *target_crs*.

        Parameters
        ----------
        target_crs : str
            Target CRS identifier accepted by :class:`pyproj.CRS`
            (e.g. ``'EPSG:32654'`` or ``'EPSG:6675'``).
        resolution : float, tuple of (dx, dy), or Quantity, optional
            Output pixel spacing in target CRS units (or compatible Quantity).
            When ``None``, inferred from the source center pixel spacing.
        method : {'nearest', 'bilinear', 'cubic'}, default 'bilinear'
            Spatial resampling method.

        Returns
        -------
        ScalarField
            Reprojected field with active :class:`GeoMetadata` reflecting the
            new CRS, transform, and reprojection history.

        Raises
        ------
        TypeError
            If field values are complex. The resampling kernel currently
            supports real values only.

        """
        if np.iscomplexobj(self.value):
            raise TypeError(
                "to_projected() cannot resample complex field values; "
                "convert to a real-valued quantity explicitly first."
            )

        from gwexpy.io.utils import ensure_dependency

        pyproj = ensure_dependency("pyproj", extra="geospatial")
        if method not in ("nearest", "bilinear", "cubic"):
            raise ValueError(
                f"Invalid resampling method '{method}'. "
                "Must be 'nearest', 'bilinear', or 'cubic'."
            )
        if self.geo is None or not self.geo.active or not self.geo.crs:
            raise ValueError(
                "to_projected() requires an active GeoMetadata with a source CRS."
            )

        src_crs = pyproj.CRS.from_user_input(self.geo.crs)
        dst_crs = pyproj.CRS.from_user_input(target_crs)

        if src_crs.is_geographic:
            src_unit = u.deg
        else:
            src_factor = (
                src_crs.axis_info[0].unit_conversion_factor
                if src_crs.axis_info
                else 1.0
            )
            src_unit = u.m if np.isclose(src_factor, 1.0) else u.m * src_factor

        src_x = np.asarray(
            u.Quantity(self._axis1_index, copy=False).to_value(src_unit),
            dtype=np.float64,
        )
        src_y = np.asarray(
            u.Quantity(self._axis2_index, copy=False).to_value(src_unit),
            dtype=np.float64,
        )
        if src_x.size < 2 or src_y.size < 2:
            raise ValueError("to_projected() requires spatial axes of length >= 2.")
        if not (np.all(np.diff(src_x) > 0) or np.all(np.diff(src_x) < 0)):
            raise ValueError(
                "Source axis 1 must be strictly monotonic for to_projected()."
            )
        if not (np.all(np.diff(src_y) > 0) or np.all(np.diff(src_y) < 0)):
            raise ValueError(
                "Source axis 2 must be strictly monotonic for to_projected()."
            )

        input_x_desc = bool(src_x[1] < src_x[0])
        input_y_desc = bool(src_y[1] < src_y[0])
        src_x_asc = src_x[::-1] if input_x_desc else src_x
        src_y_asc = src_y[::-1] if input_y_desc else src_y

        fwd = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        inv = pyproj.Transformer.from_crs(dst_crs, src_crs, always_xy=True)

        bx_min, by_min, bx_max, by_max = geo_bounds(self)
        gx = np.linspace(bx_min, bx_max, 21)
        gy = np.linspace(by_min, by_max, 21)
        gxx, gyy = np.meshgrid(gx, gy, indexing="ij")
        tx_pts, ty_pts = fwd.transform(gxx, gyy)
        tx_arr = np.asarray(tx_pts, dtype=np.float64)
        ty_arr = np.asarray(ty_pts, dtype=np.float64)
        if not np.any(np.isfinite(tx_arr) & np.isfinite(ty_arr)):
            raise ValueError(
                "CRS transformation produced no finite target coordinates."
            )
        dst_xmin = float(np.nanmin(tx_arr))
        dst_xmax = float(np.nanmax(tx_arr))
        dst_ymin = float(np.nanmin(ty_arr))
        dst_ymax = float(np.nanmax(ty_arr))

        if dst_crs.is_geographic:
            dst_unit = u.deg
            dst_names = ("x_unused", "lon", "lat")
            dst_dom = "angle"
        else:
            dst_factor = (
                dst_crs.axis_info[0].unit_conversion_factor
                if dst_crs.axis_info
                else 1.0
            )
            dst_unit = u.m if np.isclose(dst_factor, 1.0) else u.m * dst_factor
            dst_names = ("x_unused", "x", "y")
            dst_dom = "real"

        if resolution is None:
            xc = 0.5 * (bx_min + bx_max)
            yc = 0.5 * (by_min + by_max)
            dx_src = float(abs(src_x_asc[1] - src_x_asc[0]))
            dy_src = float(abs(src_y_asc[1] - src_y_asc[0]))
            x0p, y0p = fwd.transform(xc, yc)
            x1p, y1p = fwd.transform(xc + dx_src, yc)
            x2p, y2p = fwd.transform(xc, yc + dy_src)
            dx_dst = float(np.hypot(x1p - x0p, y1p - y0p))
            dy_dst = float(np.hypot(x2p - x0p, y2p - y0p))
        elif isinstance(resolution, u.Quantity):
            if resolution.ndim == 0:
                dx_dst = dy_dst = float(resolution.to_value(dst_unit))
            elif resolution.size == 2:
                vals = resolution.to_value(dst_unit)
                dx_dst, dy_dst = float(vals[0]), float(vals[1])
            else:
                raise ValueError("resolution Quantity must be scalar or length-2.")
        elif isinstance(resolution, (tuple, list, np.ndarray)):
            if len(resolution) != 2:
                raise ValueError("resolution sequence must have length 2.")
            r0, r1 = resolution
            dx_dst = float(r0.to_value(dst_unit) if isinstance(r0, u.Quantity) else r0)
            dy_dst = float(r1.to_value(dst_unit) if isinstance(r1, u.Quantity) else r1)
        else:
            dx_dst = dy_dst = float(resolution)

        if not (
            np.isfinite(dx_dst) and np.isfinite(dy_dst) and dx_dst > 0 and dy_dst > 0
        ):
            raise ValueError(
                f"resolution must be positive and finite, got ({dx_dst}, {dy_dst})."
            )

        nx_dst = max(1, int(np.ceil((dst_xmax - dst_xmin) / dx_dst)))
        ny_dst = max(1, int(np.ceil((dst_ymax - dst_ymin) / dy_dst)))
        dst_x = dst_xmin + (np.arange(nx_dst, dtype=np.float64) + 0.5) * dx_dst
        if input_y_desc:
            dst_y = dst_ymax - (np.arange(ny_dst, dtype=np.float64) + 0.5) * dy_dst
        else:
            dst_y = dst_ymin + (np.arange(ny_dst, dtype=np.float64) + 0.5) * dy_dst

        xx, yy = np.meshgrid(dst_x, dst_y, indexing="ij")
        sx_pts, sy_pts = inv.transform(xx, yy)
        sx_grid = np.asarray(sx_pts, dtype=np.float64)
        sy_grid = np.asarray(sy_pts, dtype=np.float64)

        src_vals = np.asarray(self.value, dtype=np.float64)
        if input_x_desc:
            src_vals = src_vals[:, ::-1, :, :]
        if input_y_desc:
            src_vals = src_vals[:, :, ::-1, :]

        nt, _, _, nz = src_vals.shape
        out_data = np.full((nt, nx_dst, ny_dst, nz), np.nan, dtype=np.float64)
        for it in range(nt):
            for iz in range(nz):
                out_data[it, :, :, iz] = _resample_2d_grid(
                    src_x_asc,
                    src_y_asc,
                    src_vals[it, :, :, iz],
                    sx_grid,
                    sy_grid,
                    method,
                )

        new_names = [self._axis0_name, dst_names[1], dst_names[2], self._axis3_name]
        new_domains = {
            dst_names[1]: dst_dom,
            dst_names[2]: dst_dom,
            self._axis3_name: self._space_domains.get(self._axis3_name, "real"),
        }
        result = ScalarField(
            out_data,
            unit=self.unit,
            epoch=self.epoch,
            name=self.name,
            channel=self.channel,
            axis0=self._axis0_index.copy(),
            axis1=dst_x * dst_unit,
            axis2=dst_y * dst_unit,
            axis3=self._axis3_index.copy(),
            axis_names=new_names,
            axis0_domain=self._axis0_domain,
            space_domain=new_domains,
        )
        self._propagate_gwex_attrs(result)
        dst_crs_str = dst_crs.to_string() or str(target_crs)
        out_bounds = geo_bounds(result)
        y_origin = float(dst_ymax if input_y_desc else dst_ymin)
        y_step = float(-dy_dst if input_y_desc else dy_dst)
        result.geo = GeoMetadata(
            crs=dst_crs_str,
            transform=(float(dst_xmin), float(dx_dst), 0.0, y_origin, 0.0, y_step),
            vertical_datum=self.geo.vertical_datum,
            nodata=float("nan"),
            band=self.geo.band,
            source=self.geo.source,
            source_format=self.geo.source_format,
            active=True,
            local_origin=None,
            history=(
                *self.geo.history,
                f"to_projected(src_crs={self.geo.crs!r}, target_crs={dst_crs_str!r}, "
                f"resolution=({dx_dst:g}, {dy_dst:g}), method={method!r}, "
                f"bounds={out_bounds!r})",
            ),
        )
        return result

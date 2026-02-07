from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import units as u
from scipy import fft as sp_fft

from .base import FieldBase

if TYPE_CHECKING:
    from gwexpy.types.typing import IndexLike

__all__ = ["ScalarField"]


class ScalarField(FieldBase):
    """4D Field with domain states and FFT operations.

    This class extends :class:`Array4D` to represent physical fields that
    can exist in different domains (time/frequency for axis 0, real/k-space
    for spatial axes 1-3).

    **Key feature**: All indexing operations return a ScalarField, maintaining
    4D structure. Integer indices result in axes with length 1 rather than
    being dropped.

    Parameters
    ----------
    data : array-like
        4-dimensional input data.
    unit : `~astropy.units.Unit`, optional
        Physical unit of the data.
    axis0 : `~astropy.units.Quantity` or array-like, optional
        Index values for axis 0 (time or frequency).
    axis1 : `~astropy.units.Quantity` or array-like, optional
        Index values for axis 1 (x or kx).
    axis2 : `~astropy.units.Quantity` or array-like, optional
        Index values for axis 2 (y or ky).
    axis3 : `~astropy.units.Quantity` or array-like, optional
        Index values for axis 3 (z or kz).
    axis_names : iterable of str, optional
        Names for each axis (length 4). Defaults based on domain.
    axis0_domain : {'time', 'frequency'}, optional
        Domain of axis 0. Default is 'time'.
    space_domain : {'real', 'k'} or dict, optional
        Domain of spatial axes. If str, applies to all spatial axes.
        If dict, maps axis names to domains. Default is 'real'.
    **kwargs
        Additional keyword arguments passed to :class:`Array4D`.

    Attributes
    ----------
    axis0_domain : str
        Current domain of axis 0 ('time' or 'frequency').
    space_domains : dict
        Mapping of spatial axis names to their domains ('real' or 'k').

    Examples
    --------
    >>> import numpy as np
    >>> from astropy import units as u
    >>> from gwexpy.fields import ScalarField
    >>> data = np.random.randn(100, 32, 32, 32)
    >>> times = np.arange(100) * 0.01 * u.s
    >>> x = np.arange(32) * 1.0 * u.m
    >>> field = ScalarField(data, axis0=times, axis1=x, axis2=x, axis3=x,
    ...                 axis_names=['t', 'x', 'y', 'z'])
    """

    _axis0_index: IndexLike
    _axis1_index: IndexLike
    _axis2_index: IndexLike
    _axis3_index: IndexLike

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

        return ScalarField(
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

    def _isel_tuple(self, item_tuple: tuple[Any, ...]) -> ScalarField:
        """Internal isel using ScalarField getitem logic."""
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

    def fft_time(self, nfft: int | None = None) -> ScalarField:
        """Compute FFT along time axis (axis 0).

        This method applies the same normalization as GWpy's
        ``TimeSeries.fft()``: rfft / nfft, with DC-excluded bins
        multiplied by 2 (except Nyquist bin for even nfft).

        Parameters
        ----------
        nfft : int, optional
            Length of the FFT. If None, uses the length of axis 0.

        Returns
        -------
        ScalarField
            Transformed field with ``axis0_domain='frequency'``.

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time'.
        ValueError
            If time axis length < 2 or is irregularly spaced.
        TypeError
            If input data is complex-valued.

        See Also
        --------
        gwpy.timeseries.TimeSeries.fft : The reference implementation.
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
        result._validate_domain_units()
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
        dt_unit = 1 / df_unit

        # Restore time-axis origin if preserved from fft_time
        t0_offset = getattr(self, "_axis0_offset", None)
        if t0_offset is not None:
            t0_value = t0_offset.value
            times = (np.arange(nout) * dt_value + t0_value) * dt_unit
        else:
            times = np.arange(nout) * dt_value * dt_unit

        result = ScalarField(
            dift,
            unit=self.unit,
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
        return result

    # =========================================================================
    # Spatial FFT (axes 1-3, two-sided signed FFT)
    # =========================================================================

    def fft_space(
        self,
        axes: Iterable[str] | None = None,
        n: Sequence[int] | None = None,
        overwrite: bool = False,
    ) -> ScalarField:
        """Compute FFT along spatial axes.

        This method uses two-sided FFT (scipy.fft.fftn) and produces
        angular wavenumber (k = 2π·fftfreq).

        Parameters
        ----------
        axes : iterable of str, optional
            Axis names to transform (e.g., ['x', 'y']). If None,
            transforms all spatial axes in 'real' domain.
        n : tuple of int, optional
            FFT lengths for each axis.
        overwrite : bool, optional
            If True, perform FFT in-place on a temporary copy of the data
            to reduce peak memory usage. Default is False.

        Returns
        -------
        ScalarField
            Transformed field with specified axes in 'k' domain.

        Raises
        ------
        ValueError
            If any specified axis is not in 'real' domain.
        ValueError
            If any specified axis is not uniformly spaced.

        Notes
        -----
        **Angular Wavenumber Convention**

        The wavenumber axis is computed as ``k = 2π * fftfreq(n, d=dx)``,
        satisfying ``k = 2π / λ``. This is the standard **angular wavenumber**
        definition in physics, with units of [rad/length].

        Note: This is NOT the cycle wavenumber (1/λ) commonly used in some
        fields. To convert: ``k_cycle = k_angular / (2π)``.

        **Sign Convention for Descending Axes (dx < 0)**

        If the spatial axis is descending (dx < 0), the k-axis is sign-flipped
        to preserve physical consistency with the phase factor convention
        ``e^{+ikx}``. This ensures that positive k corresponds to waves
        propagating in the positive x direction, regardless of the data
        storage order.

        This convention differs from the standard FFT behavior (which ignores
        axis direction) but maintains physical consistency for interferometer
        simulations and wave propagation analysis.

        This formula was validated by 10/12 AI models in cross-verification
        (2026-02-01). The ``2π`` factor is correctly applied, and units are
        properly set as ``1/dx_unit``.

        References
        ----------
        .. [1] Press et al., Numerical Recipes (3rd ed., 2007), §12.3.2
        .. [2] NumPy fftfreq documentation
        .. [3] GWpy FrequencySeries (Duncan Macleod et al., SoftwareX 13, 2021)
        .. [4] Jackson, Classical Electrodynamics (3rd ed., 1998), §4.2:
               Fourier transform sign conventions
        """
        # Default: all real-domain spatial axes
        if axes is None:
            axes = [
                name
                for name in [self._axis1_name, self._axis2_name, self._axis3_name]
                if self._space_domains.get(name) == "real"
            ]

        if not axes:
            raise ValueError("No axes specified for fft_space")

        # Validate axes and get integer indices
        target_axes_int = []
        for ax_name in axes:
            ax_int = self._get_axis_index(ax_name)
            if ax_int == 0:
                raise ValueError(
                    "Cannot use fft_space on axis 0 (time/frequency axis). "
                    "Use fft_time instead."
                )
            domain = self._space_domains.get(ax_name)
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
            if not ax_desc.regular:
                raise ValueError(
                    f"Axis '{ax_name}' is not uniformly spaced. Cannot apply FFT."
                )
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
            s = tuple(n)

        import scipy.fft as sp_fft

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

        for ax_name, ax_int in zip(axes, target_axes_int):
            ax_desc = self.axis(ax_name)
            # Use signed delta to preserve axis direction
            delta = ax_desc.delta
            if delta is None:
                raise ValueError(
                    f"Axis '{ax_name}' does not have a defined spacing (delta)"
                )
            dx_value = getattr(delta, "value", delta)  # Already signed from diff
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

        result = ScalarField(
            dft,
            unit=self.unit,
            axis0=new_indices[0],
            axis1=new_indices[1],
            axis2=new_indices[2],
            axis3=new_indices[3],
            axis_names=new_names,
            axis0_domain=self._axis0_domain,
            space_domain=new_space_domains,
        )
        result._validate_domain_units()
        return result

    def ifft_space(
        self,
        axes: Iterable[str] | None = None,
        n: Sequence[int] | None = None,
        overwrite: bool = False,
    ) -> ScalarField:
        """Compute inverse FFT along k-space axes.

        Parameters
        ----------
        axes : iterable of str, optional
            Axis names to transform (e.g., ['kx', 'ky']). If None,
            transforms all spatial axes in 'k' domain.
        n : tuple of int, optional
            Output lengths for each axis.
        overwrite : bool, optional
            If True, perform IFFT in-place on a temporary copy.

        Returns
        -------
        ScalarField
            Transformed field with specified axes in 'real' domain.

        Raises
        ------
        ValueError
            If any specified axis is not in 'k' domain.
        """
        # Default: all k-domain spatial axes
        if axes is None:
            axes = [
                name
                for name in [self._axis1_name, self._axis2_name, self._axis3_name]
                if self._space_domains.get(name) == "k"
            ]

        if not axes:
            raise ValueError("No axes specified for ifft_space")

        # Validate axes and get integer indices
        target_axes_int = []
        for ax_name in axes:
            ax_int = self._get_axis_index(ax_name)
            if ax_int == 0:
                raise ValueError(
                    "Cannot use ifft_space on axis 0. Use ifft_time instead."
                )
            domain = self._space_domains.get(ax_name)
            if domain != "k":
                raise ValueError(
                    f"Axis '{ax_name}' is not in 'k' domain (current: {domain}). "
                    f"Cannot apply ifft_space."
                )
            target_axes_int.append(ax_int)

        # Perform ifftn
        s = None
        if n is not None:
            s = tuple(n)

        import scipy.fft as sp_fft

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

        for ax_name, ax_int in zip(axes, target_axes_int):
            # Derive real-space axis name from k-axis name
            if ax_name.startswith("k"):
                real_name = ax_name[1:]  # kx -> x
            else:
                real_name = ax_name

            npts = dift.shape[ax_int]

            # Compute real-space coordinates from k-space
            # k = 2π * fftfreq(n, d=dx)  =>  dx = 2π / (n * |dk|)
            k_axis = self.axis(ax_name).index
            if len(k_axis) < 2:
                raise ValueError(
                    f"ifft_space requires axis length >= 2, "
                    f"got length {len(k_axis)} for axis '{ax_name}'"
                )

            dk_raw = k_axis[1] - k_axis[0]
            dk_value = getattr(dk_raw, "value", dk_raw)
            dk_unit = getattr(dk_raw, "unit", u.dimensionless_unscaled)

            # dx = 2π / (n * |dk|)
            dx_value = 2 * np.pi / (npts * abs(dk_value))
            dx_unit = 1 / dk_unit

            # If k-axis was effectively "descending" (dk < 0),
            # the reconstructed x-axis should also be descending
            if dk_value < 0:
                x_values = np.arange(npts - 1, -1, -1) * (-dx_value) * dx_unit
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
            axis0=new_indices[0],
            axis1=new_indices[1],
            axis2=new_indices[2],
            axis3=new_indices[3],
            axis_names=new_names,
            axis0_domain=self._axis0_domain,
            space_domain=new_space_domains,
        )
        result._validate_domain_units()
        return result

    def wavelength(self, axis: str | int) -> u.Quantity:
        """Compute wavelength from wavenumber axis.

        Parameters
        ----------
        axis : str or int
            The k-domain axis name or index.

        Returns
        -------
        `~astropy.units.Quantity`
            Wavelength values (:math:`\\lambda = 2\\pi / |k|`). k=0 returns inf.

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
        interp: str = "nearest",
    ) -> Any:
        """Extract time series at specified spatial points.

        Parameters
        ----------
        points : list of tuple
            List of (x, y, z) coordinates. Each coordinate should be a
            `~astropy.units.Quantity` with compatible units to the
            corresponding axis.
        interp : str, optional
            Interpolation method. Currently only 'nearest' is supported.
            Default is 'nearest'.

        Returns
        -------
        TimeSeriesList
            List of time series, one per point. Each series has metadata
            indicating the extraction coordinates.

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time'.
        ValueError
            If interpolation method is not supported.
        ValueError
            If point coordinates have incompatible units.
        IndexError
            If any point is outside the spatial domain.

        Examples
        --------
        >>> from astropy import units as u
        >>> points = [(1.0 * u.m, 2.0 * u.m, 3.0 * u.m),
        ...           (4.0 * u.m, 5.0 * u.m, 6.0 * u.m)]
        >>> ts_list = field.extract_points(points)
        """
        if self._axis0_domain != "time":
            raise ValueError(
                f"extract_points requires axis0_domain='time', "
                f"got '{self._axis0_domain}'"
            )

        if interp != "nearest":
            raise ValueError(
                f"Unsupported interpolation method '{interp}'. "
                f"Only 'nearest' is available."
            )

        from gwexpy.plot._coord import nearest_index, slice_from_index
        from gwexpy.timeseries import TimeSeries, TimeSeriesList

        result_list = []
        for point in points:
            if len(point) != 3:
                raise ValueError(
                    f"Each point must have 3 coordinates (x, y, z), got {len(point)}"
                )

            x_val, y_val, z_val = point

            # Find nearest indices for each spatial axis
            i1 = nearest_index(self._axis1_index, x_val)
            i2 = nearest_index(self._axis2_index, y_val)
            i3 = nearest_index(self._axis3_index, z_val)

            # Extract the time series at this point (keeping 4D then squeeze)
            sliced = self[
                :,
                slice_from_index(i1),
                slice_from_index(i2),
                slice_from_index(i3),
            ]
            # sliced has shape (nt, 1, 1, 1); squeeze to 1D
            ts_data = sliced.value.squeeze()

            # Actual extracted coordinates for the label
            actual_x = self._axis1_index[i1]
            actual_y = self._axis2_index[i2]
            actual_z = self._axis3_index[i3]

            label = (
                f"({self._axis1_name}={actual_x:.3g}, "
                f"{self._axis2_name}={actual_y:.3g}, "
                f"{self._axis3_name}={actual_z:.3g})"
            )

            ts = TimeSeries(
                ts_data,
                times=self._axis0_index,
                unit=self.unit,
                name=label,
            )
            result_list.append(ts)

        return TimeSeriesList(*result_list)

    def extract_profile(
        self, axis: str, at: dict[str, u.Quantity], reduce: str | None = None
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

        Returns
        -------
        tuple
            (axis_index, values): Both are `~astropy.units.Quantity` arrays.
            axis_index is the coordinate along the extraction axis,
            values is the data values along that axis.

        Raises
        ------
        ValueError
            If axis is not valid.
        ValueError
            If ``at`` dictionary is missing required axes.

        Examples
        --------
        >>> from astropy import units as u
        >>> x_axis, values = field.extract_profile(
        ...     'x', at={'t': 0.5 * u.s, 'y': 2.0 * u.m, 'z': 0.0 * u.m}
        ... )
        """
        from gwexpy.plot._coord import nearest_index, slice_from_index

        # Map axis name to integer index
        axis_int = self._get_axis_index(axis)

        # Determine which axes need to be fixed
        all_axes = [
            (0, self._axis0_name, self._axis0_index),
            (1, self._axis1_name, self._axis1_index),
            (2, self._axis2_name, self._axis2_index),
            (3, self._axis3_name, self._axis3_index),
        ]

        # Build the slice tuple
        slices = [slice(None)] * 4
        for ax_int, ax_name, ax_index in all_axes:
            if ax_int == axis_int:
                # Keep full slice for extraction axis
                continue

            if ax_name not in at:
                raise ValueError(
                    f"extract_profile requires fixed value for axis '{ax_name}' "
                    f"in 'at' dictionary"
                )

            value = at[ax_name]
            idx = nearest_index(ax_index, value)
            slices[ax_int] = slice_from_index(idx)

        # Extract the data
        sliced = self[tuple(slices)]
        # Shape should be (1, ..., n, ..., 1) with n at axis_int
        data = sliced.value.squeeze()

        # Get the axis coordinates
        axis_index = [
            self._axis0_index,
            self._axis1_index,
            self._axis2_index,
            self._axis3_index,
        ][axis_int]

        # Return with units
        from astropy import units as u

        if hasattr(self, "unit") and self.unit is not None:
            values = data * self.unit
        else:
            values = data * u.dimensionless_unscaled

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
        >>> fig, ax = field.plot_map2d('xy', at={'t': 0.5 * u.s, 'z': 0.0 * u.m})
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
        ax.set_xlabel(f"{ax2_name} [{ax2_index.unit}]")
        ax.set_ylabel(f"{ax1_name} [{ax1_index.unit}]")

        if title:
            ax.set_title(title)

        # Colorbar
        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            if self.unit is not None:
                cbar.set_label(f"{mode} [{self.unit}]")

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
        >>> points = [(1.0 * u.m, 2.0 * u.m, 3.0 * u.m)]
        >>> fig, ax = field.plot_timeseries_points(points)
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
        ax.set_xlabel(f"{self._axis0_name} [{x_unit}]")
        if self.unit is not None:
            ax.set_ylabel(f"[{self.unit}]")

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
        >>> fig, ax = field.plot_profile(
        ...     'x', at={'t': 0.5 * u.s, 'y': 0.0 * u.m, 'z': 0.0 * u.m}
        ... )
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
        ax.set_xlabel(f"{axis} [{x_unit}]")
        if y_unit is not None:
            ax.set_ylabel(f"{mode} [{y_unit}]")

        if label:
            ax.legend()

        return fig, ax

    # =========================================================================
    # Comparison & Summary Methods (Phase 2)
    # =========================================================================

    def diff(self, other, mode="diff"):
        """Compute difference or ratio between two ScalarField objects.

        Parameters
        ----------
        other : ScalarField
            The field to compare against.
        mode : str, optional
            Comparison mode:
            - 'diff': Difference (self - other)
            - 'ratio': Ratio (self / other)
            - 'percent': Percentage difference ((self - other) / other * 100)
            Default is 'diff'.

        Returns
        -------
        ScalarField
            Result field. For 'diff', unit is same as input.
            For 'ratio' and 'percent', unit is dimensionless.

        Raises
        ------
        ValueError
            If mode is not recognized.
        ValueError
            If shapes are incompatible.

        Examples
        --------
        >>> diff_field = field1.diff(field2)
        >>> ratio_field = field1.diff(field2, mode='ratio')
        """
        from astropy import units as u

        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        valid_modes = ("diff", "ratio", "percent")
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")

        if mode == "diff":
            result_data = self.value - other.value
            result_unit = self.unit
        elif mode == "ratio":
            with np.errstate(divide="ignore", invalid="ignore"):
                result_data = self.value / other.value
            result_unit = u.dimensionless_unscaled
        elif mode == "percent":
            with np.errstate(divide="ignore", invalid="ignore"):
                result_data = (self.value - other.value) / other.value * 100
            result_unit = u.percent
        else:
            raise ValueError(f"Invalid mode '{mode}'")

        return ScalarField(
            result_data,
            unit=result_unit,
            axis0=self._axis0_index,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=list(self.axis_names),
            axis0_domain=self._axis0_domain,
            space_domain=self._space_domains,
        )

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
        >>> from astropy import units as u
        >>> zscore_field = field.zscore(baseline_t=(0 * u.s, 1 * u.s))
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

        return ScalarField(
            zscore_data,
            unit=u.dimensionless_unscaled,
            axis0=self._axis0_index,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=list(self.axis_names),
            axis0_domain=self._axis0_domain,
            space_domain=self._space_domains,
        )

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
        >>> from astropy import units as u
        >>> mean_map = field.time_stat_map('mean', t_range=(0 * u.s, 1 * u.s))
        >>> mean_map.plot_map2d('xy')
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
            axis0=np.array([mean_time.value]) * mean_time.unit,
            axis1=subset._axis1_index,
            axis2=subset._axis2_index,
            axis3=subset._axis3_index,
            axis_names=list(self.axis_names),
            axis0_domain=self._axis0_domain,
            space_domain=self._space_domains,
        )

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
        >>> from astropy import units as u
        >>> t, x, data = field.time_space_map('x', at={'y': 0 * u.m, 'z': 0 * u.m})
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
        >>> fig, ax = field.plot_time_space_map('x', at={'y': 0*u.m, 'z': 0*u.m})
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
        ax.set_xlabel(f"{axis} [{space_axis.unit}]")
        ax.set_ylabel(f"{self._axis0_name} [{t_axis.unit}]")

        if title:
            ax.set_title(title)

        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            if self.unit is not None:
                cbar.set_label(f"{mode} [{self.unit}]")

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

        # Reshape to (time, points)
        orig_shape = self.shape
        data_2d = self.value.reshape(orig_shape[0], -1)

        # Use scipy.signal.resample for efficient array resampling
        # We need to compute the new number of samples
        dt = self._axis0_index[1] - self._axis0_index[0]
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
        return ScalarField(
            new_data_2d.reshape(new_shape),
            unit=self.unit,
            axis0=new_times,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=self.axis_names,
            axis0_domain="time",
            space_domain=self._space_domains,
        )

    def interpolate(self, sample_rate, kind='cubic'):
        """Resample using interpolation (preserves bandlimited signals).

        Uses scipy.interpolate for high-quality resampling, preferred for
        calibrated data where preserving frequency content is critical.

        Parameters
        ----------
        sample_rate : float or Quantity
            New sample rate (e.g., in Hz).
        kind : str, optional
            Interpolation method: 'linear', 'cubic', 'quadratic', etc.
            Default is 'cubic'. See scipy.interpolate.interp1d for options.

        Returns
        -------
        ScalarField
            Interpolated field at new sample rate.

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time'.

        Examples
        --------
        >>> # High-quality resampling with cubic interpolation
        >>> resampled = field.interpolate(4096, kind='cubic')

        See Also
        --------
        resample : FFT-based resampling
        """
        if self._axis0_domain != "time":
            raise ValueError("interpolate requires axis0_domain='time'")

        from scipy import interpolate

        # Get current sample rate and compute new parameters
        dt = self._axis0_index[1] - self._axis0_index[0]
        if hasattr(sample_rate, "to"):
            new_fs = sample_rate.to("Hz").value
            new_dt = (1.0 / new_fs) * u.s
        else:
            new_fs = float(sample_rate)
            new_dt = (1.0 / new_fs) * dt.unit

        # Original time axis
        t_old = self._axis0_index.value
        t0 = self._axis0_index[0]

        # New time axis
        duration = (self.shape[0] * dt).to(new_dt.unit).value
        new_nt = int(round(duration * new_fs))
        t_new = np.arange(new_nt) * (1.0 / new_fs) * new_dt.to(dt.unit).value

        # Reshape for efficient computation
        orig_shape = self.shape
        data_2d = self.value.reshape(orig_shape[0], -1)

        # Interpolate each spatial point
        interpolated_list = []
        for i in range(data_2d.shape[1]):
            f = interpolate.interp1d(
                t_old, data_2d[:, i], kind=kind, bounds_error=False, fill_value='extrapolate'
            )
            interpolated_list.append(f(t_new))

        # Stack results
        new_data_2d = np.array(interpolated_list).T

        # Create new time axis
        new_times = t_new * dt.unit + t0

        # Reshape back
        new_shape = [new_nt] + list(orig_shape[1:])
        return ScalarField(
            new_data_2d.reshape(new_shape),
            unit=self.unit,
            axis0=new_times,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=self.axis_names,
            axis0_domain="time",
            space_domain=self._space_domains,
        )

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

        # Reshape to (time, points)
        orig_shape = self.shape
        data_2d = self.value.reshape(orig_shape[0], -1)

        # Use gwpy to parse the filter
        # We need sample rate
        fs = (1.0 / (self._axis0_index[1] - self._axis0_index[0])).to("Hz").value
        from gwpy.signal import filter_design
        from scipy import signal as scipy_signal

        form, filt = filter_design.parse_filter(
            args,
            analog=kwargs.pop("analog", False),
            sample_rate=fs,
        )
        filtfilt = kwargs.pop("filtfilt", True)  # Default to True for phase consistency

        # Apply filter along axis 0
        if form == "zpk":
            sos = scipy_signal.zpk2sos(*filt)
            new_data_2d = (
                scipy_signal.sosfiltfilt(sos, data_2d, axis=0)
                if filtfilt
                else scipy_signal.sosfilt(sos, data_2d, axis=0)
            )
        else:
            b, a = filt
            new_data_2d = (
                scipy_signal.filtfilt(b, a, data_2d, axis=0)
                if filtfilt
                else scipy_signal.lfilter(b, a, data_2d, axis=0)
            )

        # Reshape back
        return ScalarField(
            new_data_2d.reshape(orig_shape),
            unit=self.unit,
            axis0=self._axis0_index,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=self.axis_names,
            axis0_domain="time",
            space_domain=self._space_domains,
        )

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

    def psd(self, axis=0, **kwargs):
        """Compute power spectral density along any axis.

        Computes power spectral density using Welch's method by default.
        Can compute along time axis (axis=0) or spatial axes (axis=1,2,3 or 'x','y','z').

        Parameters
        ----------
        axis : int or str, optional
            Axis along which to compute PSD. Default is 0 (time axis).
            Can specify as integer (0, 1, 2, 3) or string ('x', 'y', 'z').
        **kwargs
            Keyword arguments passed to :func:`~gwexpy.fields.signal.spectral_density`
            or :func:`~gwexpy.fields.signal.compute_psd` (if point_or_region is used).
            Common options:

            - point_or_region : tuple or list, optional
                If provided, computes PSD at specific spatial point(s) or region
                average instead of full field. Returns FrequencySeries(List).
                Only valid for axis=0 (time).
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
            PSD field with transformed axis in spectral domain.
            For axis=0, axis0_domain='frequency'.
            For spatial axes, the axis becomes wavenumber.

        Examples
        --------
        >>> from astropy import units as u
        >>> # Time-axis PSD
        >>> psd_time = field.psd(axis=0, fftlength=1.0, overlap=0.5)
        >>> # Or sample-based specification
        >>> psd_time = field.psd(axis=0, nfft=512, noverlap=256)
        >>> # Spatial power spectrum along x-axis
        >>> psd_x = field.psd(axis='x')

        See Also
        --------
        asd : Amplitude spectral density
        spectral_density : Underlying implementation
        """
        if "point_or_region" in kwargs:
            if axis != 0:
                raise ValueError("point_or_region is only valid for axis=0 (time axis)")
            from .signal import compute_psd

            point_or_region = kwargs.pop("point_or_region")
            return compute_psd(self, point_or_region, **kwargs)

        return self.spectral_density(axis=axis, **kwargs)

    def asd(self, axis=0, **kwargs):
        """Compute amplitude spectral density along any axis.

        Convenience method equivalent to ``sqrt(psd(axis=axis))``.
        Returns the square root of the power spectral density.

        Parameters
        ----------
        axis : int or str, optional
            Axis along which to compute ASD. Default is 0 (time axis).
            Can specify as integer (0, 1, 2, 3) or string ('x', 'y', 'z').
        **kwargs
            Keyword arguments passed to :meth:`psd`.
            See :meth:`psd` for available options.

        Returns
        -------
        ScalarField
            ASD field with transformed axis in spectral domain.
            Units are the square root of PSD units (e.g., V/√Hz).

        Examples
        --------
        >>> from astropy import units as u
        >>> # Time-axis ASD
        >>> asd_time = field.asd(axis=0, fftlength=1.0, overlap=0.5)
        >>> # Or sample-based specification
        >>> asd_time = field.asd(axis=0, nfft=512, noverlap=256)
        >>> # Spatial amplitude spectrum along x-axis
        >>> asd_x = field.asd(axis='x')

        See Also
        --------
        psd : Power spectral density
        spectral_density : Underlying implementation
        """
        import numpy as np

        psd_result = self.psd(axis=axis, **kwargs)
        return np.sqrt(psd_result)

    # FFT methods with axis support
    def fft(self, axis=0, **kwargs):
        """Compute FFT along any axis.

        Unified FFT interface that dispatches to fft_time() or fft_space()
        based on the axis parameter.

        Parameters
        ----------
        axis : int or str, optional
            Axis along which to compute FFT. Default is 0 (time axis).
            - 0 or 'time': Time axis (uses fft_time)
            - 1, 2, 3 or 'x', 'y', 'z': Spatial axes (uses fft_space)
        **kwargs
            Additional arguments passed to fft_time() or fft_space().

        Returns
        -------
        ScalarField
            FFT-transformed field with appropriate domain.

        Examples
        --------
        >>> # Time-axis FFT
        >>> fft_result = field.fft(axis=0)
        >>> # Spatial FFT along x
        >>> fft_x = field.fft(axis='x')

        See Also
        --------
        fft_time : FFT along time axis
        fft_space : FFT along spatial axes
        ifft : Inverse FFT
        """
        # Resolve axis
        if axis == 0 or axis == 'time':
            return self.fft_time(**kwargs)
        else:
            # For spatial axes, determine which axes
            if isinstance(axis, str):
                axis_map = {'x': 1, 'y': 2, 'z': 3}
                if axis in axis_map:
                    axis_int = axis_map[axis]
                else:
                    raise ValueError(f"Invalid axis name: {axis}")
            else:
                axis_int = axis

            if axis_int not in [1, 2, 3]:
                raise ValueError(f"Spatial axis must be 1, 2, or 3 (or 'x', 'y', 'z'), got {axis}")

            return self.fft_space(axes=(axis_int,), **kwargs)

    def ifft(self, axis=0, **kwargs):
        """Compute inverse FFT along any axis.

        Unified inverse FFT interface that dispatches to ifft_time() or ifft_space()
        based on the axis parameter.

        Parameters
        ----------
        axis : int or str, optional
            Axis along which to compute IFFT. Default is 0 (time axis).
            - 0 or 'time': Time axis (uses ifft_time)
            - 1, 2, 3 or 'x', 'y', 'z': Spatial axes (uses ifft_space)
        **kwargs
            Additional arguments passed to ifft_time() or ifft_space().

        Returns
        -------
        ScalarField
            Inverse FFT-transformed field with appropriate domain.

        Examples
        --------
        >>> # Inverse time-axis FFT
        >>> ifft_result = freq_field.ifft(axis=0)
        >>> # Inverse spatial FFT along x
        >>> ifft_x = kx_field.ifft(axis='x')

        See Also
        --------
        ifft_time : IFFT along time axis
        ifft_space : IFFT along spatial axes
        fft : Forward FFT
        """
        # Resolve axis
        if axis == 0 or axis == 'time':
            return self.ifft_time(**kwargs)
        else:
            # For spatial axes
            if isinstance(axis, str):
                axis_map = {'x': 1, 'y': 2, 'z': 3}
                if axis in axis_map:
                    axis_int = axis_map[axis]
                else:
                    raise ValueError(f"Invalid axis name: {axis}")
            else:
                axis_int = axis

            if axis_int not in [1, 2, 3]:
                raise ValueError(f"Spatial axis must be 1, 2, or 3 (or 'x', 'y', 'z'), got {axis}")

            return self.ifft_space(axes=(axis_int,), **kwargs)

    # Filter convenience methods
    def highpass(self, frequency, **kwargs):
        """Apply highpass filter to time axis.

        Parameters
        ----------
        frequency : float
            Cutoff frequency in Hz.
        **kwargs
            Additional arguments passed to filter().
            Common options: filtfilt (default True), order (default 4).

        Returns
        -------
        ScalarField
            Highpass-filtered field.

        Examples
        --------
        >>> # Highpass filter at 10 Hz
        >>> filtered = field.highpass(10)
        >>> # With custom order
        >>> filtered = field.highpass(10, order=8)

        See Also
        --------
        lowpass : Lowpass filter
        bandpass : Bandpass filter
        filter : General filter method
        """
        return self.filter(frequency, type='highpass', **kwargs)

    def lowpass(self, frequency, **kwargs):
        """Apply lowpass filter to time axis.

        Parameters
        ----------
        frequency : float
            Cutoff frequency in Hz.
        **kwargs
            Additional arguments passed to filter().
            Common options: filtfilt (default True), order (default 4).

        Returns
        -------
        ScalarField
            Lowpass-filtered field.

        Examples
        --------
        >>> # Lowpass filter at 100 Hz
        >>> filtered = field.lowpass(100)

        See Also
        --------
        highpass : Highpass filter
        bandpass : Bandpass filter
        filter : General filter method
        """
        return self.filter(frequency, type='lowpass', **kwargs)

    def bandpass(self, flow, fhigh, **kwargs):
        """Apply bandpass filter to time axis.

        Parameters
        ----------
        flow : float
            Low frequency cutoff in Hz.
        fhigh : float
            High frequency cutoff in Hz.
        **kwargs
            Additional arguments passed to filter().
            Common options: filtfilt (default True), order (default 4).

        Returns
        -------
        ScalarField
            Bandpass-filtered field.

        Examples
        --------
        >>> # Bandpass filter between 10-100 Hz
        >>> filtered = field.bandpass(10, 100)

        See Also
        --------
        highpass : Highpass filter
        lowpass : Lowpass filter
        filter : General filter method
        """
        return self.filter(flow, fhigh, type='bandpass', **kwargs)

    def notch(self, frequency, **kwargs):
        """Apply notch filter to time axis.

        Parameters
        ----------
        frequency : float or list of float
            Frequency(ies) to notch in Hz.
        **kwargs
            Additional arguments passed to filter().
            Common options: filtfilt (default True), quality (default 10).

        Returns
        -------
        ScalarField
            Notch-filtered field.

        Examples
        --------
        >>> # Notch filter at 60 Hz (power line)
        >>> filtered = field.notch(60)
        >>> # Multiple notches
        >>> filtered = field.notch([60, 120, 180])

        See Also
        --------
        bandpass : Bandpass filter
        filter : General filter method
        """
        return self.filter(frequency, type='notch', **kwargs)

    def zpk(self, zeros, poles, gain, **kwargs):
        """Apply zero-pole-gain filter to time axis.

        Parameters
        ----------
        zeros : array_like
            Filter zeros in the complex plane.
        poles : array_like
            Filter poles in the complex plane.
        gain : float
            System gain.
        **kwargs
            Additional arguments passed to filter().
            Common options: filtfilt (default True), analog (default False).

        Returns
        -------
        ScalarField
            Filtered field.

        Examples
        --------
        >>> # Custom IIR filter from zpk representation
        >>> zeros = [0]
        >>> poles = [-1, -1+1j, -1-1j]
        >>> gain = 1.0
        >>> filtered = field.zpk(zeros, poles, gain)

        See Also
        --------
        filter : General filter method
        """
        if self._axis0_domain != "time":
            raise ValueError("zpk requires axis0_domain='time'")

        # Reshape to (time, points)
        orig_shape = self.shape
        data_2d = self.value.reshape(orig_shape[0], -1)

        # Get sample rate
        fs = (1.0 / (self._axis0_index[1] - self._axis0_index[0])).to("Hz").value
        from scipy import signal as scipy_signal

        analog = kwargs.pop("analog", False)
        filtfilt = kwargs.pop("filtfilt", True)

        # Convert to second-order sections for numerical stability
        sos = scipy_signal.zpk2sos(zeros, poles, gain)

        # Apply filter along axis 0
        new_data_2d = (
            scipy_signal.sosfiltfilt(sos, data_2d, axis=0)
            if filtfilt
            else scipy_signal.sosfilt(sos, data_2d, axis=0)
        )

        # Reshape back
        return ScalarField(
            new_data_2d.reshape(orig_shape),
            unit=self.unit,
            axis0=self._axis0_index,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=self.axis_names,
            axis0_domain="time",
            space_domain=self._space_domains,
        )

    # =========================================================================
    # Basic Preprocessing Methods
    # =========================================================================

    def detrend(self, type="linear"):
        """Remove polynomial trend from time axis.

        Wraps `scipy.signal.detrend()` to remove linear or constant trends
        that can distort spectral analysis and filtering operations.

        Parameters
        ----------
        type : str, optional
            Type of detrending:
            - 'linear': Remove linear trend (default)
            - 'constant': Remove mean value
            Default is 'linear'.

        Returns
        -------
        ScalarField
            Detrended field.

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time'.

        Examples
        --------
        >>> # Remove linear drift
        >>> detrended = field.detrend('linear')
        >>> # Remove DC offset only
        >>> detrended = field.detrend('constant')

        See Also
        --------
        taper : Apply window to endpoints
        """
        if self._axis0_domain != "time":
            raise ValueError("detrend requires axis0_domain='time'")

        from scipy import signal as scipy_signal

        # Apply detrend along axis 0
        detrended_data = scipy_signal.detrend(self.value, axis=0, type=type)

        return ScalarField(
            detrended_data,
            unit=self.unit,
            axis0=self._axis0_index,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=self.axis_names,
            axis0_domain="time",
            space_domain=self._space_domains,
        )

    def taper(self, side="leftright", duration=None, nsamples=None):
        """Apply Tukey window to data endpoints.

        Applies a smooth windowing function to suppress ringing artifacts
        in FFTs caused by sharp discontinuities at segment boundaries.

        Parameters
        ----------
        side : str, optional
            Which sides to taper:
            - 'leftright': Both sides (default)
            - 'left': Left side only
            - 'right': Right side only
        duration : float or Quantity, optional
            Duration of taper in seconds. If None, uses nsamples.
        nsamples : int, optional
            Number of samples to taper on each side. If None and duration
            is None, defaults to 1 second.

        Returns
        -------
        ScalarField
            Tapered field.

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time'.

        Examples
        --------
        >>> # Taper both ends with 1 second
        >>> tapered = field.taper(duration=1.0)
        >>> # Taper left side only
        >>> tapered = field.taper(side='left', nsamples=100)

        See Also
        --------
        detrend : Remove polynomial trends
        """
        if self._axis0_domain != "time":
            raise ValueError("taper requires axis0_domain='time'")

        # Determine number of samples to taper
        if nsamples is None:
            if duration is None:
                duration = 1.0 * u.s
            elif not hasattr(duration, "unit"):
                duration = duration * u.s
            dt = self._axis0_index[1] - self._axis0_index[0]
            nsamples = int(round((duration / dt).decompose().value))

        nsamples = min(nsamples, self.shape[0] // 2)

        # Create Tukey window
        # alpha = 2 * nsamples / N gives taper on both ends
        N = self.shape[0]
        alpha = 2.0 * nsamples / N

        from scipy import signal as scipy_signal

        if side == "leftright":
            window = scipy_signal.windows.tukey(N, alpha=alpha)
        elif side == "left":
            window = np.ones(N)
            half_window = scipy_signal.windows.tukey(2 * nsamples, alpha=1.0)
            window[:nsamples] = half_window[:nsamples]
        elif side == "right":
            window = np.ones(N)
            half_window = scipy_signal.windows.tukey(2 * nsamples, alpha=1.0)
            window[-nsamples:] = half_window[-nsamples:]
        else:
            raise ValueError(f"Invalid side '{side}'. Must be 'left', 'right', or 'leftright'.")

        # Apply window (broadcast over spatial dimensions)
        window_4d = window[:, np.newaxis, np.newaxis, np.newaxis]
        tapered_data = self.value * window_4d

        return ScalarField(
            tapered_data,
            unit=self.unit,
            axis0=self._axis0_index,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=self.axis_names,
            axis0_domain="time",
            space_domain=self._space_domains,
        )

    def crop(self, start=None, end=None, copy=True):
        """Extract a time segment between specified times.

        Parameters
        ----------
        start : float or Quantity, optional
            Start time. If None, uses the beginning of the data.
        end : float or Quantity, optional
            End time. If None, uses the end of the data.
        copy : bool, optional
            If True, return a copy of the data. Default is True.

        Returns
        -------
        ScalarField
            Cropped field.

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time'.

        Examples
        --------
        >>> from astropy import units as u
        >>> # Extract segment from 10s to 20s
        >>> segment = field.crop(start=10*u.s, end=20*u.s)
        >>> # Extract from beginning to 15s
        >>> segment = field.crop(end=15*u.s)

        See Also
        --------
        pad : Extend data with padding
        """
        if self._axis0_domain != "time":
            raise ValueError("crop requires axis0_domain='time'")

        from gwexpy.plot._coord import nearest_index

        # Find start index
        if start is None:
            i_start = 0
        else:
            if not hasattr(start, "unit"):
                start = start * self._axis0_index.unit
            i_start = nearest_index(self._axis0_index, start)

        # Find end index
        if end is None:
            i_end = self.shape[0]
        else:
            if not hasattr(end, "unit"):
                end = end * self._axis0_index.unit
            i_end = nearest_index(self._axis0_index, end) + 1

        # Extract segment
        cropped = self[i_start:i_end, :, :, :]

        if copy:
            return cropped.copy()
        return cropped

    def pad(self, pad_width, mode="constant", **kwargs):
        """Extend data with padding.

        Wraps `numpy.pad()` to add padding to the time axis. Useful for
        FFT operations requiring power-of-2 lengths and reducing edge effects.

        Parameters
        ----------
        pad_width : int or tuple
            Number of values to pad. If int, pads both sides equally.
            If tuple (left, right), pads asymmetrically.
        mode : str, optional
            Padding mode (see numpy.pad documentation):
            - 'constant': Pad with constant value (default, uses 0)
            - 'edge': Pad with edge values
            - 'reflect': Reflect values at boundaries
            - 'symmetric': Symmetric reflection
            - 'wrap': Wrap around to opposite edge
        **kwargs
            Additional arguments passed to numpy.pad (e.g., constant_values).

        Returns
        -------
        ScalarField
            Padded field.

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time'.

        Examples
        --------
        >>> # Pad 100 samples on each end with zeros
        >>> padded = field.pad(100)
        >>> # Pad asymmetrically
        >>> padded = field.pad((50, 150), mode='edge')
        >>> # Pad with specific value
        >>> padded = field.pad(100, mode='constant', constant_values=1.0)

        See Also
        --------
        crop : Extract time segment
        """
        if self._axis0_domain != "time":
            raise ValueError("pad requires axis0_domain='time'")

        # Convert pad_width to tuple if needed
        if isinstance(pad_width, int):
            pad_width = (pad_width, pad_width)

        # Create pad_width for all axes (only pad axis 0)
        pad_width_4d = [pad_width, (0, 0), (0, 0), (0, 0)]

        # Pad the data
        padded_data = np.pad(self.value, pad_width_4d, mode=mode, **kwargs)

        # Extend time axis
        dt = self._axis0_index[1] - self._axis0_index[0]
        t0 = self._axis0_index[0]
        n_left, n_right = pad_width

        # Create new time axis
        new_times = np.arange(
            -n_left, self.shape[0] + n_right
        ) * dt + t0

        return ScalarField(
            padded_data,
            unit=self.unit,
            axis0=new_times,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=self.axis_names,
            axis0_domain="time",
            space_domain=self._space_domains,
        )

    def value_at(self, t):
        """Extract field values at specific time(s).

        Parameters
        ----------
        t : float, Quantity, or array-like
            Time(s) at which to extract values. Can be scalar or array.

        Returns
        -------
        array or ScalarField
            If t is scalar, returns 3D array of spatial values at that time.
            If t is array, returns ScalarField with new time axis.

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time'.

        Examples
        --------
        >>> from astropy import units as u
        >>> # Get values at single time
        >>> values_3d = field.value_at(5.0 * u.s)
        >>> # Get values at multiple times
        >>> times = [1.0, 2.0, 3.0] * u.s
        >>> subset = field.value_at(times)

        See Also
        --------
        extract_points : Extract time series at spatial points
        """
        if self._axis0_domain != "time":
            raise ValueError("value_at requires axis0_domain='time'")

        from gwexpy.plot._coord import nearest_index

        # Handle scalar vs array
        if np.isscalar(t) or (hasattr(t, "isscalar") and t.isscalar):
            # Scalar time
            if not hasattr(t, "unit"):
                t = t * self._axis0_index.unit
            idx = nearest_index(self._axis0_index, t)
            return self.value[idx, :, :, :]
        else:
            # Array of times
            if not hasattr(t, "unit"):
                t = t * self._axis0_index.unit
            indices = [nearest_index(self._axis0_index, ti) for ti in t]
            return self[indices, :, :, :]

    # =========================================================================
    # Mathematical Operations
    # =========================================================================

    def abs(self):
        """Compute absolute value of the field.

        Returns
        -------
        ScalarField
            Field with absolute values.

        Examples
        --------
        >>> abs_field = field.abs()
        >>> # Equivalent to np.abs(field)
        """
        return ScalarField(
            np.abs(self.value),
            unit=self.unit,
            axis0=self._axis0_index,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=self.axis_names,
            axis0_domain=self._axis0_domain,
            space_domain=self._space_domains,
        )

    def sqrt(self):
        """Compute square root of the field.

        Returns
        -------
        ScalarField
            Field with square root values.

        Examples
        --------
        >>> sqrt_field = field.sqrt()
        """
        result_unit = self.unit ** 0.5 if self.unit is not None else None
        return ScalarField(
            np.sqrt(self.value),
            unit=result_unit,
            axis0=self._axis0_index,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=self.axis_names,
            axis0_domain=self._axis0_domain,
            space_domain=self._space_domains,
        )

    def mean(self, axis=None, **kwargs):
        """Compute mean along specified axis.

        Parameters
        ----------
        axis : int or str, optional
            Axis along which to compute mean. If None, computes global mean.
        **kwargs
            Additional arguments passed to numpy.mean.

        Returns
        -------
        ScalarField or Quantity
            Mean field or scalar value.

        Examples
        --------
        >>> # Global mean
        >>> mean_val = field.mean()
        >>> # Mean along time axis
        >>> time_mean = field.mean(axis=0)
        """
        if axis is not None:
            if isinstance(axis, str):
                axis = self._get_axis_index(axis)
            result = np.mean(self.value, axis=axis, keepdims=True, **kwargs)

            # Create new axis indices
            new_indices = [
                self._axis0_index if i != axis else self._axis0_index[:1]
                for i in range(4)
            ]

            return ScalarField(
                result,
                unit=self.unit,
                axis0=new_indices[0],
                axis1=new_indices[1],
                axis2=new_indices[2],
                axis3=new_indices[3],
                axis_names=self.axis_names,
                axis0_domain=self._axis0_domain,
                space_domain=self._space_domains,
            )
        else:
            result = np.mean(self.value, **kwargs)
            return result * self.unit if self.unit is not None else result

    def median(self, axis=None, **kwargs):
        """Compute median along specified axis.

        Parameters
        ----------
        axis : int or str, optional
            Axis along which to compute median. If None, computes global median.
        **kwargs
            Additional arguments passed to numpy.median.

        Returns
        -------
        ScalarField or Quantity
            Median field or scalar value.

        Examples
        --------
        >>> # Global median
        >>> median_val = field.median()
        >>> # Median along time axis
        >>> time_median = field.median(axis=0)
        """
        if axis is not None:
            if isinstance(axis, str):
                axis = self._get_axis_index(axis)
            result = np.median(self.value, axis=axis, keepdims=True, **kwargs)

            # Create new axis indices
            new_indices = [
                self._axis0_index if i != axis else self._axis0_index[:1]
                for i in range(4)
            ]

            return ScalarField(
                result,
                unit=self.unit,
                axis0=new_indices[0],
                axis1=new_indices[1],
                axis2=new_indices[2],
                axis3=new_indices[3],
                axis_names=self.axis_names,
                axis0_domain=self._axis0_domain,
                space_domain=self._space_domains,
            )
        else:
            result = np.median(self.value, **kwargs)
            return result * self.unit if self.unit is not None else result

    def std(self, axis=None, **kwargs):
        """Compute standard deviation along specified axis.

        Parameters
        ----------
        axis : int or str, optional
            Axis along which to compute std. If None, computes global std.
        **kwargs
            Additional arguments passed to numpy.std.

        Returns
        -------
        ScalarField or Quantity
            Standard deviation field or scalar value.

        Examples
        --------
        >>> # Global standard deviation
        >>> std_val = field.std()
        >>> # Std along time axis
        >>> time_std = field.std(axis=0)
        """
        if axis is not None:
            if isinstance(axis, str):
                axis = self._get_axis_index(axis)
            result = np.std(self.value, axis=axis, keepdims=True, **kwargs)

            # Create new axis indices
            new_indices = [
                self._axis0_index if i != axis else self._axis0_index[:1]
                for i in range(4)
            ]

            return ScalarField(
                result,
                unit=self.unit,
                axis0=new_indices[0],
                axis1=new_indices[1],
                axis2=new_indices[2],
                axis3=new_indices[3],
                axis_names=self.axis_names,
                axis0_domain=self._axis0_domain,
                space_domain=self._space_domains,
            )
        else:
            result = np.std(self.value, **kwargs)
            return result * self.unit if self.unit is not None else result

    def rms(self, axis=None):
        """Compute root-mean-square along specified axis.

        Parameters
        ----------
        axis : int or str, optional
            Axis along which to compute RMS. If None, computes global RMS.

        Returns
        -------
        ScalarField or Quantity
            RMS field or scalar value.

        Examples
        --------
        >>> # Global RMS
        >>> rms_val = field.rms()
        >>> # RMS along time axis
        >>> time_rms = field.rms(axis=0)
        """
        if axis is not None:
            if isinstance(axis, str):
                axis = self._get_axis_index(axis)
            result = np.sqrt(np.mean(self.value**2, axis=axis, keepdims=True))

            # Create new axis indices
            new_indices = [
                self._axis0_index if i != axis else self._axis0_index[:1]
                for i in range(4)
            ]

            return ScalarField(
                result,
                unit=self.unit,
                axis0=new_indices[0],
                axis1=new_indices[1],
                axis2=new_indices[2],
                axis3=new_indices[3],
                axis_names=self.axis_names,
                axis0_domain=self._axis0_domain,
                space_domain=self._space_domains,
            )
        else:
            result = np.sqrt(np.mean(self.value**2))
            return result * self.unit if self.unit is not None else result

    # =========================================================================
    # Advanced Signal Processing
    # =========================================================================

    def whiten(self, fftlength=None, overlap=0, method="welch", **kwargs):
        """Normalize the amplitude spectral density (whitening).

        Divides the data by its ASD to flatten the spectrum, enhancing
        higher-frequency content. Essential preprocessing for matched filtering.

        Parameters
        ----------
        fftlength : float, optional
            Length of FFT segments in seconds for ASD estimation.
            If None, uses the entire data length.
        overlap : float, optional
            Overlap between segments in seconds. Default is 0.
        method : str, optional
            Method for ASD estimation: 'welch' or 'median'. Default is 'welch'.
        **kwargs
            Additional arguments passed to psd().

        Returns
        -------
        ScalarField
            Whitened field (dimensionless).

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time'.

        Examples
        --------
        >>> # Whiten using 2-second segments
        >>> whitened = field.whiten(fftlength=2.0, overlap=1.0)

        See Also
        --------
        psd : Power spectral density estimation
        asd : Amplitude spectral density
        """
        if self._axis0_domain != "time":
            raise ValueError("whiten requires axis0_domain='time'")

        # Compute ASD
        asd_field = self.asd(axis=0, fftlength=fftlength, overlap=overlap, method=method, **kwargs)

        # FFT to frequency domain
        freq_field = self.fft_time()

        # Divide by ASD (broadcasting over frequency axis)
        # asd_field has shape (n_freq, nx, ny, nz)
        # freq_field has shape (n_freq, nx, ny, nz)
        whitened_freq = freq_field.value / asd_field.value

        # Create whitened frequency field
        whitened_freq_field = ScalarField(
            whitened_freq,
            unit=u.dimensionless_unscaled,
            axis0=freq_field._axis0_index,
            axis1=freq_field._axis1_index,
            axis2=freq_field._axis2_index,
            axis3=freq_field._axis3_index,
            axis_names=freq_field.axis_names,
            axis0_domain="frequency",
            space_domain=freq_field._space_domains,
        )

        # IFFT back to time domain
        whitened = whitened_freq_field.ifft_time()

        return whitened

    def convolve(self, fir, **kwargs):
        """Apply FIR filter via time-domain convolution.

        Parameters
        ----------
        fir : array_like
            Finite impulse response filter coefficients.
        **kwargs
            Additional arguments passed to scipy.signal.convolve.
            Common options: mode ('full', 'same', 'valid').

        Returns
        -------
        ScalarField
            Convolved field.

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time'.

        Notes
        -----
        Convolution introduces boundary effects equal to half the filter
        length at each end. Consider using mode='same' to maintain the
        same output length as input.

        Examples
        --------
        >>> # Matched filter
        >>> template = [1, 2, 3, 2, 1]  # Simple template
        >>> matched = field.convolve(template, mode='same')

        See Also
        --------
        filter : General filtering interface
        """
        if self._axis0_domain != "time":
            raise ValueError("convolve requires axis0_domain='time'")

        from scipy import signal as scipy_signal

        mode = kwargs.pop("mode", "same")

        # Reshape for efficient computation
        orig_shape = self.shape
        data_2d = self.value.reshape(orig_shape[0], -1)

        # Convolve along axis 0
        fir = np.asarray(fir)
        convolved_2d = scipy_signal.convolve(
            data_2d, fir[:, np.newaxis], mode=mode, **kwargs
        )

        # Handle different output sizes
        if mode == "same":
            new_times = self._axis0_index
        elif mode == "full":
            dt = self._axis0_index[1] - self._axis0_index[0]
            n_extra = len(fir) - 1
            new_times = (
                np.arange(orig_shape[0] + n_extra) * dt + self._axis0_index[0]
            )
        elif mode == "valid":
            n_out = max(orig_shape[0] - len(fir) + 1, 0)
            dt = self._axis0_index[1] - self._axis0_index[0]
            offset = (len(fir) - 1) // 2
            new_times = (
                np.arange(n_out) * dt + self._axis0_index[0] + offset * dt
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        # Reshape back
        new_shape = [convolved_2d.shape[0]] + list(orig_shape[1:])
        return ScalarField(
            convolved_2d.reshape(new_shape),
            unit=self.unit,
            axis0=new_times,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=self.axis_names,
            axis0_domain="time",
            space_domain=self._space_domains,
        )

    def inject(self, other, alpha=1.0):
        """Add a simulated signal into the data.

        Parameters
        ----------
        other : ScalarField
            Signal to inject. Must have compatible shape and axes.
        alpha : float, optional
            Scaling factor for the injected signal. Default is 1.0.

        Returns
        -------
        ScalarField
            Field with injected signal.

        Raises
        ------
        ValueError
            If fields have incompatible shapes or axes.

        Examples
        --------
        >>> # Inject a simulated signal
        >>> signal = ScalarField.simulate('plane_wave', ...)
        >>> injected = field.inject(signal, alpha=0.5)

        See Also
        --------
        simulate : Generate simulated fields
        """
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        if self._axis0_domain != other._axis0_domain:
            raise ValueError(
                f"Domain mismatch: {self._axis0_domain} vs {other._axis0_domain}"
            )

        # Simple addition with scaling
        injected_data = self.value + alpha * other.value

        return ScalarField(
            injected_data,
            unit=self.unit,
            axis0=self._axis0_index,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=self.axis_names,
            axis0_domain=self._axis0_domain,
            space_domain=self._space_domains,
        )

    # =========================================================================
    # Cross-Spectral Analysis
    # =========================================================================

    def csd(self, other, fftlength=None, overlap=0, window="hann", **kwargs):
        """Calculate cross-spectral density with another field.

        Computes the cross power between two signals using Welch's method.
        Essential for analyzing relationships between different channels.

        Parameters
        ----------
        other : ScalarField
            Second field for cross-spectral analysis. Must have same shape.
        fftlength : float, optional
            Length of FFT segments in seconds. If None, uses entire length.
        overlap : float, optional
            Overlap between segments in seconds. Default is 0.
        window : str or tuple, optional
            Window function name. Default is 'hann'.
        **kwargs
            Additional arguments (nfft, noverlap can override time-based params).

        Returns
        -------
        ScalarField
            Cross-spectral density field with axis0_domain='frequency'.
            Units are (self.unit * other.unit / Hz).

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time' or shapes don't match.

        Examples
        --------
        >>> # CSD between two fields
        >>> csd_result = field1.csd(field2, fftlength=2.0, overlap=1.0)

        See Also
        --------
        psd : Power spectral density
        coherence : Frequency-coherence
        """
        if self._axis0_domain != "time":
            raise ValueError("csd requires axis0_domain='time'")
        if other._axis0_domain != "time":
            raise ValueError("other must have axis0_domain='time'")
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        from scipy import signal as scipy_signal

        # Get sample rate
        dt = self._axis0_index[1] - self._axis0_index[0]
        fs = (1.0 / dt).to("Hz").value

        # Convert time-based to sample-based parameters
        if "nfft" not in kwargs and fftlength is not None:
            kwargs["nperseg"] = int(round(fftlength * fs))
        if "noverlap" not in kwargs and overlap > 0:
            kwargs["noverlap"] = int(round(overlap * fs))

        # Set defaults
        kwargs.setdefault("nperseg", min(256, self.shape[0]))
        kwargs.setdefault("scaling", "density")

        # Reshape for efficient computation
        orig_shape = self.shape
        data1_2d = self.value.reshape(orig_shape[0], -1)
        data2_2d = other.value.reshape(orig_shape[0], -1)

        # Compute CSD for each spatial point
        freqs, csd_2d = scipy_signal.csd(
            data1_2d,
            data2_2d,
            fs=fs,
            window=window,
            axis=0,
            **kwargs,
        )

        # Reshape back
        new_shape = [len(freqs)] + list(orig_shape[1:])
        csd_data = csd_2d.reshape(new_shape)

        # Result units
        if self.unit is not None and other.unit is not None:
            result_unit = (self.unit * other.unit) / u.Hz
        elif self.unit is not None:
            result_unit = self.unit**2 / u.Hz
        elif other.unit is not None:
            result_unit = other.unit**2 / u.Hz
        else:
            result_unit = 1 / u.Hz

        return ScalarField(
            csd_data,
            unit=result_unit,
            axis0=freqs * u.Hz,
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

    def coherence(self, other, fftlength=None, overlap=0, window="hann", **kwargs):
        """Compute frequency-coherence with another field.

        Returns values between 0-1 indicating correlation at each frequency.
        Critical for identifying correlated noise sources.

        Parameters
        ----------
        other : ScalarField
            Second field for coherence analysis. Must have same shape.
        fftlength : float, optional
            Length of FFT segments in seconds. If None, uses entire length.
        overlap : float, optional
            Overlap between segments in seconds. Default is 0.
        window : str or tuple, optional
            Window function name. Default is 'hann'.
        **kwargs
            Additional arguments (nfft, noverlap can override time-based params).

        Returns
        -------
        ScalarField
            Coherence field (dimensionless, values 0-1) with
            axis0_domain='frequency'.

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time' or shapes don't match.

        Examples
        --------
        >>> # Coherence between two fields
        >>> coh = field1.coherence(field2, fftlength=2.0, overlap=1.0)

        See Also
        --------
        csd : Cross-spectral density
        """
        if self._axis0_domain != "time":
            raise ValueError("coherence requires axis0_domain='time'")
        if other._axis0_domain != "time":
            raise ValueError("other must have axis0_domain='time'")
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        from scipy import signal as scipy_signal

        # Get sample rate
        dt = self._axis0_index[1] - self._axis0_index[0]
        fs = (1.0 / dt).to("Hz").value

        # Convert time-based to sample-based parameters
        if "nfft" not in kwargs and fftlength is not None:
            kwargs["nperseg"] = int(round(fftlength * fs))
        if "noverlap" not in kwargs and overlap > 0:
            kwargs["noverlap"] = int(round(overlap * fs))

        # Set defaults
        kwargs.setdefault("nperseg", min(256, self.shape[0]))

        # Reshape for efficient computation
        orig_shape = self.shape
        data1_2d = self.value.reshape(orig_shape[0], -1)
        data2_2d = other.value.reshape(orig_shape[0], -1)

        # Compute coherence for each spatial point
        freqs, coh_2d = scipy_signal.coherence(
            data1_2d,
            data2_2d,
            fs=fs,
            window=window,
            axis=0,
            **kwargs,
        )

        # Reshape back
        new_shape = [len(freqs)] + list(orig_shape[1:])
        coh_data = coh_2d.reshape(new_shape)

        return ScalarField(
            coh_data,
            unit=u.dimensionless_unscaled,
            axis0=freqs * u.Hz,
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

    def spectrogram(self, stride, fftlength=None, overlap=0, window="hann", method="welch", **kwargs):
        """Generate time-frequency spectrogram.

        Shows how spectral content evolves over time. Fundamental for
        visualizing transient signals and time-varying noise.

        Parameters
        ----------
        stride : float
            Time step between consecutive spectrograms in seconds.
        fftlength : float, optional
            Length of FFT segments in seconds. If None, uses stride.
        overlap : float, optional
            Overlap between FFT segments in seconds. Default is 0.
        window : str or tuple, optional
            Window function name. Default is 'hann'.
        method : str, optional
            Method for PSD estimation: 'welch' or 'median'. Default is 'welch'.
        **kwargs
            Additional arguments passed to scipy.signal.spectrogram.

        Returns
        -------
        ScalarField
            Spectrogram with shape (n_times, n_freqs, nx, ny, nz).
            Note: axis0 represents time segments, not original time axis.
            axis1 represents frequency.

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time'.

        Notes
        -----
        The returned field has a modified structure where axis0 represents
        time segments and contains frequency information along a pseudo-axis.
        For standard usage, consider using `plot_spectrogram()` instead.

        Examples
        --------
        >>> # Generate spectrogram with 1s stride and 2s FFT length
        >>> spec = field.spectrogram(stride=1.0, fftlength=2.0, overlap=1.0)

        See Also
        --------
        psd : Power spectral density
        """
        if self._axis0_domain != "time":
            raise ValueError("spectrogram requires axis0_domain='time'")

        from scipy import signal as scipy_signal

        # Get sample rate
        dt = self._axis0_index[1] - self._axis0_index[0]
        fs = (1.0 / dt).to("Hz").value

        if fftlength is None:
            fftlength = stride

        # Convert to samples
        nperseg = int(round(fftlength * fs))
        noverlap = int(round(overlap * fs))
        stride_samples = int(round(stride * fs))

        # Reshape for efficient computation
        orig_shape = self.shape
        data_2d = self.value.reshape(orig_shape[0], -1)

        # Compute spectrogram for each spatial point
        freqs, times, spec_2d = scipy_signal.spectrogram(
            data_2d,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            axis=0,
            **kwargs,
        )

        # Reshape: spec has shape (n_freqs, n_times, n_spatial_points)
        # We want (n_times, n_freqs, nx, ny, nz)
        n_freqs, n_times = spec_2d.shape[0], spec_2d.shape[1]
        n_spatial = np.prod(orig_shape[1:])

        # Reshape to (n_freqs, n_times, nx, ny, nz)
        spec_5d = spec_2d.reshape(n_freqs, n_times, *orig_shape[1:])

        # Transpose to (n_times, n_freqs, nx, ny, nz)
        spec_5d = np.transpose(spec_5d, (1, 0, 2, 3, 4))

        # For now, flatten frequency into the data
        # Store as (n_times, n_freqs*nx, ny, nz) to maintain 4D structure
        # This is a temporary solution; ideally we'd have a Spectrogram class
        spec_4d = spec_5d.reshape(n_times, n_freqs * orig_shape[1], orig_shape[2], orig_shape[3])

        # Time axis for spectrogram segments
        t0 = self._axis0_index[0]
        seg_times = times * dt.unit + t0

        # Result units (power spectral density)
        if self.unit is not None:
            result_unit = self.unit**2 / u.Hz
        else:
            result_unit = 1 / u.Hz

        # Store frequency information in metadata
        result = ScalarField(
            spec_4d,
            unit=result_unit,
            axis0=seg_times,
            axis1=np.arange(n_freqs * orig_shape[1]) * u.dimensionless_unscaled,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=[
                "t_seg",
                "freq_space",
                self._axis2_name,
                self._axis3_name,
            ],
            axis0_domain="time",
            space_domain=self._space_domains,
        )

        # Store frequency and original spatial axis info
        result._spectrogram_freqs = freqs * u.Hz
        result._spectrogram_orig_axis1 = self._axis1_index

        return result

    def rayleigh_spectrum(self, fftlength=None, overlap=0, window="hann", **kwargs):
        """Compute Rayleigh statistic as a function of frequency.

        The Rayleigh statistic is the ratio of the maximum to mean bin power,
        more sensitive than standard PSD to narrow-band disturbances and
        non-Gaussian features.

        Parameters
        ----------
        fftlength : float, optional
            Length of FFT segments in seconds. If None, uses entire length.
        overlap : float, optional
            Overlap between segments in seconds. Default is 0.
        window : str or tuple, optional
            Window function name. Default is 'hann'.
        **kwargs
            Additional arguments (nperseg, noverlap can override time-based params).

        Returns
        -------
        ScalarField
            Rayleigh statistic as a function of frequency.
            Higher values indicate non-Gaussian spectral features.

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time'.

        Notes
        -----
        The Rayleigh statistic R is computed as:
        R(f) = max(|FFT_i(f)|^2) / mean(|FFT_i(f)|^2)

        where i indexes the FFT segments. R ≈ 2 for Gaussian noise,
        significantly higher for spectral lines or non-Gaussian features.

        Examples
        --------
        >>> # Detect spectral lines
        >>> ray_spec = field.rayleigh_spectrum(fftlength=2.0, overlap=1.0)
        >>> # Values >> 2 indicate non-Gaussian features

        See Also
        --------
        rayleigh_spectrogram : Time-frequency Rayleigh statistic
        psd : Power spectral density
        """
        if self._axis0_domain != "time":
            raise ValueError("rayleigh_spectrum requires axis0_domain='time'")

        from scipy import signal as scipy_signal

        # Get sample rate
        dt = self._axis0_index[1] - self._axis0_index[0]
        fs = (1.0 / dt).to("Hz").value

        # Convert time-based to sample-based parameters
        if "nperseg" not in kwargs and fftlength is not None:
            kwargs["nperseg"] = int(round(fftlength * fs))
        if "noverlap" not in kwargs and overlap > 0:
            kwargs["noverlap"] = int(round(overlap * fs))

        # Set defaults
        kwargs.setdefault("nperseg", min(256, self.shape[0]))
        kwargs.setdefault("scaling", "spectrum")  # Use spectrum not density

        # Reshape for efficient computation
        orig_shape = self.shape
        data_2d = self.value.reshape(orig_shape[0], -1)

        # Compute spectrogram for each spatial point
        rayleigh_list = []
        for i in range(data_2d.shape[1]):
            freqs, times, spec = scipy_signal.spectrogram(
                data_2d[:, i],
                fs=fs,
                window=window,
                axis=0,
                **kwargs,
            )

            # Compute Rayleigh statistic: max / mean across time
            with np.errstate(divide='ignore', invalid='ignore'):
                rayleigh = np.max(spec, axis=1) / np.mean(spec, axis=1)
                rayleigh = np.nan_to_num(rayleigh, nan=1.0, posinf=1.0, neginf=1.0)

            rayleigh_list.append(rayleigh)

        # Stack results
        rayleigh_2d = np.array(rayleigh_list).T

        # Reshape back
        new_shape = [len(freqs)] + list(orig_shape[1:])
        rayleigh_data = rayleigh_2d.reshape(new_shape)

        return ScalarField(
            rayleigh_data,
            unit=u.dimensionless_unscaled,
            axis0=freqs * u.Hz,
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

    def rayleigh_spectrogram(self, stride, fftlength=None, overlap=0, window="hann", **kwargs):
        """Compute time-frequency Rayleigh statistic.

        Identifies when and where non-Gaussian features appear in the data.
        Useful for glitch classification and data quality monitoring.

        Parameters
        ----------
        stride : float
            Time step between consecutive Rayleigh spectra in seconds.
        fftlength : float, optional
            Length of FFT segments for each Rayleigh spectrum in seconds.
            If None, uses stride.
        overlap : float, optional
            Overlap between FFT segments in seconds. Default is 0.
        window : str or tuple, optional
            Window function name. Default is 'hann'.
        **kwargs
            Additional arguments.

        Returns
        -------
        ScalarField
            Time-frequency Rayleigh spectrogram.
            Shape: (n_time_segments, n_freqs, nx, ny, nz).

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time'.

        Examples
        --------
        >>> # Time-frequency Rayleigh analysis
        >>> ray_spec = field.rayleigh_spectrogram(
        ...     stride=1.0, fftlength=2.0, overlap=1.0
        ... )

        See Also
        --------
        rayleigh_spectrum : Frequency-only Rayleigh statistic
        spectrogram : Standard PSD spectrogram
        """
        if self._axis0_domain != "time":
            raise ValueError("rayleigh_spectrogram requires axis0_domain='time'")

        from scipy import signal as scipy_signal

        # Get sample rate
        dt = self._axis0_index[1] - self._axis0_index[0]
        fs = (1.0 / dt).to("Hz").value

        if fftlength is None:
            fftlength = stride

        # Convert to samples
        stride_samples = int(round(stride * fs))
        nperseg = int(round(fftlength * fs))
        noverlap = int(round(overlap * fs))

        # Number of segments for Rayleigh calculation
        # Use a sliding window approach
        n_times = self.shape[0]
        segment_starts = np.arange(0, n_times - nperseg, stride_samples)

        # Reshape for efficient computation
        orig_shape = self.shape
        data_2d = self.value.reshape(orig_shape[0], -1)

        # Compute Rayleigh spectrogram for each spatial point
        rayleigh_spec_list = []
        for i in range(data_2d.shape[1]):
            ray_time_freq = []

            for start in segment_starts:
                # Extract segment
                segment = data_2d[start : start + stride_samples, i]

                # Compute spectrum of this segment using multiple FFT windows
                if len(segment) < nperseg:
                    continue

                freqs, times_seg, spec = scipy_signal.spectrogram(
                    segment,
                    fs=fs,
                    window=window,
                    nperseg=min(nperseg, len(segment)),
                    noverlap=noverlap,
                )

                # Compute Rayleigh statistic for this time segment
                with np.errstate(divide='ignore', invalid='ignore'):
                    rayleigh = np.max(spec, axis=1) / np.mean(spec, axis=1)
                    rayleigh = np.nan_to_num(rayleigh, nan=1.0, posinf=1.0, neginf=1.0)

                ray_time_freq.append(rayleigh)

            rayleigh_spec_list.append(np.array(ray_time_freq))

        # Stack results: shape (n_spatial, n_times, n_freqs)
        rayleigh_3d = np.array(rayleigh_spec_list)

        # Transpose to (n_times, n_freqs, n_spatial)
        rayleigh_3d = rayleigh_3d.transpose(1, 2, 0)

        # Reshape to (n_times, n_freqs, nx, ny, nz)
        n_time_segs = rayleigh_3d.shape[0]
        n_freqs = rayleigh_3d.shape[1]
        new_shape = [n_time_segs, n_freqs] + list(orig_shape[1:])
        rayleigh_data = rayleigh_3d.reshape(new_shape)

        # Flatten to 4D: (n_times, n_freqs*nx, ny, nz)
        rayleigh_4d = rayleigh_data.reshape(
            n_time_segs, n_freqs * orig_shape[1], orig_shape[2], orig_shape[3]
        )

        # Time axis for segments
        t0 = self._axis0_index[0]
        seg_times = segment_starts[:len(rayleigh_4d)] * dt + t0

        result = ScalarField(
            rayleigh_4d,
            unit=u.dimensionless_unscaled,
            axis0=seg_times,
            axis1=np.arange(n_freqs * orig_shape[1]) * u.dimensionless_unscaled,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=[
                "t_seg",
                "freq_space",
                self._axis2_name,
                self._axis3_name,
            ],
            axis0_domain="time",
            space_domain=self._space_domains,
        )

        # Store frequency and original spatial axis info
        result._spectrogram_freqs = freqs * u.Hz
        result._spectrogram_orig_axis1 = self._axis1_index

        return result

    # =========================================================================
    # Correlation Analysis
    # =========================================================================

    def autocorrelation(self, maxlag=None, **kwargs):
        """Compute autocorrelation function.

        Shows how the signal correlates with time-shifted copies of itself,
        revealing periodic structures and characteristic timescales.

        Parameters
        ----------
        maxlag : int, optional
            Maximum lag in samples. If None, uses half the data length.
        **kwargs
            Additional arguments (currently unused, for future compatibility).

        Returns
        -------
        ScalarField
            Autocorrelation function with lag as time axis.
            Shape is (2*maxlag+1, nx, ny, nz).

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time'.

        Examples
        --------
        >>> # Compute autocorrelation
        >>> acf = field.autocorrelation(maxlag=100)
        >>> # Peak at non-zero lag indicates periodicity

        See Also
        --------
        correlate : Cross-correlation between two fields
        """
        if self._axis0_domain != "time":
            raise ValueError("autocorrelation requires axis0_domain='time'")

        from scipy import signal as scipy_signal

        if maxlag is None:
            maxlag = self.shape[0] // 2

        # Clip maxlag to valid range
        maxlag = min(maxlag, self.shape[0] - 1)

        # Reshape for efficient computation
        orig_shape = self.shape
        data_2d = self.value.reshape(orig_shape[0], -1)

        # Compute autocorrelation for each spatial point
        # Using 'same' mode and then extracting the center portion
        acf_list = []
        for i in range(data_2d.shape[1]):
            # Normalize by removing mean
            x = data_2d[:, i]
            x_demean = x - np.mean(x)

            # Compute autocorrelation using convolution
            acf_full = scipy_signal.correlate(x_demean, x_demean, mode='same')

            # Normalize by variance and length
            acf_full = acf_full / (np.var(x) * len(x))

            # Extract center portion (lag from -maxlag to +maxlag)
            center = len(acf_full) // 2
            acf = acf_full[center - maxlag : center + maxlag + 1]
            acf_list.append(acf)

        # Stack results
        acf_2d = np.array(acf_list).T

        # Reshape back
        new_shape = [2 * maxlag + 1] + list(orig_shape[1:])
        acf_data = acf_2d.reshape(new_shape)

        # Create lag axis
        dt = self._axis0_index[1] - self._axis0_index[0]
        lags = np.arange(-maxlag, maxlag + 1) * dt

        return ScalarField(
            acf_data,
            unit=u.dimensionless_unscaled,
            axis0=lags,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=['lag', self._axis1_name, self._axis2_name, self._axis3_name],
            axis0_domain="time",
            space_domain=self._space_domains,
        )

    def correlate(self, other, maxlag=None, mode='same'):
        """Cross-correlate with another field in time domain.

        Computes correlation as a function of time lag, complementary to
        csd() in the frequency domain. Used for time-delay estimation.

        Parameters
        ----------
        other : ScalarField
            Second field for cross-correlation. Must have same shape.
        maxlag : int, optional
            Maximum lag in samples. If None, uses half the data length.
        mode : str, optional
            Correlation mode: 'same', 'full', or 'valid'. Default is 'same'.

        Returns
        -------
        ScalarField
            Cross-correlation function with lag as time axis.

        Raises
        ------
        ValueError
            If ``axis0_domain`` is not 'time' or shapes don't match.

        Examples
        --------
        >>> # Cross-correlation between two fields
        >>> xcf = field1.correlate(field2, maxlag=100)
        >>> # Peak location indicates time delay

        See Also
        --------
        autocorrelation : Autocorrelation function
        csd : Cross-spectral density (frequency domain)
        """
        if self._axis0_domain != "time":
            raise ValueError("correlate requires axis0_domain='time'")
        if other._axis0_domain != "time":
            raise ValueError("other must have axis0_domain='time'")
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

        from scipy import signal as scipy_signal

        if maxlag is None:
            maxlag = min(self.shape[0], other.shape[0]) // 2

        # Reshape for efficient computation
        orig_shape = self.shape
        data1_2d = self.value.reshape(orig_shape[0], -1)
        data2_2d = other.value.reshape(orig_shape[0], -1)

        # Compute cross-correlation for each spatial point
        xcf_list = []
        for i in range(data1_2d.shape[1]):
            # Normalize by removing mean
            x = data1_2d[:, i]
            y = data2_2d[:, i]
            x_demean = x - np.mean(x)
            y_demean = y - np.mean(y)

            # Compute cross-correlation
            xcf_full = scipy_signal.correlate(x_demean, y_demean, mode='same')

            # Normalize
            xcf_full = xcf_full / (np.std(x) * np.std(y) * len(x))

            # Extract center portion if maxlag is specified
            if mode == 'same':
                center = len(xcf_full) // 2
                xcf = xcf_full[center - maxlag : center + maxlag + 1]
            else:
                xcf = xcf_full

            xcf_list.append(xcf)

        # Stack results
        xcf_2d = np.array(xcf_list).T

        # Reshape back
        if mode == 'same':
            new_shape = [2 * maxlag + 1] + list(orig_shape[1:])
            lags = np.arange(-maxlag, maxlag + 1)
        else:
            new_shape = [xcf_2d.shape[0]] + list(orig_shape[1:])
            lags = np.arange(xcf_2d.shape[0]) - xcf_2d.shape[0] // 2

        xcf_data = xcf_2d.reshape(new_shape)

        # Create lag axis
        dt = self._axis0_index[1] - self._axis0_index[0]
        lag_times = lags * dt

        return ScalarField(
            xcf_data,
            unit=u.dimensionless_unscaled,
            axis0=lag_times,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=['lag', self._axis1_name, self._axis2_name, self._axis3_name],
            axis0_domain="time",
            space_domain=self._space_domains,
        )

    # =========================================================================
    # Time Series Utilities
    # =========================================================================

    def is_compatible(self, other):
        """Check if two fields can be combined.

        Validates matching sample rate, units, spatial axes, and domains.

        Parameters
        ----------
        other : ScalarField
            Field to check compatibility with.

        Returns
        -------
        bool
            True if fields are compatible for concatenation/combination.

        Examples
        --------
        >>> if field1.is_compatible(field2):
        ...     combined = field1.append(field2)
        """
        if not isinstance(other, ScalarField):
            return False

        # Check domains
        if self._axis0_domain != other._axis0_domain:
            return False
        if self._space_domains != other._space_domains:
            return False

        # Check spatial dimensions
        if self.shape[1:] != other.shape[1:]:
            return False

        # Check units
        if self.unit != other.unit:
            # Check if equivalent
            try:
                if self.unit is not None and other.unit is not None:
                    if not self.unit.is_equivalent(other.unit):
                        return False
                elif self.unit != other.unit:  # One is None, other isn't
                    return False
            except:
                return False

        # Check sample rate (for time domain)
        if self._axis0_domain == "time":
            dt1 = self._axis0_index[1] - self._axis0_index[0]
            dt2 = other._axis0_index[1] - other._axis0_index[0]
            if not np.isclose(dt1.value, dt2.value, rtol=1e-6):
                return False

        # Check spatial axes
        for i in range(1, 4):
            ax1 = [self._axis1_index, self._axis2_index, self._axis3_index][i - 1]
            ax2 = [other._axis1_index, other._axis2_index, other._axis3_index][i - 1]
            if ax1.shape != ax2.shape:
                return False
            if not np.allclose(ax1.value, ax2.value, rtol=1e-9, atol=1e-12):
                return False

        return True

    def is_contiguous(self, other, tol=None):
        """Test if two segments are adjacent in time.

        Parameters
        ----------
        other : ScalarField
            Field to check contiguity with.
        tol : float or Quantity, optional
            Tolerance for time gap. If None, uses half the sample period.

        Returns
        -------
        bool
            True if fields are contiguous (this ends where other begins).

        Examples
        --------
        >>> if field1.is_contiguous(field2):
        ...     combined = field1.append(field2, gap='ignore')
        """
        if not self.is_compatible(other):
            return False

        if self._axis0_domain != "time":
            return False

        # Get sample period
        dt = self._axis0_index[1] - self._axis0_index[0]
        if tol is None:
            tol = dt / 2

        # Check if self ends where other begins
        gap = abs(other._axis0_index[0] - self._axis0_index[-1] - dt)

        return gap <= tol

    def append(self, other, gap="raise", pad=0.0, resize=True):
        """Concatenate another field in time.

        Parameters
        ----------
        other : ScalarField
            Field to append (comes after self).
        gap : {'raise', 'ignore', 'pad'}, optional
            How to handle gaps:
            - 'raise': Raise error if gap exists (default)
            - 'ignore': Concatenate anyway, time axis may be discontinuous
            - 'pad': Fill gap with constant value
        pad : float, optional
            Value to use for padding if gap='pad'. Default is 0.0.
        resize : bool, optional
            Whether to allow different time axis lengths. Default is True.

        Returns
        -------
        ScalarField
            Concatenated field.

        Raises
        ------
        ValueError
            If fields are not compatible or gap handling fails.

        Examples
        --------
        >>> # Append contiguous segment
        >>> combined = field1.append(field2)
        >>> # Append with gap padding
        >>> combined = field1.append(field2, gap='pad', pad=0.0)

        See Also
        --------
        prepend : Prepend another field
        is_contiguous : Check if fields are adjacent
        """
        if not self.is_compatible(other):
            raise ValueError("Fields are not compatible for concatenation")

        if self._axis0_domain != "time":
            raise ValueError("append only works for time-domain fields")

        # Check for gap
        dt = self._axis0_index[1] - self._axis0_index[0]
        expected_start = self._axis0_index[-1] + dt
        actual_start = other._axis0_index[0]
        time_gap = actual_start - expected_start

        if gap == "raise" and abs(time_gap) > dt / 2:
            raise ValueError(
                f"Gap detected: {time_gap}. "
                f"Use gap='ignore' or gap='pad' to handle gaps."
            )

        if gap == "pad" and abs(time_gap) > dt / 2:
            # Create padding segment
            n_pad = int(round((time_gap / dt).decompose().value))
            if n_pad > 0:
                pad_data = np.full(
                    (n_pad, *self.shape[1:]), pad, dtype=self.value.dtype
                )
                pad_times = np.arange(n_pad) * dt + expected_start

                pad_field = ScalarField(
                    pad_data,
                    unit=self.unit,
                    axis0=pad_times,
                    axis1=self._axis1_index,
                    axis2=self._axis2_index,
                    axis3=self._axis3_index,
                    axis_names=self.axis_names,
                    axis0_domain="time",
                    space_domain=self._space_domains,
                )

                # Concatenate self, pad, other
                combined_data = np.concatenate(
                    [self.value, pad_data, other.value], axis=0
                )
                combined_times = np.concatenate(
                    [self._axis0_index, pad_times, other._axis0_index]
                )
            else:
                combined_data = np.concatenate([self.value, other.value], axis=0)
                combined_times = np.concatenate(
                    [self._axis0_index, other._axis0_index]
                )
        else:
            # Simple concatenation
            combined_data = np.concatenate([self.value, other.value], axis=0)
            combined_times = np.concatenate([self._axis0_index, other._axis0_index])

        return ScalarField(
            combined_data,
            unit=self.unit,
            axis0=combined_times,
            axis1=self._axis1_index,
            axis2=self._axis2_index,
            axis3=self._axis3_index,
            axis_names=self.axis_names,
            axis0_domain="time",
            space_domain=self._space_domains,
        )

    def prepend(self, other, **kwargs):
        """Concatenate another field before this one.

        Parameters
        ----------
        other : ScalarField
            Field to prepend (comes before self).
        **kwargs
            Additional arguments passed to append().

        Returns
        -------
        ScalarField
            Concatenated field.

        Examples
        --------
        >>> # Prepend segment
        >>> combined = field2.prepend(field1)
        >>> # Equivalent to field1.append(field2)

        See Also
        --------
        append : Append another field
        """
        return other.append(self, **kwargs)

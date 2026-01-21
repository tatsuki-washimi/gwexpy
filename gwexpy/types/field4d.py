"""4D Field with domain states and FFT operations."""

import numpy as np

from .array4d import Array4D

__all__ = ["Field4D"]


class Field4D(Array4D):
    """4D Field with domain states and FFT operations.

    This class extends :class:`Array4D` to represent physical fields that
    can exist in different domains (time/frequency for axis 0, real/k-space
    for spatial axes 1-3).

    **Key feature**: All indexing operations return a Field4D, maintaining
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
    >>> from gwexpy.types import Field4D
    >>> data = np.random.randn(100, 32, 32, 32)
    >>> times = np.arange(100) * 0.01 * u.s
    >>> x = np.arange(32) * 1.0 * u.m
    >>> field = Field4D(data, axis0=times, axis1=x, axis2=x, axis3=x,
    ...                 axis_names=['t', 'x', 'y', 'z'])
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

    def __new__(
        cls,
        data,
        unit=None,
        axis0=None,
        axis1=None,
        axis2=None,
        axis3=None,
        axis_names=None,
        axis0_domain="time",
        space_domain="real",
        **kwargs,
    ):
        # Set default axis names based on domain
        if axis_names is None:
            if axis0_domain == "time":
                time_name = cls._TIME_AXIS_NAME
            else:
                time_name = cls._FREQ_AXIS_NAME

            if isinstance(space_domain, dict):
                # Derive names from domain dict
                space_names = ["x", "y", "z"]  # defaults
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
            # Validate dict values
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

        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None:
            return

        if getattr(self, "_axis0_domain", None) is None:
            self._axis0_domain = getattr(obj, "_axis0_domain", "time")

        if getattr(self, "_space_domains", None) is None:
            parent_domains = getattr(obj, "_space_domains", None)
            if parent_domains is not None:
                self._space_domains = dict(parent_domains)
            else:
                # Default all to real
                self._space_domains = {
                    self._axis1_name: "real",
                    self._axis2_name: "real",
                    self._axis3_name: "real",
                }

        if getattr(self, "_axis0_offset", None) is None:
            self._axis0_offset = getattr(obj, "_axis0_offset", None)

    @property
    def axis0_domain(self):
        """Domain of axis 0: 'time' or 'frequency'."""
        return self._axis0_domain

    @property
    def space_domains(self):
        """Mapping of spatial axis names to domains."""
        return dict(self._space_domains)

    def __getitem__(self, item):
        """Get item, always returning Field4D (4D maintained).

        Integer indices are converted to length-1 slices to maintain
        4D structure.
        """
        forced_item = self._force_4d_item(item)
        return self._getitem_field4d(forced_item)

    def _force_4d_item(self, item):
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
                item[:ellipsis_idx]
                + (slice(None),) * fill
                + item[ellipsis_idx + 1 :]
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

    def _getitem_field4d(self, item):
        """Perform getitem with Field4D reconstruction.

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

        return Field4D(
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

    def _isel_tuple(self, item_tuple):
        """Internal isel using Field4D getitem logic."""
        forced_item = self._force_4d_item(item_tuple)
        return self._getitem_field4d(forced_item)

    # =========================================================================
    # Time FFT (axis=0, GWpy TimeSeries.fft compatible)
    # =========================================================================

    def _validate_axis_for_fft(self, axis_index, axis_name, domain_name):
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
        from .axis import AxisDescriptor
        ax_desc = AxisDescriptor(axis_name, axis_index)
        if not ax_desc.regular:
            raise ValueError(
                f"FFT requires regularly spaced {domain_name} axis, "
                f"but axis '{axis_name}' is irregular"
            )

    def fft_time(self, nfft=None):
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
        Field4D
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
        self._validate_axis_for_fft(
            self._axis0_index, self._axis0_name, "time"
        )

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
        dft = np.fft.rfft(self.value, n=nfft, axis=0) / nfft

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
        dt_value = dt.value
        dt_unit = dt.unit

        freqs_value = np.fft.rfftfreq(nfft, d=dt_value)
        freqs = freqs_value * (1 / dt_unit)

        result = Field4D(
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
        return result

    def ifft_time(self, nout=None):
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
        Field4D
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
        self._validate_axis_for_fft(
            self._axis0_index, self._axis0_name, "frequency"
        )

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
        dift = np.fft.irfft(array * nout, n=nout, axis=0)

        # Compute time axis
        df = self._axis0_index[1] - self._axis0_index[0]
        df_value = df.value
        df_unit = df.unit

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

        return Field4D(
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

    # =========================================================================
    # Spatial FFT (axes 1-3, two-sided signed FFT)
    # =========================================================================

    def fft_space(self, axes=None, n=None):
        """Compute FFT along spatial axes.

        This method uses two-sided FFT (numpy.fft.fftn) and produces
        angular wavenumber (k = 2π·fftfreq).

        Parameters
        ----------
        axes : iterable of str, optional
            Axis names to transform (e.g., ['x', 'y']). If None,
            transforms all spatial axes in 'real' domain.
        n : tuple of int, optional
            FFT lengths for each axis.

        Returns
        -------
        Field4D
            Transformed field with specified axes in 'k' domain.

        Raises
        ------
        ValueError
            If any specified axis is not in 'real' domain.
        ValueError
            If any specified axis is not uniformly spaced.

        Notes
        -----
        The wavenumber axis is computed as ``k = 2π * fftfreq(n, d=dx)``,
        satisfying ``k = 2π / λ``.
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
                    f"Axis '{ax_name}' is not uniformly spaced. "
                    f"Cannot apply FFT."
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
        dft = np.fft.fftn(self.value, s=s, axes=target_axes_int)

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
            dx_value = ax_desc.delta.value  # Already signed from diff
            dx_unit = ax_desc.delta.unit

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

        return Field4D(
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

    def ifft_space(self, axes=None, n=None):
        """Compute inverse FFT along k-space axes.

        Parameters
        ----------
        axes : iterable of str, optional
            Axis names to transform (e.g., ['kx', 'ky']). If None,
            transforms all spatial axes in 'k' domain.
        n : tuple of int, optional
            Output lengths for each axis.

        Returns
        -------
        Field4D
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
        dift = np.fft.ifftn(self.value, s=s, axes=target_axes_int)

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
            dk_value = dk_raw.value
            dk_unit = dk_raw.unit

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

        return Field4D(
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

    def wavelength(self, axis):
        """Compute wavelength from wavenumber axis.

        Parameters
        ----------
        axis : str or int
            The k-domain axis name or index.

        Returns
        -------
        `~astropy.units.Quantity`
            Wavelength values (λ = 2π / |k|). k=0 returns inf.

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
            wavelength_values = 2 * np.pi / np.abs(k_index.value)
        return wavelength_values * (1 / k_index.unit)

    # =========================================================================
    # Extraction API (Phase 0.3)
    # =========================================================================

    def extract_points(self, points, interp="nearest"):
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

    def extract_profile(self, axis, at, reduce=None):
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
        Field4D
            A Field4D with the non-plane axes having length=1.

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
                x_coords, y_coords, data_2d,
                vmin=vmin, vmax=vmax, cmap=cmap, **kwargs
            )
        elif method == "imshow":
            extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]
            im = ax.imshow(
                data_2d, extent=extent, origin="lower", aspect="auto",
                vmin=vmin, vmax=vmax, cmap=cmap, **kwargs
            )
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'pcolormesh' or 'imshow'.")

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
        ax.set_xlabel(f"{self._axis0_name} [{self._axis0_index.unit}]")
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

        ax.plot(axis_index.value, y_data, label=label, **kwargs)

        # Labels with units
        ax.set_xlabel(f"{axis} [{axis_index.unit}]")
        if y_unit is not None:
            ax.set_ylabel(f"{mode} [{y_unit}]")

        if label:
            ax.legend()

        return fig, ax

    # =========================================================================
    # Comparison & Summary Methods (Phase 2)
    # =========================================================================

    def diff(self, other, mode="diff"):
        """Compute difference or ratio between two Field4D objects.

        Parameters
        ----------
        other : Field4D
            The field to compare against.
        mode : str, optional
            Comparison mode:
            - 'diff': Difference (self - other)
            - 'ratio': Ratio (self / other)
            - 'percent': Percentage difference ((self - other) / other * 100)
            Default is 'diff'.

        Returns
        -------
        Field4D
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
            raise ValueError(
                f"Shape mismatch: {self.shape} vs {other.shape}"
            )

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

        return Field4D(
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
        Field4D
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

            baseline_data = self.value[i_start:i_end + 1, ...]

        # Compute mean and std along time axis
        mean = np.mean(baseline_data, axis=0, keepdims=True)
        std = np.std(baseline_data, axis=0, keepdims=True)

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore_data = (self.value - mean) / std
            zscore_data = np.nan_to_num(zscore_data, nan=0.0, posinf=0.0, neginf=0.0)

        return Field4D(
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
        Field4D
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
            result_data = np.sqrt(np.mean(data ** 2, axis=0, keepdims=True))
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

        result = Field4D(
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
                space_axis.value, t_axis.value, data_2d,
                vmin=vmin, vmax=vmax, cmap=cmap, **kwargs
            )
        elif method == "imshow":
            extent = [
                space_axis.value[0], space_axis.value[-1],
                t_axis.value[0], t_axis.value[-1]
            ]
            im = ax.imshow(
                data_2d, extent=extent, origin="lower", aspect="auto",
                vmin=vmin, vmax=vmax, cmap=cmap, **kwargs
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

        This is a convenience wrapper around :func:`~gwexpy.types.compute_psd`.

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
        gwexpy.types.compute_psd : Full documentation.
        """
        from .field4d_signal import compute_psd

        return compute_psd(self, point_or_region, **kwargs)

    def freq_space_map(self, axis, at=None, **kwargs):
        """Compute frequency-space map along a spatial axis.

        This is a convenience wrapper around :func:`~gwexpy.types.freq_space_map`.

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
        Field4D
            2D frequency-space map.

        See Also
        --------
        gwexpy.types.freq_space_map : Full documentation.
        """
        from .field4d_signal import freq_space_map

        return freq_space_map(self, axis, at=at, **kwargs)

    def compute_xcorr(self, point_a, point_b, **kwargs):
        """Compute cross-correlation between two spatial points.

        This is a convenience wrapper around :func:`~gwexpy.types.compute_xcorr`.

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
        gwexpy.types.compute_xcorr : Full documentation.
        """
        from .field4d_signal import compute_xcorr

        return compute_xcorr(self, point_a, point_b, **kwargs)

    def time_delay_map(self, ref_point, plane="xy", at=None, **kwargs):
        """Compute time delay map from a reference point.

        This is a convenience wrapper around :func:`~gwexpy.types.time_delay_map`.

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
        Field4D
            Time delay map.

        See Also
        --------
        gwexpy.types.time_delay_map : Full documentation.
        """
        from .field4d_signal import time_delay_map

        return time_delay_map(self, ref_point, plane=plane, at=at, **kwargs)

    def coherence_map(self, ref_point, plane="xy", at=None, **kwargs):
        """Compute coherence map from a reference point.

        This is a convenience wrapper around :func:`~gwexpy.types.coherence_map`.

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
        Field4D or Field4DDict
            Coherence map.

        See Also
        --------
        gwexpy.types.coherence_map : Full documentation.
        """
        from .field4d_signal import coherence_map

        return coherence_map(self, ref_point, plane=plane, at=at, **kwargs)

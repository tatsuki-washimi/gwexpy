"""4D Field with domain states and FFT operations."""

import numpy as np
from astropy.units import Quantity, dimensionless_unscaled, Unit

from .array4d import Array4D
from .axis import coerce_1d_quantity

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

    def fft_time(self, nfft=None):
        """Compute FFT along time axis (axis 0).

        This method applies the same normalization as GWpy's
        ``TimeSeries.fft()``: rfft / nfft, with DC-excluded bins
        multiplied by 2.

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

        See Also
        --------
        gwpy.timeseries.TimeSeries.fft : The reference implementation.
        """
        if self._axis0_domain != "time":
            raise ValueError(
                f"fft_time requires axis0_domain='time', got '{self._axis0_domain}'"
            )

        if nfft is None:
            nfft = self.shape[0]

        # rfft along axis 0, normalized
        dft = np.fft.rfft(self.value, n=nfft, axis=0) / nfft
        # Multiply non-DC bins by 2 (one-sided spectrum correction)
        dft[1:, ...] *= 2.0

        # Compute frequency axis
        dt = self._axis0_index[1] - self._axis0_index[0]
        dt_value = dt.value
        dt_unit = dt.unit

        freqs_value = np.fft.rfftfreq(nfft, d=dt_value)
        freqs = freqs_value * (1 / dt_unit)

        return Field4D(
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

        See Also
        --------
        gwpy.frequencyseries.FrequencySeries.ifft : Reference implementation.
        """
        if self._axis0_domain != "frequency":
            raise ValueError(
                f"ifft_time requires axis0_domain='frequency', "
                f"got '{self._axis0_domain}'"
            )

        if nout is None:
            nout = (self.shape[0] - 1) * 2

        # Undo normalization: divide non-DC by 2, multiply by nout
        array = self.value.copy()
        array[1:, ...] /= 2.0
        dift = np.fft.irfft(array * nout, n=nout, axis=0)

        # Compute time axis
        df = self._axis0_index[1] - self._axis0_index[0]
        df_value = df.value
        df_unit = df.unit

        # dt = 1 / (nout * df)
        dt_value = 1.0 / (nout * df_value)
        dt_unit = 1 / df_unit

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
                    f"Cannot use fft_space on axis 0 (time/frequency axis). "
                    f"Use fft_time instead."
                )
            domain = self._space_domains.get(ax_name)
            if domain != "real":
                raise ValueError(
                    f"Axis '{ax_name}' is not in 'real' domain (current: {domain}). "
                    f"Cannot apply fft_space."
                )
            # Check uniform spacing
            ax_desc = self.axis(ax_name)
            if not ax_desc.regular:
                raise ValueError(
                    f"Axis '{ax_name}' is not uniformly spaced. "
                    f"Cannot apply FFT."
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
            dx_value = ax_desc.delta.value
            dx_unit = ax_desc.delta.unit

            npts = dft.shape[ax_int]

            # Angular wavenumber: k = 2π * fftfreq(n, d=dx)
            k_values = 2 * np.pi * np.fft.fftfreq(npts, d=dx_value)
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
                    f"Cannot use ifft_space on axis 0. Use ifft_time instead."
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
            # k = 2π * fftfreq(n, d=dx)  =>  dx = 2π / (n * dk)
            k_axis = self.axis(ax_name).index
            if len(k_axis) > 1:
                dk = np.abs(k_axis[1] - k_axis[0])
                dk_value = dk.value
                dk_unit = dk.unit
            else:
                dk_value = 1.0
                dk_unit = k_axis.unit

            # dx = 2π / (n * dk)
            dx_value = 2 * np.pi / (npts * dk_value)
            dx_unit = 1 / dk_unit

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

"""
gwexpy.frequencyseries
----------------------

Extends gwpy.frequencyseries with matrix support and future extensibility.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from astropy import units as u
from gwpy.frequencyseries import FrequencySeries as BaseFrequencySeries

from gwexpy.fitting.mixin import FittingMixin
from gwexpy.interop import (
    from_hdf5_frequencyseries,
    from_pandas_frequencyseries,
    from_xarray_frequencyseries,
    to_hdf5_frequencyseries,
    to_pandas_frequencyseries,
    to_xarray_frequencyseries,
)
from gwexpy.interop._optional import require_optional
from gwexpy.types._stats import StatisticalMethodsMixin
from gwexpy.types.mixin import RegularityMixin, SignalAnalysisMixin

if TYPE_CHECKING:
    try:
        from gwpy.types.index import SeriesType
    except ImportError:
        pass

# Runtime fallback for SeriesType if not available from gwpy
try:
    from gwpy.types.index import SeriesType  # pragma: no cover
except ImportError:

    class SeriesType(Enum):  # type: ignore[no-redef]
        TIME = "time"
        FREQ = "freq"


# =============================
# FrequencySeries
# =============================


class FrequencySeries(
    SignalAnalysisMixin,
    RegularityMixin,
    FittingMixin,
    StatisticalMethodsMixin,
    BaseFrequencySeries,
):
    """Light wrapper of gwpy's FrequencySeries for compatibility and future extension."""

    def __new__(cls, *args: Any, **kwargs: Any) -> FrequencySeries:
        """
        Create a new FrequencySeries instance.

        This override filters out noise-generation parameters (fmin, fmax, df)
        that may be passed from gwexpy.noise functions but are not valid
        arguments for the parent FrequencySeries constructor.
        """
        # Remove noise-generation parameters that shouldn't be passed to parent
        for key in ["fmin", "fmax"]:
            kwargs.pop(key, None)

        return super().__new__(cls, *args, **kwargs)

    def __array_finalize__(self, obj: Any) -> None:
        """Propagate gwexpy-specific attributes through NumPy operations.

        This ensures that internal attributes (prefixed with `_gwex_`) are
        preserved when slicing, arithmetic operations, or other NumPy
        array manipulations create new instances.

        Parameters
        ----------
        obj : object or None
            The object from which this instance was created. None if called
            from __new__.
        """
        super().__array_finalize__(obj)
        if obj is None:
            return

        # Copy gwexpy internal attributes for FFT round-trip support
        _gwex_attrs = (
            "_gwex_fft_mode",
            "_gwex_target_nfft",
            "_gwex_pad_left",
            "_gwex_pad_right",
            "_gwex_original_n",
        )
        for attr in _gwex_attrs:
            if hasattr(obj, attr):
                setattr(self, attr, getattr(obj, attr))

    # --- Phase and Angle ---

    def phase(self, unwrap: bool = False) -> FrequencySeries:
        """
        Calculate the phase of this FrequencySeries.

        Parameters
        ----------
        unwrap : `bool`, optional
            If `True`, unwrap the phase to remove discontinuities.
            Default is `False`.

        Returns
        -------
        `FrequencySeries`
            The phase of the series, in radians.
        """
        val = np.angle(np.asarray(self.value))
        if unwrap:
            val = np.unwrap(val)

        name = self.name + "_phase" if self.name else "phase"

        return self.__class__(
            val,
            frequencies=self.frequencies,
            unit="rad",
            name=name,
            channel=self.channel,
            epoch=self.epoch,
        )

    def angle(self, unwrap: bool = False) -> FrequencySeries:
        """
        Calculate the phase angle of this FrequencySeries.

        Alias for `phase(unwrap=unwrap)`.

        Parameters
        ----------
        unwrap : `bool`, optional
            If `True`, unwrap the phase to remove discontinuities.
            Default is `False`.

        Returns
        -------
        `FrequencySeries`
            The phase of the series, in radians.
        """
        return self.phase(unwrap=unwrap)

    def degree(self, unwrap: bool = False) -> FrequencySeries:
        """
        Calculate the phase of this FrequencySeries in degrees.

        Parameters
        ----------
        unwrap : `bool`, optional
            If `True`, unwrap the phase before converting to degrees.

        Returns
        -------
        `FrequencySeries`
            The phase of the series, in degrees.
        """
        p = self.phase(unwrap=unwrap)
        val = np.rad2deg(p.value)

        name = self.name + "_phase_deg" if self.name else "phase_deg"

        return self.__class__(
            val,
            frequencies=self.frequencies,
            unit="deg",
            name=name,
            channel=self.channel,
            epoch=self.epoch,
        )

    # --- Calculus ---

    def differentiate(self, order: int = 1) -> FrequencySeries:
        """
        Differentiate the FrequencySeries in the frequency domain.

        Multiplies by (i * 2 * pi * f)^order.

        Parameters
        ----------
        order : `int`, optional
            Order of differentiation. Default is 1.

        Returns
        -------
        `FrequencySeries`
            The differentiated series.
        """
        if order == 0:
            return self.copy()

        f = self.frequencies.value
        # If frequencies has unit, we should handle it, but usually we just consider numerical differentiation
        # respective to the unit basis (Hz).
        # Typically x(t) -> dx/dt involves multiplying X(f) by i*2pi*f

        factor = (1j * 2 * np.pi * f) ** order
        val = self.value * factor

        # Update unit: unit * (Hz)^order = unit * (1/s)^order
        # Assuming f is in Hz.
        if self.unit:
            new_unit = self.unit * (u.Hz**order)
        else:
            new_unit = u.Hz**order

        name = f"d({self.name})/dt" if self.name else "derivative"
        if order > 1:
            name = (
                f"d^{order}({self.name})/dt^{order}"
                if self.name
                else f"{order}-th_derivative"
            )

        return self.__class__(
            val,
            frequencies=self.frequencies,
            unit=new_unit,
            name=name,
            channel=self.channel,
            epoch=self.epoch,
        )

    def integrate(self, order: int = 1) -> FrequencySeries:
        """
        Integrate the FrequencySeries in the frequency domain.

        Divides by (i * 2 * pi * f)^order.

        Parameters
        ----------
        order : `int`, optional
            Order of integration. Default is 1.

        Returns
        -------
        `FrequencySeries`
            The integrated series.
        """
        if order == 0:
            return self.copy()

        f = self.frequencies.value
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = (1j * 2 * np.pi * f) ** (-order)

        # Handle DC component (f=0): usually set to 0 or leave as inf/nan?
        # Standard practice often is 0 for DC integration or just ignore warning.
        # numpy will produce inf for f=0.
        # Let's set f=0 term to 0 to be safe/clean? Or leave it?
        # Usually zeroing out DC is safer for subsequent operations.
        if f[0] == 0:
            factor[0] = 0

        val = self.value * factor

        if self.unit:
            new_unit = self.unit * (u.s**order)
        else:
            new_unit = u.s**order

        name = f"int({self.name})dt" if self.name else "integral"
        if order > 1:
            name = (
                f"int^{order}({self.name})dt^{order}"
                if self.name
                else f"{order}-th_integral"
            )

        return self.__class__(
            val,
            frequencies=self.frequencies,
            unit=new_unit,
            name=name,
            channel=self.channel,
            epoch=self.epoch,
        )

    # --- dB / Logarithmic ---

    def to_db(self, ref: Any = 1.0, amplitude: bool = True) -> FrequencySeries:
        """
        Convert this series to decibels.

        Parameters
        ----------
        ref : `float` or `Quantity`, optional
            Reference value for 0 dB. Default is 1.0.
        amplitude : `bool`, optional
            If `True` (default), treat data as amplitude (20 * log10).
            If `False`, treat data as power (10 * log10).

        Returns
        -------
        `FrequencySeries`
            The series in dB.
        """
        val = self.value
        if isinstance(ref, u.Quantity):
            ref = ref.value

        # Handle magnitude
        mag = np.abs(val)

        # Avoid log(0)
        # We could use a small epsilon or let numpy handle -inf
        with np.errstate(divide="ignore"):
            log_val = np.log10(mag / ref)

        factor = 20.0 if amplitude else 10.0
        db_val = factor * log_val

        name = self.name + "_db" if self.name else "db"

        return self.__class__(
            db_val,
            frequencies=self.frequencies,
            unit="dB",
            name=name,
            channel=self.channel,
            epoch=self.epoch,
        )

    def plot(self, **kwargs: Any) -> Any:
        """Plot this FrequencySeries. Delegates to gwexpy.plot.Plot."""
        from gwexpy.plot import Plot

        return Plot(self, **kwargs)

    # --- Interop helpers ---
    def filterba(self, *args, **kwargs):
        """Apply a [b, a] filter to this FrequencySeries.
        Inherited from gwpy.
        """
        return super().filterba(*args, **kwargs)

    def to_pandas(
        self, index: str = "frequency", *, name: str | None = None, copy: bool = False
    ) -> Any:
        """Convert to pandas.Series."""
        return to_pandas_frequencyseries(self, index=index, name=name, copy=copy)

    @classmethod
    def from_pandas(cls: type[FrequencySeries], series: Any, **kwargs: Any) -> Any:
        """Create FrequencySeries from pandas.Series."""
        return from_pandas_frequencyseries(cls, series, **kwargs)

    # ===============================
    # 5. External Library Interop
    # ===============================

    def to_polars(
        self,
        name: str | None = None,
        as_dataframe: bool = True,
        frequencies: str = "frequency",
    ) -> Any:
        """
        Convert this series to a polars.DataFrame or polars.Series.

        Parameters
        ----------
        name : str, optional
            Name for the polars Series/Column.
        as_dataframe : bool, default True
            If True, returns a DataFrame with a 'frequency' column.
            If False, returns a raw Series of values.
        frequencies : str, default "frequency"
            Name of the frequency column (only if as_dataframe=True).

        Returns
        -------
        polars.DataFrame or polars.Series
        """
        require_optional("polars")
        if as_dataframe:
            from gwexpy.interop import to_polars_frequencyseries

            return to_polars_frequencyseries(self, index_column=frequencies)
        else:
            from gwexpy.interop import to_polars_series

            return to_polars_series(self, name=name)

    @classmethod
    def from_polars(
        cls: type[FrequencySeries],
        data: Any,
        frequencies: str | None = "frequency",
        **kwargs: Any,
    ) -> Any:
        """
        Create a FrequencySeries from a polars.DataFrame or polars.Series.

        Parameters
        ----------
        data : polars.DataFrame or polars.Series
            Input data.
        frequencies : str, optional
            If data is a DataFrame, name of the column to use as frequency.
        **kwargs
            Additional arguments passed to frequency series constructor.

        Returns
        -------
        FrequencySeries
        """
        pl = require_optional("polars")
        if isinstance(data, pl.DataFrame):
            # We reuse from_polars_dataframe but might need specific frequency logic
            from gwexpy.interop import from_polars_dataframe

            return from_polars_dataframe(cls, data, index_column=frequencies, **kwargs)
        else:
            from gwexpy.interop import from_polars_series

            return from_polars_series(cls, data, **kwargs)

    def to_tgraph(self, error: Any | None = None) -> Any:
        """
        Convert to ROOT TGraph or TGraphErrors.
        """
        from gwexpy.interop import to_tgraph

        return to_tgraph(self, error=error)

    def to_th1d(self, error: Any | None = None) -> Any:
        """
        Convert to ROOT TH1D.
        """
        from gwexpy.interop import to_th1d

        return to_th1d(self, error=error)

    @classmethod
    def from_root(
        cls: type[FrequencySeries], obj: Any, return_error: bool = False, **kwargs: Any
    ) -> Any:
        """
        Create FrequencySeries from ROOT TGraph or TH1.
        """
        from gwexpy.interop import from_root

        return from_root(cls, obj, return_error=return_error, **kwargs)

    def to_xarray(self, freq_coord: str = "Hz") -> Any:
        """Convert to xarray.DataArray."""
        return to_xarray_frequencyseries(self, freq_coord=freq_coord)

    @classmethod
    def from_xarray(cls: type[FrequencySeries], da: Any, **kwargs: Any) -> Any:
        """Create FrequencySeries from xarray.DataArray."""
        return from_xarray_frequencyseries(cls, da, **kwargs)

    def to_hdf5_dataset(
        self,
        group: Any,
        path: str,
        *,
        overwrite: bool = False,
        compression: str | None = None,
        compression_opts: Any = None,
    ) -> Any:
        """Write to HDF5 dataset within a group."""
        return to_hdf5_frequencyseries(
            self,
            group,
            path,
            overwrite=overwrite,
            compression=compression,
            compression_opts=compression_opts,
        )

    @classmethod
    def from_hdf5_dataset(cls: type[FrequencySeries], group: Any, path: str) -> Any:
        """Read FrequencySeries from HDF5 dataset."""
        return from_hdf5_frequencyseries(cls, group, path)

    # --- Time Calculus ---

    def ifft(
        self,
        *,
        mode: str = "auto",
        trim: bool = True,
        original_n: int | None = None,
        pad_left: int | None = None,
        pad_right: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Inverse FFT returning a gwexpy TimeSeries, supporting transient round-trip.

        Parameters
        ----------
        mode : {"auto", "gwpy", "transient"}
            auto: use transient restoration if `_gwex_fft_mode=="transient"` is detected, otherwise GWpy compatible.
        trim : bool
            Whether to remove padding and trim to original length during transient mode.
        original_n : int, optional
            Explicitly specify the length after restoration (takes priority).
        pad_right, pad_left : int, optional
            Specify padding lengths for transient mode to override defaults.
        """
        self._check_regular("ifft")
        from gwexpy.timeseries import TimeSeries

        mode_to_use = mode
        if mode == "auto":
            mode_to_use = (
                "transient"
                if getattr(self, "_gwex_fft_mode", None) == "transient"
                else "gwpy"
            )

        if mode_to_use == "gwpy":
            ts = super().ifft(**kwargs)
            return TimeSeries(
                ts.value,
                times=ts.times,
                unit=ts.unit,
                name=ts.name,
                channel=ts.channel,
                epoch=ts.epoch,
            )

        if mode_to_use != "transient":
            raise ValueError(f"Unknown ifft mode: {mode}")

        # transient inverse: undo factor2 and 1/n normalization
        n_freq = len(self)
        target_nfft = getattr(self, "_gwex_target_nfft", None) or (n_freq - 1) * 2
        spectrum = self.value.copy()
        if spectrum.shape[-1] > 1:
            spectrum[..., 1:] /= 2.0
        time_data = np.fft.irfft(spectrum * target_nfft, n=target_nfft)

        # derive dt from df and nfft
        if hasattr(self, "df") and self.df is not None:
            if hasattr(self.df, "unit"):
                dt = (1 / (self.df * target_nfft)).to("s")
            else:
                dt = u.Quantity(1.0 / (float(self.df) * target_nfft), "s")
        else:
            dt = None

        # trimming / pad removal
        pad_l = (
            pad_left
            if pad_left is not None
            else getattr(self, "_gwex_pad_left", 0) or 0
        )
        pad_r = (
            pad_right
            if pad_right is not None
            else getattr(self, "_gwex_pad_right", 0) or 0
        )
        data_trim = time_data
        if trim and (pad_l or pad_r):
            if pad_r == 0:
                data_trim = data_trim[pad_l:]
            else:
                data_trim = data_trim[pad_l:-pad_r]

        target_n = (
            original_n
            if original_n is not None
            else getattr(self, "_gwex_original_n", None)
        )
        if trim and target_n is not None:
            data_trim = data_trim[:target_n]

        return TimeSeries(
            data_trim,
            t0=getattr(self, "epoch", None),
            dt=dt,
            unit=self.unit,
            name=self.name,
            channel=self.channel,
        )

    def idct(self, type: int = 2, norm: str = "ortho", *, n: int | None = None) -> Any:
        """
        Compute the Inverse Discrete Cosine Transform (IDCT).

        Reconstructs a time-domain signal from DCT coefficients.

        Parameters
        ----------
        type : int, optional
            DCT type (1, 2, 3, or 4). Should match the type used for
            the forward DCT. Default is 2.
        norm : str, optional
            Normalization mode: 'ortho' for orthonormal, None for standard.
            Default is 'ortho'.
        n : int, optional
            Length of the output time series. If None, uses the stored
            `original_n` attribute if available.

        Returns
        -------
        TimeSeries
            The reconstructed time series.

        Notes
        -----
        For a proper roundtrip, use the same `type` and `norm` as the
        forward DCT transform.

        Examples
        --------
        >>> fs = FrequencySeries(dct_coeffs, ...)
        >>> ts = fs.idct()
        """
        self._check_regular("idct")

        scipy_fft = require_optional("scipy.fft")

        # Check metadata if available
        meta_n = getattr(self, "original_n", None)
        meta_dt = getattr(self, "dt", None)

        target_n = n if n is not None else meta_n

        # IDCT
        out_data = scipy_fft.idct(self.value, type=type, norm=norm, n=target_n)

        # Determine dt
        if meta_dt is not None:
            dt = meta_dt
        elif self.df is not None:
            N_out = len(out_data)
            # Handle quantity df
            if isinstance(self.df, u.Quantity):
                df_val = self.df.to("Hz").value
            else:
                df_val = self.df

            with np.errstate(divide="ignore"):
                if df_val > 0:
                    dt_val = 1.0 / (2 * N_out * df_val)
                    dt = u.Quantity(dt_val, "s")
                else:
                    dt = 1.0 * u.s  # Fallback
        else:
            dt = 1.0 * u.s

        from gwexpy.timeseries import TimeSeries

        return TimeSeries(
            out_data,
            t0=self.epoch,
            dt=dt,
            unit=self.unit,
            name=self.name + "_idct" if self.name else "idct",
            channel=self.channel,
        )

    def differentiate_time(self) -> Any:
        """
        Apply time differentiation in frequency domain.

        Multiplies by (2 * pi * i * f).
        Converting Displacement -> Velocity -> Acceleration.

        Returns
        -------
        `FrequencySeries`
        """
        f = self.frequencies.to("Hz").value
        omega = 2 * np.pi * f

        # Apply factor (j * omega)
        factor = 1j * omega

        new_val = self.value * factor

        # Update unit: multiply by Hz (1/s)
        new_unit = self.unit * u.Hz if self.unit else u.Hz

        name = f"d({self.name})/dt" if self.name else "differentiation"

        return self.__class__(
            new_val,
            frequencies=self.frequencies,
            unit=new_unit,
            name=name,
            channel=self.channel,
            epoch=self.epoch,
        )

    def integrate_time(self) -> Any:
        """
        Apply time integration in frequency domain.

        Divides by (2 * pi * i * f).
        Converting Acceleration -> Velocity -> Displacement.

        Returns
        -------
        `FrequencySeries`
        """
        f = self.frequencies.to("Hz").value
        omega = 2 * np.pi * f

        # Avoid division by zero at DC
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = 1.0 / (1j * omega)

        # Handle DC (0 Hz) -> set to 0 or leave as inf/nan?
        # Standard practice is often to set DC to 0 for integration results or keep it if known.
        # Here we let numpy handle it (likely inf), but the user might want to mask it.
        if f[0] == 0:
            factor[0] = 0  # Set DC term to 0 explicitly to avoid NaN propagation

        new_val = self.value * factor

        # Update unit: divide by Hz (multiply by s)
        new_unit = self.unit * u.s if self.unit else u.s

        name = f"int({self.name})dt" if self.name else "integration"

        return self.__class__(
            new_val,
            frequencies=self.frequencies,
            unit=new_unit,
            name=name,
            channel=self.channel,
            epoch=self.epoch,
        )

    # smooth() and find_peaks() are now inherited from SignalAnalysisMixin

    def quadrature_sum(self, other: Any) -> Any:
        """
        Compute sqrt(self^2 + other^2) assuming checking independence.

        Operates on magnitude. Phase information is lost (returns real).

        Parameters
        ----------
        other : `FrequencySeries`
            The other series to add.

        Returns
        -------
        `FrequencySeries`
            Magnitude combined series.
        """
        mag_self = np.abs(self.value)
        mag_other = np.abs(other.value)

        # Check compatibility?

        val = np.sqrt(mag_self**2 + mag_other**2)

        return self.__class__(
            val,
            frequencies=self.frequencies,
            unit=self.unit,
            name=f"sqrt({self.name}^2 + {other.name}^2)",
            epoch=self.epoch,
        )

    def group_delay(self) -> Any:
        """
        Calculate the group delay of the series.

        Group delay is defined as -d(phase)/d(omega), where omega = 2 * pi * f.
        It represents the time delay of the envelope of a signal at a given frequency.

        Returns
        -------
        `FrequencySeries`
            A new FrequencySeries representing the group delay in seconds.
        """
        # Gradient of unwrapped phase w.r.t frequency
        orig_phase = self.phase(unwrap=True).value
        freqs = self.frequencies.value

        # d(phi)/dw = d(phi) / (2pi * df)
        # derivative w.r.t. frequency in Hz
        d_phi_d_f = np.gradient(orig_phase, freqs)

        # group delay = - d(phi)/dw = - (1/2pi) * d(phi)/df
        gd = -1 / (2 * np.pi) * d_phi_d_f

        return self.__class__(
            gd,
            frequencies=self.frequencies,
            unit="s",
            name=self.name + "_group_delay" if self.name else "group_delay",
            channel=self.channel,
            epoch=self.epoch,
        )

    def rebin(self, width: float | u.Quantity) -> FrequencySeries:
        """
        Rebin the FrequencySeries to a new resolution.

        Parameters
        ----------
        width : `float` or `Quantity`
            New bin width in Hz.

        Returns
        -------
        `FrequencySeries`
            The rebinned series.
        """
        if isinstance(width, u.Quantity):
            width = width.to("Hz").value

        df = self.df.to("Hz").value if hasattr(self.df, "to") else self.df
        bin_size = int(round(width / df))

        if bin_size <= 1:
            return self.copy()

        data = self.value
        n_freq = len(data)
        n_freq_new = n_freq // bin_size

        # Truncate
        if n_freq_new * bin_size != n_freq:
            data = data[: n_freq_new * bin_size]

        # Rebin
        data_rebinned = data.reshape(n_freq_new, bin_size).mean(axis=1)

        # New frequency axis
        freqs = self.frequencies
        if n_freq_new * bin_size != n_freq:
            freqs = freqs[: n_freq_new * bin_size]
        freqs_rebinned = freqs.reshape(n_freq_new, bin_size).mean(axis=1)

        return self.__class__(
            data_rebinned,
            frequencies=freqs_rebinned,
            unit=self.unit,
            name=self.name,
            channel=self.channel,
            epoch=self.epoch,
        )

    def to_control_frd(self, frequency_unit: str = "Hz") -> Any:
        """Convert to control.FRD."""
        from gwexpy.interop import to_control_frd

        return to_control_frd(self, frequency_unit=frequency_unit)

    @classmethod
    def from_control_frd(
        cls: type[FrequencySeries], frd: Any, *, frequency_unit: str = "Hz"
    ) -> Any:
        """Create from control.FRD."""
        from gwexpy.interop import from_control_frd

        return from_control_frd(cls, frd, frequency_unit=frequency_unit)

    # --- ML Framework Interop ---

    def to_torch(
        self,
        device: str | None = None,
        dtype: Any = None,
        requires_grad: bool = False,
        copy: bool = False,
    ) -> Any:
        """
        Convert to torch.Tensor.

        Parameters
        ----------
        device : `str` or `torch.device`, optional
            Target device (e.g. 'cpu', 'cuda').
        dtype : `torch.dtype`, optional
            Target data type. Defaults to preserving complex64/128 or float32/64.
        requires_grad : `bool`, optional
            If `True`, enable gradient tracking.
        copy : `bool`, optional
            If `True`, force a copy of the data.

        Returns
        -------
        `torch.Tensor`
        """
        from gwexpy.interop.torch_ import to_torch

        return to_torch(
            self, device=device, dtype=dtype, requires_grad=requires_grad, copy=copy
        )

    @classmethod
    def from_torch(
        cls: type[FrequencySeries],
        tensor: Any,
        frequencies: Any,
        unit: Any | None = None,
    ) -> Any:
        """
        Create FrequencySeries from torch.Tensor.

        Parameters
        ----------
        tensor : `torch.Tensor`
            Input tensor.
        frequencies : `Array` or `Quantity`
            Frequency array matching the tensor size.
        unit : `Unit` or `str`, optional
            Data unit.

        Returns
        -------
        `FrequencySeries`
        """
        # Wrapper to adapt signature (from_torch calls cls(data, t0, dt) for TimeSeries)
        # We need a custom implementation for FrequencySeries or adapt from_torch logic.
        # Since from_torch in interop is tailored for TimeSeries (takes t0, dt),
        # we pull the data extraction logic specifically.

        data = tensor.detach().cpu().resolve_conj().resolve_neg().numpy()
        return cls(data, frequencies=frequencies, unit=unit)

    def to_tensorflow(self, dtype: Any = None) -> Any:
        """
        Convert to tensorflow.Tensor.

        Returns
        -------
        `tensorflow.Tensor`
        """
        from gwexpy.interop.tensorflow_ import to_tf

        return to_tf(self, dtype=dtype)

    @classmethod
    def from_tensorflow(
        cls: type[FrequencySeries],
        tensor: Any,
        frequencies: Any,
        unit: Any | None = None,
    ) -> Any:
        """Create FrequencySeries from tensorflow.Tensor."""
        data = tensor.numpy()
        return cls(data, frequencies=frequencies, unit=unit)

    def to_jax(self, dtype: Any = None) -> Any:
        """
        Convert to JAX array.

        Returns
        -------
        `jax.Array`
        """
        from gwexpy.interop.jax_ import to_jax

        return to_jax(self, dtype=dtype)

    @classmethod
    def from_jax(
        cls: type[FrequencySeries],
        array: Any,
        frequencies: Any,
        unit: Any | None = None,
    ) -> Any:
        """Create FrequencySeries from JAX array."""
        import numpy as np

        data = np.array(array)
        return cls(data, frequencies=frequencies, unit=unit)

    def to_cupy(self, dtype: Any = None) -> Any:
        """
        Convert to CuPy array.

        Returns
        -------
        `cupy.ndarray`
        """
        from gwexpy.interop.cupy_ import to_cupy

        return to_cupy(self, dtype=dtype)

    @classmethod
    def from_cupy(
        cls: type[FrequencySeries],
        array: Any,
        frequencies: Any,
        unit: Any | None = None,
    ) -> Any:
        """Create FrequencySeries from CuPy array."""
        cp = require_optional("cupy")
        data = cp.asnumpy(array)
        return cls(data, frequencies=frequencies, unit=unit)

    # --- Domain Specific Interop ---

    def to_quantities(self, units: str | None = None) -> Any:
        """
        Convert to quantities.Quantity (Elephant/Neo compatible).

        Parameters
        ----------
        units : str or quantities.UnitQuantity, optional
            Target units.

        Returns
        -------
        quantities.Quantity
        """
        from gwexpy.interop import to_quantity

        return to_quantity(self, units=units)

    @classmethod
    def from_quantities(cls: type[FrequencySeries], q: Any, frequencies: Any) -> Any:
        """
        Create FrequencySeries from quantities.Quantity.

        Parameters
        ----------
        q : quantities.Quantity
            Input data.
        frequencies : array-like
            Frequencies corresponding to the data.

        Returns
        -------
        FrequencySeries
        """
        from gwexpy.interop import from_quantity

        return from_quantity(cls, q, frequencies=frequencies)

    def to_mne(self, info: Any | None = None) -> Any:
        """
        Convert to MNE-Python object.

        Parameters
        ----------
        info : mne.Info, optional
            MNE Info object.

        Returns
        -------
        mne.time_frequency.SpectrumArray
        """
        from gwexpy.interop import to_mne

        return to_mne(self, info=info)

    @classmethod
    def from_mne(cls: type[FrequencySeries], spectrum: Any, **kwargs: Any) -> Any:
        """
        Create FrequencySeries from MNE-Python Spectrum object.

        Parameters
        ----------
        spectrum : mne.time_frequency.Spectrum
            Input spectrum data.
        **kwargs
            Additional arguments passed to constructor.

        Returns
        -------
        FrequencySeries or FrequencySeriesDict
        """
        from gwexpy.interop import from_mne

        return from_mne(cls, spectrum, **kwargs)

    def to_obspy(self, **kwargs: Any) -> Any:
        """
        Convert to Obspy Trace.

        Returns
        -------
        obspy.Trace
        """
        from gwexpy.interop import to_obspy

        return to_obspy(self, **kwargs)

    @classmethod
    def from_obspy(cls, trace: Any, **kwargs: Any) -> Any:
        """
        Create FrequencySeries from Obspy Trace.

        Parameters
        ----------
        trace : obspy.Trace
            Input trace.
        **kwargs
            Additional arguments.

        Returns
        -------
        FrequencySeries
        """
        from gwexpy.interop import from_obspy

        return from_obspy(cls, trace, **kwargs)

    def to_simpeg(
        self, location=None, rx_type="PointElectricField", orientation="x", **kwargs
    ) -> Any:
        """
        Convert to SimPEG Data object.

        Parameters
        ----------
        location : array_like, optional
            Rx location (x, y, z). Default is [0, 0, 0].
        rx_type : str, optional
            Receiver class name. Default "PointElectricField".
        orientation : str, optional
            Receiver orientation ('x', 'y', 'z'). Default 'x'.

        Returns
        -------
        simpeg.data.Data
        """
        from gwexpy.interop import to_simpeg

        return to_simpeg(
            self, location=location, rx_type=rx_type, orientation=orientation, **kwargs
        )

    @classmethod
    def from_simpeg(cls, data_obj: Any, **kwargs: Any) -> Any:
        """
        Create FrequencySeries from SimPEG Data object.

        Parameters
        ----------
        data_obj : simpeg.data.Data
            Input SimPEG Data.

        Returns
        -------
        FrequencySeries
        """
        from gwexpy.interop import from_simpeg

        return from_simpeg(cls, data_obj, **kwargs)

    def to_specutils(self, **kwargs):
        """
        Convert to specutils.Spectrum1D.

        Parameters
        ----------
        **kwargs
            Arguments passed to Spectrum1D constructor.

        Returns
        -------
        specutils.Spectrum1D
        """
        from gwexpy.interop import to_specutils

        return to_specutils(self, **kwargs)

    @classmethod
    def from_specutils(cls, spectrum, **kwargs):
        """
        Create FrequencySeries from specutils.Spectrum1D.

        Parameters
        ----------
        spectrum : specutils.Spectrum1D
            Input spectrum.

        Returns
        -------
        FrequencySeries
        """
        from gwexpy.interop import from_specutils

        return from_specutils(cls, spectrum, **kwargs)

    def to_pyspeckit(self, **kwargs):
        """
        Convert to pyspeckit.Spectrum.

        Parameters
        ----------
        **kwargs
            Arguments passed to pyspeckit.Spectrum constructor.

        Returns
        -------
        pyspeckit.Spectrum
        """
        from gwexpy.interop import to_pyspeckit

        return to_pyspeckit(self, **kwargs)

    @classmethod
    def from_pyspeckit(cls, spectrum, **kwargs):
        """
        Create FrequencySeries from pyspeckit.Spectrum.

        Parameters
        ----------
        spectrum : pyspeckit.Spectrum
            Input spectrum.

        Returns
        -------
        FrequencySeries
        """
        from gwexpy.interop import from_pyspeckit

        return from_pyspeckit(cls, spectrum, **kwargs)


# =============================
# Helpers
# =============================


def as_series_dict_class(seriesclass):
    """Decorate a `dict` class as the `DictClass` for a series class.

    This mirrors `gwpy.timeseries.core.as_series_dict_class` and allows
    `FrequencySeries.DictClass` to point to the matching dict container.
    """

    def decorate_class(cls: type[FrequencySeries]) -> type[FrequencySeries]:
        seriesclass.DictClass = cls
        return cls

    return decorate_class


# =============================
# FrequencySeries containers (MVP)
# =============================

_FS = TypeVar("_FS", bound=FrequencySeries)

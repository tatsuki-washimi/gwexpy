"""
gwexpy.frequencyseries
----------------------

Extends gwpy.frequencyseries with matrix support and future extensibility.
"""

from __future__ import annotations

from collections import OrderedDict
from enum import Enum
from typing import Any, Iterable, Optional, TypeVar

import numpy as np
from astropy import units as u
import scipy.signal

from gwpy.frequencyseries import FrequencySeries as BaseFrequencySeries
from gwexpy.types.seriesmatrix import SeriesMatrix
from gwexpy.interop import (
    to_pandas_frequencyseries,
    from_pandas_frequencyseries,
    to_xarray_frequencyseries,
    from_xarray_frequencyseries,
    to_hdf5_frequencyseries,
    from_hdf5_frequencyseries,
)

try:
    from gwpy.types.index import SeriesType  # pragma: no cover - optional in gwpy
except ImportError:
    class SeriesType(Enum):
        TIME = "time"
        FREQ = "freq"

# =============================
# FrequencySeries
# =============================

class FrequencySeries(BaseFrequencySeries):
    """Light wrapper of gwpy's FrequencySeries for compatibility and future extension."""

    # --- Phase and Angle ---

    def phase(self, unwrap=False):
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
        val = np.angle(self.value)
        if unwrap:
            val = np.unwrap(val)
        
        name = self.name + "_phase" if self.name else "phase"
        
        return self.__class__(
            val,
            frequencies=self.frequencies,
            unit="rad",
            name=name,
            channel=self.channel,
            epoch=self.epoch
        )

    def angle(self, unwrap=False):
        """Alias for `phase(unwrap=unwrap)`."""
        return self.phase(unwrap=unwrap)
    
    def degree(self, unwrap=False):
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
            epoch=self.epoch
        )

    # --- dB / Logarithmic ---

    def to_db(self, ref=1.0, amplitude=True):
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
        with np.errstate(divide='ignore'):
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
            epoch=self.epoch
        )

    # --- Interop helpers ---
    def to_pandas(self, index="frequency", *, name=None, copy=False):
        return to_pandas_frequencyseries(self, index=index, name=name, copy=copy)

    @classmethod
    def from_pandas(cls, series, **kwargs):
        return from_pandas_frequencyseries(cls, series, **kwargs)

    def to_xarray(self, freq_coord="Hz"):
        return to_xarray_frequencyseries(self, freq_coord=freq_coord)

    @classmethod
    def from_xarray(cls, da, **kwargs):
        return from_xarray_frequencyseries(cls, da, **kwargs)

    def to_hdf5_dataset(self, group, path, *, overwrite=False, compression=None, compression_opts=None):
        return to_hdf5_frequencyseries(
            self,
            group,
            path,
            overwrite=overwrite,
            compression=compression,
            compression_opts=compression_opts,
        )

    @classmethod
    def from_hdf5_dataset(cls, group, path):
        return from_hdf5_frequencyseries(cls, group, path)

    # --- Time Calculus ---

    def ifft(self, *, mode="auto", trim=True, original_n=None, pad_left=None, pad_right=None, **kwargs):
        """
        Inverse FFT returning a gwexpy TimeSeries, supporting transient round-trip.

        Parameters
        ----------
        mode : {"auto","gwpy","transient"}
            auto: use transient復元 if `_gwex_fft_mode=="transient"` を検出、なければ gwpy互換。
        trim : bool
            transient時にパディングを除去し、元長へトリムするか。
        original_n : int, optional
            明示的に復元後の長さを指定（指定が優先）。
        pad_left, pad_right : int, optional
            transient用のパディング長を上書きしたい場合に指定。
        """
        from gwexpy.timeseries import TimeSeries
        mode_to_use = mode
        if mode == "auto":
            mode_to_use = "transient" if getattr(self, "_gwex_fft_mode", None) == "transient" else "gwpy"

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
        pad_l = pad_left if pad_left is not None else getattr(self, "_gwex_pad_left", 0) or 0
        pad_r = pad_right if pad_right is not None else getattr(self, "_gwex_pad_right", 0) or 0
        data_trim = time_data
        if trim and (pad_l or pad_r):
            if pad_r == 0:
                data_trim = data_trim[pad_l:]
            else:
                data_trim = data_trim[pad_l:-pad_r]

        target_n = original_n if original_n is not None else getattr(self, "_gwex_original_n", None)
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

    def differentiate_time(self):
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
            epoch=self.epoch
        )

    def integrate_time(self):
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
        with np.errstate(divide='ignore', invalid='ignore'):
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
            epoch=self.epoch
        )

    # --- Analysis & Smoothing ---

    def smooth(self, width, method='amplitude'):
        """
        Smooth the frequency series.
        
        Parameters
        ----------
        width : `int`
            Number of samples for the smoothing winow (e.g. rolling mean size).
        method : `str`, optional
            Smoothing target:
            - 'amplitude': Smooth absolute value |X|, keep phase 'original' (or 0?). 
                           Actually, strictly smoothing magnitude destroys phase coherence. 
                           Returns REAL series (magnitude only).
            - 'power': Smooth power |X|^2. Returns REAL series.
            - 'complex': Smooth real and imaginary parts separately. Preserves complex.
            - 'db': Smooth dB values. Returns REAL series (in dB).
            
        Returns
        -------
        `FrequencySeries`
        """
        if method == 'complex':
            # Convolve/filter real and imag separately
            w = np.ones(width) / width
            # Use 'same' to keep size, but handle boundary effects
            # scipy.ndimage.uniform_filter1d or regular convolution
            # Let's use simple convolution for MVP
            
            # Boundary handling: 'valid' shrinks, 'same' has edge effects.
            # GWpy generally prefers exact preservation or careful handling.
            # We will use 'same' mode convolution.
            from scipy.ndimage import uniform_filter1d
            
            re = uniform_filter1d(self.value.real, size=width)
            im = uniform_filter1d(self.value.imag, size=width)
            val = re + 1j * im
            unit = self.unit
            
        elif method == 'amplitude':
            mag = np.abs(self.value)
            from scipy.ndimage import uniform_filter1d
            val = uniform_filter1d(mag, size=width)
            unit = self.unit
            
        elif method == 'power':
            pwr = np.abs(self.value)**2
            from scipy.ndimage import uniform_filter1d
            val = uniform_filter1d(pwr, size=width)
            unit = self.unit**2
            
        elif method == 'db':
            # To dB
            mag = np.abs(self.value)
            with np.errstate(divide='ignore'):
                 db = 20 * np.log10(mag)
            from scipy.ndimage import uniform_filter1d
            # Handle -inf in dB (0 magnitude) by masking? 
            # For now just smooth what we have
            val = uniform_filter1d(db, size=width)
            unit = u.Unit('dB')
            
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

        return self.__class__(
            val,
            frequencies=self.frequencies,
            unit=unit,
            name=self.name,
            channel=self.channel,
            epoch=self.epoch
        )

    def find_peaks(self, threshold=None, method='amplitude', **kwargs):
        """
        Find peaks in the series.
        
        Wraps `scipy.signal.find_peaks`.
        
        Parameters
        ----------
        threshold : `float` or `str`
            Height threshold.
        method : `str`, optional
            'amplitude', 'power', 'db'. Defines what metric to search on.
        **kwargs
            Passed to `scipy.signal.find_peaks` (e.g. distance, prominence).
            
        Returns
        -------
        `tuple`
            (peaks_indices, peaks_properties)
        """
        # Prepare target array
        if method == 'amplitude':
            target = np.abs(self.value)
        elif method == 'power':
            target = np.abs(self.value)**2
        elif method == 'db':
            target = 20 * np.log10(np.abs(self.value))
        else:
            raise ValueError(f"Unknown method {method}")
            
        if threshold is not None:
            if hasattr(threshold, 'unit'):  # astropy.units.Quantity
                if method == 'amplitude':
                    threshold = threshold.to(self.unit).value
                elif method == 'power':
                    threshold = threshold.to(self.unit**2).value
                elif method == 'db':
                    # dB is logically dimensionless but let's just use .value if it's already dB-like?
                    # Usually users pass plain float for dB.
                    threshold = threshold.value
            kwargs['height'] = threshold
             
        return scipy.signal.find_peaks(target, **kwargs)

    def quadrature_sum(self, other):
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
            epoch=self.epoch
        )

    def group_delay(self):
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
            epoch=self.epoch
        )
        
    def to_control_frd(self, frequency_unit="Hz"):
        """Convert to control.FRD."""
        from gwexpy.interop import to_control_frd
        return to_control_frd(self, frequency_unit=frequency_unit)
        
    @classmethod
    def from_control_frd(cls, frd, *, frequency_unit="Hz"):
        """Create from control.FRD."""
        from gwexpy.interop import from_control_frd
        return from_control_frd(cls, frd, frequency_unit=frequency_unit)

# =============================
# Helpers
# =============================

def as_series_dict_class(seriesclass):
    """Decorate a `dict` class as the `DictClass` for a series class.

    This mirrors `gwpy.timeseries.core.as_series_dict_class` and allows
    `FrequencySeries.DictClass` to point to the matching dict container.
    """

    def decorate_class(cls):
        seriesclass.DictClass = cls
        return cls

    return decorate_class


# =============================
# FrequencySeries containers (MVP)
# =============================

_FS = TypeVar("_FS", bound=FrequencySeries)


class FrequencySeriesBaseDict(OrderedDict[str, _FS]):
    """Ordered mapping container for `FrequencySeries` objects.

    This is a lightweight GWpy-inspired container:
    - enforces `EntryClass` on insertion/update
    - provides map-style helpers (`copy`, `crop`, `plot`)
    - default values for setdefault() must be FrequencySeries (None not allowed)

    Non-trivial operations (I/O, fetching, axis coercion, joins) are
    intentionally out-of-scope for this MVP.
    """

    EntryClass = FrequencySeries

    @property
    def span(self):
        """Frequency extent across all elements (based on xspan)."""
        from gwpy.segments import SegmentList

        span = SegmentList([val.xspan for val in self.values()])
        try:
            return span.extent()
        except ValueError as exc:  # empty
            exc.args = (f"cannot calculate span for empty {type(self).__name__}",)
            raise

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        if args or kwargs:
            self.update(*args, **kwargs)

    def __setitem__(self, key: str, value: _FS) -> None:
        if not isinstance(value, self.EntryClass):
            raise TypeError(
                f"Cannot set key '{key}' to type '{type(value).__name__}' in {type(self).__name__}"
            )
        super().__setitem__(key, value)

    def setdefault(self, key: str, default: Optional[_FS] = None) -> _FS:  # type: ignore[override]
        if key in self:
            return self[key]
        if default is None:
            raise TypeError(
                f"Cannot set default None for {type(self).__name__}; expected {self.EntryClass.__name__}"
            )
        if not isinstance(default, self.EntryClass):
            raise TypeError(
                f"Cannot set default type '{type(default).__name__}' in {type(self).__name__}"
            )
        self[key] = default
        return default

    def copy(self) -> "FrequencySeriesBaseDict[_FS]":
        new = self.__class__()
        for key, val in self.items():
            new[key] = val.copy()
        return new

    def crop(
        self, start: Any = None, end: Any = None, copy: bool = False
    ) -> "FrequencySeriesBaseDict[_FS]":
        for key, val in list(self.items()):
            self[key] = val.crop(start=start, end=end, copy=copy)
        return self

    def plot(
        self,
        label: str = "key",
        method: str = "plot",
        figsize: Optional[Any] = None,
        **kwargs: Any,
    ):
        from gwpy.plot import Plot

        kwargs = dict(kwargs)
        separate = kwargs.get("separate", False)
        if figsize is not None:
            kwargs.setdefault("figsize", figsize)
        kwargs.update({"label": label, "method": method})

        if separate:
            plot = Plot(*self.values(), **kwargs)
        else:
            plot = Plot(self.values(), **kwargs)

        artmap = {"plot": "lines", "scatter": "collections"}
        artists = [
            artist
            for ax in plot.axes
            for artist in getattr(ax, artmap.get(method, "lines"))
        ]

        label_key = label.lower()
        for key, artist in zip(self, artists):
            if label_key == "name":
                lab = self[key].name
            elif label_key == "key":
                lab = key
            else:
                lab = label
            artist.set_label(lab)

        return plot

    def plot_all(self, *args: Any, **kwargs: Any):
        return self.plot(*args, **kwargs)


@as_series_dict_class(FrequencySeries)
class FrequencySeriesDict(FrequencySeriesBaseDict[FrequencySeries]):
    """Ordered mapping of `FrequencySeries` objects keyed by label."""

    EntryClass = FrequencySeries


class FrequencySeriesBaseList(list[_FS]):
    """List container for `FrequencySeries` objects with type enforcement."""

    EntryClass = FrequencySeries

    def __init__(self, *items: _FS):
        super().__init__()
        for item in items:
            self.append(item)

    @property
    def segments(self):
        """Frequency spans of each element (xspan)."""
        from gwpy.segments import SegmentList

        return SegmentList([item.xspan for item in self])

    def _validate(self, item: Any, *, op: str) -> None:
        if not isinstance(item, self.EntryClass):
            raise TypeError(
                f"Cannot {op} type '{type(item).__name__}' to {type(self).__name__}"
            )

    def append(self, item: _FS):  # type: ignore[override]
        self._validate(item, op="append")
        super().append(item)
        return self

    def extend(self, items: Iterable[_FS]) -> None:  # type: ignore[override]
        validated = self.__class__(*items)
        super().extend(validated)

    def insert(self, index: int, item: _FS) -> None:  # type: ignore[override]
        self._validate(item, op="insert")
        super().insert(index, item)

    def __setitem__(self, index, item) -> None:  # type: ignore[override]
        if isinstance(index, slice):
            validated = self.__class__(*item)
            super().__setitem__(index, validated)
            return
        self._validate(item, op="set")
        super().__setitem__(index, item)

    def __getitem__(self, index):  # type: ignore[override]
        if isinstance(index, slice):
            return self.__class__(*super().__getitem__(index))
        return super().__getitem__(index)

    def copy(self) -> "FrequencySeriesBaseList[_FS]":
        return self.__class__(*(item.copy() for item in self))

    def plot(self, **kwargs: Any):
        from gwpy.plot import Plot

        return Plot(self, **kwargs)

    def plot_all(self, *args: Any, **kwargs: Any):
        return self.plot(*args, **kwargs)


class FrequencySeriesList(FrequencySeriesBaseList[FrequencySeries]):
    """List of `FrequencySeries` objects."""

    EntryClass = FrequencySeries


# =============================
# FrequencySeriesMatrix
# =============================

class FrequencySeriesMatrix(SeriesMatrix):
    """
    Matrix container for multiple FrequencySeries objects.

    Inherits from SeriesMatrix and returns FrequencySeries instances when indexed.
    """
    series_class = FrequencySeries
    series_type = SeriesType.FREQ
    default_xunit = "Hz"
    default_yunit = None
    _default_plot_method = "plot"

    def __new__(cls, data=None, frequencies=None, df=None, f0=None, **kwargs):
        channel_names = kwargs.pop("channel_names", None)

        # Map frequency-specific arguments to SeriesMatrix generic arguments
        if frequencies is not None:
            kwargs['xindex'] = frequencies
        if df is not None:
            kwargs['dx'] = df
        if f0 is not None:
            kwargs['x0'] = f0
        elif frequencies is None and df is not None and 'x0' not in kwargs:
            # Default f0 to 0 if not specified but df is provided (requiring explicit axis)
            kwargs['x0'] = 0
        
        # Set default xunit to Hz if not specified
        if 'xunit' not in kwargs:
            kwargs['xunit'] = cls.default_xunit

        obj = super().__new__(cls, data, **kwargs)
        if channel_names is not None:
            obj.channel_names = list(channel_names)
        return obj

    # --- Properties mapping to SeriesMatrix attributes ---

    @property
    def df(self):
        """Frequency spacing (dx)."""
        return self.dx

    @property
    def f0(self):
        """Starting frequency (x0)."""
        return self.x0

    @property
    def frequencies(self):
        """Frequency array (xindex)."""
        return self.xindex

    def _repr_string_(self):
        if self.size > 0:
            u = self.meta[0, 0].unit
        else:
            u = None
        return f"<FrequencySeriesMatrix shape={self.shape}, df={self.df}, unit={u}>"

    @frequencies.setter
    def frequencies(self, value):
        self.xindex = value

    # --- Methods ---

    def __getitem__(self, item):
        """
        Return FrequencySeries for single element access, or FrequencySeriesMatrix for slicing.
        """
        if isinstance(item, tuple) and len(item) == 2:
            r, c = item
            is_scalar_r = isinstance(r, (int, np.integer, str))
            is_scalar_c = isinstance(c, (int, np.integer, str))

            if is_scalar_r and is_scalar_c:
                ri = self.row_index(r) if isinstance(r, str) else r
                ci = self.col_index(c) if isinstance(c, str) else c

                val = self._value[ri, ci]
                meta = self.meta[ri, ci]

                return self.series_class(
                    val,
                    frequencies=self.frequencies,
                    unit=meta.unit,
                    name=meta.name,
                    channel=meta.channel,
                    epoch=getattr(self, "epoch", None),
                )

        ret = super().__getitem__(item)
        if isinstance(ret, SeriesMatrix) and not isinstance(ret, FrequencySeriesMatrix):
            return ret.view(FrequencySeriesMatrix)
        return ret

    def ifft(self):
        """
        Compute the inverse FFT of this frequency-domain matrix.
        
        Matches GWpy FrequencySeries.ifft normalization.

        Returns
        -------
        TimeSeriesMatrix
            The time-domain matrix resulting from the inverse FFT.
        """
        import numpy.fft as fft
        from astropy import units as u
        from gwexpy.timeseries import TimeSeriesMatrix

        n_freq = self.shape[-1]
        nout = (n_freq - 1) * 2

        # Undo normalization from TimeSeries.fft (GWpy logic):
        # the DC component does not have the factor of two applied.
        spectrum = self.value.copy()
        spectrum[..., 1:] /= 2.0
        time_data = fft.irfft(spectrum * nout, n=nout, axis=-1)

        # 4. Metadata
        # dt = 1 / (df * nout)
        if isinstance(self.df, u.Quantity):
            dt = (1 / (self.df * nout)).to("s")
        else:
            dt = u.Quantity(1.0 / (float(self.df) * nout), "s")

        return TimeSeriesMatrix(
            time_data,
            meta=self.meta,
            dt=dt,
            epoch=self.epoch,
            name=getattr(self, 'name', ""),
            rows=self.rows,
            cols=self.cols,
            xunit='s'
        )

    def filter(self, *filt, **kwargs):
        """
        Apply a filter to the FrequencySeriesMatrix.
        
        Matches GWpy FrequencySeries.filter behavior (magnitude-only response)
        by delegating to gwpy.frequencyseries._fdcommon.fdfilter.
        Use apply_response() for complex response application.

        Parameters
        ----------
        *filt : filter arguments
            Filter definition.
        inplace : bool, optional
            If True, modify in-place. Default False.
        **kwargs :
            Additional arguments passed to fdfilter (e.g. analog=True).

        Returns
        -------
        FrequencySeriesMatrix
            Filtered matrix.
        """
        from gwpy.frequencyseries._fdcommon import fdfilter
        return fdfilter(self, *filt, **kwargs)

    def apply_response(self, response, inplace=False):
        """
        Apply a complex frequency response to the matrix.
        
        Extension method (not in GWpy) to support complex filtering/calibration.
        
        Parameters
        ----------
        response : array-like or Quantity
            Complex frequency response array aligned with self.frequencies.
        inplace : bool, optional
            If True, modify in-place.
        """
        from astropy import units as u
        import numpy as np

        if isinstance(response, u.Quantity):
            h = response
        else:
            h = u.Quantity(np.asarray(response), u.dimensionless_unscaled)
            
        if inplace:
            self *= h
            return self
        else:
            return self * h

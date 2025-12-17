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
from gwpy.frequencyseries import FrequencySeries as BaseFrequencySeries
from gwexpy.types.seriesmatrix import SeriesMatrix

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
    pass


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

        return super().__new__(cls, data, **kwargs)

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

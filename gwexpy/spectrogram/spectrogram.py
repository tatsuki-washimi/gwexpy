from collections import UserList
from typing import Any, Optional
try:
    from collections import UserDict
except ImportError:
    from collections import MutableMapping as UserDict

import numpy as np
# h5py is imported lazily where needed
from astropy import units as u

from gwpy.spectrogram import Spectrogram as BaseSpectrogram
from gwexpy.plot import Plot

from gwexpy.interop._optional import require_optional

# Optional dependencies handled lazily

# We can reuse SeriesMatrix if we want, but SpectrogramMatrix has different dimensions (Time, Freq)

class Spectrogram(BaseSpectrogram):
    """
    Extends gwpy.spectrogram.Spectrogram with additional interop methods.
    """
    def bootstrap_asd(
        self,
        n_boot=1000,
        average="median",
        ci=0.68,
        window="hann",
        nperseg=None,
        noverlap=None,
        ignore_nan=True,
    ):
        """
        Estimate robust ASD from this spectrogram using bootstrap resampling.

        This is a convenience wrapper around `gwexpy.spectral.bootstrap_spectrogram`.
        """
        from gwexpy.spectral import bootstrap_spectrogram

        return bootstrap_spectrogram(
            self,
            n_boot=n_boot,
            average=average,
            ci=ci,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            ignore_nan=ignore_nan,
        )

    def to_th2d(self, error=None):
        """
        Convert to ROOT TH2D.
        """
        from gwexpy.interop import to_th2d
        return to_th2d(self, error=error)

    def to_quantities(self, units=None):
        """
        Convert to quantities.Quantity (Elephant/Neo compatible).
        """
        from gwexpy.interop import to_quantity
        return to_quantity(self, units=units)

    @classmethod
    def from_quantities(cls, q, times, frequencies):
        """
        Create Spectrogram from quantities.Quantity.
        
        Parameters
        ----------
        q : quantities.Quantity
            Input data (Time x Frequency matrix).
        times : array-like
            Time axis.
        frequencies : array-like
            Frequency axis.
        """
        from gwexpy.interop import from_quantity
        return from_quantity(cls, q, times=times, frequencies=frequencies)

    @classmethod
    def from_root(cls, obj, return_error=False):
        """
        Create Spectrogram from ROOT TH2D.
        """
        from gwexpy.interop import from_root
        return from_root(cls, obj, return_error=return_error)

    def to_mne(self, info: Optional[Any] = None) -> Any:
        """
        Convert to MNE-Python object.

        Parameters
        ----------
        info : mne.Info, optional
            MNE Info object.

        Returns
        -------
        mne.time_frequency.EpochsTFRArray
        """
        from gwexpy.interop import to_mne
        return to_mne(self, info=info)

    @classmethod
    def from_mne(cls, tfr: Any, **kwargs: Any) -> Any:
        """
        Create Spectrogram from MNE-Python TFR object.

        Parameters
        ----------
        tfr : mne.time_frequency.EpochsTFR or AverageTFR
            Input TFR data.
        **kwargs
            Additional arguments passed to constructor.

        Returns
        -------
        Spectrogram or SpectrogramDict
        """
        from gwexpy.interop import from_mne
        return from_mne(cls, tfr, **kwargs)

    def to_obspy(self, **kwargs: Any) -> Any:
        """
        Convert to Obspy Stream.
        
        Returns
        -------
        obspy.Stream
        """
        from gwexpy.interop import to_obspy
        return to_obspy(self, **kwargs)
        
    @classmethod
    def from_obspy(cls, stream: Any, **kwargs: Any) -> Any:
        """
        Create Spectrogram from Obspy Stream.
        
        Parameters
        ----------
        stream : obspy.Stream
            Input stream.
        **kwargs
            Additional arguments.
            
        Returns
        -------
        Spectrogram
        """
        from gwexpy.interop import from_obspy
        return from_obspy(cls, stream, **kwargs)

    def imshow(self, **kwargs):
        """Plot using imshow. Inherited from gwpy."""
        return super().imshow(**kwargs)

    def pcolormesh(self, **kwargs):
        """Plot using pcolormesh. Inherited from gwpy."""
        return super().pcolormesh(**kwargs)

class SpectrogramMatrix(np.ndarray):
    """
    Matrix of Spectrogram data.
    Shape is typically (N, Time, Frequencies) for a list of Spectrograms,
    or (N, M, Time, Frequencies) for a matrix of Spectrograms.

    Attributes
    ----------
    times : array-like
    frequencies : array-like
    unit : Unit
    name : str
    """
    def __new__(cls, input_array, times=None, frequencies=None, unit=None, name=None,
                rows=None, cols=None, meta=None):
        obj = np.asarray(input_array).view(cls)
        obj.times = times
        obj.frequencies = frequencies
        obj.unit = unit
        obj.name = name

        # Metadata for indexing rows/cols if applicable
        from gwexpy.types.metadata import MetaDataDict, MetaDataMatrix

        def _entries_len(entries):
            if entries is None:
                return None
            try:
                return len(entries)
            except TypeError:
                return None

        if obj.ndim == 3:
             N = obj.shape[0]
             row_len = _entries_len(rows)
             col_len = _entries_len(cols)
             use_grid = False

             if row_len is not None or col_len is not None:
                 if row_len is None and col_len:
                     if N % col_len == 0:
                         row_len = N // col_len
                 if col_len is None and row_len:
                     if N % row_len == 0:
                         col_len = N // row_len
                 if row_len and col_len and row_len * col_len == N:
                     use_grid = True

             if use_grid:
                 obj.rows = MetaDataDict(rows, expected_size=row_len, key_prefix='row')
                 obj.cols = MetaDataDict(cols, expected_size=col_len, key_prefix='col')
                 obj.meta = MetaDataMatrix(meta, shape=(row_len, col_len))
             else:
                 obj.rows = MetaDataDict(rows, expected_size=N, key_prefix='batch')
                 obj.cols = MetaDataDict(None, expected_size=1, key_prefix='col') # Not really used for 3D but for consistency
                 obj.meta = MetaDataMatrix(meta, shape=(N, 1))
        elif obj.ndim == 4:
             nrow, ncol = obj.shape[:2]
             obj.rows = MetaDataDict(rows, expected_size=nrow, key_prefix='row')
             obj.cols = MetaDataDict(cols, expected_size=ncol, key_prefix='col')
             obj.meta = MetaDataMatrix(meta, shape=(nrow, ncol))
        else:
             obj.rows = None
             obj.cols = None
             obj.meta = None

        return obj

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            if isinstance(row_key, (int, str, np.integer)) and isinstance(col_key, (int, str, np.integer)):
                i = self.row_index(row_key) if isinstance(row_key, str) else int(row_key)
                j = self.col_index(col_key) if isinstance(col_key, str) else int(col_key)
                return super().__getitem__((i, j))
        return super().__getitem__(key)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.times = getattr(obj, 'times', None)
        self.frequencies = getattr(obj, 'frequencies', None)
        self.unit = getattr(obj, 'unit', None)
        self.name = getattr(obj, 'name', None)
        self.rows = getattr(obj, 'rows', None)
        self.cols = getattr(obj, 'cols', None)
        self.meta = getattr(obj, 'meta', None)

    def to_series_2Dlist(self):
        """Convert matrix to a 2D nested list of Spectrogram objects."""
        from gwexpy.spectrogram import Spectrogram
        r_keys = self.row_keys()
        c_keys = self.col_keys()
        return [[Spectrogram(self[i, j], times=self.times, frequencies=self.frequencies, 
                            unit=self.meta[i, j].unit, name=self.meta[i, j].name)
                 for j in range(len(c_keys))] for i in range(len(r_keys))]

    def to_series_1Dlist(self):
        """Convert matrix to a flat 1D list of Spectrogram objects."""
        from gwexpy.spectrogram import Spectrogram
        r_keys = self.row_keys()
        c_keys = self.col_keys()
        results = []
        # Use view(np.ndarray) to pass raw data to Spectrogram constructor
        raw_data = self.view(np.ndarray)
        if self.ndim == 3:
            # (N_rows, Time, Freq)
            for i in range(len(r_keys)):
                unit = self.meta[i, 0].unit if self.meta is not None else self.unit
                name = self.meta[i, 0].name if self.meta is not None else self.name
                results.append(Spectrogram(raw_data[i], times=self.times, frequencies=self.frequencies,
                                          unit=unit, name=name))
        elif self.ndim == 4:
            # (N_rows, N_cols, Time, Freq)
            for i in range(len(r_keys)):
                for j in range(len(c_keys)):
                    unit = self.meta[i, j].unit
                    name = self.meta[i, j].name
                    results.append(Spectrogram(raw_data[i, j], times=self.times, frequencies=self.frequencies,
                                              unit=unit, name=name))
        else:
            raise ValueError(f"Unsupported SpectrogramMatrix dimension: {self.ndim}")
        return results

    def _all_element_units_equivalent(self):
        """Check whether all element units are mutually equivalent."""
        if self.meta is None:
            return True, self.unit
        ref_unit = self.meta[0, 0].unit
        for meta in self.meta.flat:
            if not meta.unit.is_equivalent(ref_unit):
                return False, ref_unit
        return True, ref_unit

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        res = super().mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)
        # If result is 1D and matches frequency length, it might be a spectrum
        # Typically (N, Time, Freq) -> mean(axis=1) -> (N, Freq)
        return res

    def row_keys(self):
        return tuple(self.rows.keys()) if self.rows else tuple()

    def col_keys(self):
        return tuple(self.cols.keys()) if self.cols else tuple()

    def row_index(self, key):
        if not self.rows:
            raise KeyError(f"Invalid row key: {key}")
        try:
            return list(self.row_keys()).index(key)
        except ValueError:
            raise KeyError(f"Invalid row key: {key}")

    def col_index(self, key):
        if not self.cols:
            raise KeyError(f"Invalid column key: {key}")
        try:
            return list(self.col_keys()).index(key)
        except ValueError:
            raise KeyError(f"Invalid column key: {key}")

    def plot(self, **kwargs):
        """Plot the matrix data using gwexpy.plot.Plot."""
        from gwexpy.plot import Plot
        return Plot(self, **kwargs)

    def plot_summary(self, **kwargs):
        """
        Plot Matrix as side-by-side Spectrograms and percentile summaries.
        """
        from gwexpy.plot.plot import plot_summary
        return plot_summary(self, **kwargs)

    def to_torch(self, device=None, dtype=None, requires_grad=False, copy=False):
        """Convert to PyTorch tensor."""
        from gwexpy.interop.torch_ import to_torch
        return to_torch(self, device=device, dtype=dtype, requires_grad=requires_grad, copy=copy)

    def to_cupy(self, dtype=None):
        """Convert to CuPy array."""
        from gwexpy.interop.cupy_ import to_cupy
        return to_cupy(self, dtype=dtype)


class SpectrogramList(UserList):
    """
    List of Spectrogram objects.
    Reference: similar to TimeSeriesList but for 2D Spectrograms.

    .. note::
       Spectrogram objects can be very large in memory.
       Use `inplace=True` where possible to avoid deep copies.
    """
    def __init__(self, initlist=None):
        super().__init__(initlist)
        if initlist:
             self._validate_items(self.data)

    def _validate_items(self, items):
        for i, item in enumerate(items):
            if not isinstance(item, Spectrogram):
                if isinstance(item, BaseSpectrogram):
                    # Auto-convert to gwexpy Spectrogram
                    items[i] = item.view(Spectrogram)
                else:
                    raise TypeError(f"Items must be of type Spectrogram, not {type(item)}")

    def __setitem__(self, index, item):
        if not isinstance(item, Spectrogram):
            if isinstance(item, BaseSpectrogram):
                item = item.view(Spectrogram)
            else:
                raise TypeError("Value must be a Spectrogram")
        self.data[index] = item

    def append(self, item):
        if not isinstance(item, Spectrogram):
            if isinstance(item, BaseSpectrogram):
                item = item.view(Spectrogram)
            else:
                raise TypeError("Can only append Spectrogram objects")
        super().append(item)

    def extend(self, other):
        self._validate_items(other)
        super().extend(other)

    def read(self, source, *args, **kwargs):
        """Read spectrograms into the list from HDF5."""
        format = kwargs.get("format", "hdf5")
        new_list = self.__class__()
        if format == "hdf5":
             h5py = require_optional("h5py")
             with h5py.File(source, "r") as f:
                  keys = sorted(f.keys(), key=lambda x: int(x) if x.isdigit() else x)
                  for k in keys:
                       try:
                           new_list.append(Spectrogram.read(f[k], format="hdf5"))
                       except (TypeError, ValueError, AttributeError):
                           pass
        else:
             raise NotImplementedError(f"Format {format} not supported")
        self.extend(new_list)
        return self

    def write(self, target, *args, **kwargs):
        """Write list to file."""
        format = kwargs.get("format", "hdf5")
        mode = kwargs.get("mode", "w")
        if format == "root" or (isinstance(target, str) and target.endswith(".root")):
             from gwexpy.interop import write_root_file
             return write_root_file(self, target, **kwargs)
        if format == "hdf5":
             import h5py
             with h5py.File(target, mode) as f:
                  for i, s in enumerate(self):
                       grp = f.create_group(str(i))
                       s.write(grp, format="hdf5")
        else:
             raise NotImplementedError(f"Format {format} not supported")

    def crop(self, t0, t1, inplace=False):
        """Crop each spectrogram."""
        if inplace:
             target = self
        else:
             target = self.__class__()

        for i, s in enumerate(self):
             res = s.crop(t0, t1)
             if inplace:
                  self[i] = res
             else:
                  target.append(res)
        if inplace:
             return self
        return target

    def crop_frequencies(self, f0, f1, inplace=False):
        """Crop frequencies."""
        if inplace:
             target = self
        else:
             target = self.__class__()

        for i, s in enumerate(self):
             if hasattr(s, 'crop_frequencies'):
                  res = s.crop_frequencies(f0, f1)
             else:
                  if isinstance(f0, u.Quantity):
                      f0 = f0.to(s.yunit).value
                  if isinstance(f1, u.Quantity):
                      f1 = f1.to(s.yunit).value
                  res = s.crop_frequencies(f0, f1)

             if inplace:
                  self[i] = res
             else:
                  target.append(res)
        if inplace:
             return self
        return target

    def rebin(self, dt, df, inplace=False):
        """Rebin each spectrogram."""
        if inplace:
             target = self
        else:
             target = self.__class__()

        for i, s in enumerate(self):
             if hasattr(s, 'rebin'):
                  res = s.rebin(dt, df)
                  if inplace:
                       self[i] = res
                  else:
                       target.append(res)
             else:
                  raise NotImplementedError("rebin not supported")
        if inplace:
             return self
        return target

    def interpolate(self, dt, df, inplace=False):
        """Interpolate each spectrogram."""
        if inplace:
             target = self
        else:
             target = self.__class__()
        for i, s in enumerate(self):
             if hasattr(s, 'interpolate'):
                  res = s.interpolate(dt, df)
                  if inplace:
                       self[i] = res
                  else:
                       target.append(res)
             else:
                  raise NotImplementedError("interpolate not supported")
        if inplace:
             return self
        return target

    def plot(self, **kwargs):
        """Plot all spectrograms stacked vertically."""
        from gwexpy.plot import Plot
        # We pass self directly to Plot
        return Plot(self, **kwargs)

    def plot_summary(self, **kwargs):
        """
        Plot List as side-by-side Spectrograms and percentile summaries.
        """
        from gwexpy.plot.plot import plot_summary
        return plot_summary(self, **kwargs)

    def to_matrix(self):
        """Convert to SpectrogramMatrix (N, Time, Freq)."""
        if not self:
             return SpectrogramMatrix(np.empty((0,0,0)))

        shape0 = self[0].shape
        for s in self[1:]:
             if s.shape != shape0:
                 raise ValueError("Shape mismatch in SpectrogramList elements")

        arr = np.stack([s.value for s in self])
        s0 = self[0]

        return SpectrogramMatrix(
             arr,
             times=s0.times,
             frequencies=s0.frequencies,
             unit=s0.unit,
             name=getattr(s0, 'name', None)
        )

    def to_torch(self, device=None, dtype=None):
        """Convert each spectrogram to PyTorch tensor. Returns a list."""
        torch = require_optional("torch")
        return [torch.tensor(s.value, device=device, dtype=dtype) for s in self]

    def to_cupy(self, dtype=None):
        """Convert each spectrogram to CuPy array. Returns a list."""
        cupy = require_optional("cupy")
        return [cupy.array(s.value, dtype=dtype) for s in self]

    def bootstrap_asd(self, *args, **kwargs):
        """Estimate robust ASD from each spectrogram in the list (returns FrequencySeriesList)."""
        from gwexpy.frequencyseries import FrequencySeriesList
        new_list = FrequencySeriesList()
        for v in self:
            new_list.append(v.bootstrap_asd(*args, **kwargs))
        return new_list

class SpectrogramDict(UserDict):
    """
    Dictionary of Spectrogram objects.

    .. note::
       Spectrogram objects can be very large in memory.
       Use `inplace=True` where possible to update container in-place.
    """
    def __init__(self, dict=None, **kwargs):
        self.data = {}
        if dict is not None:
             self.update(dict)
        if kwargs:
             self.update(kwargs)

    def __setitem__(self, key, item):
        if not isinstance(item, Spectrogram):
            if isinstance(item, BaseSpectrogram):
                item = item.view(Spectrogram)
            else:
                raise TypeError("Value must be a Spectrogram")
        self.data[key] = item

    def update(self, other=None, **kwargs):
        if other is not None:
             if isinstance(other, dict):
                  for k, v in other.items():
                       self[k] = v
             elif hasattr(other, 'keys'):
                  for k in other.keys():
                       self[k] = other[k]
             else:
                  for k, v in other:
                       self[k] = v
        for k, v in kwargs.items():
             self[k] = v

    def read(self, source, *args, **kwargs):
        """Read dictionary from HDF5 file keys -> dict keys."""
        format = kwargs.get("format", "hdf5")
        if format == "hdf5":
             h5py = require_optional("h5py")
             with h5py.File(source, "r") as f:
                  for k in f.keys():
                       try:
                           s = Spectrogram.read(f[k], format="hdf5")
                           self[k] = s
                       except (TypeError, ValueError, AttributeError):
                           pass
        else:
             raise NotImplementedError(f"Format {format} not supported")
        return self

    def write(self, target, *args, **kwargs):
        """Write dictionary to file."""
        format = kwargs.get("format", "hdf5")
        mode = kwargs.get("mode", "w")
        if format == "root" or (isinstance(target, str) and target.endswith(".root")):
             from gwexpy.interop import write_root_file
             return write_root_file(self, target, **kwargs)
        if format == "hdf5":
             h5py = require_optional("h5py")
             with h5py.File(target, mode) as f:
                  for k, s in self.items():
                       grp = f.create_group(str(k))
                       s.write(grp, format="hdf5")
        else:
             raise NotImplementedError(f"Format {format} not supported")

    def crop(self, t0, t1, inplace=False):
        """Crop each spectrogram in time.

        Parameters
        ----------
        t0, t1 : float
            Start and end times.
        inplace : bool, optional
            If True, modify in place.

        Returns
        -------
        SpectrogramDict
        """
        if inplace:
             target = self
        else:
             target = self.__class__()
        for k, v in self.items():
            res = v.crop(t0, t1)
            if inplace:
                 self[k] = res
            else:
                 target[k] = res
        if inplace:
             return self
        return target

    def crop_frequencies(self, f0, f1, inplace=False):
        """Crop each spectrogram in frequency.

        Parameters
        ----------
        f0, f1 : float or Quantity
            Start and end frequencies.
        inplace : bool, optional
            If True, modify in place.

        Returns
        -------
        SpectrogramDict
        """
        if inplace:
             target = self
        else:
             target = self.__class__()
        for k, v in self.items():
            if hasattr(v, 'crop_frequencies'):
                res = v.crop_frequencies(f0, f1)
            else:
                 if isinstance(f0, u.Quantity):
                     f0 = f0.to(v.yunit).value
                 if isinstance(f1, u.Quantity):
                     f1 = f1.to(v.yunit).value
                 res = v.crop_frequencies(f0, f1)

            if inplace:
                 self[k] = res
            else:
                 target[k] = res
        if inplace:
             return self
        return target

    def rebin(self, dt, df, inplace=False):
        """Rebin each spectrogram to new time/frequency resolution.

        Parameters
        ----------
        dt : float
            New time bin size.
        df : float
            New frequency bin size.
        inplace : bool, optional
            If True, modify in place.

        Returns
        -------
        SpectrogramDict
        """
        if inplace:
             target = self
        else:
             target = self.__class__()
        for k, v in self.items():
            if hasattr(v, 'rebin'):
                res = v.rebin(dt, df)
                if inplace:
                     self[k] = res
                else:
                     target[k] = res
            else:
                raise NotImplementedError("rebin not supported")
        if inplace:
             return self
        return target

    def interpolate(self, dt, df, inplace=False):
        """Interpolate each spectrogram to new resolution.

        Parameters
        ----------
        dt : float
            New time resolution.
        df : float
            New frequency resolution.
        inplace : bool, optional
            If True, modify in place.

        Returns
        -------
        SpectrogramDict
        """
        if inplace:
             target = self
        else:
             target = self.__class__()
        for k, v in self.items():
            if hasattr(v, 'interpolate'):
                res = v.interpolate(dt, df)
                if inplace:
                     self[k] = res
                else:
                     target[k] = res
            else:
                raise NotImplementedError("interpolate not supported")
        if inplace:
             return self
        return target

    def plot(self, **kwargs):
        """Plot all spectrograms stacked vertically."""
        from gwexpy.plot import Plot
        # Pass self directly, Plot will unpack values
        return Plot(self, **kwargs)

    def plot_summary(self, **kwargs):
        """
        Plot Dictionary as side-by-side Spectrograms and percentile summaries.
        """
        from gwexpy.plot.plot import plot_summary
        return plot_summary(self, **kwargs)

    def to_matrix(self):
        """Convert to SpectrogramMatrix.

        Returns
        -------
        SpectrogramMatrix
            3D array of (N, Time, Freq).
        """
        vals = list(self.values())
        if not vals:
             return SpectrogramMatrix(np.empty((0,0,0)))

        shape0 = vals[0].shape
        for s in vals[1:]:
             if s.shape != shape0:
                 raise ValueError("Mismatch shape")

        arr = np.stack([s.value for s in vals])
        s0 = vals[0]
        matrix = SpectrogramMatrix(
             arr,
             times=s0.times,
             frequencies=s0.frequencies,
             unit=s0.unit,
             name=None
        )
        return matrix

    def to_torch(self, device=None, dtype=None):
        """Convert to dict of PyTorch tensors."""
        if torch is None:
             raise ImportError("torch")
        return {k: torch.tensor(v.value, device=device, dtype=dtype) for k, v in self.items()}

    def to_cupy(self, dtype=None):
        """Convert to dict of CuPy arrays."""
        if cupy is None:
             raise ImportError("cupy")
        return {k: cupy.array(v.value, dtype=dtype) for k, v in self.items()}
    def bootstrap_asd(self, *args, **kwargs):
        """Estimate robust ASD from each spectrogram in the dict (returns FrequencySeriesDict)."""
        from gwexpy.frequencyseries import FrequencySeriesDict
        new_dict = FrequencySeriesDict()
        for k, v in self.items():
            new_dict[k] = v.bootstrap_asd(*args, **kwargs)
        return new_dict

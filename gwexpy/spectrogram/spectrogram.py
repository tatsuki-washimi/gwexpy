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

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        res = super().mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)
        # If result is 1D and matches frequency length, it might be a spectrum
        # Typically (N, Time, Freq) -> mean(axis=1) -> (N, Freq)
        return res

    def row_keys(self):
        """Return list of row metadata keys."""
        return list(self.rows.keys()) if self.rows else []

    def col_keys(self):
        """Return list of column metadata keys."""
        return list(self.cols.keys()) if self.cols else []

    def plot(self, **kwargs):
        """
        Plot the matrix data.

        If it is 3D (Batch, Time, Freq), plots a vertical list of spectrograms.
        If it is 4D (Row, Col, Time, Freq), plots a grid of spectrograms.
        If row/col metadata implies a grid for 3D, that grid is used instead
        of a single column.

        Optional Kwargs:
            monitor: int or (row, col) to plot a single element
            method: 'pcolormesh' (default)
            separate: bool (default True for 4D)
            geometry: tuple (default based on shape)
            yscale: 'log' (default)
            xscale: 'linear' (default)
        """
        kwargs.setdefault("method", "pcolormesh")
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("xscale", "linear")

        def _normalize_names(names, keys, count):
            if names is not None and len(names) == count:
                if any(name not in (None, "") for name in names):
                    return [str(name) if name is not None else None for name in names]
            if keys is not None and len(keys) == count:
                return [str(key) for key in keys]
            return [None] * count

        def _spectrogram_name(row_name, col_name):
            if row_name and col_name:
                return f"{row_name},{col_name}"
            if row_name:
                return str(row_name)
            if col_name:
                return str(col_name)
            return None

        def _apply_grid_labels(plot_obj, nrow, ncol, row_names, col_names):
            axes = plot_obj.axes[: nrow * ncol]
            if 'ylabel' not in kwargs:
                for i, name in enumerate(row_names):
                    idx = i * ncol
                    if idx < len(axes) and name:
                        axes[idx].set_ylabel(str(name))
            if 'title' not in kwargs:
                for j, name in enumerate(col_names):
                    if j < len(axes) and name:
                        axes[j].set_title(str(name), pad=8)
            if kwargs.get("constrained_layout", True):
                try:
                    plot_obj.set_constrained_layout(True)
                except (TypeError, ValueError, AttributeError):
                    pass
            return plot_obj

        monitor = kwargs.pop("monitor", None)

        if self.ndim == 3:
             index = monitor
             if index is not None:
                  data = self[index]
                  title = self.rows.names[index] if self.rows else f"Channel {index}"
                  s = Spectrogram(
                       data,
                       times=self.times,
                       frequencies=self.frequencies,
                       unit=self.unit,
                       name=self.name or title
                  )
                  return s.plot(**kwargs)
             else:
                  # If no monitor, plot as a grid if row/col metadata implies one
                  n_items = self.shape[0]
                  geometry = kwargs.get("geometry")
                  if geometry is not None:
                      nrow, ncol = geometry
                  else:
                      row_count = len(self.rows) if self.rows else 0
                      col_count = len(self.cols) if self.cols else 0
                      if row_count and col_count:
                          if row_count * col_count == n_items:
                              nrow, ncol = row_count, col_count
                          elif col_count == n_items:
                              nrow, ncol = 1, col_count
                          elif row_count == n_items:
                              nrow, ncol = row_count, 1
                          else:
                              nrow, ncol = n_items, 1
                      else:
                          nrow, ncol = n_items, 1

                  row_names = _normalize_names(
                      self.rows.names if self.rows else None,
                      list(self.rows.keys()) if self.rows else None,
                      nrow,
                  )
                  col_names = _normalize_names(
                      self.cols.names if self.cols else None,
                      list(self.cols.keys()) if self.cols else None,
                      ncol,
                  )

                  specs = []
                  for i in range(nrow):
                       for j in range(ncol):
                            idx = i * ncol + j
                            if idx >= n_items:
                                break
                            name = _spectrogram_name(row_names[i], col_names[j])
                            s = Spectrogram(
                                 self[idx],
                                 times=self.times,
                                 frequencies=self.frequencies,
                                 unit=self.unit,
                                 name=name
                            )
                            specs.append(s)
                  kwargs.setdefault("separate", True)
                  kwargs.setdefault("geometry", (nrow, ncol))
                  plot = Plot(*specs, **kwargs)
                  return _apply_grid_labels(plot, nrow, ncol, row_names, col_names)

        elif self.ndim == 4:
             nrow, ncol = self.shape[:2]
             r_names = _normalize_names(
                 self.rows.names if self.rows else None,
                 list(self.rows.keys()) if self.rows else None,
                 nrow,
             )
             c_names = _normalize_names(
                 self.cols.names if self.cols else None,
                 list(self.cols.keys()) if self.cols else None,
                 ncol,
             )

             if monitor is not None:
                  if isinstance(monitor, tuple) and len(monitor) == 2:
                       row, col = monitor
                  elif isinstance(monitor, (int, np.integer)):
                       if ncol == 1:
                            row, col = int(monitor), 0
                       elif nrow == 1:
                            row, col = 0, int(monitor)
                       else:
                            row = int(monitor) // ncol
                            col = int(monitor) % ncol
                  else:
                       raise TypeError("monitor must be int or (row, col)")

                  if row < 0 or col < 0 or row >= nrow or col >= ncol:
                       raise IndexError("monitor index out of range")

                  name = _spectrogram_name(r_names[row], c_names[col])
                  s = Spectrogram(
                       self[row, col],
                       times=self.times,
                       frequencies=self.frequencies,
                       unit=self.unit,
                       name=name
                  )
                  return s.plot(**kwargs)

             # Expand into flat list of Spectrogram objects
             specs = []
             for i in range(nrow):
                  for j in range(ncol):
                       s = Spectrogram(
                            self[i, j],
                            times=self.times,
                            frequencies=self.frequencies,
                            unit=self.unit,
                            name=_spectrogram_name(r_names[i], c_names[j])
                       )
                       specs.append(s)
             kwargs.setdefault("separate", True)
             kwargs.setdefault("geometry", (nrow, ncol))
             plot = Plot(*specs, **kwargs)
             return _apply_grid_labels(plot, nrow, ncol, r_names, c_names)

        return Plot()

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
        kwargs.setdefault("method", "pcolormesh")
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("xscale", "linear")
        if not self:
             return Plot()

        # Default to vertical stacking
        kwargs.setdefault("separate", True)
        kwargs.setdefault("geometry", (len(self), 1))

        return Plot(*self, **kwargs)

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
        kwargs.setdefault("method", "pcolormesh")
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("xscale", "linear")
        if not self:
             return Plot()

        # Default to vertical stacking
        kwargs.setdefault("separate", True)
        kwargs.setdefault("geometry", (len(self), 1))

        return Plot(*self.values(), **kwargs)

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

from __future__ import annotations

from collections import UserList
try:
    from collections import UserDict
except ImportError:
    from collections import MutableMapping as UserDict

from astropy import units as u
from gwpy.spectrogram import Spectrogram as BaseSpectrogram

from gwexpy.types.mixin import PhaseMethodsMixin
from gwexpy.interop._optional import require_optional
from .spectrogram import Spectrogram


class SpectrogramList(PhaseMethodsMixin, UserList):
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
             import h5py  # noqa: F401 - availability check
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
        import numpy as np
        from .matrix import SpectrogramMatrix
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

    def to_torch(self, *args, **kwargs) -> list:
        """Convert each item to torch.Tensor. Returns a list."""
        return [s.to_torch(*args, **kwargs) for s in self]

    def to_tensorflow(self, *args, **kwargs) -> list:
        """Convert each item to tensorflow.Tensor. Returns a list."""
        return [s.to_tensorflow(*args, **kwargs) for s in self]

    def to_jax(self, *args, **kwargs) -> list:
        """Convert each item to jax.Array. Returns a list."""
        return [s.to_jax(*args, **kwargs) for s in self]

    def to_cupy(self, *args, **kwargs) -> list:
        """Convert each item to cupy.ndarray. Returns a list."""
        return [s.to_cupy(*args, **kwargs) for s in self]

    def to_dask(self, *args, **kwargs) -> list:
        """Convert each item to dask.array. Returns a list."""
        return [s.to_dask(*args, **kwargs) for s in self]

    def bootstrap_asd(self, *args, **kwargs):
        """Estimate robust ASD from each spectrogram in the list (returns FrequencySeriesList)."""
        from gwexpy.frequencyseries import FrequencySeriesList
        new_list = FrequencySeriesList()
        for v in self:
            new_list.append(v.bootstrap_asd(*args, **kwargs))
        return new_list

    def radian(self, unwrap: bool = False) -> "SpectrogramList":
        """Compute phase (in radians) of each spectrogram."""
        return self.__class__([s.radian(unwrap=unwrap) for s in self])

    def degree(self, unwrap: bool = False) -> "SpectrogramList":
        """Compute phase (in degrees) of each spectrogram."""
        return self.__class__([s.degree(unwrap=unwrap) for s in self])


class SpectrogramDict(PhaseMethodsMixin, UserDict):
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
        import numpy as np
        from .matrix import SpectrogramMatrix
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

    def to_torch(self, *args, **kwargs) -> dict:
        """Convert each item to torch.Tensor. Returns a dict."""
        return {k: v.to_torch(*args, **kwargs) for k, v in self.items()}

    def to_tensorflow(self, *args, **kwargs) -> dict:
        """Convert each item to tensorflow.Tensor. Returns a dict."""
        return {k: v.to_tensorflow(*args, **kwargs) for k, v in self.items()}

    def to_jax(self, *args, **kwargs) -> dict:
        """Convert each item to jax.Array. Returns a dict."""
        return {k: v.to_jax(*args, **kwargs) for k, v in self.items()}

    def to_cupy(self, *args, **kwargs) -> dict:
        """Convert each item to cupy.ndarray. Returns a dict."""
        return {k: v.to_cupy(*args, **kwargs) for k, v in self.items()}

    def to_dask(self, *args, **kwargs) -> dict:
        """Convert each item to dask.array. Returns a dict."""
        return {k: v.to_dask(*args, **kwargs) for k, v in self.items()}

    def bootstrap_asd(self, *args, **kwargs):
        """Estimate robust ASD from each spectrogram in the dict (returns FrequencySeriesDict)."""
        from gwexpy.frequencyseries import FrequencySeriesDict
        new_dict = FrequencySeriesDict()
        for k, v in self.items():
            new_dict[k] = v.bootstrap_asd(*args, **kwargs)
        return new_dict

    def radian(self, unwrap: bool = False) -> "SpectrogramDict":
        """Compute phase (in radians) of each spectrogram."""
        new_dict = self.__class__()
        for k, v in self.items():
            new_dict[k] = v.radian(unwrap=unwrap)
        return new_dict

    def degree(self, unwrap: bool = False) -> "SpectrogramDict":
        """Compute phase (in degrees) of each spectrogram."""
        new_dict = self.__class__()
        for k, v in self.items():
            new_dict[k] = v.degree(unwrap=unwrap)
        return new_dict

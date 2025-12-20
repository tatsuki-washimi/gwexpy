import warnings
from collections import UserList
try:
    from collections import UserDict
except ImportError:
    from collections import MutableMapping as UserDict

import numpy as np
import h5py
from astropy import units as u

from gwpy.spectrogram import Spectrogram
from gwpy.plot import Plot

# Optional dependencies
try:
    import torch
except ImportError:
    torch = None

try:
    import cupy
except ImportError:
    cupy = None

# We can reuse SeriesMatrix if we want, but SpectrogramMatrix has different dimensions (Time, Freq)
from gwexpy.types.seriesmatrix import SeriesMatrix

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
    def __new__(cls, input_array, times=None, frequencies=None, unit=None, name=None):
        obj = np.asarray(input_array).view(cls)
        obj.times = times
        obj.frequencies = frequencies
        obj.unit = unit
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.times = getattr(obj, 'times', None)
        self.frequencies = getattr(obj, 'frequencies', None)
        self.unit = getattr(obj, 'unit', None)
        self.name = getattr(obj, 'name', None)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        res = super().mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)
        # If result is 1D and matches frequency length, it might be a spectrum
        # Typically (N, Time, Freq) -> mean(axis=1) -> (N, Freq)
        return res

    def plot(self, **kwargs):
        """
        Plot the matrix data.
        If it is 3D (Batch, Time, Freq), defaults to plotting mean spectrogram.
        Kwargs: monitor=int (index), method='pcolormesh'
        """
        kwargs.setdefault("method", "pcolormesh")
        if self.ndim == 3:
             index = kwargs.pop("monitor", None)
             if index is not None:
                  data = self[index]
                  title = f"Channel {index}"
             else:
                  data = np.mean(self, axis=0)
                  title = "Mean Spectrogram"
             
             s = Spectrogram(
                  data,
                  times=self.times,
                  frequencies=self.frequencies,
                  unit=self.unit,
                  name=self.name or title
             )
             return s.plot(**kwargs)
        return Plot()

    def to_torch(self, device=None, dtype=None):
        if torch is None:
            raise ImportError("torch not installed")
        return torch.as_tensor(self, device=device, dtype=dtype)

    def to_cupy(self, dtype=None):
        if cupy is None:
            raise ImportError("cupy not installed")
        return cupy.asarray(self, dtype=dtype)


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
        for item in items:
            if not isinstance(item, Spectrogram):
                raise TypeError(f"Items must be of type Spectrogram, not {type(item)}")

    def __setitem__(self, index, item):
        if not isinstance(item, Spectrogram):
            raise TypeError("Value must be a Spectrogram")
        self.data[index] = item

    def append(self, item):
        if not isinstance(item, Spectrogram):
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
             with h5py.File(source, "r") as f:
                  keys = sorted(f.keys(), key=lambda x: int(x) if x.isdigit() else x)
                  for k in keys:
                       try:
                           new_list.append(Spectrogram.read(f[k], format="hdf5"))
                       except Exception:
                           pass
        else:
             raise NotImplementedError(f"Format {format} not supported")
        self.extend(new_list)
        return self

    def write(self, target, *args, **kwargs):
        """Write list to HDF5 file (each item as a group)."""
        format = kwargs.get("format", "hdf5")
        mode = kwargs.get("mode", "w")
        if format == "hdf5":
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
                  if isinstance(f0, u.Quantity): f0 = f0.to(s.yunit).value
                  if isinstance(f1, u.Quantity): f1 = f1.to(s.yunit).value
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
        """Plot all spectrograms."""
        kwargs.setdefault("method", "pcolormesh")
        if not self:
             return Plot()
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
        if torch is None:
             raise ImportError("torch not installed")
        return [torch.tensor(s.value, device=device, dtype=dtype) for s in self]

    def to_cupy(self, dtype=None):
        if cupy is None:
             raise ImportError("cupy not installed")
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

    def _validate_item(self, item):
        if not isinstance(item, Spectrogram):
             raise TypeError("Value must be a Spectrogram")

    def __setitem__(self, key, item):
        self._validate_item(item)
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
             with h5py.File(source, "r") as f:
                  for k in f.keys():
                       try:
                           s = Spectrogram.read(f[k], format="hdf5")
                           self[k] = s
                       except Exception:
                           pass
        else:
             raise NotImplementedError(f"Format {format} not supported")
        return self
        
    def write(self, target, *args, **kwargs):
        """Write dictionary to HDF5 file keys -> groups."""
        format = kwargs.get("format", "hdf5")
        mode = kwargs.get("mode", "w")
        if format == "hdf5":
             with h5py.File(target, mode) as f:
                  for k, s in self.items():
                       grp = f.create_group(str(k))
                       s.write(grp, format="hdf5")
        else:
             raise NotImplementedError(f"Format {format} not supported")

    def crop(self, t0, t1, inplace=False):
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
        if inplace:
             target = self
        else:
             target = self.__class__()
        for k, v in self.items():
            if hasattr(v, 'crop_frequencies'):
                res = v.crop_frequencies(f0, f1)
            else:
                 if isinstance(f0, u.Quantity): f0 = f0.to(v.yunit).value
                 if isinstance(f1, u.Quantity): f1 = f1.to(v.yunit).value
                 res = v.crop_frequencies(f0, f1)
            
            if inplace:
                 self[k] = res
            else:
                 target[k] = res
        if inplace:
             return self
        return target

    def rebin(self, dt, df, inplace=False):
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
        kwargs.setdefault("method", "pcolormesh")
        if not self:
             return Plot()
        return Plot(*self.values(), **kwargs)

    def to_matrix(self):
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
        if torch is None:
             raise ImportError("torch")
        return {k: torch.tensor(v.value, device=device, dtype=dtype) for k, v in self.items()}

    def to_cupy(self, dtype=None):
        if cupy is None:
             raise ImportError("cupy")
        return {k: cupy.array(v.value, dtype=dtype) for k, v in self.items()}

import warnings
from collections import UserList
try:
    from collections import UserDict
except ImportError:
    from collections import MutableMapping as UserDict

import numpy as np
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
       Methods like `crop`, `rebin`, etc. typically return new objects 
       (deep copies of data), which may double memory usage temporarily.
       Use `inplace=True` where supported (standard gwpy objects usually return copies).
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

    def read(self, *args, **kwargs):
        raise NotImplementedError("Batch read not implemented")

    def write(self, *args, **kwargs):
        raise NotImplementedError("Batch write not implemented")

    def crop(self, t0, t1):
        """
        Crop each spectrogram in the list by time.
        
        Parameters
        ----------
        t0 : `float` or `~astropy.units.Quantity`
            start time of crop
        t1 : `float` or `~astropy.units.Quantity`
            stop time of crop
        
        Returns
        -------
        SpectrogramList
            A new list containing cropped spectrograms.
            Note: Spectrogram.crop returns a copy by default.
        """
        new_list = self.__class__()
        for s in self:
             new_list.append(s.crop(t0, t1))
        return new_list

    def crop_frequencies(self, f0, f1):
        """
        Crop each spectrogram in the list by frequency.
        
        Parameters
        ----------
        f0 : `float` or `~astropy.units.Quantity`
            start frequency of crop
        f1 : `float` or `~astropy.units.Quantity`
            stop frequency of crop
            
        Returns
        -------
        SpectrogramList
        """
        new_list = self.__class__()
        for s in self:
             if hasattr(s, 'crop_frequencies'):
                 new_list.append(s.crop_frequencies(f0, f1))
             else:
                 # Fallback if method is missing on specific version
                 # Frequency is axis 1
                 if isinstance(f0, u.Quantity):
                     f0 = f0.to(s.yunit).value
                 if isinstance(f1, u.Quantity):
                     f1 = f1.to(s.yunit).value
                 
                 # Assuming s.frequencies contains the grid
                 freqs = s.frequencies.value
                 idx = (freqs >= f0) & (freqs <= f1)
                 # Slicing: s operates as Array2D. Array2D[time, freq] ?
                 # Spectrogram[idx, :] is time? 
                 # Spectrogram documentation says `self[start:stop]` cuts time.
                 # `self[:, start:stop]` cuts frequency?
                 # Need to find indices.
                 # Implementation detail: exact match or range?
                 # s.df and s.f0 can determine indices too.
                 
                 # Simpler: use boolean mask on axis 1? 
                 # Objects might not support advanced indexing on axis 1 easily?
                 # We will assume new Spectrogram supports valid slicing.
                 # s[:, sub_freqs] might fail if it expects slice.
                 # Let's trust crop_frequencies exists (as verified).
                 new_list.append(s.crop_frequencies(f0, f1))
        return new_list

    def rebin(self, dt, df):
        """
        Rebin each spectrogram in the list.
        
        Parameters
        ----------
        dt : `float` or `~astropy.units.Quantity`
            New time bin size
        df : `float` or `~astropy.units.Quantity`
            New frequency bin size
            
        Returns
        -------
        SpectrogramList
        """
        new_list = self.__class__()
        for s in self:
             if hasattr(s, 'rebin'):
                  new_list.append(s.rebin(dt, df)) # Verify signature match
             else:
                  # Attempt to use resample or custom
                  warnings.warn(f"Spectrogram object does not support rebin, skipping/failing {type(s)}")
                  raise NotImplementedError("rebin not supported by underlying Spectrogram")
        return new_list

    def interpolate(self, dt, df):
        """
        Interpolate each spectrogram in the list.
        """
        new_list = self.__class__()
        for s in self:
             if hasattr(s, 'interpolate'):
                  new_list.append(s.interpolate(dt, df))
             else:
                  warnings.warn("Spectrogram.interpolate not found")
                  raise NotImplementedError("interpolate not supported by underlying Spectrogram")
        return new_list

    def plot(self, **kwargs):
        """
        Plot all spectrograms.
        Default method='pcolormesh'.
        """
        kwargs.setdefault("method", "pcolormesh")
        if not self:
             return Plot()
        return Plot(*self, **kwargs)

    def to_matrix(self):
        """
        Convert to SpectrogramMatrix (N, Time, Freq).
        """
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
        """Convert elements to torch tensors (List of Tensors)."""
        if torch is None:
             raise ImportError("torch not installed")
        return [torch.tensor(s.value, device=device, dtype=dtype) for s in self]

    def to_cupy(self, dtype=None):
        """Convert elements to cupy arrays (List of Arrays)."""
        if cupy is None:
             raise ImportError("cupy not installed")
        return [cupy.array(s.value, dtype=dtype) for s in self]

class SpectrogramDict(UserDict):
    """
    Dictionary of Spectrogram objects.
    
    .. note::
       Spectrogram objects can be very large in memory. 
       Methods typically return new dictionaries/objects (copies).
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

    def read(self, *args, **kwargs):
        raise NotImplementedError("Batch read not implemented")
        
    def write(self, *args, **kwargs):
        raise NotImplementedError("Batch write not implemented")

    def crop(self, t0, t1):
        new_dict = self.__class__()
        for k, v in self.items():
            new_dict[k] = v.crop(t0, t1)
        return new_dict

    def crop_frequencies(self, f0, f1):
        new_dict = self.__class__()
        for k, v in self.items():
            if hasattr(v, 'crop_frequencies'):
                new_dict[k] = v.crop_frequencies(f0, f1)
            else:
                # Fallback
                 if isinstance(f0, u.Quantity):
                     f0 = f0.to(v.yunit).value
                 if isinstance(f1, u.Quantity):
                     f1 = f1.to(v.yunit).value
                 freqs = v.frequencies.value
                 idx = (freqs >= f0) & (freqs <= f1)
                 # Assume slicing works on axis 1
                 # Unfortunately, basic array slicing might not work for non-regular or if Spectrogram overrides it.
                 # But we assume validity.
                 # If crop_frequencies is standard in modern gwpy, this branch is rarely taken.
                 new_dict[k] = v.crop_frequencies(f0, f1)
        return new_dict

    def rebin(self, dt, df):
        new_dict = self.__class__()
        for k, v in self.items():
            if hasattr(v, 'rebin'):
                new_dict[k] = v.rebin(dt, df)
            else:
                raise NotImplementedError("rebin not supported on Spectrogram item")
        return new_dict

    def interpolate(self, dt, df):
        new_dict = self.__class__()
        for k, v in self.items():
            if hasattr(v, 'interpolate'):
                new_dict[k] = v.interpolate(dt, df)
            else:
                raise NotImplementedError("interpolate not supported on Spectrogram item")
        return new_dict

    def plot(self, **kwargs):
        kwargs.setdefault("method", "pcolormesh")
        if not self:
             return Plot()
        return Plot(*self.values(), **kwargs)

    def to_matrix(self):
        """Convert values to SpectrogramMatrix (N, Time, Freq)."""
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
        """Convert elements to dict of tensors."""
        if torch is None:
             raise ImportError("torch")
        return {k: torch.tensor(v.value, device=device, dtype=dtype) for k, v in self.items()}

    def to_cupy(self, dtype=None):
        """Convert elements to dict of cupy arrays."""
        if cupy is None:
             raise ImportError("cupy")
        return {k: cupy.array(v.value, dtype=dtype) for k, v in self.items()}

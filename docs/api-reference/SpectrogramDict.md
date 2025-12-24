# SpectrogramDict

**Inherits from:** UserDict


Dictionary of Spectrogram objects.

.. note::
   Spectrogram objects can be very large in memory.
   Use `inplace=True` where possible to update container in-place.


## Methods

### `__init__`

```python
__init__(self, dict=None, **kwargs)
```

Initialize self.  See help(type(self)) for accurate signature.

*(Inherited from `MutableMapping`)*

### `crop`

```python
crop(self, t0, t1, inplace=False)
```

Crop each spectrogram in time.

Parameters
----------
t0, t1 : float
    Start and end times.
inplace : bool, optional
    If True, modify in place.

Returns
-------
SpectrogramDict


### `crop_frequencies`

```python
crop_frequencies(self, f0, f1, inplace=False)
```

Crop each spectrogram in frequency.

Parameters
----------
f0, f1 : float or Quantity
    Start and end frequencies.
inplace : bool, optional
    If True, modify in place.

Returns
-------
SpectrogramDict


### `interpolate`

```python
interpolate(self, dt, df, inplace=False)
```

Interpolate each spectrogram to new resolution.

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


### `plot`

```python
plot(self, **kwargs)
```

Plot all spectrograms stacked vertically.

### `read`

```python
read(self, source, *args, **kwargs)
```

Read dictionary from HDF5 file keys -> dict keys.

### `rebin`

```python
rebin(self, dt, df, inplace=False)
```

Rebin each spectrogram to new time/frequency resolution.

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


### `to_cupy`

```python
to_cupy(self, dtype=None)
```

Convert to dict of CuPy arrays.

### `to_matrix`

```python
to_matrix(self)
```

Convert to SpectrogramMatrix.

Returns
-------
SpectrogramMatrix
    3D array of (N, Time, Freq).


### `to_torch`

```python
to_torch(self, device=None, dtype=None)
```

Convert to dict of PyTorch tensors.

### `update`

```python
update(self, other=None, **kwargs)
```

 D.update([E, ]**F) -> None.  Update D from mapping/iterable E and F.
If E present and has a .keys() method, does:     for k in E: D[k] = E[k]
If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v
In either case, this is followed by: for k, v in F.items(): D[k] = v


*(Inherited from `UserDict`)*

### `write`

```python
write(self, target, *args, **kwargs)
```

Write dictionary to file.


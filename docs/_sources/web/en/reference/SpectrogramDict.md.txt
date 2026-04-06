# SpectrogramDict

**Inherits from:** PhaseMethodsMixin, UserDict


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

*(Inherited from `PhaseMethodsMixin`)*

### `angle`

```python
angle(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any
```

Alias for `phase(unwrap=unwrap, deg=deg)`.

### `bootstrap_asd`

```python
bootstrap_asd(self, *args, **kwargs)
```

Estimate robust ASD from each spectrogram in the dict (returns FrequencySeriesDict).

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


### `degree`

```python
degree(self, unwrap: 'bool' = False) -> "'SpectrogramDict'"
```

Compute phase (in degrees) of each spectrogram.

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


### `phase`

```python
phase(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any
```


Calculate the phase of the data.

Parameters
----------
unwrap : `bool`, optional
    If `True`, unwrap the phase to remove discontinuities.
    Default is `False`.
deg : `bool`, optional
    If `True`, return the phase in degrees.
    Default is `False` (radians).
**kwargs
    Additional arguments passed to the underlying calculation.

Returns
-------
`Series` or `Matrix` or `Collection`
    The phase of the data.


### `plot`

```python
plot(self, **kwargs)
```

Plot all spectrograms stacked vertically.

### `plot_summary`

```python
plot_summary(self, **kwargs)
```


Plot Dictionary as side-by-side Spectrograms and percentile summaries.


### `radian`

```python
radian(self, unwrap: 'bool' = False) -> "'SpectrogramDict'"
```

Compute phase (in radians) of each spectrogram.

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
to_cupy(self, *args, **kwargs) -> 'dict'
```

Convert each item to cupy.ndarray. Returns a dict.

### `to_dask`

```python
to_dask(self, *args, **kwargs) -> 'dict'
```

Convert each item to dask.array. Returns a dict.

### `to_jax`

```python
to_jax(self, *args, **kwargs) -> 'dict'
```

Convert each item to jax.Array. Returns a dict.

### `to_matrix`

```python
to_matrix(self)
```

Convert to SpectrogramMatrix.

Returns
-------
SpectrogramMatrix
    3D array of (N, Time, Freq).


### `to_tensorflow`

```python
to_tensorflow(self, *args, **kwargs) -> 'dict'
```

Convert each item to tensorflow.Tensor. Returns a dict.

### `to_torch`

```python
to_torch(self, *args, **kwargs) -> 'dict'
```

Convert each item to torch.Tensor. Returns a dict.

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

For HDF5 output you can choose a layout (default is GWpy-compatible dataset-per-entry).

```python
sgd.write("out.h5", format="hdf5")               # GWpy-compatible (default)
sgd.write("out.h5", format="hdf5", layout="group")  # legacy group-per-entry
```

HDF5 dataset names (for GWpy `path=`):
- Keys are sanitized to be HDF5-friendly (e.g. `H1:SPEC` -> `H1_SPEC`).
- If multiple keys sanitize to the same name, a suffix like `__1` is added.
- The original keys are stored in file attributes, and `gwexpy` restores them on read.

.. warning::
   Never unpickle data from untrusted sources. ``pickle``/``shelve`` can execute
   arbitrary code on load.

Pickle portability note: pickled gwexpy `SpectrogramDict` unpickles as a built-in
`dict` of GWpy `Spectrogram` (gwexpy not required on the loading side).

# SpectrogramList

**Inherits from:** PhaseMethodsMixin, UserList


List of Spectrogram objects.
Reference: similar to TimeSeriesList but for 2D Spectrograms.

.. note::
   Spectrogram objects can be very large in memory.
   Use `inplace=True` where possible to avoid deep copies.


## Methods

### `__init__`

```python
__init__(self, initlist=None)
```

Initialize self.  See help(type(self)) for accurate signature.

*(Inherited from `PhaseMethodsMixin`)*

### `angle`

```python
angle(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any
```

Alias for `phase(unwrap=unwrap, deg=deg)`.

### `append`

```python
append(self, item)
```

S.append(value) -- append value to the end of the sequence

*(Inherited from `MutableSequence`)*

### `bootstrap_asd`

```python
bootstrap_asd(self, *args, **kwargs)
```

Estimate robust ASD from each spectrogram in the list (returns FrequencySeriesList).

### `crop`

```python
crop(self, t0, t1, inplace=False)
```

Crop each spectrogram.

### `crop_frequencies`

```python
crop_frequencies(self, f0, f1, inplace=False)
```

Crop frequencies.

### `degree`

```python
degree(self, unwrap: 'bool' = False) -> "'SpectrogramList'"
```

Compute phase (in degrees) of each spectrogram.

### `extend`

```python
extend(self, other)
```

S.extend(iterable) -- extend sequence by appending elements from the iterable

*(Inherited from `MutableSequence`)*

### `interpolate`

```python
interpolate(self, dt, df, inplace=False)
```

Interpolate each spectrogram.

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


Plot List as side-by-side Spectrograms and percentile summaries.


### `radian`

```python
radian(self, unwrap: 'bool' = False) -> "'SpectrogramList'"
```

Compute phase (in radians) of each spectrogram.

### `read`

```python
read(self, source, *args, **kwargs)
```

Read spectrograms into the list from HDF5.

### `rebin`

```python
rebin(self, dt, df, inplace=False)
```

Rebin each spectrogram.

### `to_cupy`

```python
to_cupy(self, *args, **kwargs) -> 'list'
```

Convert each item to cupy.ndarray. Returns a list.

### `to_dask`

```python
to_dask(self, *args, **kwargs) -> 'list'
```

Convert each item to dask.array. Returns a list.

### `to_jax`

```python
to_jax(self, *args, **kwargs) -> 'list'
```

Convert each item to jax.Array. Returns a list.

### `to_matrix`

```python
to_matrix(self)
```

Convert to SpectrogramMatrix (N, Time, Freq).

### `to_tensorflow`

```python
to_tensorflow(self, *args, **kwargs) -> 'list'
```

Convert each item to tensorflow.Tensor. Returns a list.

### `to_torch`

```python
to_torch(self, *args, **kwargs) -> 'list'
```

Convert each item to torch.Tensor. Returns a list.

### `write`

```python
write(self, target, *args, **kwargs)
```

Write list to file.


# SpectrogramList

**Inherits from:** UserList


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

*(Inherited from `MutableSequence`)*

### `append`

```python
append(self, item)
```

S.append(value) -- append value to the end of the sequence

*(Inherited from `MutableSequence`)*

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

### `plot`

```python
plot(self, **kwargs)
```

Plot all spectrograms stacked vertically.

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
to_cupy(self, dtype=None)
```

Convert each spectrogram to CuPy array. Returns a list.

### `to_matrix`

```python
to_matrix(self)
```

Convert to SpectrogramMatrix (N, Time, Freq).

### `to_torch`

```python
to_torch(self, device=None, dtype=None)
```

Convert each spectrogram to PyTorch tensor. Returns a list.

### `write`

```python
write(self, target, *args, **kwargs)
```

Write list to file.


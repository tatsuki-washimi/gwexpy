# SpectrogramMatrix

**Inherits from:** ndarray


Matrix of Spectrogram data.
Shape is typically (N, Time, Frequencies) for a list of Spectrograms,
or (N, M, Time, Frequencies) for a matrix of Spectrograms.

Attributes
----------
times : array-like
frequencies : array-like
unit : Unit
name : str


## Methods

### `col_keys`

```python
col_keys(self)
```

Return list of column metadata keys.

### `mean`

```python
mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True)
```

a.mean(axis=None, dtype=None, out=None, keepdims=False, *, where=True)

Returns the average of the array elements along given axis.

Refer to `numpy.mean` for full documentation.

See Also
--------
numpy.mean : equivalent function

*(Inherited from `ndarray`)*

### `plot`

```python
plot(self, **kwargs)
```


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


### `row_keys`

```python
row_keys(self)
```

Return list of row metadata keys.

### `to_cupy`

```python
to_cupy(self, dtype=None)
```

Convert to CuPy array.

### `to_torch`

```python
to_torch(self, device=None, dtype=None, requires_grad=False, copy=False)
```

Convert to PyTorch tensor.


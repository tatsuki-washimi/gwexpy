# Field4DDict

**Inherits from:** dict

A dictionary-like collection of `Field4D` objects.

## Validation

When `validate=True`, the dict checks that all values share:
- Unit, axis names, axis0 domain, and space domains.
- Axis coordinate arrays (shape, unit, and values within tolerance).

## Methods

### `fft_time_all`

```python
fft_time_all(self, **kwargs)
```

Apply `fft_time()` to all fields in the dictionary.

Returns
-------
Field4DDict
    Dictionary of transformed fields.

### `ifft_time_all`

```python
ifft_time_all(self, **kwargs)
```

Apply `ifft_time()` to all fields in the dictionary.

Returns
-------
Field4DDict
    Dictionary of transformed fields.

### `fft_space_all`

```python
fft_space_all(self, **kwargs)
```

Apply `fft_space()` to all fields in the dictionary.

Returns
-------
Field4DDict
    Dictionary of transformed fields.

### `ifft_space_all`

```python
ifft_space_all(self, **kwargs)
```

Apply `ifft_space()` to all fields in the dictionary.

Returns
-------
Field4DDict
    Dictionary of transformed fields.

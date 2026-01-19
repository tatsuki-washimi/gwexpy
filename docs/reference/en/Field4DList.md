# Field4DList

**Inherits from:** list

A list-like collection of `Field4D` objects.

## Methods

### `fft_time_all`

```python
fft_time_all(self, **kwargs)
```

Apply `fft_time()` to all fields in the list.

Returns
-------
Field4DList
    List of transformed fields.

### `ifft_time_all`

```python
ifft_time_all(self, **kwargs)
```

Apply `ifft_time()` to all fields in the list.

Returns
-------
Field4DList
    List of transformed fields.

### `fft_space_all`

```python
fft_space_all(self, **kwargs)
```

Apply `fft_space()` to all fields in the list.

Returns
-------
Field4DList
    List of transformed fields.

### `ifft_space_all`

```python
ifft_space_all(self, **kwargs)
```

Apply `ifft_space()` to all fields in the list.

Returns
-------
Field4DList
    List of transformed fields.

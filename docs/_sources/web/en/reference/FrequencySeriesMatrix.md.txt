# FrequencySeriesMatrix

**Inherits from:** FrequencySeriesMatrixCoreMixin, FrequencySeriesMatrixAnalysisMixin, SeriesMatrix


Matrix container for multiple FrequencySeries objects.

Inherits from SeriesMatrix and returns FrequencySeries instances when indexed.


## Methods

### `MetaDataMatrix`

Metadata matrix containing per-element metadata.

### `N_samples`

Number of samples along the x-axis.

### `T`

Transpose of the matrix (rows and columns swapped).

### `append`

```python
append(self, other, inplace=True, pad=None, gap=None, resize=True)
```

Append another matrix along the sample axis.

### `append_exact`

```python
append_exact(self, other, inplace=False, pad=None, gap=None, tol=3.814697265625e-06)
```

Append another matrix with strict contiguity checking.

### `apply_response`

```python
apply_response(self, response: 'Any', inplace: 'bool' = False) -> 'Any'
```


Apply a complex frequency response to the matrix.

Extension method (not in GWpy) to support complex filtering/calibration.

Parameters
----------
response : array-like or Quantity
    Complex frequency response array aligned with self.frequencies.
inplace : bool, optional
    If True, modify in-place.


### `astype`

```python
astype(self, dtype, copy=True)
```

Cast matrix data to a specified type.

### `channel_names`

Flattened list of all element names.

### `channels`

2D array of channel identifiers for each matrix element.

### `col_index`

```python
col_index(self, key: 'Any') -> 'int'
```

Get the integer index for a column key.

### `col_keys`

```python
col_keys(self) -> 'tuple[Any, ...]'
```

Get the keys (labels) for all columns.

### `conj`

```python
conj(self)
```

Complex conjugate of the matrix.

### `copy`

```python
copy(self, order='C')
```

Create a deep copy of this matrix.

### `crop`

```python
crop(self, start=None, end=None, copy=False)
```

Crop the matrix to a specified range along the sample axis.

### `det`

```python
det(self)
```

Compute the determinant of the matrix at each sample point.

### `df`

Frequency spacing (dx).

### `diagonal`

```python
diagonal(self, output: 'str' = 'list')
```

Extract diagonal elements from the matrix.

### `dict_class`

```python
dict_class(*args: 'Any', **kwargs: 'Any')
```

Ordered mapping of `FrequencySeries` objects keyed by label.

### `diff`

```python
diff(self, n=1, axis=None)
```

Calculate the n-th discrete difference along the sample axis.

### `duration`

Duration covered by the samples.

### `dx`

Step size between samples on the x-axis.

### `f0`

Starting frequency (x0).

### `filter`

```python
filter(self, *filt: 'Any', **kwargs: 'Any') -> 'Any'
```


Apply a filter to the FrequencySeriesMatrix.

Matches GWpy FrequencySeries.filter behavior (magnitude-only response)
by delegating to gwpy.frequencyseries._fdcommon.fdfilter.
Use apply_response() for complex response application.

Parameters
----------
*filt : filter arguments
    Filter definition.
inplace : bool, optional
    If True, modify in-place. Default False.
**kwargs :
    Additional arguments passed to fdfilter (e.g. analog=True).

Returns
-------
FrequencySeriesMatrix
    Filtered matrix.


### `frequencies`

Frequency array (xindex).

### `get_index`

```python
get_index(self, key_row: 'Any', key_col: 'Any') -> 'tuple[int, int]'
```

Get the (row, col) integer indices for given keys.

### `ifft`

```python
ifft(self) -> 'Any'
```


Compute the inverse FFT of this frequency-domain matrix.

Matches GWpy FrequencySeries.ifft normalization.

Returns
-------
TimeSeriesMatrix
    The time-domain matrix resulting from the inverse FFT.


### `imag`

```python
imag(self)
```

Imaginary part of the matrix.

### `interpolate`

```python
interpolate(self, xindex, **kwargs)
```

Interpolate the matrix to a new sample axis.

### `inv`

```python
inv(self, swap_rowcol: 'bool' = True)
```

Compute the matrix inverse at each sample point.

### `is_compatible`

```python
is_compatible(self, other: 'Any') -> 'bool'
```

Compatibility check.

### `is_compatible_exact`

```python
is_compatible_exact(self, other: 'Any') -> 'bool'
```

Check strict compatibility with another matrix.

### `is_contiguous`

```python
is_contiguous(self, other: 'Any', tol: 'float' = 3.814697265625e-06) -> 'int'
```

Check if this matrix is contiguous with another.

### `is_contiguous_exact`

```python
is_contiguous_exact(self, other: 'Any', tol: 'float' = 3.814697265625e-06) -> 'int'
```

Check contiguity with strict shape matching.

### `is_regular`

Return True if this series has a regular grid (constant spacing).

### `keys`

```python
keys(self) -> 'tuple[tuple[Any, ...], tuple[Any, ...]]'
```

Get both row and column keys.

### `list_class`

```python
list_class(*items: 'Union[_FS, Iterable[_FS]]')
```

List of `FrequencySeries` objects.

### `loc`

Label-based indexer for direct value access.

### `max`

```python
max(self, axis=None, out=None, keepdims=False, initial=None, where=True, ignore_nan=True)
```

a.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

Return the maximum along a given axis.

Refer to `numpy.amax` for full documentation.

See Also
--------
numpy.amax : equivalent function

*(Inherited from `ndarray`)*

### `mean`

```python
mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True, ignore_nan=True)
```

a.mean(axis=None, dtype=None, out=None, keepdims=False, *, where=True)

Returns the average of the array elements along given axis.

Refer to `numpy.mean` for full documentation.

See Also
--------
numpy.mean : equivalent function

*(Inherited from `ndarray`)*

### `median`

```python
median(self, axis=None, out=None, overwrite_input=False, keepdims=False, ignore_nan=True)
```

_No documentation available._

### `min`

```python
min(self, axis=None, out=None, keepdims=False, initial=None, where=True, ignore_nan=True)
```

a.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

Return the minimum along a given axis.

Refer to `numpy.amin` for full documentation.

See Also
--------
numpy.amin : equivalent function

*(Inherited from `ndarray`)*

### `names`

2D array of names for each matrix element. Alias for channel_names if 1D.

### `pad`

```python
pad(self, pad_width, **kwargs)
```

Pad the matrix along the sample axis.

### `plot`

```python
plot(self, **kwargs: 'Any') -> 'Any'
```

Plot FrequencySeriesMatrix.

### `prepend`

```python
prepend(self, other, inplace=True, pad=None, gap=None, resize=True)
```

Prepend another matrix at the beginning along the sample axis.

### `prepend_exact`

```python
prepend_exact(self, other, inplace=False, pad=None, gap=None, tol=3.814697265625e-06)
```

Prepend another matrix with strict contiguity checking.

### `read`

```python
read(source, format=None, **kwargs)
```

Read a SeriesMatrix from file.

Parameters
----------
source : str or path-like
    Path to file to read.
format : str, optional
    File format. If None, inferred from extension.
**kwargs
    Additional arguments passed to the reader.

Returns
-------
SeriesMatrix
    The loaded matrix.

The available built-in formats are:

======== ==== ===== =============
 Format  Read Write Auto-identify
======== ==== ===== =============
     ats  Yes    No            No
  dttxml  Yes    No            No
     gbd  Yes    No            No
    gse2  Yes    No            No
    knet  Yes    No            No
      li  Yes    No            No
     lsf  Yes    No            No
     mem  Yes    No            No
miniseed  Yes    No            No
     orf  Yes    No            No
     sac  Yes    No            No
     sdb  Yes    No            No
  sqlite  Yes    No            No
 sqlite3  Yes    No            No
 taffmat  Yes    No            No
    tdms  Yes    No            No
     wav  Yes    No            No
     wdf  Yes    No            No
     win  Yes    No            No
   win32  Yes    No            No
     wvf  Yes    No            No
======== ==== ===== =============

### `real`

```python
real(self)
```

Real part of the matrix.

### `reshape`

```python
reshape(self, shape, order='C')
```

Reshape the matrix dimensions.

### `rms`

```python
rms(self, axis=None, keepdims=False, ignore_nan=True)
```

_No documentation available._

### `row_index`

```python
row_index(self, key: 'Any') -> 'int'
```

Get the integer index for a row key.

### `row_keys`

```python
row_keys(self) -> 'tuple[Any, ...]'
```

Get the keys (labels) for all rows.

### `schur`

```python
schur(self, keep_rows, keep_cols=None, eliminate_rows=None, eliminate_cols=None)
```

Compute the Schur complement of a block matrix.

### `series_class`

```python
series_class(data, unit=None, f0=None, df=None, frequencies=None, name=None, epoch=None, channel=None, **kwargs)
```

Light wrapper of gwpy's FrequencySeries for compatibility and future extension.

### `shape3D`

Shape of the matrix as a 3-tuple (n_rows, n_cols, n_samples).
For 4D matrices (spectrograms), the last dimension is likely frequency,
so n_samples is determined by _x_axis_index.


### `shift`

```python
shift(self, delta)
```

Shift the sample axis by a constant offset.

### `smooth`

```python
smooth(self, width: 'int', method: 'str' = 'amplitude', ignore_nan: 'bool' = True) -> 'Any'
```


Smooth the frequency series matrix along the frequency axis.

Parameters
----------
width : int
    Full width of the smoothing window in samples.
method : str, optional
    Smoothing method: 'amplitude', 'power', 'complex', 'db'.
    Default is 'amplitude'.
ignore_nan : bool, optional
    If True, ignore NaNs during smoothing. Default is True.


### `std`

```python
std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, ignore_nan=True)
```

a.std(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)

Returns the standard deviation of the array elements along given axis.

Refer to `numpy.std` for full documentation.

See Also
--------
numpy.std : equivalent function

*(Inherited from `ndarray`)*

### `step`

```python
step(self, where: 'str' = 'post', **kwargs: 'Any') -> 'Any'
```

Plot the matrix as a step function.

### `submatrix`

```python
submatrix(self, row_keys, col_keys)
```

Extract a submatrix by selecting specific rows and columns.

### `to_cupy`

```python
to_cupy(self, dtype=None) -> Any
```

Convert to CuPy Array.

### `to_dask`

```python
to_dask(self, chunks='auto') -> Any
```

Convert to Dask Array.

### `to_dict`

```python
to_dict(self) -> 'Any'
```


Convert matrix to an appropriate collection dict (e.g. TimeSeriesDict).
Follows the matrix structure (row, col) unless it's a 1-column matrix.


### `to_dict_flat`

```python
to_dict_flat(self) -> 'dict[str, Series]'
```

Convert matrix to a flat dictionary mapping name to Series.

### `to_hdf5`

```python
to_hdf5(self, filepath, **kwargs)
```

Write matrix to HDF5 file.

### `to_jax`

```python
to_jax(self) -> Any
```

Convert to JAX Array.

### `to_list`

```python
to_list(self) -> 'Any'
```

Convert matrix to an appropriate collection list (e.g. TimeSeriesList).

### `to_pandas`

```python
to_pandas(self, format='wide')
```

Convert matrix to a pandas DataFrame.

### `to_series_1Dlist`

```python
to_series_1Dlist(self) -> 'list[Series]'
```

Convert matrix to a flat 1D list of Series objects.

### `to_series_2Dlist`

```python
to_series_2Dlist(self) -> 'list[list[Series]]'
```

Convert matrix to a 2D nested list of Series objects.

### `to_tensorflow`

```python
to_tensorflow(self, dtype: Any = None) -> Any
```

Convert to tensorflow.Tensor.

### `to_torch`

```python
to_torch(self, device: Optional[str] = None, dtype: Any = None, requires_grad: bool = False, copy: bool = False) -> Any
```

Convert to torch.Tensor.

### `to_zarr`

```python
to_zarr(self, store, path=None, **kwargs) -> Any
```

Save to Zarr storage.

### `trace`

```python
trace(self)
```

Compute the trace of the matrix (sum of diagonal elements).

### `transpose`

```python
transpose(self, *axes)
```

Transpose rows and columns, preserving sample axis as 2.

### `units`

2D array of units for each matrix element.

### `update`

```python
update(self, other, inplace=True, pad=None, gap=None)
```

Update matrix by appending without resizing (rolling buffer style).

### `value`

Underlying numpy array of data values.

### `value_at`

```python
value_at(self, x)
```

Get the matrix values at a specific x-axis location.

### `var`

```python
var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, ignore_nan=True)
```

a.var(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)

Returns the variance of the array elements, along given axis.

Refer to `numpy.var` for full documentation.

See Also
--------
numpy.var : equivalent function

*(Inherited from `ndarray`)*

### `write`

```python
write(self, target, format=None, **kwargs)
```

Write matrix to file.

### `x0`

Starting value of the sample axis.

### `xarray`

Return the sample axis values.

### `xindex`

Sample axis index array.

### `xspan`

Full extent of the sample axis as a tuple (start, end).

### `xunit`

_No documentation available._


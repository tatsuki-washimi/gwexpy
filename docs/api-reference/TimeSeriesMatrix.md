# TimeSeriesMatrix

**Inherits from:** SeriesMatrix


Matrix container for multiple TimeSeries objects.

Provides dt, t0, times aliases and constructs FrequencySeriesMatrix via FFT.


## Methods

### `MetaDataMatrix`

Metadata matrix containing per-element metadata.

Returns
-------
MetaDataMatrix
    A 2D matrix of MetaData objects, one for each (row, col) element.


### `N_samples`

Number of samples along the x-axis.

Returns
-------
int
    The length of the sample axis (third dimension of the matrix).


### `T`

View of the transposed array.

Same as ``self.transpose()``.

Examples
--------
>>> a = np.array([[1, 2], [3, 4]])
>>> a
array([[1, 2],
       [3, 4]])
>>> a.T
array([[1, 3],
       [2, 4]])

>>> a = np.array([1, 2, 3, 4])
>>> a
array([1, 2, 3, 4])
>>> a.T
array([1, 2, 3, 4])

See Also
--------
transpose

*(Inherited from `ndarray`)*

### `abs`

```python
abs(self)
```

Compute element-wise absolute value.

For complex-valued matrices, this returns the magnitude.

Returns
-------
SeriesMatrix
    A new matrix with real-valued absolute values.


### `angle`

```python
angle(self, deg: bool = False)
```

Compute element-wise phase angle.

For complex-valued matrices, this returns the phase angle (argument).

Parameters
----------
deg : bool, optional
    If True, return angle in degrees; otherwise radians (default).

Returns
-------
SeriesMatrix
    A new matrix with phase angles, unit set to degrees or radians.


### `append`

```python
append(self, other, inplace=True, pad=None, gap=None, resize=True)
```

Append another matrix along the sample axis.

Parameters
----------
other : SeriesMatrix
    Matrix to append. Must have compatible row/column structure.
inplace : bool, optional
    If True (default), modify this matrix in place.
pad : float or None, optional
    Value to use for padding gaps. If None, gaps raise an error.
gap : str, float, or None, optional
    Gap handling: 'raise' (error on gap), 'pad' (fill with pad value),
    'ignore' (simple concatenation), or a numeric tolerance.
resize : bool, optional
    If True (default), allow the matrix to grow. If False, keep
    the original length by trimming from the start.

Returns
-------
SeriesMatrix
    The concatenated matrix (self if inplace=True).

Raises
------
ValueError
    If matrices overlap or have incompatible gaps.


### `append_exact`

```python
append_exact(self, other, inplace=False, pad=None, gap=None, tol=3.814697265625e-06)
```

Append another matrix with strict contiguity checking.

This method performs exact sample-by-sample appending with tolerance
checking for gaps between the matrices.

Parameters
----------
other : SeriesMatrix
    Matrix to append.
inplace : bool, optional
    If True, modify this matrix in place. Default is False.
pad : float or None, optional
    Value to use for padding gaps.
gap : float or None, optional
    Maximum allowed gap size for padding.
tol : float, optional
    Tolerance for contiguity check (default: 1/2^18 ≈ 3.8e-6).

Returns
-------
SeriesMatrix
    The concatenated matrix.

Raises
------
ValueError
    If shapes don't match or gap exceeds tolerance.


### `asd`

```python
asd(self, **kwargs: 'Any') -> 'Any'
```


Compute ASD of each element.
Returns FrequencySeriesMatrix.


### `astype`

```python
astype(self, dtype, copy=True)
```

Cast matrix data to a specified type.

Parameters
----------
dtype : str or numpy.dtype
    Target data type.
copy : bool, optional
    If True (default), return a copy even if dtype is unchanged.

Returns
-------
SeriesMatrix
    A new matrix with the specified data type.


### `auto_coherence`

```python
auto_coherence(self, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.auto_coherence`.

### `bandpass`

```python
bandpass(self, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.bandpass`.

### `channels`

2D array of channel identifiers for each matrix element.

Returns
-------
numpy.ndarray
    A 2D object array of channel names or Channel objects.


### `coherence`

```python
coherence(self, other, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.coherence` with another TimeSeries.

### `col_index`

```python
col_index(self, key: Any) -> int
```

Get the integer index for a column key.

Parameters
----------
key : Any
    The column key to look up.

Returns
-------
int
    The zero-based index of the column.

Raises
------
KeyError
    If the key is not found.


### `col_keys`

```python
col_keys(self) -> list[typing.Any]
```

Get the keys (labels) for all columns.

Returns
-------
tuple
    A tuple of column keys in order.


### `copy`

```python
copy(self, order='C')
```

Create a deep copy of this matrix.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    Memory layout of the copy. Default is 'C' (row-major).

Returns
-------
SeriesMatrix
    A new matrix with copied data and metadata.


### `crop`

```python
crop(self, start: 'Any' = None, end: 'Any' = None, copy: 'bool' = False) -> "'TimeSeriesMatrix'"
```


Crop this matrix to the given GPS start and end times.
Accepts any time format supported by gwexpy.time.to_gps (str, datetime, pandas, obspy, etc).


### `csd`

```python
csd(self, other, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.csd` with another TimeSeries.

### `dagger`

Conjugate transpose (Hermitian adjoint) of the matrix.

Returns the complex conjugate of the transposed matrix,
often denoted as A† in matrix notation.

Returns
-------
SeriesMatrix
    The Hermitian adjoint of this matrix.


### `det`

```python
det(self)
```

Compute the determinant of the matrix at each sample point.

Returns
-------
Series
    A Series containing the determinant values.

Raises
------
ValueError
    If the matrix is not square.
UnitConversionError
    If element units are not equivalent.


### `detrend`

```python
detrend(self, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.detrend`.

### `diagonal`

```python
diagonal(self, output: str = 'list')
```

Extract diagonal elements from the matrix.

Parameters
----------
output : {'list', 'vector', 'matrix'}, optional
    Output format:
    - 'list': Return a list of Series (default)
    - 'vector': Return as column vector (n x 1 matrix)
    - 'matrix': Return full matrix with off-diagonal zeroed

Returns
-------
list, SeriesMatrix
    Diagonal elements in the requested format.


### `diff`

```python
diff(self, n=1, axis=2)
```

Calculate the n-th discrete difference along the sample axis.

Parameters
----------
n : int, optional
    Number of times to difference. Default is 1.
axis : int, optional
    Must be 2 (sample axis). Other values raise ValueError.

Returns
-------
SeriesMatrix
    Differenced matrix with length reduced by n.


### `dt`

Time spacing (dx).

### `duration`


Duration covered by the samples.


### `dx`

Step size between samples on the x-axis.

For regularly-sampled data, this is the constant spacing between
consecutive samples. For time series, this equals ``1/sample_rate``.

Returns
-------
dx : `~astropy.units.Quantity`
    The sample spacing with appropriate units.

Raises
------
AttributeError
    If xindex is not set or if the series has irregular sampling.


### `fft`

```python
fft(self, **kwargs: 'Any') -> 'Any'
```


Compute FFT of each element.
Returns FrequencySeriesMatrix.


### `filter`

```python
filter(self, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.filter`.

### `from_neo_analogsignal`

```python
from_neo_analogsignal(sig: 'Any') -> 'Any'
```

Create from neo.AnalogSignal.

### `get_index`

```python
get_index(self, key_row: Any, key_col: Any) -> tuple[int, int]
```

Get the (row, col) integer indices for given keys.

Parameters
----------
key_row : Any
    The row key.
key_col : Any
    The column key.

Returns
-------
tuple of int
    A 2-tuple of (row_index, col_index).


### `highpass`

```python
highpass(self, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.highpass`.

### `ica`

```python
ica(self, return_model: 'bool' = False, **kwargs: 'Any') -> 'Any'
```

Fit and transform ICA.

### `ica_fit`

```python
ica_fit(self, **kwargs: 'Any') -> 'Any'
```

Fit ICA.

### `ica_inverse_transform`

```python
ica_inverse_transform(self, ica_res: 'Any', sources: 'Any') -> 'Any'
```

Inverse transform ICA sources.

### `ica_transform`

```python
ica_transform(self, ica_res: 'Any') -> 'Any'
```

Transform using ICA.

### `imag`

Imaginary part of the matrix.

For complex matrices, returns only the imaginary component.
For real matrices, returns zeros.

Returns
-------
SeriesMatrix
    Matrix with imaginary components as real values.


### `impute`

```python
impute(self, *, method: 'str' = 'linear', limit: 'Optional[int]' = None, axis: 'str' = 'time', max_gap: 'Optional[float]' = None, **kwargs: 'Any') -> 'Any'
```

Impute missing values in the matrix.

### `inv`

```python
inv(self, swap_rowcol: bool = True)
```

Compute the matrix inverse at each sample point.

Parameters
----------
swap_rowcol : bool, optional
    If True (default), swap row/col labels in the result.

Returns
-------
SeriesMatrix
    The inverse matrix with unit^-1.

Raises
------
ValueError
    If the matrix is not square.
UnitConversionError
    If element units are not equivalent.


### `is_compatible`

```python
is_compatible(self, other: Any) -> bool
```


Compatibility check.


### `is_compatible_exact`

```python
is_compatible_exact(self, other)
```

Check strict compatibility with another matrix.

Requires identical shape, xindex values, row/col keys,
and element units.

Parameters
----------
other : SeriesMatrix
    The other matrix to check compatibility with.

Returns
-------
bool
    True if compatible.

Raises
------
ValueError
    If any compatibility check fails.


### `is_contiguous`

```python
is_contiguous(self, other, tol=3.814697265625e-06)
```

Check if this matrix is contiguous with another (gwpy-like semantics).

Two matrices are contiguous if the end of one aligns with the start
of the other within the specified tolerance.

Parameters
----------
other : SeriesMatrix
    The other matrix to check contiguity with.
tol : float, optional
    Tolerance for the contiguity check (default: 1/2^18 ≈ 3.8e-6).

Returns
-------
int
    1 if self ends where other starts,
    -1 if other ends where self starts,
    0 if not contiguous.


### `is_contiguous_exact`

```python
is_contiguous_exact(self, other, tol=3.814697265625e-06)
```

Check contiguity with strict shape matching.

Unlike :meth:`is_contiguous`, this method requires identical shapes.

Parameters
----------
other : SeriesMatrix
    The other matrix to check contiguity with.
tol : float, optional
    Tolerance for the contiguity check (default: 1/2^18 ≈ 3.8e-6).

Returns
-------
int
    1 if self ends where other starts,
    -1 if other ends where self starts,
    0 if not contiguous.

Raises
------
ValueError
    If shapes do not match.


### `is_regular`

Return True if this TimeSeriesMatrix has a regular sample rate.

### `keys`

```python
keys(self) -> list[tuple[typing.Any, typing.Any]]
```

Get both row and column keys.

Returns
-------
tuple
    A 2-tuple of (row_keys, col_keys).


### `loc`

Label-based indexer for direct value access.

Provides pandas-like `.loc` accessor for getting/setting values
directly in the underlying data array using index notation.

Returns
-------
_LocAccessor
    An accessor object supporting ``[]`` indexing.

Examples
--------
>>> matrix.loc[0, 1, :] = new_values  # Set all samples at row 0, col 1
>>> vals = matrix.loc[0, 1, :]  # Get all samples at row 0, col 1


### `lock_in`

```python
lock_in(self, **kwargs: 'Any') -> 'Any'
```


Apply lock-in amplification element-wise.

Returns
-------
TimeSeriesMatrix or tuple of TimeSeriesMatrix
    If output='amp_phase' (default) or 'iq', returns (matrix1, matrix2).
    If output='complex', returns a single complex TimeSeriesMatrix.


### `lowpass`

```python
lowpass(self, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.lowpass`.

### `names`

2D array of names for each matrix element.

Returns
-------
numpy.ndarray
    A 2D object array of string names.


### `notch`

```python
notch(self, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.notch`.

### `pad`

```python
pad(self, pad_width, **kwargs)
```

Pad the matrix along the sample axis.

Parameters
----------
pad_width : int or tuple of (int, int)
    Number of samples to pad. If int, pads equally on both sides.
    If tuple, specifies (before, after) padding.
**kwargs
    Additional arguments passed to numpy.pad. Default mode is 'constant'.

Returns
-------
SeriesMatrix
    Padded matrix with extended sample axis.


### `pca`

```python
pca(self, return_model: 'bool' = False, **kwargs: 'Any') -> 'Any'
```

Fit and transform PCA.

### `pca_fit`

```python
pca_fit(self, **kwargs: 'Any') -> 'Any'
```

Fit PCA.

### `pca_inverse_transform`

```python
pca_inverse_transform(self, pca_res: 'Any', scores: 'Any') -> 'Any'
```

Inverse transform PCA scores.

### `pca_transform`

```python
pca_transform(self, pca_res: 'Any', **kwargs: 'Any') -> 'Any'
```

Transform using PCA.

### `plot`

```python
plot(self, **kwargs: 'Any') -> 'Any'
```


Plot the matrix data.


### `prepend`

```python
prepend(self, other, inplace=True, pad=None, gap=None, resize=True)
```

Prepend another matrix at the beginning along the sample axis.

Parameters
----------
other : SeriesMatrix
    Matrix to prepend.
inplace : bool, optional
    If True (default), modify this matrix in place.
pad : float or None, optional
    Value to use for padding gaps.
gap : str, float, or None, optional
    Gap handling strategy (see :meth:`append`).
resize : bool, optional
    If True (default), allow the matrix to grow.

Returns
-------
SeriesMatrix
    The concatenated matrix.


### `prepend_exact`

```python
prepend_exact(self, other, inplace=False, pad=None, gap=None, tol=3.814697265625e-06)
```

Prepend another matrix with strict contiguity checking.

See :meth:`append_exact` for parameter details.


### `psd`

```python
psd(self, **kwargs: 'Any') -> 'Any'
```


Compute PSD of each element.
Returns FrequencySeriesMatrix.


### `q_transform`

```python
q_transform(self, *args: 'Any', **kwargs: 'Any') -> 'Any'
```


Compute Q-transform of each element.
Returns SpectrogramMatrix.


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
 taffmat  Yes    No            No
     wdf  Yes    No            No
     win  Yes    No            No
   win32  Yes    No            No
     wvf  Yes    No            No
======== ==== ===== =============

### `real`

Real part of the matrix.

For complex matrices, returns only the real component.
For real matrices, returns a copy.

Returns
-------
SeriesMatrix
    Matrix with real-valued elements.


### `resample`

```python
resample(self, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.resample`.

### `rolling_max`

```python
rolling_max(self, window: 'Any', *, center: 'bool' = False, min_count: 'int' = 1, nan_policy: 'str' = 'omit', backend: 'str' = 'auto') -> 'Any'
```

Rolling maximum along the time axis.

### `rolling_mean`

```python
rolling_mean(self, window: 'Any', *, center: 'bool' = False, min_count: 'int' = 1, nan_policy: 'str' = 'omit', backend: 'str' = 'auto') -> 'Any'
```

Rolling mean along the time axis.

### `rolling_median`

```python
rolling_median(self, window: 'Any', *, center: 'bool' = False, min_count: 'int' = 1, nan_policy: 'str' = 'omit', backend: 'str' = 'auto') -> 'Any'
```

Rolling median along the time axis.

### `rolling_min`

```python
rolling_min(self, window: 'Any', *, center: 'bool' = False, min_count: 'int' = 1, nan_policy: 'str' = 'omit', backend: 'str' = 'auto') -> 'Any'
```

Rolling minimum along the time axis.

### `rolling_std`

```python
rolling_std(self, window: 'Any', *, center: 'bool' = False, min_count: 'int' = 1, nan_policy: 'str' = 'omit', backend: 'str' = 'auto', ddof: 'int' = 0) -> 'Any'
```

Rolling standard deviation along the time axis.

### `row_index`

```python
row_index(self, key: Any) -> int
```

Get the integer index for a row key.

Parameters
----------
key : Any
    The row key to look up.

Returns
-------
int
    The zero-based index of the row.

Raises
------
KeyError
    If the key is not found.


### `row_keys`

```python
row_keys(self) -> list[typing.Any]
```

Get the keys (labels) for all rows.

Returns
-------
tuple
    A tuple of row keys in order.


### `sample_rate`

Sampling rate (1/dt).

### `schur`

```python
schur(self, keep_rows, keep_cols=None, eliminate_rows=None, eliminate_cols=None)
```

Compute the Schur complement of a block matrix.

The Schur complement is computed as: A - B @ D^(-1) @ C
where the matrix is partitioned into blocks [[A, B], [C, D]].

Parameters
----------
keep_rows : list
    Row keys/indices to keep (block A rows).
keep_cols : list, optional
    Column keys/indices to keep. If None, same as keep_rows.
eliminate_rows : list, optional
    Row keys/indices to eliminate (block D rows). If None, auto-inferred.
eliminate_cols : list, optional
    Column keys/indices to eliminate. If None, auto-inferred.

Returns
-------
SeriesMatrix
    The Schur complement matrix.

Raises
------
ValueError
    If eliminate sets have different sizes or keep sets are empty.
UnitConversionError
    If element units are not equivalent.


### `series_class`

```python
series_class(data, *args, **kwargs)
```


Extended TimeSeries with all gwexpy functionality.

This class combines functionality from multiple modules:
- Core operations: is_regular, _check_regular, tail, crop, append, find_peaks
- Spectral transforms: fft, psd, cwt, laplace, etc.
- Signal processing: analytic_signal, mix_down, xcorr, etc.
- Analysis: impute, standardize, rolling_*, etc.
- Interoperability: to_pandas, to_torch, to_xarray, etc.

Inherits from gwpy.timeseries.TimeSeries for full compatibility.


### `shape3D`

Shape of the matrix as a 3-tuple (n_rows, n_cols, n_samples).

Returns
-------
tuple
    A 3-tuple of integers representing (rows, columns, samples).


### `shift`

```python
shift(self, delta)
```

Shift the sample axis by a constant offset.

Parameters
----------
delta : float or Quantity
    Amount to shift the x-axis. Positive values shift forward.

Returns
-------
self
    Returns self for method chaining (modifies in place).


### `span`

Time span (xspan).

### `spectrogram`

```python
spectrogram(self, *args: 'Any', **kwargs: 'Any') -> 'Any'
```


Compute spectrogram of each element.
Returns SpectrogramMatrix.


### `spectrogram2`

```python
spectrogram2(self, *args: 'Any', **kwargs: 'Any') -> 'Any'
```


Compute spectrogram2 of each element.
Returns SpectrogramMatrix.


### `standardize`

```python
standardize(self, *, axis: 'str' = 'time', method: 'str' = 'zscore', ddof: 'int' = 0, **kwargs: 'Any') -> 'Any'
```


Standardize the matrix.
See gwexpy.timeseries.preprocess.standardize_matrix.


### `step`

```python
step(self, where='post', **kwargs)
```

Plot the matrix as a step function.

Parameters
----------
where : {'pre', 'post', 'mid'}, optional
    Where to place the step. Default is 'post'.
**kwargs
    Additional arguments passed to plot().

Returns
-------
Plot
    The matplotlib figure.


### `submatrix`

```python
submatrix(self, row_keys, col_keys)
```

Extract a submatrix by selecting specific rows and columns.

Parameters
----------
row_keys : list
    List of row keys to include in the submatrix.
col_keys : list
    List of column keys to include in the submatrix.

Returns
-------
SeriesMatrix
    A new matrix containing only the selected rows and columns.


### `t0`

Start time (x0).

### `taper`

```python
taper(self, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.taper`.

### `times`

Time array (xindex).

### `to_dict`

```python
to_dict(self) -> "'TimeSeriesDict'"
```

Convert to TimeSeriesDict.

### `to_hdf5`

```python
to_hdf5(self, filepath, **kwargs)
```

Write matrix to HDF5 file.

Parameters
----------
filepath : str or path-like
    Output file path.
**kwargs
    Additional arguments passed to h5py.File.

Notes
-----
The HDF5 file structure includes:
- /data: 3D array of values
- /xindex: Sample axis information
- /meta: Per-element metadata (units, names, channels)
- /rows, /cols: Row and column metadata


### `to_list`

```python
to_list(self) -> "'TimeSeriesList'"
```

Convert to TimeSeriesList.

### `to_mne_raw`

```python
to_mne_raw(self, info: 'Any' = None) -> 'Any'
```

Alias for :meth:`to_mne_rawarray`.

### `to_mne_rawarray`

```python
to_mne_rawarray(self, info: 'Any' = None) -> 'Any'
```

Convert to mne.io.RawArray.

### `to_neo_analogsignal`

```python
to_neo_analogsignal(self, units: 'Optional[str]' = None) -> 'Any'
```

Convert to neo.AnalogSignal.

### `to_pandas`

```python
to_pandas(self, format='wide')
```

Convert matrix to a pandas DataFrame.

Parameters
----------
format : {'wide', 'long'}, optional
    Output format. 'wide' creates columns for each row-col combination.
    'long' creates a tidy format with row, col, value columns.
    Default is 'wide'.

Returns
-------
pandas.DataFrame
    The data as a DataFrame.


### `to_series_1Dlist`

```python
to_series_1Dlist(self)
```

Convert matrix to a flat 1D list of Series objects.

Returns
-------
list
    A flat list of Series, ordered by columns then rows.


### `to_series_2Dlist`

```python
to_series_2Dlist(self)
```

Convert matrix to a 2D nested list of Series objects.

Returns
-------
list of list
    A 2D list where ``result[i][j]`` is the Series at row i, column j.


### `to_torch`

```python
to_torch(self, device: 'Optional[str]' = None, dtype: 'Any' = None, requires_grad: 'bool' = False, copy: 'bool' = False) -> 'Any'
```


Convert matrix values to a torch.Tensor (shape preserved).


### `trace`

```python
trace(self)
```

Compute the trace of the matrix (sum of diagonal elements).

Returns
-------
Series
    A Series containing the trace values at each sample point.

Raises
------
ValueError
    If the matrix is not square.
UnitConversionError
    If diagonal elements have incompatible units.


### `transfer_function`

```python
transfer_function(self, other, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.transfer_function` with another TimeSeries.

### `transpose`

```python
transpose(self)
```

a.transpose(*axes)

Returns a view of the array with axes transposed.

Refer to `numpy.transpose` for full documentation.

Parameters
----------
axes : None, tuple of ints, or `n` ints

 * None or no argument: reverses the order of the axes.

 * tuple of ints: `i` in the `j`-th place in the tuple means that the
   array's `i`-th axis becomes the transposed array's `j`-th axis.

 * `n` ints: same as an n-tuple of the same ints (this form is
   intended simply as a "convenience" alternative to the tuple form).

Returns
-------
p : ndarray
    View of the array with its axes suitably permuted.

See Also
--------
transpose : Equivalent function.
ndarray.T : Array property returning the array transposed.
ndarray.reshape : Give a new shape to an array without changing its data.

Examples
--------
>>> a = np.array([[1, 2], [3, 4]])
>>> a
array([[1, 2],
       [3, 4]])
>>> a.transpose()
array([[1, 3],
       [2, 4]])
>>> a.transpose((1, 0))
array([[1, 3],
       [2, 4]])
>>> a.transpose(1, 0)
array([[1, 3],
       [2, 4]])

>>> a = np.array([1, 2, 3, 4])
>>> a
array([1, 2, 3, 4])
>>> a.transpose()
array([1, 2, 3, 4])

*(Inherited from `ndarray`)*

### `units`

2D array of units for each matrix element.

Returns
-------
numpy.ndarray
    A 2D object array of `astropy.units.Unit` instances.


### `update`

```python
update(self, other, inplace=True, pad=None, gap=None)
```

Update matrix by appending without resizing (rolling buffer style).

Appends `other` and trims from the start to maintain original length.

Parameters
----------
other : SeriesMatrix
    Matrix to append.
inplace : bool, optional
    If True (default), modify this matrix in place.
pad : float or None, optional
    Value to use for padding gaps.
gap : str, float, or None, optional
    Gap handling strategy.

Returns
-------
SeriesMatrix
    The updated matrix with same length as original.


### `value`

Underlying numpy array of data values.

Returns
-------
numpy.ndarray
    3D array of shape (n_rows, n_cols, n_samples) containing the data.


### `value_at`

```python
value_at(self, x)
```

Get the matrix values at a specific x-axis location.

Parameters
----------
x : float or Quantity
    The x-axis value to look up.

Returns
-------
numpy.ndarray
    2D array of shape (n_rows, n_cols) with values at that x.

Raises
------
ValueError
    If x is not in xindex.


### `whiten`

```python
whiten(self, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.whiten`.

### `whiten_channels`

```python
whiten_channels(self, *, method: 'str' = 'pca', eps: 'float' = 1e-12, n_components: 'Optional[int]' = None, return_model: 'bool' = True) -> 'Any'
```


Whiten the matrix (channels/components).
Returns (whitened_matrix, WhiteningModel) by default.
Set return_model=False to return only the whitened matrix.
See gwexpy.timeseries.preprocess.whiten_matrix.


### `write`

```python
write(self, target, format=None, **kwargs)
```

Write matrix to file.

Parameters
----------
target : str or path-like
    Output file path.
format : {'hdf5', 'csv', 'parquet'} or None, optional
    Output format. If None, inferred from file extension.
**kwargs
    Additional arguments passed to the writer.

Returns
-------
None or result from underlying writer.

Notes
-----
Supported extensions:
- .h5, .hdf5, .hdf → HDF5 format
- .csv → CSV format (wide layout)
- .parquet, .pq → Parquet format (wide layout)


### `x0`

Starting value of the sample axis.

Returns
-------
x0 : `~astropy.units.Quantity`
    The first value of xindex, representing the start time/frequency.


### `xarray`


Return the sample axis values.


### `xindex`

Sample axis index array.

This is the array of x-axis values (e.g., time or frequency) corresponding
to each sample in the matrix. For time series, this represents timestamps;
for frequency series, this represents frequency bins.

Returns
-------
xindex : `~gwpy.types.Index`, `~astropy.units.Quantity`, or `numpy.ndarray`
    The sample axis values with appropriate units.


### `xspan`

Full extent of the sample axis as a tuple (start, end).

The end value is calculated as the last sample plus one step (dx),
representing the exclusive upper bound of the data range.

Returns
-------
xspan : tuple
    A 2-tuple of (start, end) values with appropriate units.


### `xunit`

Physical unit of the sample axis.

Returns
-------
xunit : `~astropy.units.Unit`
    The unit of the x-axis (e.g., seconds for time series, Hz for frequency series).



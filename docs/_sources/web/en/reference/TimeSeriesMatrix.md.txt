# TimeSeriesMatrix

**Inherits from:** PhaseMethodsMixin, TimeSeriesMatrixCoreMixin, TimeSeriesMatrixAnalysisMixin, TimeSeriesMatrixSpectralMixin, TimeSeriesMatrixInteropMixin, SeriesMatrix


2D Matrix container for multiple TimeSeries objects sharing a common time axis.

This class represents a 2-dimensional array (rows x columns) where each element corresponds to a TimeSeries data stream. Crucially, **all elements in the matrix share the same time array** (same `t0`, `dt`, and number of samples). It behaves like a multivariate time series organized in a grid structure.

Provides dt, t0, times aliases and constructs FrequencySeriesMatrix via FFT.


## Methods

### `MetaDataMatrix`

Metadata matrix containing per-element metadata.

### `N_samples`

Number of samples along the x-axis.

### `T`

Transpose of the matrix (rows and columns swapped).

### `angle`

```python
angle(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any
```

Alias for `phase(unwrap=unwrap, deg=deg)`.

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

### `channel_names`

Flattened list of all element names.

### `channels`

2D array of channel identifiers for each matrix element.

### `coherence`

```python
coherence(self, other, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.coherence` with another TimeSeries.

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

### `correlation_vector`

```python
correlation_vector(self, target_timeseries, method='mic', nproc=None)
```

Calculate correlation between a target TimeSeries and all channels in this Matrix.

Notes
-----
- `method="pearson"` uses a vectorized path (fast for many channels).
- For small matrices, `nproc` may be throttled to avoid process overhead.

### `partial_correlation_matrix`

```python
partial_correlation_matrix(self, *, estimator: str = 'empirical', shrinkage: float | str | None = None, eps: float = 1e-08, return_precision: bool = False) -> Any
```

Compute the partial-correlation matrix across all channels (flattened).


### `crop`

```python
crop(self, start: 'Any' = None, end: 'Any' = None, copy: 'bool' = False) -> 'Any'
```


Crop this matrix to the given GPS start and end times.
Accepts any time format supported by gwexpy.time.to_gps (str, datetime, pandas, obspy, etc).


### `csd`

```python
csd(self, other, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.csd` with another TimeSeries.

### `degree`

```python
degree(self, unwrap: 'bool' = False, **kwargs: 'Any') -> 'Any'
```


Calculate the instantaneous phase of the matrix in degrees.

Parameters
----------
unwrap : bool, optional
    If True, unwrap the phase.
**kwargs
    Passed to analytic_signal.

Returns
-------
TimeSeriesMatrix
    The phase of the matrix elements, in degrees.


### `det`

```python
det(self)
```

Compute the determinant of the matrix at each sample point.

### `detrend`

```python
detrend(self, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.detrend`.

### `diagonal`

```python
diagonal(self, output: 'str' = 'list')
```

Extract diagonal elements from the matrix.

### `dict_class`

```python
dict_class(...)
```

Dictionary of TimeSeries objects.

### `diff`

```python
diff(self, n=1, axis=None)
```

Calculate the n-th discrete difference along the sample axis.

### `dt`

Time spacing (dx).

### `duration`

Duration covered by the samples.

### `dx`

Step size between samples on the x-axis.

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

### `find_peaks`

```python
find_peaks(self, threshold: 'Optional[float]' = None, method: 'str' = 'amplitude', **kwargs: 'Any') -> 'Any'
```

Find peaks in each element of the matrix. Element-wise delegate to `TimeSeries.find_peaks`.

### `from_neo`

```python
from_neo(sig: 'Any') -> 'Any'
```


Create TimeSeriesMatrix from neo.AnalogSignal.

Parameters
----------
sig : neo.core.AnalogSignal
    Input signal.

Returns
-------
TimeSeriesMatrix


### `get_index`

```python
get_index(self, key_row: 'Any', key_col: 'Any') -> 'tuple[int, int]'
```

Get the (row, col) integer indices for given keys.

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

```python
imag(self)
```

Imaginary part of the matrix.

### `impute`

```python
impute(self, *, method: 'str' = 'linear', limit: 'Optional[int]' = None, axis: 'str' = 'time', max_gap: 'Optional[float]' = None, **kwargs: 'Any') -> 'Any'
```

Impute missing values in the matrix.

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
list_class(*items)
```

List of TimeSeries objects.

### `loc`

Label-based indexer for direct value access.

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
plot(self, **kwargs: 'Any') -> 'Any'
```

Plot the matrix data.

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


### `radian`

```python
radian(self, unwrap: 'bool' = False, **kwargs: 'Any') -> 'Any'
```


Calculate the instantaneous phase of the matrix in radians.

Parameters
----------
unwrap : bool, optional
    If True, unwrap the phase.
**kwargs
    Passed to analytic_signal.

Returns
-------
TimeSeriesMatrix
    The phase of the matrix elements, in radians.


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

### `resample`

```python
resample(self, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.resample`.

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

### `rolling_max`

```python
rolling_max(self, window: 'Any', *, center: 'bool' = False, min_count: 'int' = 1, nan_policy: 'str' = 'omit', backend: 'str' = 'auto', ignore_nan: 'Optional[bool]' = None) -> 'Any'
```

Rolling maximum along the time axis.

### `rolling_mean`

```python
rolling_mean(self, window: 'Any', *, center: 'bool' = False, min_count: 'int' = 1, nan_policy: 'str' = 'omit', backend: 'str' = 'auto', ignore_nan: 'Optional[bool]' = None) -> 'Any'
```

Rolling mean along the time axis.

### `rolling_median`

```python
rolling_median(self, window: 'Any', *, center: 'bool' = False, min_count: 'int' = 1, nan_policy: 'str' = 'omit', backend: 'str' = 'auto', ignore_nan: 'Optional[bool]' = None) -> 'Any'
```

Rolling median along the time axis.

### `rolling_min`

```python
rolling_min(self, window: 'Any', *, center: 'bool' = False, min_count: 'int' = 1, nan_policy: 'str' = 'omit', backend: 'str' = 'auto', ignore_nan: 'Optional[bool]' = None) -> 'Any'
```

Rolling minimum along the time axis.

### `rolling_std`

```python
rolling_std(self, window: 'Any', *, center: 'bool' = False, min_count: 'int' = 1, nan_policy: 'str' = 'omit', backend: 'str' = 'auto', ddof: 'int' = 0, ignore_nan: 'Optional[bool]' = None) -> 'Any'
```

Rolling standard deviation along the time axis.

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

### `sample_rate`

Sampling rate (1/dt).

### `schur`

```python
schur(self, keep_rows, keep_cols=None, eliminate_rows=None, eliminate_cols=None)
```

Compute the Schur complement of a block matrix.

### `series_class`

```python
series_class(data, *args, **kwargs)
```


Extended TimeSeries with all gwexpy functionality.

### `smooth`

```python
smooth(self, width: 'Any', method: 'str' = 'amplitude', ignore_nan: 'bool' = True) -> 'Any'
```

Smooth the data. Element-wise delegate to `TimeSeries.smooth`.

This class combines functionality from multiple modules:
- Core operations: is_regular, _check_regular, tail, crop, append, find_peaks
- Spectral transforms: fft, psd, cwt, laplace, etc.
- Signal processing: analytic_signal, mix_down, xcorr, etc.
- Analysis: impute, standardize, rolling_*, etc.
- Interoperability: to_pandas, to_torch, to_xarray, etc.

Inherits from gwpy.timeseries.TimeSeries for full compatibility.


### `shape3D`

Shape of the matrix as a 3-tuple (n_rows, n_cols, n_samples).
For 4D matrices (spectrograms), the last dimension is likely frequency,
so n_samples is determined by _x_axis_index.


### `shift`

```python
shift(self, delta)
```

Shift the sample axis by a constant offset.

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

### `t0`

Start time (x0).

### `taper`

```python
taper(self, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.taper`.

### `times`

Time array (xindex).

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

### `to_mne`

```python
to_mne(self, info: 'Any' = None) -> 'Any'
```

Convert to mne.io.RawArray.

### `to_neo`

```python
to_neo(self, units: 'Any' = None) -> 'Any'
```


Convert to neo.AnalogSignal.

Returns
-------
neo.core.AnalogSignal


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

### `transfer_function`

```python
transfer_function(self, other, *args, **kwargs)
```

Element-wise delegate to `TimeSeries.transfer_function` with another TimeSeries.

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

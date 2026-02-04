# TimeSeriesList

**Inherits from:** PhaseMethodsMixin, TimeSeriesList

List of TimeSeries objects.

## Methods

### `__init__`

```python
__init__(self, *items)
```

Initialise a new list
        

### `EntryClass`

```python
EntryClass(data, unit=None, t0=None, dt=None, sample_rate=None, times=None, channel=None, name=None, **kwargs)
```

A time-domain data array.

Parameters
----------
value : array-like
    input data array

unit : `~astropy.units.Unit`, optional
    physical unit of these data

t0 : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
    GPS epoch associated with these data,
    any input parsable by `~gwpy.time.to_gps` is fine

dt : `float`, `~astropy.units.Quantity`, optional
    time between successive samples (seconds), can also be given inversely
    via `sample_rate`

sample_rate : `float`, `~astropy.units.Quantity`, optional
    the rate of samples per second (Hertz), can also be given inversely
    via `dt`

times : `array-like`
    the complete array of GPS times accompanying the data for this series.
    This argument takes precedence over `t0` and `dt` so should be given
    in place of these if relevant, not alongside

name : `str`, optional
    descriptive title for this array

channel : `~gwpy.detector.Channel`, `str`, optional
    source data stream for these data

dtype : `~numpy.dtype`, optional
    input data type

copy : `bool`, optional
    choose to copy the input data to new memory

subok : `bool`, optional
    allow passing of sub-classes by the array generator

Notes
-----
The necessary metadata to reconstruct timing information are recorded
in the `epoch` and `sample_rate` attributes. This time-stamps can be
returned via the :attr:`~TimeSeries.times` property.

All comparison operations performed on a `TimeSeries` will return a
`~gwpy.timeseries.StateTimeSeries` - a boolean array
with metadata copied from the starting `TimeSeries`.

Examples
--------
>>> from gwpy.timeseries import TimeSeries

To create an array of random numbers, sampled at 100 Hz, in units of
'metres':

>>> from numpy import random
>>> series = TimeSeries(random.random(1000), sample_rate=100, unit='m')

which can then be simply visualised via

>>> plot = series.plot()
>>> plot.show()


### `analytic_signal`

```python
analytic_signal(self, *args, **kwargs)
```

Apply analytic_signal to each item.

### `angle`

```python
angle(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any
```

Alias for `phase(unwrap=unwrap, deg=deg)`.

### `append`

```python
append(self, item)
```

Append object to the end of the list.

### `asd`

```python
asd(self, *args, **kwargs)
```


Compute ASD for each TimeSeries in the list.
Returns a FrequencySeriesList.


### `average_fft`

```python
average_fft(self, *args, **kwargs)
```


Apply average_fft to each TimeSeries in the list.
Returns a FrequencySeriesList.


### `baseband`

```python
baseband(self, *args, **kwargs)
```

Apply baseband to each item.

### `coalesce`

```python
coalesce(self)
```

Merge contiguous elements of this list into single objects

This method implicitly sorts and potentially shortens this list.


### `coherence`

```python
coherence(self, other=None, *, fftlength=None, overlap=None, window='hann', symmetric=True, include_diagonal=True, diagonal_value=1.0, **kwargs)
```


Compute coherence for each element or as a matrix depending on `other`.


### `coherence_matrix`

```python
coherence_matrix(self, other=None, *, fftlength=None, overlap=None, window='hann', symmetric=True, include_diagonal=True, diagonal_value=1.0, **kwargs)
```


Compute Coherence Matrix.

Parameters
----------
other : TimeSeriesDict or TimeSeriesList, optional
    Other collection.
fftlength, overlap, window :
    See TimeSeries.coherence().
symmetric : bool, default=True
    If True and other is None, compute only upper triangle and copy to lower.
include_diagonal : bool, default=True
    Include diagonal.
diagonal_value : float or None, default=1.0
    Value to fill diagonal if include_diagonal is True. If None, compute diagonal coherence.

Returns
-------
FrequencySeriesMatrix


### `copy`

```python
copy(self)
```

Return a copy of this list with each element copied to new memory
        

### `crop`

```python
crop(self, start=None, end=None, copy=False) -> 'TimeSeriesList'
```


Crop each TimeSeries in the list.
Accepts any time format supported by gwexpy.time.to_gps (str, datetime, pandas, obspy, etc).
Returns a new TimeSeriesList.


### `csd`

```python
csd(self, other=None, *, fftlength=None, overlap=None, window='hann', hermitian=True, include_diagonal=True, **kwargs)
```


Compute CSD for each element or as a matrix depending on `other`.


### `csd_matrix`

```python
csd_matrix(self, other=None, *, fftlength=None, overlap=None, window='hann', hermitian=True, include_diagonal=True, **kwargs)
```


Compute Cross Spectral Density Matrix.

Parameters
----------
other : TimeSeriesDict or TimeSeriesList, optional
    Other collection for cross-CSD.
fftlength, overlap, window :
    See TimeSeries.csd() arguments.
hermitian : bool, default=True
    If True and other is None, compute only upper triangle and conjugate fill lower.
include_diagonal : bool, default=True
    Whether to compute diagonal elements.

Returns
-------
FrequencySeriesMatrix


### `decimate`

```python
decimate(self, *args, **kwargs) -> 'TimeSeriesList'
```


Decimate each TimeSeries in the list.
Returns a new TimeSeriesList.


### `degree`

```python
degree(self, *args, **kwargs) -> 'TimeSeriesList'
```

Compute instantaneous phase (in degrees) of each item.

### `detrend`

```python
detrend(self, *args, **kwargs) -> 'TimeSeriesList'
```


Detrend each TimeSeries in the list.
Returns a new TimeSeriesList.


### `envelope`

```python
envelope(self, *args, **kwargs)
```

Apply envelope to each item.

### `extend`

```python
extend(self, item)
```

Extend list by appending elements from the iterable.

### `fft`

```python
fft(self, *args, **kwargs)
```


Apply FFT to each TimeSeries in the list.
Returns a FrequencySeriesList.


### `filter`

```python
filter(self, *args, **kwargs) -> 'TimeSeriesList'
```


Filter each TimeSeries in the list.
Returns a new TimeSeriesList.


### `gate`

```python
gate(self, *args, **kwargs) -> 'TimeSeriesList'
```


Gate each TimeSeries in the list.
Returns a new TimeSeriesList.


### `heterodyne`

```python
heterodyne(self, *args, **kwargs)
```

Apply heterodyne to each item.

### `hilbert`

```python
hilbert(self, *args, **kwargs)
```

Alias for analytic_signal.

### `ica`

```python
ica(self, *args, **kwargs)
```

Perform ICA decomposition across channels.

### `impute`

```python
impute(self, *, method='interpolate', limit=None, axis='time', max_gap=None, **kwargs)
```

Impute missing data (NaNs) in each TimeSeries.

Parameters
----------
method : str, optional
    Imputation method ('interpolate', 'fill', etc.).
limit : int, optional
    Maximum number of consecutive NaNs to fill.
axis : str, optional
    Axis to impute along.
max_gap : float, optional
    Maximum gap size to fill (in seconds).
**kwargs
    Passed to TimeSeries.impute().

Returns
-------
TimeSeriesList


### `instantaneous_frequency`

```python
instantaneous_frequency(self, *args, **kwargs)
```

Apply instantaneous_frequency to each item.

### `instantaneous_phase`

```python
instantaneous_phase(self, *args, **kwargs)
```

Apply instantaneous_phase to each item.

### `is_contiguous`

```python
is_contiguous(self, *args, **kwargs)
```

Check contiguity with another object for each TimeSeries.

### `join`

```python
join(self, pad=None, gap=None)
```

Concatenate all of the elements of this list into a single object

Parameters
----------
pad : `float`, optional
    value with which to fill gaps in the source data,
    by default gaps will result in a `ValueError`.

gap : `str`, optional, default: `'raise'`
    what to do if there are gaps in the data, one of

    - ``'raise'`` - raise a `ValueError`
    - ``'ignore'`` - remove gap and join data
    - ``'pad'`` - pad gap with zeros

    If `pad` is given and is not `None`, the default is ``'pad'``,
    otherwise ``'raise'``.

Returns
-------
series : `gwpy.types.TimeSeriesBase` subclass
     a single series containing all data from each entry in this list

See also
--------
TimeSeries.append
    for details on how the individual series are concatenated together


### `lock_in`

```python
lock_in(self, *args, **kwargs)
```

Apply lock_in to each item.

### `mask`

```python
mask(self, *args, **kwargs) -> 'TimeSeriesList'
```


Mask each TimeSeries in the list.
Returns a new TimeSeriesList.


### `max`

```python
max(self, *args, **kwargs)
```

Compute maximum for each TimeSeries. Returns list of scalars.

### `mean`

```python
mean(self, *args, **kwargs)
```

Compute mean for each TimeSeries. Returns list of scalars.

### `min`

```python
min(self, *args, **kwargs)
```

Compute minimum for each TimeSeries. Returns list of scalars.

### `mix_down`

```python
mix_down(self, *args, **kwargs)
```

Apply mix_down to each item.

### `notch`

```python
notch(self, *args, **kwargs) -> 'TimeSeriesList'
```


Notch filter each TimeSeries in the list.
Returns a new TimeSeriesList.


### `pca`

```python
pca(self, *args, **kwargs)
```

Perform PCA decomposition across channels.

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
plot(self, **kwargs: Any)
```

Plot all series. Delegates to gwexpy.plot.Plot.

### `plot_all`

```python
plot_all(self, *args: Any, **kwargs: Any)
```

Alias for plot(). Plots all series.

### `psd`

```python
psd(self, *args, **kwargs)
```


Compute PSD for each TimeSeries in the list.
Returns a FrequencySeriesList.


### `q_transform`

```python
q_transform(self, *args, **kwargs)
```


Compute Q-transform for each TimeSeries in the list.
Returns a SpectrogramList.


### `radian`

```python
radian(self, *args, **kwargs) -> 'TimeSeriesList'
```

Compute instantaneous phase (in radians) of each item.

### `resample`

```python
resample(self, *args, **kwargs) -> 'TimeSeriesList'
```


Resample each TimeSeries in the list.
Returns a new TimeSeriesList.


### `rms`

```python
rms(self, *args, **kwargs)
```

Compute RMS for each TimeSeries. Returns list of scalars.

### `rolling_max`

```python
rolling_max(self, window, *, center=False, min_count=1, nan_policy='omit', backend='auto', ignore_nan=None)
```

Apply rolling max to each element.

### `rolling_mean`

```python
rolling_mean(self, window, *, center=False, min_count=1, nan_policy='omit', backend='auto', ignore_nan=None)
```

Apply rolling mean to each element.

### `rolling_median`

```python
rolling_median(self, window, *, center=False, min_count=1, nan_policy='omit', backend='auto', ignore_nan=None)
```

Apply rolling median to each element.

### `rolling_min`

```python
rolling_min(self, window, *, center=False, min_count=1, nan_policy='omit', backend='auto', ignore_nan=None)
```

Apply rolling min to each element.

### `rolling_std`

```python
rolling_std(self, window, *, center=False, min_count=1, nan_policy='omit', backend='auto', ddof=0, ignore_nan=None)
```

Apply rolling std to each element.

### `segments`

The `span` of each series in this list
        

### `shift`

```python
shift(self, *args, **kwargs) -> 'TimeSeriesList'
```


Shift each TimeSeries in the list.
Returns a new TimeSeriesList.


### `spectrogram`

```python
spectrogram(self, *args, **kwargs)
```


Compute spectrogram for each TimeSeries in the list.
Returns a SpectrogramList.


### `spectrogram2`

```python
spectrogram2(self, *args, **kwargs)
```


Compute spectrogram2 for each TimeSeries in the list.
Returns a SpectrogramList.


### `state_segments`

```python
state_segments(self, *args, **kwargs)
```

Run state_segments on each element (returns list of SegmentLists).

### `std`

```python
std(self, *args, **kwargs)
```

Compute standard deviation for each TimeSeries. Returns list of scalars.

### `stlt`

```python
stlt(self, *args, **kwargs)
```

Apply stlt to each item. Returns a list of TimePlaneTransforms.

Notes
-----
All arguments are forwarded to `TimeSeries.stlt()`. In particular, you can pass
`frequencies` (Hz) to evaluate STLT at arbitrary frequency points instead of the
FFT grid.

### `taper`

```python
taper(self, *args, **kwargs) -> 'TimeSeriesList'
```


Taper each TimeSeries in the list.
Returns a new TimeSeriesList.


### `to_matrix`

```python
to_matrix(self, *, align='intersection', **kwargs)
```

Convert list to TimeSeriesMatrix with alignment.

Parameters
----------
align : str, optional
    Alignment strategy ('intersection', 'union', etc.). Default 'intersection'.
**kwargs
    Additional arguments passed to alignment function.

Returns
-------
TimeSeriesMatrix
    Matrix with all series aligned to common time axis.


### `to_pandas`

```python
to_pandas(self, **kwargs)
```


Convert TimeSeriesList to pandas DataFrame.
Each element becomes a column.
ASSUMES common time axis.


### `to_tmultigraph`

```python
to_tmultigraph(self, name: Optional[str] = None) -> Any
```

Convert to ROOT TMultiGraph.

### `unwrap_phase`

```python
unwrap_phase(self, *args, **kwargs)
```

Apply unwrap_phase to each item.

### `value_at`

```python
value_at(self, *args, **kwargs)
```

Get value at a specific time for each TimeSeries.

### `whiten`

```python
whiten(self, *args, **kwargs) -> 'TimeSeriesList'
```


Whiten each TimeSeries in the list.
Returns a new TimeSeriesList.


### `write`

```python
write(self, target: str, *args: Any, **kwargs: Any) -> Any
```

Write TimeSeriesList to file (HDF5, ROOT, etc.).

CSV/TXT output for multi-channel data uses a directory layout (one file per entry).

```python
tsl.write("out_dir", format="txt")  # writes per-series TXT under out_dir/
```

For HDF5 output you can choose a layout (default is GWpy-compatible dataset-per-entry).

```python
tsl.write("out.h5", format="hdf5")               # GWpy-compatible (default)
tsl.write("out.h5", format="hdf5", layout="group")  # legacy group-per-entry
```

.. warning::
   Never unpickle data from untrusted sources. ``pickle``/``shelve`` can execute
   arbitrary code on load.

Pickle portability note: pickled gwexpy `TimeSeriesList` unpickles as a **GWpy**
`TimeSeriesList` (gwexpy not required on the loading side).

### `zpk`

```python
zpk(self, *args, **kwargs) -> 'TimeSeriesList'
```


ZPK filter each TimeSeries in the list.
Returns a new TimeSeriesList.

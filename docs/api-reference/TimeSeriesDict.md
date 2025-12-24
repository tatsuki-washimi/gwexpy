# TimeSeriesDict

**Inherits from:** TimeSeriesDict

Dictionary of TimeSeries objects.

## Methods

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

### `append`

```python
append(self, other, copy=True, **kwargs) -> 'TimeSeriesDict'
```


Append another mapping of TimeSeries or a single TimeSeries to each item.


### `asd`

```python
asd(self, fftlength=4, overlap=2)
```

Compute Amplitude Spectral Density for each TimeSeries.

Parameters
----------
fftlength : float, optional
    FFT length in seconds (default 4).
overlap : float, optional
    Overlap between segments in seconds (default 2).

Returns
-------
FrequencySeriesDict
    Dictionary of ASD results.


### `asfreq`

```python
asfreq(self, rule, **kwargs)
```


Apply asfreq to each TimeSeries in the dict.
Returns a new TimeSeriesDict.


### `average_fft`

```python
average_fft(self, *args, **kwargs)
```


Apply averge_fft to each TimeSeries in the dict.
Returns a FrequencySeriesDict.


### `baseband`

```python
baseband(self, *args, **kwargs)
```

Apply baseband to each item.

### `coherence`

```python
coherence(self, other=None, *, fftlength=None, overlap=None, window='hann', symmetric=True, include_diagonal=True, diagonal_value=1.0, **kwargs)
```


Compute coherence for each element or as a matrix depending on `other`.


### `coherence_matrix`

```python
coherence_matrix(self, other=None, *, fftlength=None, overlap=None, window='hann', symmetric=True, include_diagonal=True, diagonal_value=1.0, **kwargs)
```

Compute coherence matrix for all pairs.

Parameters
----------
other : TimeSeriesDict or TimeSeriesList, optional
    Another collection for cross-coherence.
fftlength : float, optional
    FFT length in seconds.
overlap : float, optional
    Overlap between segments in seconds.
window : str, optional
    Window function name (default 'hann').
symmetric : bool, optional
    If True, exploit symmetry (default True).
include_diagonal : bool, optional
    Whether to include diagonal elements (default True).
diagonal_value : float, optional
    Value for diagonal elements (default 1.0).

Returns
-------
FrequencySeriesMatrix
    The coherence matrix.


### `copy`

```python
copy(self)
```

Return a copy of this dict with each value copied to new memory
        

### `crop`

```python
crop(self, start=None, end=None, copy=False) -> 'TimeSeriesDict'
```


Crop each TimeSeries in the dict.
Accepts any time format supported by gwexpy.time.to_gps (str, datetime, pandas, obspy, etc).
Returns a new TimeSeriesDict.


### `csd`

```python
csd(self, other=None, *, fftlength=None, overlap=None, window='hann', hermitian=True, include_diagonal=True, **kwargs)
```


Compute CSD for each element or as a matrix depending on `other`.


### `csd_matrix`

```python
csd_matrix(self, other=None, *, fftlength=None, overlap=None, window='hann', hermitian=True, include_diagonal=True, **kwargs)
```

Compute Cross-Spectral Density matrix for all pairs.

Parameters
----------
other : TimeSeriesDict or TimeSeriesList, optional
    Another collection for cross-CSD. If None, compute self-CSD matrix.
fftlength : float, optional
    FFT length in seconds.
overlap : float, optional
    Overlap between segments in seconds.
window : str, optional
    Window function name (default 'hann').
hermitian : bool, optional
    If True, exploit Hermitian symmetry (default True).
include_diagonal : bool, optional
    Whether to include diagonal elements (default True).

Returns
-------
FrequencySeriesMatrix
    The CSD matrix.


### `decimate`

```python
decimate(self, *args, **kwargs) -> 'TimeSeriesDict'
```


Decimate each TimeSeries in the dict.
Returns a new TimeSeriesDict.


### `detrend`

```python
detrend(self, *args, **kwargs) -> 'TimeSeriesDict'
```


Detrend each TimeSeries in the dict.
Returns a new TimeSeriesDict.


### `envelope`

```python
envelope(self, *args, **kwargs)
```

Apply envelope to each item.

### `fetch`

```python
fetch(channels, start, end, host=None, port=None, verify=False, verbose=False, connection=None, pad=None, scaled=None, allow_tape=None, type=None, dtype=None)
```

Fetch data from NDS for a number of channels.

Parameters
----------
channels : `list`
    required data channels.

start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
    GPS start time of required data,
    any input parseable by `~gwpy.time.to_gps` is fine

end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
    GPS end time of required data, defaults to end of data found;
    any input parseable by `~gwpy.time.to_gps` is fine

host : `str`, optional
    URL of NDS server to use, if blank will try any server
    (in a relatively sensible order) to get the data

port : `int`, optional
    port number for NDS server query, must be given with `host`.

verify : `bool`, optional, default: `True`
    check channels exist in database before asking for data

verbose : `bool`, optional
    print verbose output about NDS download progress, if ``verbose``
    is specified as a string, this defines the prefix for the
    progress meter

connection : `nds2.connection`, optional
    open NDS connection to use.

scaled : `bool`, optional
    apply slope and bias calibration to ADC data, for non-ADC data
    this option has no effect.

allow_tape : `bool`, optional
    allow data access from slow tapes. If `host` or `connection` is
    given, the default is to do whatever the server default is,
    otherwise servers will be searched in logical order allowing tape
    access if necessary to retrieve the data

type : `int`, `str`, optional
    NDS2 channel type integer or string name to match.

dtype : `numpy.dtype`, `str`, `type`, or `dict`
    NDS2 data type to match

Returns
-------
data : :class:`~gwpy.timeseries.TimeSeriesBaseDict`
    a new `TimeSeriesBaseDict` of (`str`, `TimeSeries`) pairs fetched
    from NDS.


### `fft`

```python
fft(self, *args, **kwargs)
```


Apply FFT to each TimeSeries in the dict.
Returns a FrequencySeriesDict.


### `filter`

```python
filter(self, *args, **kwargs) -> 'TimeSeriesDict'
```


Filter each TimeSeries in the dict.
Returns a new TimeSeriesDict.


### `find`

```python
find(channels, start, end, frametype=None, frametype_match=None, pad=None, scaled=None, nproc=1, verbose=False, allow_tape=True, observatory=None, **readargs)
```

Find and read data from frames for a number of channels.

This method uses :mod:`gwdatafind` to discover the (`file://`) URLs
that provide the requested data, then reads those files using
:meth:`TimeSeriesDict.read()`.

Parameters
----------
channels : `list`
    Required data channels.

start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
    GPS start time of required data,
    any input parseable by `~gwpy.time.to_gps` is fine

end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
    GPS end time of required data, defaults to end of data found;
    any input parseable by `~gwpy.time.to_gps` is fine

frametype : `str`
    Name of frametype in which this channel is stored; if not given
    all frametypes discoverable via GWDataFind will be searched for
    the required channels.

frametype_match : `str`
    Regular expression to use for frametype matching.

pad : `float`
    Value with which to fill gaps in the source data,
    by default gaps will result in a `ValueError`.

scaled : `bool`
    Apply slope and bias calibration to ADC data, for non-ADC data
    this option has no effect.

nproc : `int`
    Number of parallel processes to use.

allow_tape : `bool`
    Allow reading from frame files on (slow) magnetic tape.

verbose : `bool`, optional
    Print verbose output about read progress, if ``verbose``
    is specified as a string, this defines the prefix for the
    progress meter.

readargs
    Any other keyword arguments to be passed to `.read()`.

Raises
------
requests.exceptions.HTTPError
    If the GWDataFind query fails for any reason.

RuntimeError
    If no files are found to read, or if the read operation
    fails.


### `from_mne_raw`

```python
from_mne_raw(raw, *, unit_map=None)
```

Create from mne.io.Raw.

### `from_nds2_buffers`

```python
from_nds2_buffers(buffers, scaled=None, copy=True, **metadata)
```

Construct a new dict from a list of `nds2.buffer` objects

**Requires:** |nds2|_

Parameters
----------
buffers : `list` of `nds2.buffer`
    the input NDS2-client buffers to read

scaled : `bool`, optional
    apply slope and bias calibration to ADC data, for non-ADC data
    this option has no effect.

copy : `bool`, optional
    if `True`, copy the contained data array to new  to a new array

**metadata
    any other metadata keyword arguments to pass to the `TimeSeries`
    constructor

Returns
-------
dict : `TimeSeriesDict`
    a new `TimeSeriesDict` containing the data from the given buffers


### `from_pandas`

```python
from_pandas(df, *, unit_map=None, t0=None, dt=None)
```

Create TimeSeriesDict from pandas.DataFrame.

### `from_polars`

```python
from_polars(df, *, time_column='time', unit_map=None)
```

Create TimeSeriesDict from polars.DataFrame.

### `gate`

```python
gate(self, *args, **kwargs) -> 'TimeSeriesDict'
```


Gate each TimeSeries in the dict.
Returns a new TimeSeriesDict.


### `get`

```python
get(channels, start, end, pad=None, scaled=None, dtype=None, verbose=False, allow_tape=None, **kwargs)
```

Retrieve data for multiple channels from frames or NDS

This method dynamically accesses either frames on disk, or a
remote NDS2 server to find and return data for the given interval

Parameters
----------
channels : `list`
    required data channels.

start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
    GPS start time of required data,
    any input parseable by `~gwpy.time.to_gps` is fine

end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
    GPS end time of required data, defaults to end of data found;
    any input parseable by `~gwpy.time.to_gps` is fine

frametype : `str`, optional
    name of frametype in which this channel is stored, by default
    will search for all required frame types

pad : `float`, optional
    value with which to fill gaps in the source data,
    by default gaps will result in a `ValueError`.

scaled : `bool`, optional
    apply slope and bias calibration to ADC data, for non-ADC data
    this option has no effect.

nproc : `int`, optional, default: `1`
    number of parallel processes to use, serial process by
    default.

allow_tape : `bool`, optional, default: `None`
    allow the use of frames that are held on tape, default is `None`
    to attempt to allow the `TimeSeries.fetch` method to
    intelligently select a server that doesn't use tapes for
    data storage (doesn't always work), but to eventually allow
    retrieving data from tape if required

verbose : `bool`, optional
    print verbose output about data access progress, if ``verbose``
    is specified as a string, this defines the prefix for the
    progress meter

**kwargs
    other keyword arguments to pass to either
    `TimeSeriesBaseDict.find` (for direct GWF file access) or
    `TimeSeriesBaseDict.fetch` for remote NDS2 access


### `hht`

```python
hht(self, *args, **kwargs)
```


Apply hht (Hilbert-Huang Transform) to each item.
Returns a dict of results (either dicts or Spectrograms).


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

Apply impute to each item.

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

### `lock_in`

```python
lock_in(self, *args, **kwargs)
```


Apply lock_in to each item.
Returns TimeSeriesDict (if output='complex') or tuple of TimeSeriesDicts.


### `mask`

```python
mask(self, *args, **kwargs) -> 'TimeSeriesDict'
```


Mask each TimeSeries in the dict.
Returns a new TimeSeriesDict.


### `max`

```python
max(self, *args, **kwargs)
```

Compute maximum for each TimeSeries. Returns pandas.Series of scalars.

### `mean`

```python
mean(self, *args, **kwargs)
```

Compute mean for each TimeSeries. Returns pandas.Series of scalars.

### `min`

```python
min(self, *args, **kwargs)
```

Compute minimum for each TimeSeries. Returns pandas.Series of scalars.

### `mix_down`

```python
mix_down(self, *args, **kwargs)
```

Apply mix_down to each item.

### `notch`

```python
notch(self, *args, **kwargs) -> 'TimeSeriesDict'
```


Notch filter each TimeSeries in the dict.
Returns a new TimeSeriesDict.


### `pca`

```python
pca(self, *args, **kwargs)
```

Perform PCA decomposition across channels.

### `plot`

```python
plot(self, label='key', method='plot', figsize=(12, 4), xscale='auto-gps', **kwargs)
```

Plot the data for this `TimeSeriesBaseDict`.

Parameters
----------
label : `str`, optional
    labelling system to use, or fixed label for all elements
    Special values include

    - ``'key'``: use the key of the `TimeSeriesBaseDict`,
    - ``'name'``: use the :attr:`~TimeSeries.name` of each element

    If anything else, that fixed label will be used for all lines.

**kwargs
    all other keyword arguments are passed to the plotter as
    appropriate


### `prepend`

```python
prepend(self, *args, **kwargs) -> 'TimeSeriesDict'
```


Prepend to each TimeSeries in the dict (in-place).
Returns self.


### `psd`

```python
psd(self, *args, **kwargs)
```


Compute PSD for each TimeSeries in the dict.
Returns a FrequencySeriesDict.


### `q_transform`

```python
q_transform(self, *args, **kwargs)
```


Compute Q-transform for each TimeSeries in the dict.
Returns a SpectrogramDict.


### `read`

```python
read(source, *args, **kwargs)
```

Read data for multiple channels into a `TimeSeriesDict`

Parameters
----------
source : `str`, `list`
    Source of data, any of the following:

    - `str` path of single data file,
    - `str` path of LAL-format cache file,
    - `list` of paths.

channels : `~gwpy.detector.channel.ChannelList`, `list`
    a list of channels to read from the source.

start : `~gwpy.time.LIGOTimeGPS`, `float`, `str` optional
    GPS start time of required data, anything parseable by
    :func:`~gwpy.time.to_gps` is fine

end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
    GPS end time of required data, anything parseable by
    :func:`~gwpy.time.to_gps` is fine

format : `str`, optional
    source format identifier. If not given, the format will be
    detected if possible. See below for list of acceptable
    formats.

nproc : `int`, optional
    number of parallel processes to use, serial process by
    default.

pad : `float`, optional
    value with which to fill gaps in the source data,
    by default gaps will result in a `ValueError`.

Returns
-------
tsdict : `TimeSeriesDict`
    a `TimeSeriesDict` of (`channel`, `TimeSeries`) pairs. The keys
    are guaranteed to be the ordered list `channels` as given.

Notes
-----
The available built-in formats are:

======== ==== ===== =============
 Format  Read Write Auto-identify
======== ==== ===== =============
  dttxml  Yes    No            No
     gbd  Yes    No            No
    gse2  Yes   Yes            No
    knet  Yes    No            No
      li  Yes    No            No
     lsf  Yes    No            No
     mem  Yes    No            No
miniseed  Yes   Yes            No
     orf  Yes    No            No
     sac  Yes   Yes            No
     sdb  Yes    No            No
 taffmat  Yes    No            No
     wdf  Yes    No            No
     win  Yes    No            No
   win32  Yes    No            No
     wvf  Yes    No            No
======== ==== ===== =============

### `resample`

```python
resample(self, rate, **kwargs)
```


Resample items in the TimeSeriesDict.
In-place operation (updates the dict contents).

If rate is time-like, performs time-bin resampling.
Otherwise performs signal processing resampling (gwpy's native behavior).


### `rms`

```python
rms(self, *args, **kwargs)
```

Compute RMS for each TimeSeries. Returns pandas.Series of scalars.

### `rolling_max`

```python
rolling_max(self, window, *, center=False, min_count=1, nan_policy='omit', backend='auto')
```

Apply rolling max to each item.

### `rolling_mean`

```python
rolling_mean(self, window, *, center=False, min_count=1, nan_policy='omit', backend='auto')
```

Apply rolling mean to each item.

### `rolling_median`

```python
rolling_median(self, window, *, center=False, min_count=1, nan_policy='omit', backend='auto')
```

Apply rolling median to each item.

### `rolling_min`

```python
rolling_min(self, window, *, center=False, min_count=1, nan_policy='omit', backend='auto')
```

Apply rolling min to each item.

### `rolling_std`

```python
rolling_std(self, window, *, center=False, min_count=1, nan_policy='omit', backend='auto', ddof=0)
```

Apply rolling std to each item.

### `shift`

```python
shift(self, *args, **kwargs) -> 'TimeSeriesDict'
```


Shift each TimeSeries in the dict.
Returns a new TimeSeriesDict.


### `span`

The GPS ``[start, stop)`` extent of data in this `dict`

:type: `~gwpy.segments.Segment`


### `spectrogram`

```python
spectrogram(self, *args, **kwargs)
```


Compute spectrogram for each TimeSeries in the dict.
Returns a SpectrogramDict.


### `spectrogram2`

```python
spectrogram2(self, *args, **kwargs)
```


Compute spectrogram2 for each TimeSeries in the dict.
Returns a SpectrogramDict.


### `state_segments`

```python
state_segments(self, *args, **kwargs)
```

Run state_segments on each item (returns Series of SegmentLists).

### `std`

```python
std(self, *args, **kwargs)
```

Compute standard deviation for each TimeSeries. Returns pandas.Series of scalars.

### `step`

```python
step(self, label='key', where='post', figsize=(12, 4), xscale='auto-gps', **kwargs)
```

Create a step plot of this dict.

Parameters
----------
label : `str`, optional
    labelling system to use, or fixed label for all elements
    Special values include

    - ``'key'``: use the key of the `TimeSeriesBaseDict`,
    - ``'name'``: use the :attr:`~TimeSeries.name` of each element

    If anything else, that fixed label will be used for all lines.

**kwargs
    all other keyword arguments are passed to the plotter as
    appropriate


### `taper`

```python
taper(self, *args, **kwargs) -> 'TimeSeriesDict'
```


Taper each TimeSeries in the dict.
Returns a new TimeSeriesDict.


### `to_matrix`

```python
to_matrix(self, *, align='intersection', **kwargs)
```


Convert dictionary to TimeSeriesMatrix with alignment.


### `to_mne_raw`

```python
to_mne_raw(self, info=None, picks=None)
```

Alias for :meth:`to_mne_rawarray`.

### `to_mne_rawarray`

```python
to_mne_rawarray(self, info=None, picks=None)
```

Convert to mne.io.RawArray.

### `to_pandas`

```python
to_pandas(self, index='datetime', *, copy=False)
```

Convert to pandas.DataFrame.

### `to_polars`

```python
to_polars(self, time_column='time', time_unit='datetime')
```

Convert to polars.DataFrame.

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
whiten(self, *args, **kwargs) -> 'TimeSeriesDict'
```


Whiten each TimeSeries in the dict.
Returns a new TimeSeriesDict.


### `write`

```python
write(self, target: str, *args: Any, **kwargs: Any) -> Any
```

Write this `TimeSeriesDict` to a file

Arguments and keywords depend on the output format, see the
online documentation for full details for each format.

Parameters
----------
target : `str`
    output filename

format : `str`, optional
    output format identifier. If not given, the format will be
    detected if possible. See below for list of acceptable
    formats.

Notes
-----
The available built-in formats are:

============ ==== ===== =============
   Format    Read Write Auto-identify
============ ==== ===== =============
         gwf  Yes   Yes           Yes
gwf.framecpp  Yes   Yes            No
  gwf.framel  Yes   Yes            No
gwf.lalframe  Yes   Yes            No
        hdf5  Yes   Yes            No
============ ==== ===== =============

*(Inherited from `TimeSeriesDict`)*

### `zpk`

```python
zpk(self, *args, **kwargs) -> 'TimeSeriesDict'
```


Apply ZPK filter to each TimeSeries in the dict.
Returns a new TimeSeriesDict.



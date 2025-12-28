# TimeSeries

**Inherits from:** TimeSeriesInteropMixin, TimeSeriesAnalysisMixin, TimeSeriesResamplingMixin, TimeSeriesSignalMixin, TimeSeriesSpectralMixin, TimeSeries


Extended TimeSeries with all gwexpy functionality.

This class combines functionality from multiple modules:
- Core operations: is_regular, _check_regular, tail, crop, append, find_peaks
- Spectral transforms: fft, psd, cwt, laplace, etc.
- Signal processing: analytic_signal, mix_down, xcorr, etc.
- Analysis: impute, standardize, rolling_*, etc.
- Interoperability: to_pandas, to_torch, to_xarray, etc.

Inherits from gwpy.timeseries.TimeSeries for full compatibility.


## Methods

### `DictClass`

```python
DictClass(...)
```

Ordered key-value mapping of named `TimeSeries` objects

This object is designed to hold data for many different sources (channels)
for a single time span.

The main entry points for this object are the
:meth:`~TimeSeriesDict.read` and :meth:`~TimeSeriesDict.fetch`
data access methods.


### `abs`

```python
abs(self, axis=None, **kwargs)
```

absolute(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])

Calculate the absolute value element-wise.

``np.abs`` is a shorthand for this function.

Parameters
----------
x : array_like
    Input array.
out : ndarray, None, or tuple of ndarray and None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
where : array_like, optional
    This condition is broadcast over the input. At locations where the
    condition is True, the `out` array will be set to the ufunc result.
    Elsewhere, the `out` array will retain its original value.
    Note that if an uninitialized `out` array is created via the default
    ``out=None``, locations within it where the condition is False will
    remain uninitialized.
**kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

Returns
-------
absolute : ndarray
    An ndarray containing the absolute value of
    each element in `x`.  For complex input, ``a + ib``, the
    absolute value is :math:`\sqrt{ a^2 + b^2 }`.
    This is a scalar if `x` is a scalar.

Examples
--------
>>> x = np.array([-1.2, 1.2])
>>> np.absolute(x)
array([ 1.2,  1.2])
>>> np.absolute(1.2 + 1j)
1.5620499351813308

Plot the function over ``[-10, 10]``:

>>> import matplotlib.pyplot as plt

>>> x = np.linspace(start=-10, stop=10, num=101)
>>> plt.plot(x, np.absolute(x))
>>> plt.show()

Plot the function over the complex plane:

>>> xx = x + 1j * x[:, np.newaxis]
>>> plt.imshow(np.abs(xx), extent=[-10, 10, -10, 10], cmap='gray')
>>> plt.show()

The `abs` function can be used as a shorthand for ``np.absolute`` on
ndarrays.

>>> x = np.array([-1.2, 1.2])
>>> abs(x)
array([1.2, 1.2])

### `analytic_signal`

```python
analytic_signal(self, pad: 'Any' = None, pad_mode: 'str' = 'reflect', pad_value: 'float' = 0.0, nan_policy: 'str' = 'raise', copy: 'bool' = True) -> "'TimeSeriesSignalMixin'"
```


Compute the analytic signal (Hilbert transform) of the TimeSeries.

If input is real, returns complex analytic signal z(t) = x(t) + i H[x(t)].
If input is complex, returns a copy (casting to complex if needed).


### `append`

```python
append(self, other: 'Any', inplace: 'bool' = True, pad: 'Any' = None, gap: 'Any' = None, resize: 'bool' = True) -> "'TimeSeries'"
```


Append another TimeSeries (GWpy-compatible), returning gwexpy TimeSeries.


### `asd`

```python
asd(self, *args: 'Any', **kwargs: 'Any') -> 'Any'
```

Compute ASD of the TimeSeries.

### `asfreq`

```python
asfreq(self, rule: 'Any', method: 'Optional[str]' = None, fill_value: 'Any' = nan, *, origin: 'str' = 't0', offset: 'Any' = None, align: 'str' = 'ceil', tolerance: 'Optional[float]' = None, max_gap: 'Optional[float]' = None, copy: 'bool' = True) -> "'TimeSeriesResamplingMixin'"
```


Reindex the TimeSeries to a new fixed-interval grid associated with the given rule.

Parameters
----------
rule : str, float, or Quantity
    Target time interval (e.g., '1s', 0.1, 0.5*u.s).
method : str or None
    Interpolation method: None (exact), 'interpolate', 'ffill', 'bfill', 'nearest'.
fill_value : scalar
    Value for missing data points.
origin : str
    Reference point for grid alignment: 't0' or 'gps0'.
offset : Quantity
    Time offset from origin.
align : str
    Grid alignment: 'ceil' or 'floor'.
tolerance : float
    Tolerance for matching times.
max_gap : float
    Maximum gap for interpolation.
copy : bool
    Whether to copy data.

Returns
-------
TimeSeries
    Reindexed series.


### `auto_coherence`

```python
auto_coherence(self, dt, fftlength=None, overlap=None, window='hann', **kwargs)
```

Calculate the frequency-coherence between this `TimeSeries`
and a time-shifted copy of itself.

The standard :meth:`TimeSeries.coherence` is calculated between
the input `TimeSeries` and a :meth:`cropped <TimeSeries.crop>`
copy of itself. Since the cropped version will be shorter, the
input series will be shortened to match.

Parameters
----------
dt : `float`
    duration (in seconds) of time-shift

fftlength : `float`, optional
    number of seconds in single FFT, defaults to a single FFT
    covering the full duration

overlap : `float`, optional
    number of seconds of overlap between FFTs, defaults to the
    recommended overlap for the given window (if given), or 0

window : `str`, `numpy.ndarray`, optional
    window function to apply to timeseries prior to FFT,
    see :func:`scipy.signal.get_window` for details on acceptable
    formats

**kwargs
    any other keyword arguments accepted by
    :func:`matplotlib.mlab.cohere` except ``NFFT``, ``window``,
    and ``noverlap`` which are superceded by the above keyword
    arguments

Returns
-------
coherence : `~gwpy.frequencyseries.FrequencySeries`
    the coherence `FrequencySeries` of this `TimeSeries`
    with the other

Notes
-----
The :meth:`TimeSeries.auto_coherence` will perform best when
``dt`` is approximately ``fftlength / 2``.

See also
--------
matplotlib.mlab.cohere
    for details of the coherence calculator


### `average_fft`

```python
average_fft(self, fftlength=None, overlap=0, window=None)
```

Compute the averaged one-dimensional DFT of this `TimeSeries`.

This method computes a number of FFTs of duration ``fftlength``
and ``overlap`` (both given in seconds), and returns the mean
average. This method is analogous to the Welch average method
for power spectra.

Parameters
----------
fftlength : `float`
    number of seconds in single FFT, default, use
    whole `TimeSeries`

overlap : `float`, optional
    number of seconds of overlap between FFTs, defaults to the
    recommended overlap for the given window (if given), or 0

window : `str`, `numpy.ndarray`, optional
    window function to apply to timeseries prior to FFT,
    see :func:`scipy.signal.get_window` for details on acceptable
    formats

Returns
-------
out : complex-valued `~gwpy.frequencyseries.FrequencySeries`
    the transformed output, with populated frequencies array
    metadata

See also
--------
TimeSeries.fft
   The FFT method used.


### `bandpass`

```python
bandpass(self, flow, fhigh, gpass=2, gstop=30, fstop=None, type='iir', filtfilt=True, **kwargs)
```

Filter this `TimeSeries` with a band-pass filter.

Parameters
----------
flow : `float`
    lower corner frequency of pass band

fhigh : `float`
    upper corner frequency of pass band

gpass : `float`
    the maximum loss in the passband (dB).

gstop : `float`
    the minimum attenuation in the stopband (dB).

fstop : `tuple` of `float`, optional
    `(low, high)` edge-frequencies of stop band

type : `str`
    the filter type, either ``'iir'`` or ``'fir'``

**kwargs
    other keyword arguments are passed to
    :func:`gwpy.signal.filter_design.bandpass`

Returns
-------
bpseries : `TimeSeries`
    a band-passed version of the input `TimeSeries`

See also
--------
gwpy.signal.filter_design.bandpass
    for details on the filter design
TimeSeries.filter
    for details on how the filter is applied


### `baseband`

```python
baseband(self, *, phase: 'Any' = None, f0: 'Any' = None, fdot: 'Any' = 0.0, fddot: 'Any' = 0.0, phase_epoch: 'Any' = None, phase0: 'float' = 0.0, lowpass: 'Optional[float]' = None, lowpass_kwargs: 'Optional[dict[str, Any]]' = None, output_rate: 'Optional[float]' = None, singlesided: 'bool' = False) -> "'TimeSeriesSignalMixin'"
```


Demodulate the TimeSeries to baseband, optionally applying lowpass filter and resampling.


### `cepstrum`

```python
cepstrum(self, kind: 'str' = 'real', *, window: 'Any' = None, detrend: 'bool' = False, eps: 'Optional[float]' = None, fft_mode: 'str' = 'gwpy') -> 'Any'
```


Compute the Cepstrum of the TimeSeries.

Parameters
----------
kind : {"real", "power", "complex"}, optional
    Type of cepstrum. Default is "real".
window : `str`, `numpy.ndarray`, or `None`, optional
    Window function to apply.
detrend : `bool`, optional
    If `True`, detrend (linear).
eps : `float`, optional
    Small value to avoid log(0).
fft_mode : `str`, optional
    Mode for the underlying FFT.

Returns
-------
out : `FrequencySeries`
    The cepstrum, with frequencies interpreted as quefrency (seconds).


### `channel`

Instrumental channel associated with these data

:type: `~gwpy.detector.Channel`


### `coherence`

```python
coherence(self, *args: 'Any', **kwargs: 'Any') -> 'Any'
```

Compute coherence of the TimeSeries.

### `coherence_spectrogram`

```python
coherence_spectrogram(self, other, stride, fftlength=None, overlap=None, window='hann', nproc=1)
```

Calculate the coherence spectrogram between this `TimeSeries`
and other.

Parameters
----------
other : `TimeSeries`
    the second `TimeSeries` in this CSD calculation

stride : `float`
    number of seconds in single PSD (column of spectrogram)

fftlength : `float`
    number of seconds in single FFT

overlap : `float`, optional
    number of seconds of overlap between FFTs, defaults to the
    recommended overlap for the given window (if given), or 0

window : `str`, `numpy.ndarray`, optional
    window function to apply to timeseries prior to FFT,
    see :func:`scipy.signal.get_window` for details on acceptable
    formats

nproc : `int`
    number of parallel processes to use when calculating
    individual coherence spectra.

Returns
-------
spectrogram : `~gwpy.spectrogram.Spectrogram`
    time-frequency coherence spectrogram as generated from the
    input time-series.


### `convolve`

```python
convolve(self, fir, window='hann')
```

Convolve this `TimeSeries` with an FIR filter using the
   overlap-save method

Parameters
----------
fir : `numpy.ndarray`
    the time domain filter to convolve with

window : `str`, optional
    window function to apply to boundaries, default: ``'hann'``
    see :func:`scipy.signal.get_window` for details on acceptable
    formats

Returns
-------
out : `TimeSeries`
    the result of the convolution

See also
--------
scipy.signal.fftconvolve
    for details on the convolution scheme used here
TimeSeries.filter
    for an alternative method designed for short filters

Notes
-----
The output `TimeSeries` is the same length and has the same timestamps
as the input.

Due to filter settle-in, a segment half the length of `fir` will be
corrupted at the left and right boundaries. To prevent spectral leakage
these segments will be windowed before convolving.


### `copy`

```python
copy(self, order='C')
```

a.copy(order='C')

Return a copy of the array.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    Controls the memory layout of the copy. 'C' means C-order,
    'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
    'C' otherwise. 'K' means match the layout of `a` as closely
    as possible. (Note that this function and :func:`numpy.copy` are very
    similar but have different default values for their order=
    arguments, and this function always passes sub-classes through.)

See also
--------
numpy.copy : Similar function with different default behavior
numpy.copyto

Notes
-----
This function is the preferred method for creating an array copy.  The
function :func:`numpy.copy` is similar, but it defaults to using order 'K',
and will not pass sub-classes through by default.

Examples
--------
>>> x = np.array([[1,2,3],[4,5,6]], order='F')

>>> y = x.copy()

>>> x.fill(0)

>>> x
array([[0, 0, 0],
       [0, 0, 0]])

>>> y
array([[1, 2, 3],
       [4, 5, 6]])

>>> y.flags['C_CONTIGUOUS']
True

### `correlate`

```python
correlate(self, mfilter, window='hann', detrend='linear', whiten=False, wduration=2, highpass=None, **asd_kw)
```

Cross-correlate this `TimeSeries` with another signal

Parameters
----------
mfilter : `TimeSeries`
    the time domain signal to correlate with

window : `str`, optional
    window function to apply to timeseries prior to FFT,
    default: ``'hann'``
    see :func:`scipy.signal.get_window` for details on acceptable
    formats

detrend : `str`, optional
    type of detrending to do before FFT (see `~TimeSeries.detrend`
    for more details), default: ``'linear'``

whiten : `bool`, optional
    boolean switch to enable (`True`) or disable (`False`) data
    whitening, default: `False`

wduration : `float`, optional
    duration (in seconds) of the time-domain FIR whitening filter,
    only used if `whiten=True`, defaults to 2 seconds

highpass : `float`, optional
    highpass corner frequency (in Hz) of the FIR whitening filter,
    only used if `whiten=True`, default: `None`

**asd_kw
    keyword arguments to pass to `TimeSeries.asd` to generate
    an ASD, only used if `whiten=True`

Returns
-------
snr : `TimeSeries`
    the correlated signal-to-noise ratio (SNR) timeseries

See also
--------
TimeSeries.asd
    for details on the ASD calculation
TimeSeries.convolve
    for details on convolution with the overlap-save method

Notes
-----
The `window` argument is used in ASD estimation, whitening, and
preventing spectral leakage in the output. It is not used to condition
the matched-filter, which should be windowed before passing to this
method.

Due to filter settle-in, a segment half the length of `mfilter` will be
corrupted at the beginning and end of the output. See
`~TimeSeries.convolve` for more details.

The input and matched-filter will be detrended, and the output will be
normalised so that the SNR measures number of standard deviations from
the expected mean.


### `correlation`

```python
correlation(self, other, method='pearson', **kwargs)
```

Calculate correlation coefficient with another TimeSeries.

Args:
    other (TimeSeries): The series to compare with.
    method (str): 'pearson', 'kendall', or 'mic'.
    **kwargs: Additional arguments passed to the underlying function.

Returns:
    float: The correlation coefficient.



### `crop`

```python
crop(self, start: 'Any' = None, end: 'Any' = None, copy: 'bool' = False) -> "'TimeSeries'"
```


Crop this series to the given GPS start and end times.
Accepts any time format supported by gwexpy.time.to_gps.


### `csd`

```python
csd(self, *args: 'Any', **kwargs: 'Any') -> 'Any'
```

Compute CSD of the TimeSeries.

### `csd_spectrogram`

```python
csd_spectrogram(self, other, stride, fftlength=None, overlap=0, window='hann', nproc=1, **kwargs)
```

Calculate the cross spectral density spectrogram of this
   `TimeSeries` with 'other'.

Parameters
----------
other : `~gwpy.timeseries.TimeSeries`
    second time-series for cross spectral density calculation

stride : `float`
    number of seconds in single PSD (column of spectrogram).

fftlength : `float`
    number of seconds in single FFT.

overlap : `float`, optional
    number of seconds of overlap between FFTs, defaults to the
    recommended overlap for the given window (if given), or 0

window : `str`, `numpy.ndarray`, optional
    window function to apply to timeseries prior to FFT,
    see :func:`scipy.signal.get_window` for details on acceptable
    formats

nproc : `int`
    maximum number of independent frame reading processes, default
    is set to single-process file reading.

Returns
-------
spectrogram : `~gwpy.spectrogram.Spectrogram`
    time-frequency cross spectrogram as generated from the
    two input time-series.


### `cwt`

```python
cwt(self, wavelet: 'str' = 'cmor1.5-1.0', widths: 'Any' = None, frequencies: 'Any' = None, *, window: 'Any' = None, detrend: 'bool' = False, output: 'str' = 'spectrogram', chunk_size: 'Optional[int]' = None, **kwargs: 'Any') -> 'Any'
```


Compute the Continuous Wavelet Transform (CWT) using PyWavelets.

Parameters
----------
wavelet : `str`, optional
    Wavelet name (default "cmor1.5-1.0" for complex morlet).
widths : `array-like`, optional
    Scales to use for CWT.
frequencies : `array-like`, optional
     Frequencies to use (Hz). If provided, converts to scales.
window : `str`, `numpy.ndarray`, or `None`
     Window function to apply before CWT.
detrend : `bool`
     If True, detrend the time series.
output : {"spectrogram", "ndarray"}
     Output format.
chunk_size : `int`, optional
     If provided, compute in chunks to save memory.
**kwargs :
     Additional arguments passed to `pywt.cwt`.

Returns
-------
out : `gwpy.spectrogram.Spectrogram` or `(ndarray, freqs)`


### `ktau`

```python
ktau(self, other)
```

Calculate Kendall's Rank Correlation Coefficient.

Robust against outliers and non-parametric.
Note: Calculation is O(N log N).


### `kurtosis`

```python
kurtosis(self, fisher=True, nan_policy='propagate')
```

Compute the kurtosis (Fisher or Pearson) of the time series data.

Kurtosis is a measure of the "tailedness" of the probability distribution 
of a real-valued random variable.

Parameters
----------
fisher : `bool`, optional
    If True, Fisher's definition is used (normal ==> 0.0). 
    If False, Pearson's definition is used (normal ==> 3.0).
nan_policy : `str`, optional
    'propagate', 'raise', 'omit'.

Returns
-------
float
    The kurtosis.



### `mic`

```python
mic(self, other, alpha=0.6, c=15, est="mic_approx")
```

Calculate Maximal Information Coefficient (MIC) using minepy.

MIC is robust against non-linear relationships.
Reference: Reshef et al. (2011), Jung et al. (2022).

Args:
    other (TimeSeries): The series to compare with.
    alpha (float): Exponent for B = n^alpha. Default 0.6.
    c (float): Clumping parameter. Default 15.
    est (str): Estimator name. Default "mic_approx".


### `pcc`

```python
pcc(self, other)
```

Calculate Pearson Correlation Coefficient.

Linear correlation coefficient.


### `dct`

```python
dct(self, type: 'int' = 2, norm: 'str' = 'ortho', *, window: 'Any' = None, detrend: 'bool' = False) -> 'Any'
```


Compute the Discrete Cosine Transform (DCT) of the TimeSeries.

Parameters
----------
type : `int`, optional
    Type of the DCT (1, 2, 3, 4). Default is 2.
norm : `str`, optional
    Normalization mode. Default is "ortho".
window : `str`, `numpy.ndarray`, or `None`, optional
    Window function to apply before DCT.
detrend : `bool`, optional
    If `True`, remove the mean (or detrend) from the data before DCT.

Returns
-------
out : `FrequencySeries`
     The DCT of the TimeSeries.


### `demodulate`

```python
demodulate(self, f, stride=1, exp=False, deg=True)
```

Compute the average magnitude and phase of this `TimeSeries`
once per stride at a given frequency

Parameters
----------
f : `float`
    frequency (Hz) at which to demodulate the signal

stride : `float`, optional
    stride (seconds) between calculations, defaults to 1 second

exp : `bool`, optional
    return the magnitude and phase trends as one `TimeSeries` object
    representing a complex exponential, default: False

deg : `bool`, optional
    if `exp=False`, calculates the phase in degrees

Returns
-------
mag, phase : `TimeSeries`
    if `exp=False`, returns a pair of `TimeSeries` objects representing
    magnitude and phase trends with `dt=stride`

out : `TimeSeries`
    if `exp=True`, returns a single `TimeSeries` with magnitude and
    phase trends represented as `mag * exp(1j*phase)` with `dt=stride`

Examples
--------
Demodulation is useful when trying to examine steady sinusoidal
signals we know to be contained within data. For instance,
we can download some data from GWOSC to look at trends of the
amplitude and phase of LIGO Livingston's calibration line at 331.3 Hz:

>>> from gwpy.timeseries import TimeSeries
>>> data = TimeSeries.fetch_open_data('L1', 1131350417, 1131357617)

We can demodulate the `TimeSeries` at 331.3 Hz with a stride of one
minute:

>>> amp, phase = data.demodulate(331.3, stride=60)

We can then plot these trends to visualize fluctuations in the
amplitude of the calibration line:

>>> from gwpy.plot import Plot
>>> plot = Plot(amp)
>>> ax = plot.gca()
>>> ax.set_ylabel('Strain Amplitude at 331.3 Hz')
>>> plot.show()

See also
--------
TimeSeries.heterodyne
    for the underlying heterodyne detection method


### `detrend`

```python
detrend(self, detrend='constant')
```

Remove the trend from this `TimeSeries`

This method just wraps :func:`scipy.signal.detrend` to return
an object of the same type as the input.

Parameters
----------
detrend : `str`, optional
    the type of detrending.

Returns
-------
detrended : `TimeSeries`
    the detrended input series

See also
--------
scipy.signal.detrend
    for details on the options for the `detrend` argument, and
    how the operation is done


### `diff`

```python
diff(self, n=1, axis=-1)
```

Calculate the n-th order discrete difference along given axis.

The first order difference is given by ``out[n] = a[n+1] - a[n]`` along
the given axis, higher order differences are calculated by using `diff`
recursively.

Parameters
----------
n : int, optional
    The number of times values are differenced.
axis : int, optional
    The axis along which the difference is taken, default is the
    last axis.

Returns
-------
diff : `Series`
    The `n` order differences. The shape of the output is the same
    as the input, except along `axis` where the dimension is
    smaller by `n`.

See also
--------
numpy.diff
    for documentation on the underlying method


### `distance_correlation`

```python
distance_correlation(self, other)
```

Calculate Distance Correlation (dCor).

Distance correlation is a measure of dependence between two random vectors 
that is 0 if and only if the random vectors are independent. 
It can detect non-linear relationships.

Args:
    other (TimeSeries): The series to compare with.

Returns:
    float: The distance correlation (0 to 1).



### `dt`

X-axis sample separation

:type: `~astropy.units.Quantity` scalar


### `dumps`

```python
dumps(self)
```

a.dumps()

Returns the pickle of the array as a string.
pickle.loads will convert the string back to an array.

Parameters
----------
None

### `duration`

Duration of this series in seconds

:type: `~astropy.units.Quantity` scalar


### `dx`

X-axis sample separation

:type: `~astropy.units.Quantity` scalar


### `emd`

```python
emd(self, *, method: 'str' = 'eemd', max_imf: 'Optional[int]' = None, sift_max_iter: 'int' = 1000, stopping_criterion: 'Any' = 'default', eemd_noise_std: 'float' = 0.2, eemd_trials: 'int' = 100, random_state: 'Optional[int]' = None, return_residual: 'bool' = True) -> 'Any'
```


Perform Empirical Mode Decomposition (EMD) or EEMD.

Parameters
----------
method : {"emd", "eemd"}, optional
    Decomposition method. Default is "eemd".
max_imf : `int`, optional
    Maximum number of IMFs to extract.
sift_max_iter : `int`, optional
    Maximum number of sifting iterations.
stopping_criterion : `str` or `dict`, optional
    Stopping criterion configuration.
eemd_noise_std : `float`, optional
    Noise standard deviation for EEMD. Default 0.2.
eemd_trials : `int`, optional
    Number of trials for EEMD. Default 100.
random_state : `int` or `np.random.RandomState`, optional
    Seed for reproducibility.
return_residual : `bool`, optional
    If True, include residual in the output.

Returns
-------
imfs : `TimeSeriesDict`
    A dictionary of extracted IMFs (keys: "IMF1", "IMF2", ..., "residual").


### `envelope`

```python
envelope(self, *args: 'Any', **kwargs: 'Any') -> "'TimeSeriesSignalMixin'"
```

Compute the envelope of the TimeSeries.

### `epoch`

GPS epoch for these data.

This attribute is stored internally by the `t0` attribute

:type: `~astropy.time.Time`


### `fetch`

```python
fetch(channel, start, end, host=None, port=None, verbose=False, connection=None, verify=False, pad=None, allow_tape=None, scaled=None, type=None, dtype=None)
```

Fetch data from NDS

Parameters
----------
channel : `str`, `~gwpy.detector.Channel`
    the data channel for which to query

start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
    GPS start time of required data,
    any input parseable by `~gwpy.time.to_gps` is fine

end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
    GPS end time of required data,
    any input parseable by `~gwpy.time.to_gps` is fine

host : `str`, optional
    URL of NDS server to use, if blank will try any server
    (in a relatively sensible order) to get the data

port : `int`, optional
    port number for NDS server query, must be given with `host`

verify : `bool`, optional, default: `False`
    check channels exist in database before asking for data

scaled : `bool`, optional
    apply slope and bias calibration to ADC data, for non-ADC data
    this option has no effect

connection : `nds2.connection`, optional
    open NDS connection to use

verbose : `bool`, optional
    print verbose output about NDS progress, useful for debugging;
    if ``verbose`` is specified as a string, this defines the
    prefix for the progress meter

type : `int`, optional
    NDS2 channel type integer or string name to match

dtype : `type`, `numpy.dtype`, `str`, optional
    NDS2 data type to match


### `fetch_open_data`

```python
fetch_open_data(ifo, start, end, sample_rate=4096, version=None, format='hdf5', host='https://gwosc.org', verbose=False, cache=None, **kwargs)
```

Fetch open-access data from GWOSC.

Parameters
----------
ifo : `str`
    the two-character prefix of the IFO in which you are interested,
    e.g. `'L1'`

start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
    GPS start time of required data, defaults to start of data found;
    any input parseable by `~gwpy.time.to_gps` is fine

end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
    GPS end time of required data, defaults to end of data found;
    any input parseable by `~gwpy.time.to_gps` is fine

sample_rate : `float`, optional,
    the sample rate of desired data; most data are stored
    by GWOSC at 4096 Hz, however there may be event-related
    data releases with a 16384 Hz rate, default: `4096`

version : `int`, optional
    version of files to download, defaults to highest discovered
    version

format : `str`, optional
    the data format to download and parse, default: ``'h5py'``

    - ``'hdf5'``
    - ``'gwf'`` - requires |LDAStools.frameCPP|_

host : `str`, optional
    HTTP host name of GWOSC server to access

verbose : `bool`, optional, default: `False`
    print verbose output while fetching data

cache : `bool`, optional
    save/read a local copy of the remote URL, default: `False`;
    useful if the same remote data are to be accessed multiple times.
    Set `GWPY_CACHE=1` in the environment to auto-cache.

timeout : `float`, optional
    the time to wait for a response from the GWOSC server.

**kwargs
    any other keyword arguments are passed to the `TimeSeries.read`
    method that parses the file that was downloaded

Examples
--------
>>> from gwpy.timeseries import (TimeSeries, StateVector)
>>> print(TimeSeries.fetch_open_data('H1', 1126259446, 1126259478))
TimeSeries([  2.17704028e-19,  2.08763900e-19,  2.39681183e-19,
            ...,   3.55365541e-20,  6.33533516e-20,
              7.58121195e-20]
           unit: Unit(dimensionless),
           t0: 1126259446.0 s,
           dt: 0.000244140625 s,
           name: Strain,
           channel: None)
>>> print(StateVector.fetch_open_data('H1', 1126259446, 1126259478))
StateVector([127,127,127,127,127,127,127,127,127,127,127,127,
             127,127,127,127,127,127,127,127,127,127,127,127,
             127,127,127,127,127,127,127,127]
            unit: Unit(dimensionless),
            t0: 1126259446.0 s,
            dt: 1.0 s,
            name: Data quality,
            channel: None,
            bits: Bits(0: data present
                       1: passes cbc CAT1 test
                       2: passes cbc CAT2 test
                       3: passes cbc CAT3 test
                       4: passes burst CAT1 test
                       5: passes burst CAT2 test
                       6: passes burst CAT3 test,
                       channel=None,
                       epoch=1126259446.0))

For the `StateVector`, the naming of the bits will be
``format``-dependent, because they are recorded differently by GWOSC
in different formats.

Notes
-----
`StateVector` data are not available in ``txt.gz`` format.


### `fft`

```python
fft(self, nfft: 'Optional[int]' = None, *, mode: 'str' = 'gwpy', pad_mode: 'str' = 'zero', pad_left: 'int' = 0, pad_right: 'int' = 0, nfft_mode: 'Optional[str]' = None, other_length: 'Optional[int]' = None, **kwargs: 'Any') -> 'Any'
```


Compute the one-dimensional discrete Fourier Transform.

Parameters
----------
nfft : `int`, optional
    Length of the FFT. If `None`, the length of the TimeSeries is used.
mode : `str`, optional, default: "gwpy"
    "gwpy": Standard behavior (circular convolution assumption).
    "transient": Transient-friendly mode with padding options.
pad_mode : `str`, optional, default: "zero"
    Padding mode for "transient" mode. "zero" or "reflect".
pad_left : `int`, optional, default: 0
    Number of samples to pad on the left (for "transient" mode).
pad_right : `int`, optional, default: 0
    Number of samples to pad on the right (for "transient" mode).
nfft_mode : `str`, optional, default: None
    "next_fast_len": Use scipy.fft.next_fast_len to optimize FFT size.
    None: Use exact calculated length.
other_length : `int`, optional, default: None
    If provided in "transient" mode, the target length is calculated as
    len(self) + other_length - 1 (linear convolution size).
**kwargs
    Keyword arguments passed to the `FrequencySeries` constructor.

Returns
-------
out : `FrequencySeries`
    The DFT of the TimeSeries.


### `fftgram`

```python
fftgram(self, fftlength, overlap=None, window='hann', **kwargs)
```

Calculate the Fourier-gram of this `TimeSeries`.

At every ``stride``, a single, complex FFT is calculated.

Parameters
----------
fftlength : `float`
    number of seconds in single FFT.

overlap : `float`, optional
    number of seconds of overlap between FFTs, defaults to the
    recommended overlap for the given window (if given), or 0

window : `str`, `numpy.ndarray`, optional
    window function to apply to timeseries prior to FFT,
    see :func:`scipy.signal.get_window` for details on acceptable


Returns
-------
    a Fourier-gram


### `filter`

```python
filter(self, *filt, **kwargs)
```

Filter this `TimeSeries` with an IIR or FIR filter

Parameters
----------
*filt : filter arguments
    1, 2, 3, or 4 arguments defining the filter to be applied,

        - an ``Nx1`` `~numpy.ndarray` of FIR coefficients
        - an ``Nx6`` `~numpy.ndarray` of SOS coefficients
        - ``(numerator, denominator)`` polynomials
        - ``(zeros, poles, gain)``
        - ``(A, B, C, D)`` 'state-space' representation

filtfilt : `bool`, optional
    filter forward and backwards to preserve phase,
    default: `False`

analog : `bool`, optional
    if `True`, filter coefficients will be converted from Hz
    to Z-domain digital representation, default: `False`

inplace : `bool`, optional
    if `True`, this array will be overwritten with the filtered
    version, default: `False`

**kwargs
    other keyword arguments are passed to the filter method

Returns
-------
result : `TimeSeries`
    the filtered version of the input `TimeSeries`

Notes
-----
IIR filters are converted into cascading second-order sections before
being applied to this `TimeSeries`.

FIR filters are passed directly to :func:`scipy.signal.lfilter` or
:func:`scipy.signal.filtfilt` without any conversions.

See also
--------
scipy.signal.sosfilt
    for details on filtering with second-order sections

scipy.signal.sosfiltfilt
    for details on forward-backward filtering with second-order
    sections

scipy.signal.lfilter
    for details on filtering (without SOS)

scipy.signal.filtfilt
    for details on forward-backward filtering (without SOS)

Raises
------
ValueError
    if ``filt`` arguments cannot be interpreted properly

Examples
--------
We can design an arbitrarily complicated filter using
:mod:`gwpy.signal.filter_design`

>>> from gwpy.signal import filter_design
>>> bp = filter_design.bandpass(50, 250, 4096.)
>>> notches = [filter_design.notch(f, 4096.) for f in (60, 120, 180)]
>>> zpk = filter_design.concatenate_zpks(bp, *notches)

And then can download some data from GWOSC to apply it using
`TimeSeries.filter`:

>>> from gwpy.timeseries import TimeSeries
>>> data = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)
>>> filtered = data.filter(zpk, filtfilt=True)

We can plot the original signal, and the filtered version, cutting
off either end of the filtered data to remove filter-edge artefacts

>>> from gwpy.plot import Plot
>>> plot = Plot(data, filtered[128:-128], separate=True)
>>> plot.show()


### `find`

```python
find(channel, start, end, frametype=None, pad=None, scaled=None, nproc=1, verbose=False, **readargs)
```

Find and read data from frames for a channel

Parameters
----------
channel : `str`, `~gwpy.detector.Channel`
    the name of the channel to read, or a `Channel` object.

start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
    GPS start time of required data,
    any input parseable by `~gwpy.time.to_gps` is fine

end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
    GPS end time of required data,
    any input parseable by `~gwpy.time.to_gps` is fine

frametype : `str`, optional
    name of frametype in which this channel is stored, will search
    for containing frame types if necessary

nproc : `int`, optional, default: `1`
    number of parallel processes to use, serial process by
    default.

pad : `float`, optional
    value with which to fill gaps in the source data,
    by default gaps will result in a `ValueError`.

allow_tape : `bool`, optional, default: `True`
    allow reading from frame files on (slow) magnetic tape

verbose : `bool`, optional
    print verbose output about read progress, if ``verbose``
    is specified as a string, this defines the prefix for the
    progress meter

**readargs
    any other keyword arguments to be passed to `.read()`


### `find_gates`

```python
find_gates(self, tzero=1.0, whiten=True, threshold=50.0, cluster_window=0.5, **whiten_kwargs)
```

Identify points that should be gates using a provided threshold
and clustered within a provided time window.

Parameters
----------
tzero : `int`, optional
    half-width time duration (seconds) in which the timeseries is
    set to zero

whiten : `bool`, optional
    if True, data will be whitened before gating points are discovered,
    use of this option is highly recommended

threshold : `float`, optional
    amplitude threshold, if the data exceeds this value a gating window
    will be placed

cluster_window : `float`, optional
    time duration (seconds) over which gating points will be clustered

**whiten_kwargs
    other keyword arguments that will be passed to the
    `TimeSeries.whiten` method if it is being used when discovering
    gating points

Returns
-------
out : `~gwpy.segments.SegmentList`
    a list of segments that should be gated based on the
    provided parameters

See also
--------
TimeSeries.gate
    for a method that applies the identified gates


### `find_peaks`

```python
find_peaks(self, height: 'Any' = None, threshold: 'Any' = None, distance: 'Any' = None, prominence: 'Any' = None, width: 'Any' = None, wlen: 'Optional[int]' = None, rel_height: 'float' = 0.5, plateau_size: 'Any' = None) -> "tuple['TimeSeries', dict[str, Any]]"
```


Find peaks in the TimeSeries.

Wraps `scipy.signal.find_peaks`.

Returns
-------
`tuple`
    (peaks, props) where `peaks` is a new `TimeSeries` containing the peak values and times, and `props` is a dictionary of properties.


### `fit`

```python
fit(self, model: 'str', p0: 'Optional[dict[str, float]]' = None, method: 'str' = 'leastsq', **kwargs: 'Any') -> 'Any'
```


Fit the TimeSeries to a model function.

Parameters
----------
model : `str` or `callable`
    Name of built-in model (e.g., 'gaussian', 'exponential', 'damped_oscillation') or a callable function.
p0 : `dict`, optional
    Initial guess for parameters.
method : `str`, optional
    Fitting method: 'leastsq' (default) or 'mcmc' (requires `emcee`).
**kwargs
    Additional arguments passed to the fitter (e.g., `nwalkers`, `nsteps` for MCMC).

Returns
-------
`FitResult`
    Object containing fit results, parameters, and plotting methods.


### `fit_arima`

```python
fit_arima(self, order: 'tuple' = (1, 0, 0), **kwargs: 'Any') -> 'Any'
```


Fit ARIMA model to the series.

Parameters
----------
order : tuple
    (p, d, q) order of the ARIMA model.
**kwargs
    Additional arguments passed to the ARIMA fitting function.

Returns
-------
ARIMAResult
    Fitted model result.

See Also
--------
gwexpy.timeseries.arima.fit_arima


### `flatten`

```python
flatten(self, order='C')
```

Return a copy of the array collapsed into one dimension.

Any index information is removed as part of the flattening,
and the result is returned as a `~astropy.units.Quantity` array.

Parameters
----------
order : {'C', 'F', 'A', 'K'}, optional
    'C' means to flatten in row-major (C-style) order.
    'F' means to flatten in column-major (Fortran-
    style) order. 'A' means to flatten in column-major
    order if `a` is Fortran *contiguous* in memory,
    row-major order otherwise. 'K' means to flatten
    `a` in the order the elements occur in memory.
    The default is 'C'.

Returns
-------
y : `~astropy.units.Quantity`
    A copy of the input array, flattened to one dimension.

See also
--------
ravel : Return a flattened array.
flat : A 1-D flat iterator over the array.

Examples
--------
>>> a = Array([[1,2], [3,4]], unit='m', name='Test')
>>> a.flatten()
<Quantity [1., 2., 3., 4.] m>


### `from_astropy_timeseries`

```python
from_astropy_timeseries(ap_ts: 'Any', column: 'str' = 'value', unit: 'Optional[Any]' = None) -> 'Any'
```


Create from astropy.timeseries.TimeSeries.

Parameters
----------
ap_ts : astropy.timeseries.TimeSeries
    Input TimeSeries.
column : str
    Column name containing data.
unit : Unit, optional
    Physical unit.

Returns
-------
TimeSeries


### `from_cupy`

```python
from_cupy(array: 'Any', *, t0: 'Any' = None, dt: 'Any' = None, unit: 'Optional[Any]' = None) -> 'Any'
```


Create from cupy array.

Parameters
----------
array : cupy.ndarray
    Input array.
t0, dt : required
    Time parameters.
unit : Unit, optional
    Physical unit.

Returns
-------
TimeSeries


### `from_dask`

```python
from_dask(array: 'Any', *, t0: 'Any' = None, dt: 'Any' = None, unit: 'Optional[Any]' = None, compute: 'bool' = True) -> 'Any'
```


Create from dask.array.

Parameters
----------
array : dask.array.Array
    Input array.
t0, dt : required
    Time parameters.
unit : Unit, optional
    Physical unit.
compute : bool
    Whether to compute immediately.

Returns
-------
TimeSeries


### `from_hdf5_dataset`

```python
from_hdf5_dataset(group: 'Any', path: 'str') -> 'Any'
```


Read from HDF5 group/dataset.

Parameters
----------
group : h5py.Group or h5py.File
    Source group.
path : str
    Dataset path.

Returns
-------
TimeSeries


### `from_jax`

```python
from_jax(array: 'Any', *, t0: 'Any' = None, dt: 'Any' = None, unit: 'Optional[Any]' = None) -> 'Any'
```


Create from jax array.

Parameters
----------
array : jax.numpy.ndarray
    Input array.
t0, dt : required
    Time parameters.
unit : Unit, optional
    Physical unit.

Returns
-------
TimeSeries


### `from_lal`

```python
from_lal(lalts, copy=True)
```

Generate a new TimeSeries from a LAL TimeSeries of any type.
        

### `from_nds2_buffer`

```python
from_nds2_buffer(buffer_, scaled=None, copy=True, **metadata)
```

Construct a new series from an `nds2.buffer` object

**Requires:** |nds2|_

Parameters
----------
buffer_ : `nds2.buffer`
    the input NDS2-client buffer to read

scaled : `bool`, optional
    apply slope and bias calibration to ADC data, for non-ADC data
    this option has no effect

copy : `bool`, optional
    if `True`, copy the contained data array to new  to a new array

**metadata
    any other metadata keyword arguments to pass to the `TimeSeries`
    constructor

Returns
-------
timeseries : `TimeSeries`
    a new `TimeSeries` containing the data from the `nds2.buffer`,
    and the appropriate metadata


### `from_netcdf4`

```python
from_netcdf4(ds: 'Any', var_name: 'str') -> 'Any'
```


Read from netCDF4 Dataset.

Parameters
----------
ds : netCDF4.Dataset
    Source dataset.
var_name : str
    Variable name.

Returns
-------
TimeSeries


### `from_obspy`

```python
from_obspy(tr: 'Any', *, unit: 'Optional[Any]' = None, name_policy: 'str' = 'id') -> 'Any'
```


Create TimeSeries from obspy.Trace.

Parameters
----------
tr : obspy.Trace
    Input trace.
unit : Unit, optional
    Physical unit.
name_policy : str
    How to derive name: 'id', 'station', etc.

Returns
-------
TimeSeries


### `from_pandas`

```python
from_pandas(series: 'Any', *, unit: 'Optional[Any]' = None, t0: 'Any' = None, dt: 'Any' = None) -> 'Any'
```


Create TimeSeries from Pandas Series.

### `from_simpeg`

```python
from_simpeg(cls, data_obj, **kwargs)
```


Create TimeSeries from SimPEG Data object (TDEM).

Parameters
----------
cls : type
    The class itself (for class method).
data_obj : SimPEG.data.Data
    Input SimPEG data object.
unit : Unit, optional
    Physical unit of the data.
t0 : Quantity or float, optional
    Start time.
dt : Quantity or float, optional
    Sample interval.

Returns
-------
TimeSeries


### `from_polars`

```python
from_polars(data: 'Any', time_column: 'Optional[str]' = 'time', unit: 'Optional[Any]' = None) -> 'Any'
```


Create TimeSeries from polars.DataFrame or polars.Series.

Parameters
----------
data : polars.DataFrame or polars.Series
    Input data.
time_column : str, optional
    If data is a DataFrame, name of the column to use as time.
unit : Unit, optional
    Physical unit.

Returns
-------
TimeSeries


### `from_pycbc`

```python
from_pycbc(pycbcseries, copy=True)
```

Convert a `pycbc.types.timeseries.TimeSeries` into a `TimeSeries`

Parameters
----------
pycbcseries : `pycbc.types.timeseries.TimeSeries`
    the input PyCBC `~pycbc.types.timeseries.TimeSeries` array

copy : `bool`, optional, default: `True`
    if `True`, copy these data to a new array

Returns
-------
timeseries : `TimeSeries`
    a GWpy version of the input timeseries


### `from_pydub`

```python
from_pydub(seg: 'Any', *, unit: 'Optional[Any]' = None) -> 'Any'
```


Create from pydub.AudioSegment.

Parameters
----------
seg : pydub.AudioSegment
    Input audio segment.
unit : Unit, optional
    Physical unit.

Returns
-------
TimeSeries


### `from_root`

```python
from_root(obj: 'Any', return_error: 'bool' = False) -> 'Any'
```


Create TimeSeries from ROOT TGraph or TH1.

Parameters
----------
obj : ROOT.TGraph or ROOT.TH1
    Input ROOT object.
return_error : bool, default False
    If True, return (series, error_series).

Returns
-------
TimeSeries or tuple of TimeSeries


### `from_sqlite`

```python
from_sqlite(conn: 'Any', series_id: 'Any') -> 'Any'
```


Load from sqlite3 database.

Parameters
----------
conn : sqlite3.Connection
    Database connection.
series_id : str
    Identifier for the series.

Returns
-------
TimeSeries


### `from_tensorflow`

```python
from_tensorflow(tensor: 'Any', *, t0: 'Any' = None, dt: 'Any' = None, unit: 'Optional[Any]' = None) -> 'Any'
```


Create from tensorflow.Tensor.

Parameters
----------
tensor : tensorflow.Tensor
    Input tensor.
t0, dt : required
    Time parameters.
unit : Unit, optional
    Physical unit.

Returns
-------
TimeSeries


### `from_torch`

```python
from_torch(tensor: 'Any', *, t0: 'Any' = None, dt: 'Any' = None, unit: 'Optional[Any]' = None) -> 'Any'
```


Create from torch.Tensor.

Parameters
----------
tensor : torch.Tensor
    Input tensor.
t0 : Quantity or float
    Start time (required).
dt : Quantity or float
    Sample interval (required).
unit : Unit, optional
    Physical unit.

Returns
-------
TimeSeries


### `from_xarray`

```python
from_xarray(da: 'Any', *, unit: 'Optional[Any]' = None) -> 'Any'
```


Create TimeSeries from xarray.DataArray.

Parameters
----------
da : xarray.DataArray
    Input DataArray.
unit : Unit, optional
    Physical unit.

Returns
-------
TimeSeries


### `from_zarr`

```python
from_zarr(store: 'Any', path: 'str') -> 'Any'
```


Read from Zarr array.

Parameters
----------
store : str or zarr.Store
    Source store.
path : str
    Array path.

Returns
-------
TimeSeries


### `gate`

```python
gate(self, tzero=1.0, tpad=0.5, whiten=True, threshold=50.0, cluster_window=0.5, **whiten_kwargs)
```

Removes high amplitude peaks from data using inverse Planck window.

Points will be discovered automatically using a provided threshold
and clustered within a provided time window.

Parameters
----------
tzero : `int`, optional
    half-width time duration (seconds) in which the timeseries is
    set to zero

tpad : `int`, optional
    half-width time duration (seconds) in which the Planck window
    is tapered

whiten : `bool`, optional
    if True, data will be whitened before gating points are discovered,
    use of this option is highly recommended

threshold : `float`, optional
    amplitude threshold, if the data exceeds this value a gating window
    will be placed

cluster_window : `float`, optional
    time duration (seconds) over which gating points will be clustered

**whiten_kwargs
    other keyword arguments that will be passed to the
    `TimeSeries.whiten` method if it is being used when discovering
    gating points

Returns
-------
out : `~gwpy.timeseries.TimeSeries`
    a copy of the original `TimeSeries` that has had gating windows
    applied

Examples
--------
Read data into a `TimeSeries`

>>> from gwpy.timeseries import TimeSeries
>>> data = TimeSeries.fetch_open_data('H1', 1135148571, 1135148771)

Apply gating using custom arguments

>>> gated = data.gate(tzero=1.0, tpad=1.0, threshold=10.0,
                      fftlength=4, overlap=2, method='median')

Plot the original data and the gated data, whiten both for
visualization purposes

>>> overlay = data.whiten(4,2,method='median').plot(dpi=150,
                          label='Ungated', color='dodgerblue',
                          zorder=2)
>>> ax = overlay.gca()
>>> ax.plot(gated.whiten(4,2,method='median'), label='Gated',
            color='orange', zorder=3)
>>> ax.set_xlim(1135148661, 1135148681)
>>> ax.legend()
>>> overlay.show()

See also
--------
TimeSeries.mask
    for the method that masks out unwanted data
TimeSeries.find_gates
    for the method that identifies gating points
TimeSeries.whiten
    for the whitening filter used to identify gating points


### `get`

```python
get(channel, start, end, pad=None, scaled=None, dtype=None, verbose=False, allow_tape=None, **kwargs)
```

Get data for this channel from frames or NDS

This method dynamically accesses either frames on disk, or a
remote NDS2 server to find and return data for the given interval

Parameters
----------
channel : `str`, `~gwpy.detector.Channel`
    the name of the channel to read, or a `Channel` object.

start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
    GPS start time of required data,
    any input parseable by `~gwpy.time.to_gps` is fine

end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
    GPS end time of required data,
    any input parseable by `~gwpy.time.to_gps` is fine

pad : `float`, optional
    value with which to fill gaps in the source data,
    by default gaps will result in a `ValueError`.

scaled : `bool`, optional
    apply slope and bias calibration to ADC data, for non-ADC data
    this option has no effect

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
    :meth:`.find` (for direct GWF file access) or
    :meth:`.fetch` for remote NDS2 access

See also
--------
TimeSeries.fetch
    for grabbing data from a remote NDS2 server
TimeSeries.find
    for discovering and reading data from local GWF files


### `heterodyne`

```python
heterodyne(self, phase, stride=1, singlesided=False)
```

Compute the average magnitude and phase of this `TimeSeries`
once per stride after heterodyning with a given phase series

Parameters
----------
phase : `array_like`
    an array of phase measurements (radians) with which to heterodyne
    the signal

stride : `float`, optional
    stride (seconds) between calculations, defaults to 1 second

singlesided : `bool`, optional
    Boolean switch to return single-sided output (i.e., to multiply by
    2 so that the signal is distributed across positive frequencies
    only), default: False

Returns
-------
out : `TimeSeries`
    magnitude and phase trends, represented as
    `mag * exp(1j*phase)` with `dt=stride`

Notes
-----
This is similar to the :meth:`~gwpy.timeseries.TimeSeries.demodulate`
method, but is more general in that it accepts a varying phase
evolution, rather than a fixed frequency.

Unlike :meth:`~gwpy.timeseries.TimeSeries.demodulate`, the complex
output is double-sided by default, so is not multiplied by 2.

Examples
--------
Heterodyning can be useful in analysing quasi-monochromatic signals
with a known phase evolution, such as continuous-wave signals
from rapidly rotating neutron stars. These sources radiate at a
frequency that slowly decreases over time, and is Doppler modulated
due to the Earth's rotational and orbital motion.

To see an example of heterodyning in action, we can simulate a signal
whose phase evolution is described by the frequency and its first
derivative with respect to time. We can download some O1 era
LIGO-Livingston data from GWOSC, inject the simulated signal, and
recover its amplitude.

>>> from gwpy.timeseries import TimeSeries
>>> data = TimeSeries.fetch_open_data('L1', 1131350417, 1131354017)

We now need to set the signal parameters, generate the expected
phase evolution, and create the signal:

>>> import numpy
>>> f0 = 123.456789  # signal frequency (Hz)
>>> fdot = -9.87654321e-7  # signal frequency derivative (Hz/s)
>>> fepoch = 1131350417  # phase epoch
>>> amp = 1.5e-22  # signal amplitude
>>> phase0 = 0.4  # signal phase at the phase epoch
>>> times = data.times.value - fepoch
>>> phase = 2 * numpy.pi * (f0 * times + 0.5 * fdot * times**2)
>>> signal = TimeSeries(amp * numpy.cos(phase + phase0),
>>>                     sample_rate=data.sample_rate, t0=data.t0)
>>> data = data.inject(signal)

To recover the signal, we can bandpass the injected data around the
signal frequency, then heterodyne using our phase model with a stride
of 60 seconds:

>>> filtdata = data.bandpass(f0 - 0.5, f0 + 0.5)
>>> het = filtdata.heterodyne(phase, stride=60, singlesided=True)

We can then plot signal amplitude over time (cropping the first two
minutes to remove the filter response):

>>> plot = het.crop(het.x0.value + 180).abs().plot()
>>> ax = plot.gca()
>>> ax.set_ylabel("Strain amplitude")
>>> plot.show()

See also
--------
TimeSeries.demodulate
    for a method to heterodyne at a fixed frequency


### `hht`

```python
hht(self, *, emd_method: 'str' = 'eemd', emd_kwargs: 'Optional[dict[str, Any]]' = None, hilbert_kwargs: 'Optional[dict[str, Any]]' = None, output: 'str' = 'dict') -> 'Any'
```


Perform Hilbert-Huang Transform (HHT): EMD + HSA.

Parameters
----------
emd_method : `str`
    "emd" or "eemd".
emd_kwargs : `dict`, optional
    Arguments for `emd()`.
hilbert_kwargs : `dict`, optional
    Arguments for `hilbert_analysis()`.
output : `str`
    "dict" (returns IMFs and properties) or "spectrogram" (returns Spectrogram).

Returns
-------
result : `dict` or `Spectrogram`


### `highpass`

```python
highpass(self, frequency, gpass=2, gstop=30, fstop=None, type='iir', filtfilt=True, **kwargs)
```

Filter this `TimeSeries` with a high-pass filter.

Parameters
----------
frequency : `float`
    high-pass corner frequency

gpass : `float`
    the maximum loss in the passband (dB).

gstop : `float`
    the minimum attenuation in the stopband (dB).

fstop : `float`
    stop-band edge frequency, defaults to `frequency * 1.5`

type : `str`
    the filter type, either ``'iir'`` or ``'fir'``

**kwargs
    other keyword arguments are passed to
    :func:`gwpy.signal.filter_design.highpass`

Returns
-------
hpseries : `TimeSeries`
    a high-passed version of the input `TimeSeries`

See also
--------
gwpy.signal.filter_design.highpass
    for details on the filter design
TimeSeries.filter
    for details on how the filter is applied


### `hilbert`

```python
hilbert(self, *args: 'Any', **kwargs: 'Any') -> "'TimeSeriesSignalMixin'"
```

Alias for analytic_signal.

### `hilbert_analysis`

```python
hilbert_analysis(self, *, unwrap_phase: 'bool' = True, frequency_unit: 'str' = 'Hz') -> 'dict[str, Any]'
```


Perform Hilbert Spectral Analysis (HSA) to get instantaneous properties.

Returns
-------
results : `dict`
    {
        "analytic": TimeSeries (complex),
        "amplitude": TimeSeries (real),
        "phase": TimeSeries (rad),
        "frequency": TimeSeries (Hz)
    }


### `hurst`

```python
hurst(self, **kwargs: 'Any') -> 'Any'
```


Compute Hurst exponent.

The Hurst exponent is a measure of long-term memory of a time series.
H < 0.5: anti-persistent (mean-reverting)
H = 0.5: random walk
H > 0.5: persistent (trending)

Returns
-------
float
    Hurst exponent.

See Also
--------
gwexpy.timeseries.hurst.hurst


### `impute`

```python
impute(self, *, method: 'str' = 'linear', limit: 'Optional[int]' = None, axis: 'str' = 'time', max_gap: 'Optional[float]' = None, **kwargs: 'Any') -> 'Any'
```


Impute missing values.

Parameters
----------
method : str
    Interpolation method: 'linear', 'nearest', 'ffill', 'bfill', etc.
limit : int, optional
    Maximum number of consecutive NaNs to fill.
axis : str
    Axis along which to interpolate.
max_gap : float, optional
    Maximum gap (in time units) to interpolate across.
**kwargs
    Additional arguments passed to the imputation function.

Returns
-------
TimeSeries
    Imputed series.

See Also
--------
gwexpy.timeseries.preprocess.impute_timeseries


### `inject`

```python
inject(self, other)
```

Add two compatible `Series` along their shared x-axis values.

Parameters
----------
other : `Series`
    a `Series` whose xindex intersects with `self.xindex`

Returns
-------
out : `Series`
    the sum of `self` and `other` along their shared x-axis values

Raises
------
ValueError
    if `self` and `other` have incompatible units or xindex intervals

Notes
-----
If `other.xindex` and `self.xindex` do not intersect, this method will
return a copy of `self`. If the series have uniformly offset indices,
this method will raise a warning.

If `self.xindex` is an array of timestamps, and if `other.xspan` is
not a subset of `self.xspan`, then `other` will be cropped before
being adding to `self`.

Users who wish to taper or window their `Series` should do so before
passing it to this method. See :meth:`TimeSeries.taper` and
:func:`~gwpy.signal.window.planck` for more information.


### `instantaneous_frequency`

```python
instantaneous_frequency(self, unwrap: 'bool' = True, smooth: 'Any' = None, **kwargs: 'Any') -> "'TimeSeriesSignalMixin'"
```


Compute the instantaneous frequency of the TimeSeries.
Returns unit 'Hz'.


### `instantaneous_phase`

```python
instantaneous_phase(self, deg: 'bool' = False, unwrap: 'bool' = False, **kwargs: 'Any') -> "'TimeSeriesSignalMixin'"
```


Compute the instantaneous phase of the TimeSeries.


### `is_compatible`

```python
is_compatible(self, other)
```

Check whether this series and other have compatible metadata

This method tests that the `sample size <Series.dx>`, and the
`~Series.unit` match.


### `is_contiguous`

```python
is_contiguous(self, other, tol=3.814697265625e-06)
```

Check whether other is contiguous with self.

Parameters
----------
other : `Series`, `numpy.ndarray`
    another series of the same type to test for contiguity

tol : `float`, optional
    the numerical tolerance of the test

Returns
-------
1
    if `other` is contiguous with this series, i.e. would attach
    seamlessly onto the end
-1
    if `other` is anti-contiguous with this seires, i.e. would attach
    seamlessly onto the start
0
    if `other` is completely dis-contiguous with thie series

Notes
-----
if a raw `numpy.ndarray` is passed as other, with no metadata, then
the contiguity check will always pass


### `is_regular`

Return True if this TimeSeries has a regular sample rate.

### `laplace`

```python
laplace(self, *, sigma: 'Any' = 0.0, frequencies: 'Any' = None, t_start: 'Any' = None, t_stop: 'Any' = None, window: 'Any' = None, detrend: 'bool' = False, normalize: 'str' = 'integral', dtype: 'Any' = None, chunk_size: 'Optional[int]' = None, **kwargs: 'Any') -> 'Any'
```


One-sided finite-interval Laplace transform.

Define s = sigma + i*2*pi*f (f in Hz).
Compute on a cropped segment [t_start, t_stop] but shift time origin to t_start:
    L(s) = integral_0^T x(tau) exp(-(sigma + i 2pi f) tau) dtau
Discrete approximation uses uniform dt.

Parameters
----------
sigma : `float` or `astropy.units.Quantity`, optional
    Real part of the Laplace variable s (1/s). Default 0.0.
frequencies : `numpy.ndarray` or `astropy.units.Quantity`, optional
    Frequencies for the imaginary part of s (Hz).
    If None, defaults to `np.fft.rfftfreq(n, d=dt)`.
t_start : `float` or `astropy.units.Quantity`, optional
    Start time relative to the beginning of the TimeSeries (seconds).
t_stop : `float` or `astropy.units.Quantity`, optional
    Stop time relative to the beginning of the TimeSeries (seconds).
window : `str`, `numpy.ndarray`, or `None`, optional
    Window function to apply.
detrend : `bool`, optional
    If True, remove the mean before transformation.
normalize : `str`, optional
    "integral": Sum * dt (standard Laplace integral approximation).
    "mean": Sum / n.
dtype : `numpy.dtype`, optional
    Output data type.
chunk_size : `int`, optional
    If provided, compute the transform in chunks along the frequency axis
    to save memory.

Returns
-------
out : `FrequencySeries`
    The Laplace transform result (complex).


### `local_hurst`

```python
local_hurst(self, window: 'Any', **kwargs: 'Any') -> 'Any'
```


Compute local Hurst exponent over a sliding window.

Parameters
----------
window : str, float, or Quantity
    Window size for local computation.
**kwargs
    Additional arguments.

Returns
-------
TimeSeries
    Local Hurst exponent series.

See Also
--------
gwexpy.timeseries.hurst.local_hurst


### `lock_in`

```python
lock_in(self, *, phase: 'Any' = None, f0: 'Any' = None, fdot: 'Any' = 0.0, fddot: 'Any' = 0.0, phase_epoch: 'Any' = None, phase0: 'float' = 0.0, stride: 'float' = 1.0, singlesided: 'bool' = True, output: 'str' = 'amp_phase', deg: 'bool' = True) -> 'Any'
```


Perform lock-in amplification (demodulation + averaging).


### `lowpass`

```python
lowpass(self, frequency, gpass=2, gstop=30, fstop=None, type='iir', filtfilt=True, **kwargs)
```

Filter this `TimeSeries` with a Butterworth low-pass filter.

Parameters
----------
frequency : `float`
    low-pass corner frequency

gpass : `float`
    the maximum loss in the passband (dB).

gstop : `float`
    the minimum attenuation in the stopband (dB).

fstop : `float`
    stop-band edge frequency, defaults to `frequency * 1.5`

type : `str`
    the filter type, either ``'iir'`` or ``'fir'``

**kwargs
    other keyword arguments are passed to
    :func:`gwpy.signal.filter_design.lowpass`

Returns
-------
lpseries : `TimeSeries`
    a low-passed version of the input `TimeSeries`

See also
--------
gwpy.signal.filter_design.lowpass
    for details on the filter design
TimeSeries.filter
    for details on how the filter is applied


### `mask`

```python
mask(self, deadtime=None, flag=None, query_open_data=False, const=nan, tpad=0.5, **kwargs)
```

Mask away portions of this `TimeSeries` that fall within a given
list of time segments

Parameters
----------
deadtime : `SegmentList`, optional
    a list of time segments defining the deadtime (i.e., masked
    portions) of the output, will supersede `flag` if given

flag : `str`, optional
    the name of a data-quality flag for which to query, required if
    `deadtime` is not given

query_open_data : `bool`, optional
    if `True`, will query for publicly released data-quality segments
    through the Gravitational-wave Open Science Center (GWOSC),
    default: `False`

const : `float`, optional
    constant value with which to mask deadtime data,
    default: `~numpy.nan`

tpad : `float`, optional
    length of time (in seconds) over which to taper off data at
    mask segment boundaries, default: 0.5 seconds

**kwargs : `dict`, optional
    additional keyword arguments to
    `~gwpy.segments.DataQualityFlag.query` or
    `~gwpy.segments.DataQualityFlag.fetch_open_data`,
    see "Notes" below

Returns
-------
out : `TimeSeries`
    the masked version of this `TimeSeries`

Notes
-----
If `tpad` is nonzero, the Planck-taper window is used to smoothly
ramp data down to zero over a timescale `tpad` approaching every
segment boundary in `deadtime`. However, this does not apply to
the left or right bounds of the original `TimeSeries`.

The `deadtime` segment list will always be coalesced and restricted to
the limits of `self.span`. In particular, when querying a data-quality
flag, this means the `start` and `end` arguments to
`~gwpy.segments.DataQualityFlag.query` will effectively be reset and
therefore need not be given.

If `flag` is interpreted positively, i.e. if `flag` being active
corresponds to a "good" state, then its complement in `self.span`
will be used to define the deadtime for masking.

See also
--------
gwpy.segments.DataQualityFlag.query
    for the method to query segments of a given data-quality flag
gwpy.segments.DataQualityFlag.fetch_open_data
    for the method to query data-quality flags from the GWOSC database
gwpy.signal.window.planck
    for the generic Planck-taper window


### `median`

```python
median(self, axis=None, **kwargs)
```


Compute the median along the specified axis.

Returns the median of the array elements.

Parameters
----------
a : array_like
    Input array or object that can be converted to an array.
axis : {int, sequence of int, None}, optional
    Axis or axes along which the medians are computed. The default
    is to compute the median along a flattened version of the array.
    A sequence of axes is supported since version 1.9.0.
out : ndarray, optional
    Alternative output array in which to place the result. It must
    have the same shape and buffer length as the expected output,
    but the type (of the output) will be cast if necessary.
overwrite_input : bool, optional
   If True, then allow use of memory of input array `a` for
   calculations. The input array will be modified by the call to
   `median`. This will save memory when you do not need to preserve
   the contents of the input array. Treat the input as undefined,
   but it will probably be fully or partially sorted. Default is
   False. If `overwrite_input` is ``True`` and `a` is not already an
   `ndarray`, an error will be raised.
keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the original `arr`.

    .. versionadded:: 1.9.0

Returns
-------
median : ndarray
    A new array holding the result. If the input contains integers
    or floats smaller than ``float64``, then the output data-type is
    ``np.float64``.  Otherwise, the data-type of the output is the
    same as that of the input. If `out` is specified, that array is
    returned instead.

See Also
--------
mean, percentile

Notes
-----
Given a vector ``V`` of length ``N``, the median of ``V`` is the
middle value of a sorted copy of ``V``, ``V_sorted`` - i
e., ``V_sorted[(N-1)/2]``, when ``N`` is odd, and the average of the
two middle values of ``V_sorted`` when ``N`` is even.

Examples
--------
>>> a = np.array([[10, 7, 4], [3, 2, 1]])
>>> a
array([[10,  7,  4],
       [ 3,  2,  1]])
>>> np.median(a)
3.5
>>> np.median(a, axis=0)
array([6.5, 4.5, 2.5])
>>> np.median(a, axis=1)
array([7.,  2.])
>>> m = np.median(a, axis=0)
>>> out = np.zeros_like(m)
>>> np.median(a, axis=0, out=m)
array([6.5,  4.5,  2.5])
>>> m
array([6.5,  4.5,  2.5])
>>> b = a.copy()
>>> np.median(b, axis=1, overwrite_input=True)
array([7.,  2.])
>>> assert not np.all(a==b)
>>> b = a.copy()
>>> np.median(b, axis=None, overwrite_input=True)
3.5
>>> assert not np.all(a==b)



### `mix_down`

```python
mix_down(self, *, phase: 'Any' = None, f0: 'Any' = None, fdot: 'Any' = 0.0, fddot: 'Any' = 0.0, phase_epoch: 'Any' = None, phase0: 'float' = 0.0, singlesided: 'bool' = False, copy: 'bool' = True) -> "'TimeSeriesSignalMixin'"
```


Mix the TimeSeries with a complex oscillator.


### `name`

Name for this data set

:type: `str`


### `notch`

```python
notch(self, frequency, type='iir', filtfilt=True, **kwargs)
```

Notch out a frequency in this `TimeSeries`.

Parameters
----------
frequency : `float`, `~astropy.units.Quantity`
    frequency (default in Hertz) at which to apply the notch

type : `str`, optional
    type of filter to apply, currently only 'iir' is supported

**kwargs
    other keyword arguments to pass to `scipy.signal.iirdesign`

Returns
-------
notched : `TimeSeries`
   a notch-filtered copy of the input `TimeSeries`

See also
--------
TimeSeries.filter
   for details on the filtering method
scipy.signal.iirdesign
    for details on the IIR filter design method


### `override_unit`

```python
override_unit(self, unit, parse_strict='raise')
```

Forcefully reset the unit of these data

Use of this method is discouraged in favour of `to()`,
which performs accurate conversions from one unit to another.
The method should really only be used when the original unit of the
array is plain wrong.

Parameters
----------
unit : `~astropy.units.Unit`, `str`
    the unit to force onto this array
parse_strict : `str`, optional
    how to handle errors in the unit parsing, default is to
    raise the underlying exception from `astropy.units`

Raises
------
ValueError
    if a `str` cannot be parsed as a valid unit


### `pad`

```python
pad(self, pad_width, **kwargs)
```

Pad this series to a new size

Parameters
----------
pad_width : `int`, pair of `ints`
    number of samples by which to pad each end of the array;
    given a single `int` to pad both ends by the same amount,
    or a (before, after) `tuple` for assymetric padding

**kwargs
    see :meth:`numpy.pad` for kwarg documentation

Returns
-------
series : `Series`
    the padded version of the input

See also
--------
numpy.pad
    for details on the underlying functionality


### `plot`

```python
plot(self, method='plot', figsize=(12, 4), xscale='auto-gps', **kwargs)
```

Plot the data for this timeseries

Returns
-------
figure : `~matplotlib.figure.Figure`
    the newly created figure, with populated Axes.

See also
--------
matplotlib.pyplot.figure
    for documentation of keyword arguments used to create the
    figure
matplotlib.figure.Figure.add_subplot
    for documentation of keyword arguments used to create the
    axes
matplotlib.axes.Axes.plot
    for documentation of keyword arguments used in rendering the data


### `prepend`

```python
prepend(self, other, inplace=True, pad=None, gap=None, resize=True)
```

Connect another series onto the start of the current one.

Parameters
----------
other : `Series`
    another series of the same type as this one

inplace : `bool`, optional
    perform operation in-place, modifying current series,
    otherwise copy data and return new series, default: `True`

    .. warning::

       `inplace` prepend bypasses the reference check in
       `numpy.ndarray.resize`, so be carefully to only use this
       for arrays that haven't been sharing their memory!

pad : `float`, optional
    value with which to pad discontiguous series,
    by default gaps will result in a `ValueError`.

gap : `str`, optional
    action to perform if there's a gap between the other series
    and this one. One of

    - ``'raise'`` - raise a `ValueError`
    - ``'ignore'`` - remove gap and join data
    - ``'pad'`` - pad gap with zeros

    If `pad` is given and is not `None`, the default is ``'pad'``,
    otherwise ``'raise'``.

resize : `bool`, optional
    resize this array to accommodate new data, otherwise shift the
    old data to the left (potentially falling off the start) and
    put the new data in at the end, default: `True`.

Returns
-------
series : `TimeSeries`
    time-series containing joined data sets


### `psd`

```python
psd(self, *args: 'Any', **kwargs: 'Any') -> 'Any'
```

Compute PSD of the TimeSeries.

### `q_gram`

```python
q_gram(self, qrange=(4, 64), frange=(0, inf), mismatch=0.2, snrthresh=5.5, **kwargs)
```

Scan a `TimeSeries` using the multi-Q transform and return an
`EventTable` of the most significant tiles

Parameters
----------
qrange : `tuple` of `float`, optional
    `(low, high)` range of Qs to scan

frange : `tuple` of `float`, optional
    `(low, high)` range of frequencies to scan

mismatch : `float`, optional
    maximum allowed fractional mismatch between neighbouring tiles

snrthresh : `float`, optional
    lower inclusive threshold on individual tile SNR to keep in the
    table

**kwargs
    other keyword arguments to be passed to :meth:`QTiling.transform`,
    including ``'epoch'`` and ``'search'``

Returns
-------
qgram : `EventTable`
    a table of time-frequency tiles on the most significant `QPlane`

See also
--------
TimeSeries.q_transform
    for a method to interpolate the raw Q-transform over a regularly
    gridded spectrogram
gwpy.signal.qtransform
    for code and documentation on how the Q-transform is implemented
gwpy.table.EventTable.tile
    to render this `EventTable` as a collection of polygons

Notes
-----
Only tiles with signal energy greater than or equal to
`snrthresh ** 2 / 2` will be stored in the output `EventTable`. The
table columns are ``'time'``, ``'duration'``, ``'frequency'``,
``'bandwidth'``, and ``'energy'``.


### `q_transform`

```python
q_transform(self, qrange=(4, 64), frange=(0, inf), gps=None, search=0.5, tres='<default>', fres='<default>', logf=False, norm='median', mismatch=0.2, outseg=None, whiten=True, fduration=2, highpass=None, **asd_kw)
```

Scan a `TimeSeries` using the multi-Q transform and return an
interpolated high-resolution spectrogram

By default, this method returns a high-resolution spectrogram in
both time and frequency, which can result in a large memory
footprint. If you know that you only need a subset of the output
for, say, a figure, consider using ``outseg`` and the other
keyword arguments to restrict the size of the returned data.

Parameters
----------
qrange : `tuple` of `float`, optional
    `(low, high)` range of Qs to scan

frange : `tuple` of `float`, optional
    `(log, high)` range of frequencies to scan

gps : `float`, optional
    central time of interest for determine loudest Q-plane

search : `float`, optional
    window around `gps` in which to find peak energies, only
    used if `gps` is given

tres : `float`, optional
    desired time resolution (seconds) of output `Spectrogram`,
    default is `abs(outseg) / 1000.`

fres : `float`, `int`, `None`, optional
    desired frequency resolution (Hertz) of output `Spectrogram`,
    or, if ``logf=True``, the number of frequency samples;
    give `None` to skip this step and return the original resolution,
    default is 0.5 Hz or 500 frequency samples

logf : `bool`, optional
    boolean switch to enable (`True`) or disable (`False`) use of
    log-sampled frequencies in the output `Spectrogram`,
    if `True` then `fres` is interpreted as a number of frequency
    samples, default: `False`

norm : `bool`, `str`, optional
    whether to normalize the returned Q-transform output, or how,
    default: `True` (``'median'``), other options: `False`,
    ``'mean'``

mismatch : `float`
    maximum allowed fractional mismatch between neighbouring tiles

outseg : `~gwpy.segments.Segment`, optional
    GPS `[start, stop)` segment for output `Spectrogram`,
    default is the full duration of the input

whiten : `bool`, `~gwpy.frequencyseries.FrequencySeries`, optional
    boolean switch to enable (`True`) or disable (`False`) data
    whitening, or an ASD `~gwpy.freqencyseries.FrequencySeries`
    with which to whiten the data

fduration : `float`, optional
    duration (in seconds) of the time-domain FIR whitening filter,
    only used if `whiten` is not `False`, defaults to 2 seconds

highpass : `float`, optional
    highpass corner frequency (in Hz) of the FIR whitening filter,
    used only if `whiten` is not `False`, default: `None`

**asd_kw
    keyword arguments to pass to `TimeSeries.asd` to generate
    an ASD to use when whitening the data

Returns
-------
out : `~gwpy.spectrogram.Spectrogram`
    output `Spectrogram` of normalised Q energy

See also
--------
TimeSeries.asd
    for documentation on acceptable `**asd_kw`
TimeSeries.whiten
    for documentation on how the whitening is done
gwpy.signal.qtransform
    for code and documentation on how the Q-transform is implemented

Notes
-----
This method will return a `Spectrogram` of dtype ``float32`` if
``norm`` is given, and ``float64`` otherwise.

To optimize plot rendering with `~matplotlib.axes.Axes.pcolormesh`,
the output `~gwpy.spectrogram.Spectrogram` can be given a log-sampled
frequency axis by passing `logf=True` at runtime. The `fres` argument
is then the number of points on the frequency axis. Note, this is
incompatible with `~matplotlib.axes.Axes.imshow`.

It is also highly recommended to use the `outseg` keyword argument
when only a small window around a given GPS time is of interest. This
will speed up this method a little, but can greatly speed up
rendering the resulting `Spectrogram` using `pcolormesh`.

If you aren't going to use `pcolormesh` in the end, don't worry.

Examples
--------
>>> from numpy.random import normal
>>> from scipy.signal import gausspulse
>>> from gwpy.timeseries import TimeSeries

Generate a `TimeSeries` containing Gaussian noise sampled at 4096 Hz,
centred on GPS time 0, with a sine-Gaussian pulse ('glitch') at
500 Hz:

>>> noise = TimeSeries(normal(loc=1, size=4096*4), sample_rate=4096, epoch=-2)
>>> glitch = TimeSeries(gausspulse(noise.times.value, fc=500) * 4, sample_rate=4096)
>>> data = noise + glitch

Compute and plot the Q-transform of these data:

>>> q = data.q_transform()
>>> plot = q.plot()
>>> ax = plot.gca()
>>> ax.set_xlim(-.2, .2)
>>> ax.set_epoch(0)
>>> plot.show()


### `rayleigh_spectrogram`

```python
rayleigh_spectrogram(self, stride, fftlength=None, overlap=0, window='hann', nproc=1, **kwargs)
```

Calculate the Rayleigh statistic spectrogram of this `TimeSeries`

Parameters
----------
stride : `float`
    number of seconds in single PSD (column of spectrogram).

fftlength : `float`
    number of seconds in single FFT.

overlap : `float`, optional
    number of seconds of overlap between FFTs, passing `None` will
    choose based on the window method, default: ``0``

window : `str`, `numpy.ndarray`, optional
    window function to apply to timeseries prior to FFT,
    see :func:`scipy.signal.get_window` for details on acceptable
    formats

nproc : `int`, optional
    maximum number of independent frame reading processes, default
    default: ``1``

Returns
-------
spectrogram : `~gwpy.spectrogram.Spectrogram`
    time-frequency Rayleigh spectrogram as generated from the
    input time-series.

See also
--------
TimeSeries.rayleigh
    for details of the statistic calculation


### `rayleigh_spectrum`

```python
rayleigh_spectrum(self, fftlength=None, overlap=0, window='hann')
```

Calculate the Rayleigh `FrequencySeries` for this `TimeSeries`.

The Rayleigh statistic is calculated as the ratio of the standard
deviation and the mean of a number of periodograms.

Parameters
----------
fftlength : `float`
    number of seconds in single FFT, defaults to a single FFT
    covering the full duration

overlap : `float`, optional
    number of seconds of overlap between FFTs, passing `None` will
    choose based on the window method, default: ``0``

window : `str`, `numpy.ndarray`, optional
    window function to apply to timeseries prior to FFT,
    see :func:`scipy.signal.get_window` for details on acceptable
    formats

Returns
-------
psd :  `~gwpy.frequencyseries.FrequencySeries`
    a data series containing the PSD.


### `read`

```python
read(source, *args, **kwargs)
```

Read data into a `TimeSeries`

Arguments and keywords depend on the output format, see the
online documentation for full details for each format, the parameters
below are common to most formats.

Parameters
----------
source : `str`, `list`
    Source of data, any of the following:

    - `str` path of single data file,
    - `str` path of LAL-format cache file,
    - `list` of paths.

name : `str`, `~gwpy.detector.Channel`
    the name of the channel to read, or a `Channel` object.

start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
    GPS start time of required data, defaults to start of data found;
    any input parseable by `~gwpy.time.to_gps` is fine

end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
    GPS end time of required data, defaults to end of data found;
    any input parseable by `~gwpy.time.to_gps` is fine

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

Raises
------
IndexError
    if ``source`` is an empty list

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
resample(self, rate: 'Any', *args: 'Any', **kwargs: 'Any') -> "'TimeSeriesResamplingMixin'"
```


Resample the TimeSeries.

If 'rate' is a time-string (e.g. '1s') or time Quantity, performs time-bin aggregation.
Otherwise, performs signal processing resampling (gwpy standard).


### `rfft`

```python
rfft(self, *args: 'Any', **kwargs: 'Any') -> 'Any'
```

Compute RFFT of the TimeSeries.

### `rms`

```python
rms(self, stride=1)
```

Calculate the root-mean-square value of this `TimeSeries`
once per stride.

Parameters
----------
stride : `float`
    stride (seconds) between RMS calculations

Returns
-------
rms : `TimeSeries`
    a new `TimeSeries` containing the RMS value with dt=stride


### `rolling_max`

```python
rolling_max(self, window: 'Any', *, center: 'bool' = False, min_count: 'int' = 1, nan_policy: 'str' = 'omit', backend: 'str' = 'auto') -> 'Any'
```


Rolling maximum over time.

Parameters
----------
window : str, float, or Quantity
    Window size.
center : bool
    If True, center the window on each point.
min_count : int
    Minimum number of observations required.
nan_policy : str
    How to handle NaN values.
backend : str
    Computation backend.

Returns
-------
TimeSeries
    Rolling maximum.


### `rolling_mean`

```python
rolling_mean(self, window: 'Any', *, center: 'bool' = False, min_count: 'int' = 1, nan_policy: 'str' = 'omit', backend: 'str' = 'auto') -> 'Any'
```


Rolling mean over time.

Parameters
----------
window : str, float, or Quantity
    Window size.
center : bool
    If True, center the window on each point.
min_count : int
    Minimum number of observations required.
nan_policy : str
    How to handle NaN values.
backend : str
    Computation backend.

Returns
-------
TimeSeries
    Rolling mean.


### `rolling_median`

```python
rolling_median(self, window: 'Any', *, center: 'bool' = False, min_count: 'int' = 1, nan_policy: 'str' = 'omit', backend: 'str' = 'auto') -> 'Any'
```


Rolling median over time.

Parameters
----------
window : str, float, or Quantity
    Window size.
center : bool
    If True, center the window on each point.
min_count : int
    Minimum number of observations required.
nan_policy : str
    How to handle NaN values.
backend : str
    Computation backend.

Returns
-------
TimeSeries
    Rolling median.


### `rolling_min`

```python
rolling_min(self, window: 'Any', *, center: 'bool' = False, min_count: 'int' = 1, nan_policy: 'str' = 'omit', backend: 'str' = 'auto') -> 'Any'
```


Rolling minimum over time.

Parameters
----------
window : str, float, or Quantity
    Window size.
center : bool
    If True, center the window on each point.
min_count : int
    Minimum number of observations required.
nan_policy : str
    How to handle NaN values.
backend : str
    Computation backend.

Returns
-------
TimeSeries
    Rolling minimum.


### `rolling_std`

```python
rolling_std(self, window: 'Any', *, center: 'bool' = False, min_count: 'int' = 1, nan_policy: 'str' = 'omit', backend: 'str' = 'auto', ddof: 'int' = 0) -> 'Any'
```


Rolling standard deviation over time.

Parameters
----------
window : str, float, or Quantity
    Window size.
center : bool
    If True, center the window on each point.
min_count : int
    Minimum number of observations required.
nan_policy : str
    How to handle NaN values.
backend : str
    Computation backend.
ddof : int
    Delta degrees of freedom.

Returns
-------
TimeSeries
    Rolling standard deviation.


### `sample_rate`

Data rate for this `TimeSeries` in samples per second (Hertz).

This attribute is stored internally by the `dx` attribute

:type: `~astropy.units.Quantity` scalar


### `shift`

```python
shift(self, delta)
```

Shift this `Series` forward on the X-axis by ``delta``

This modifies the series in-place.

Parameters
----------
delta : `float`, `~astropy.units.Quantity`, `str`
    The amount by which to shift (in x-axis units if `float`), give
    a negative value to shift backwards in time

Examples
--------
>>> from gwpy.types import Series
>>> a = Series([1, 2, 3, 4, 5], x0=0, dx=1, xunit='m')
>>> print(a.x0)
0.0 m
>>> a.shift(5)
>>> print(a.x0)
5.0 m
>>> a.shift('-1 km')
-995.0 m


### `span`

X-axis [low, high) segment encompassed by these data

:type: `~gwpy.segments.Segment`


### `spectral_variance`

```python
spectral_variance(self, stride, fftlength=None, overlap=None, method='median', window='hann', nproc=1, filter=None, bins=None, low=None, high=None, nbins=500, log=False, norm=False, density=False)
```

Calculate the `SpectralVariance` of this `TimeSeries`.

Parameters
----------
stride : `float`
    number of seconds in single PSD (column of spectrogram)

fftlength : `float`
    number of seconds in single FFT

method : `str`, optional
    FFT-averaging method (default: ``'median'``),
    see *Notes* for more details

overlap : `float`, optional
    number of seconds of overlap between FFTs, defaults to the
    recommended overlap for the given window (if given), or 0

window : `str`, `numpy.ndarray`, optional
    window function to apply to timeseries prior to FFT,
    see :func:`scipy.signal.get_window` for details on acceptable
    formats

nproc : `int`
    maximum number of independent frame reading processes, default
    is set to single-process file reading.

bins : `numpy.ndarray`, optional, default `None`
    array of histogram bin edges, including the rightmost edge

low : `float`, optional
    left edge of lowest amplitude bin, only read
    if ``bins`` is not given

high : `float`, optional
    right edge of highest amplitude bin, only read
    if ``bins`` is not given

nbins : `int`, optional
    number of bins to generate, only read if ``bins`` is not
    given

log : `bool`, optional
    calculate amplitude bins over a logarithmic scale, only
    read if ``bins`` is not given

norm : `bool`, optional
    normalise bin counts to a unit sum

density : `bool`, optional
    normalise bin counts to a unit integral

Returns
-------
specvar : `SpectralVariance`
    2D-array of spectral frequency-amplitude counts

See also
--------
numpy.histogram
    for details on specifying bins and weights

Notes
-----
The accepted ``method`` arguments are:

- ``'bartlett'`` : a mean average of non-overlapping periodograms
- ``'median'`` : a median average of overlapping periodograms
- ``'welch'`` : a mean average of overlapping periodograms


### `spectrogram`

```python
spectrogram(self, stride, fftlength=None, overlap=None, window='hann', method='median', nproc=1, **kwargs)
```

Calculate the average power spectrogram of this `TimeSeries`
using the specified average spectrum method.

Each time-bin of the output `Spectrogram` is calculated by taking
a chunk of the `TimeSeries` in the segment
`[t - overlap/2., t + stride + overlap/2.)` and calculating the
:meth:`~gwpy.timeseries.TimeSeries.psd` of those data.

As a result, each time-bin is calculated using `stride + overlap`
seconds of data.

Parameters
----------
stride : `float`
    number of seconds in single PSD (column of spectrogram).

fftlength : `float`
    number of seconds in single FFT.

overlap : `float`, optional
    number of seconds of overlap between FFTs, defaults to the
    recommended overlap for the given window (if given), or 0

window : `str`, `numpy.ndarray`, optional
    window function to apply to timeseries prior to FFT,
    see :func:`scipy.signal.get_window` for details on acceptable
    formats

method : `str`, optional
    FFT-averaging method (default: ``'median'``),
    see *Notes* for more details

nproc : `int`
    number of CPUs to use in parallel processing of FFTs

Returns
-------
spectrogram : `~gwpy.spectrogram.Spectrogram`
    time-frequency power spectrogram as generated from the
    input time-series.

Notes
-----
The accepted ``method`` arguments are:

- ``'bartlett'`` : a mean average of non-overlapping periodograms
- ``'median'`` : a median average of overlapping periodograms
- ``'welch'`` : a mean average of overlapping periodograms


### `spectrogram2`

```python
spectrogram2(self, fftlength, overlap=None, window='hann', **kwargs)
```

Calculate the non-averaged power `Spectrogram` of this `TimeSeries`

Parameters
----------
fftlength : `float`
    number of seconds in single FFT.

overlap : `float`, optional
    number of seconds of overlap between FFTs, defaults to the
    recommended overlap for the given window (if given), or 0

window : `str`, `numpy.ndarray`, optional
    window function to apply to timeseries prior to FFT,
    see :func:`scipy.signal.get_window` for details on acceptable
    formats

scaling : [ 'density' | 'spectrum' ], optional
    selects between computing the power spectral density ('density')
    where the `Spectrogram` has units of V**2/Hz if the input is
    measured in V and computing the power spectrum ('spectrum')
    where the `Spectrogram` has units of V**2 if the input is
    measured in V. Defaults to 'density'.

**kwargs
    other parameters to be passed to `scipy.signal.periodogram` for
    each column of the `Spectrogram`

Returns
-------
spectrogram: `~gwpy.spectrogram.Spectrogram`
    a power `Spectrogram` with `1/fftlength` frequency resolution and
    (fftlength - overlap) time resolution.

See also
--------
scipy.signal.periodogram
    for documentation on the Fourier methods used in this calculation

Notes
-----
This method calculates overlapping periodograms for all possible
chunks of data entirely containing within the span of the input
`TimeSeries`, then normalises the power in overlapping chunks using
a triangular window centred on that chunk which most overlaps the
given `Spectrogram` time sample.


### `standardize`

```python
standardize(self, *, method: 'str' = 'zscore', ddof: 'int' = 0, robust: 'bool' = False) -> 'Any'
```


Standardize the series.

Parameters
----------
method : str
    Standardization method: 'zscore', 'minmax', etc.
ddof : int
    Delta degrees of freedom for std calculation.
robust : bool
    If True, use median/IQR instead of mean/std.

Returns
-------
TimeSeries
    Standardized series.

See Also
--------
gwexpy.timeseries.preprocess.standardize_timeseries


### `step`

```python
step(self, **kwargs)
```

Create a step plot of this series
        

### `stlt`

```python
stlt(self, stride: 'Any', window: 'Any', **kwargs: 'Any') -> 'Any'
```


Compute Short-Time Laplace Transform (STLT).

Produces a 3D time-frequency-frequency representation wrapped in
a TimePlaneTransform container.

Parameters
----------
stride : str or Quantity
    Time step duration (e.g. '1s').
window : str or Quantity
    Analysis window duration (e.g. '2s').

Returns
-------
TimePlaneTransform
    3D transform result with shape (time, axis1, axis2).


### `t0`

X-axis coordinate of the first data point

:type: `~astropy.units.Quantity` scalar


### `tail`

```python
tail(self, n: 'int' = 5) -> "'TimeSeries'"
```

Return the last `n` samples of this series.

### `taper`

```python
taper(self, side='leftright', duration=None, nsamples=None)
```

Taper the ends of this `TimeSeries` smoothly to zero.

Parameters
----------
side : `str`, optional
    the side of the `TimeSeries` to taper, must be one of `'left'`,
    `'right'`, or `'leftright'`

duration : `float`, optional
    the duration of time to taper, will override `nsamples`
    if both are provided as arguments

nsamples : `int`, optional
    the number of samples to taper, will be overridden by `duration`
    if both are provided as arguments

Returns
-------
out : `TimeSeries`
    a copy of `self` tapered at one or both ends

Raises
------
ValueError
    if `side` is not one of `('left', 'right', 'leftright')`

Examples
--------
To see the effect of the Planck-taper window, we can taper a
sinusoidal `TimeSeries` at both ends:

>>> import numpy
>>> from gwpy.timeseries import TimeSeries
>>> t = numpy.linspace(0, 1, 2048)
>>> series = TimeSeries(numpy.cos(10.5*numpy.pi*t), times=t)
>>> tapered = series.taper()

We can plot it to see how the ends now vary smoothly from 0 to 1:

>>> from gwpy.plot import Plot
>>> plot = Plot(series, tapered, separate=True, sharex=True)
>>> plot.show()

Notes
-----
The :meth:`TimeSeries.taper` automatically tapers from the second
stationary point (local maximum or minimum) on the specified side
of the input. However, the method will never taper more than half
the full width of the `TimeSeries`, and will fail if there are no
stationary points.

See :func:`~gwpy.signal.window.planck` for the generic Planck taper
window, and see :func:`scipy.signal.get_window` for other common
window formats.


### `times`

Positions of the data on the x-axis

:type: `~astropy.units.Quantity` array


### `to_astropy_timeseries`

```python
to_astropy_timeseries(self, column: 'str' = 'value', time_format: 'str' = 'gps') -> 'Any'
```


Convert to astropy.timeseries.TimeSeries.

Parameters
----------
column : str
    Column name for the data values.
time_format : str
    Time format ('gps', 'unix', etc.).

Returns
-------
astropy.timeseries.TimeSeries


### `to_cupy`

```python
to_cupy(self, dtype: 'Any' = None) -> 'Any'
```


Convert to cupy.array.

Parameters
----------
dtype : cupy.dtype, optional
    Output dtype.

Returns
-------
cupy.ndarray


### `to_dask`

```python
to_dask(self, chunks: 'Any' = 'auto') -> 'Any'
```


Convert to dask.array.

Parameters
----------
chunks : int or 'auto'
    Chunk size for the dask array.

Returns
-------
dask.array.Array


### `to_hdf5_dataset`

```python
to_hdf5_dataset(self, group: 'Any', path: 'str', *, overwrite: 'bool' = False, compression: 'Optional[str]' = None, compression_opts: 'Any' = None) -> 'None'
```


Write to HDF5 group/dataset.

Parameters
----------
group : h5py.Group or h5py.File
    Target group.
path : str
    Dataset path within group.
overwrite : bool
    Whether to overwrite existing dataset.
compression : str, optional
    Compression filter.
compression_opts : int, optional
    Compression level.


### `to_jax`

```python
to_jax(self, dtype: 'Any' = None) -> 'Any'
```


Convert to jax.numpy.array.

Parameters
----------
dtype : jax.numpy.dtype, optional
    Output dtype.

Returns
-------
jax.numpy.ndarray


### `to_lal`

```python
to_lal(self)
```

Convert this `TimeSeries` into a LAL TimeSeries.

.. note::

   This operation always copies data to new memory.


### `to_librosa`

```python
to_librosa(self, y_dtype: 'Any' = <class 'numpy.float32'>) -> 'Any'
```


Export to librosa-compatible numpy array.

Parameters
----------
y_dtype : dtype
    Output dtype (librosa expects float32).

Returns
-------
tuple
    (y, sr) where y is the audio signal and sr is sample rate.


### `to_mne`

```python
to_mne(self, info: 'Any' = None) -> 'Any'
```

Alias for :meth:`to_mne`.

### `to_mne`

```python
to_mne(self, info: 'Any' = None) -> 'Any'
```


Convert to ``mne.io.RawArray`` (single-channel).

Parameters
----------
info : mne.Info, optional
    Channel information. Created if not provided.

Returns
-------
mne.io.RawArray


### `to_netcdf4`

```python
to_netcdf4(self, ds: 'Any', var_name: 'str', **kwargs: 'Any') -> 'None'
```


Write to netCDF4 Dataset.

Parameters
----------
ds : netCDF4.Dataset
    Target dataset.
var_name : str
    Variable name.
**kwargs
    Additional arguments for createVariable.


### `to_obspy`

```python
to_obspy(self, *, stats_extra: 'Optional[dict[str, Any]]' = None, dtype: 'Any' = None) -> 'Any'
```


Convert to obspy.Trace.

Parameters
----------
stats_extra : dict, optional
    Extra stats to add to the Trace.
dtype : dtype, optional
    Output data type.

Returns
-------
obspy.Trace


### `to_pandas`

```python
to_pandas(self, index: 'str' = 'datetime', *, name: 'Optional[str]' = None, copy: 'bool' = False) -> 'Any'
```


Convert TimeSeries to pandas.Series.

Parameters
----------
index : str, default "datetime"
    Index type: "datetime" (UTC aware), "seconds" (unix), or "gps".
name : str, optional
    Name for the pandas Series.
copy : bool, default False
    Whether to guarantee a copy.

Returns
-------
pandas.Series


### `to_simpeg`

```python
to_simpeg(self, location=None, rx_type='PointElectricField', orientation='x', **kwargs) -> 'simpeg.data.Data'
```

Convert to SimPEG Data object.


### `to_polars`

```python
to_polars(self, name: 'Optional[str]' = None, as_dataframe: 'bool' = True, time_column: 'str' = 'time', time_unit: 'str' = 'datetime') -> 'Any'
```


Convert TimeSeries to polars object.

Parameters
----------
name : str, optional
    Name for the polars Series/Column.
as_dataframe : bool, default True
    If True, returns a DataFrame with a time column.
    If False, returns a raw Series of values.
time_column : str, default "time"
    Name of the time column (only if as_dataframe=True).
time_unit : str, default "datetime"
    Format of the time column: "datetime", "gps", or "unix".

Returns
-------
polars.DataFrame or polars.Series


### `to_pycbc`

```python
to_pycbc(self, copy=True)
```

Convert this `TimeSeries` into a PyCBC
`~pycbc.types.timeseries.TimeSeries`

Parameters
----------
copy : `bool`, optional, default: `True`
    if `True`, copy these data to a new array

Returns
-------
timeseries : `~pycbc.types.timeseries.TimeSeries`
    a PyCBC representation of this `TimeSeries`


### `to_pydub`

```python
to_pydub(self, sample_width: 'int' = 2, channels: 'int' = 1) -> 'Any'
```


Export to pydub.AudioSegment.

Parameters
----------
sample_width : int
    Bytes per sample (1, 2, or 4).
channels : int
    Number of audio channels.

Returns
-------
pydub.AudioSegment


### `to_sqlite`

```python
to_sqlite(self, conn: 'Any', series_id: 'Optional[str]' = None, *, overwrite: 'bool' = False) -> 'Any'
```


Save to sqlite3 database.

Parameters
----------
conn : sqlite3.Connection
    Database connection.
series_id : str, optional
    Identifier for the series.
overwrite : bool
    Whether to overwrite existing.

Returns
-------
str
    The series_id used.


### `to_tensorflow`

```python
to_tensorflow(self, dtype: 'Any' = None) -> 'Any'
```


Convert to tensorflow.Tensor.

Parameters
----------
dtype : tf.dtype, optional
    Output dtype.

Returns
-------
tensorflow.Tensor


### `to_tgraph`

```python
to_tgraph(self, error: 'Optional[Any]' = None) -> 'Any'
```


Convert to ROOT TGraph or TGraphErrors.

Parameters
----------
error : Series, Quantity, or array-like, optional
    Error bars for the y-axis.

Returns
-------
ROOT.TGraph or ROOT.TGraphErrors


### `to_th1d`

```python
to_th1d(self, error: 'Optional[Any]' = None) -> 'Any'
```


Convert to ROOT TH1D.

Parameters
----------
error : Series, Quantity, or array-like, optional
    Bin errors.

Returns
-------
ROOT.TH1D


### `to_torch`

```python
to_torch(self, device: 'Optional[str]' = None, dtype: 'Any' = None, requires_grad: 'bool' = False, copy: 'bool' = False) -> 'Any'
```


Convert to torch.Tensor.

Parameters
----------
device : str, optional
    Target device ('cpu', 'cuda', etc.).
dtype : torch.dtype, optional
    Output dtype.
requires_grad : bool
    Whether to enable gradient tracking.
copy : bool
    Whether to force a copy.

Returns
-------
torch.Tensor


### `to_xarray`

```python
to_xarray(self, time_coord: 'str' = 'datetime') -> 'Any'
```


Convert to xarray.DataArray.

Parameters
----------
time_coord : str
    Name of the time coordinate.

Returns
-------
xarray.DataArray


### `to_zarr`

```python
to_zarr(self, store: 'Any', path: 'str', chunks: 'Any' = None, compressor: 'Any' = None, overwrite: 'bool' = False) -> 'None'
```


Write to Zarr array.

Parameters
----------
store : str or zarr.Store
    Target store.
path : str
    Array path within store.
chunks : int, optional
    Chunk size.
compressor : numcodecs.Codec, optional
    Compression codec.
overwrite : bool
    Whether to overwrite existing.


### `tostring`

```python
tostring(self, order='C')
```

a.tobytes(order='C')

Construct Python bytes containing the raw data bytes in the array.

Constructs Python bytes showing a copy of the raw contents of
data memory. The bytes object is produced in C-order by default.
This behavior is controlled by the ``order`` parameter.

.. versionadded:: 1.9.0

Parameters
----------
order : {'C', 'F', 'A'}, optional
    Controls the memory layout of the bytes object. 'C' means C-order,
    'F' means F-order, 'A' (short for *Any*) means 'F' if `a` is
    Fortran contiguous, 'C' otherwise. Default is 'C'.

Returns
-------
s : bytes
    Python bytes exhibiting a copy of `a`'s raw data.

See also
--------
frombuffer
    Inverse of this operation, construct a 1-dimensional array from Python
    bytes.

Examples
--------
>>> x = np.array([[0, 1], [2, 3]], dtype='<u2')
>>> x.tobytes()
b'\x00\x00\x01\x00\x02\x00\x03\x00'
>>> x.tobytes('C') == x.tobytes()
True
>>> x.tobytes('F')
b'\x00\x00\x02\x00\x01\x00\x03\x00'

### `transfer_function`

```python
transfer_function(self, other: 'Any', fftlength: 'Optional[float]' = None, overlap: 'Optional[float]' = None, window: 'Any' = 'hann', average: 'str' = 'mean', *, method: 'str' = 'gwpy', fft_kwargs: 'Optional[dict[str, Any]]' = None, downsample: 'Optional[float]' = None, align: 'str' = 'intersection', **kwargs: 'Any') -> 'Any'
```


Compute the transfer function between this TimeSeries and another.

Parameters
----------
other : `TimeSeries`
    The input TimeSeries.
fftlength : `int`, optional
    Length of the FFT, in seconds (default) or samples.
overlap : `int`, optional
    Overlap between segments, in seconds (default) or samples.
window : `str`, `numpy.ndarray`, optional
    Window function to apply.
average : `str`, optional
    Method to average viewing periods.
method : `str`, optional
    "gwpy" or "csd_psd": Use GWpy CSD/PSD estimator.
    "fft": Use direct FFT ratio (other.fft() / self.fft()).
    "auto": Use "fft" if fftlength is None, else "gwpy".

Returns
-------
out : `FrequencySeries`
    Transfer function.


### `unit`

The physical unit of these data

:type: `~astropy.units.UnitBase`


### `unwrap_phase`

```python
unwrap_phase(self, deg: 'bool' = False, **kwargs: 'Any') -> "'TimeSeriesSignalMixin'"
```

Alias for instantaneous_phase(unwrap=True).

### `update`

```python
update(self, other, inplace=True)
```

Update this series by appending new data from an other
and dropping the same amount of data off the start.

This is a convenience method that just calls `~Series.append` with
`resize=False`.


### `value_at`

```python
value_at(self, x)
```

Return the value of this `Series` at the given `xindex` value

Parameters
----------
x : `float`, `~astropy.units.Quantity`
    the `xindex` value at which to search

Returns
-------
y : `~astropy.units.Quantity`
    the value of this Series at the given `xindex` value


### `whiten`

```python
whiten(self, fftlength=None, overlap=0, method='median', window='hann', detrend='constant', asd=None, fduration=2, highpass=None, **kwargs)
```

Whiten this `TimeSeries` using inverse spectrum truncation

Parameters
----------
fftlength : `float`, optional
    FFT integration length (in seconds) for ASD estimation,
    default: choose based on sample rate

overlap : `float`, optional
    number of seconds of overlap between FFTs, defaults to the
    recommended overlap for the given window (if given), or 0

method : `str`, optional
    FFT-averaging method (default: ``'median'``)

window : `str`, `numpy.ndarray`, optional
    window function to apply to timeseries prior to FFT,
    default: ``'hann'``
    see :func:`scipy.signal.get_window` for details on acceptable
    formats

detrend : `str`, optional
    type of detrending to do before FFT (see `~TimeSeries.detrend`
    for more details), default: ``'constant'``

asd : `~gwpy.frequencyseries.FrequencySeries`, optional
    the amplitude spectral density using which to whiten the data,
    overrides other ASD arguments, default: `None`

fduration : `float`, optional
    duration (in seconds) of the time-domain FIR whitening filter,
    must be no longer than `fftlength`, default: 2 seconds

highpass : `float`, optional
    highpass corner frequency (in Hz) of the FIR whitening filter,
    default: `None`

**kwargs
    other keyword arguments are passed to the `TimeSeries.asd`
    method to estimate the amplitude spectral density
    `FrequencySeries` of this `TimeSeries`

Returns
-------
out : `TimeSeries`
    a whitened version of the input data with zero mean and unit
    variance

See also
--------
TimeSeries.asd
    for details on the ASD calculation
TimeSeries.convolve
    for details on convolution with the overlap-save method
gwpy.signal.filter_design.fir_from_transfer
    for FIR filter design through spectrum truncation

Notes
-----
The accepted ``method`` arguments are:

- ``'bartlett'`` : a mean average of non-overlapping periodograms
- ``'median'`` : a median average of overlapping periodograms
- ``'welch'`` : a mean average of overlapping periodograms

The ``window`` argument is used in ASD estimation, FIR filter design,
and in preventing spectral leakage in the output.

Due to filter settle-in, a segment of length ``0.5*fduration`` will be
corrupted at the beginning and end of the output. See
`~TimeSeries.convolve` for more details.

The input is detrended and the output normalised such that, if the
input is stationary and Gaussian, then the output will have zero mean
and unit variance.

For more on inverse spectrum truncation, see arXiv:gr-qc/0509116.


### `write`

```python
write(self, target, *args, **kwargs)
```

Write this `TimeSeries` to a file

Parameters
----------
target : `str`
    path of output file

format : `str`, optional
    output format identifier. If not given, the format will be
    detected if possible. See below for list of acceptable
    formats.

Notes
-----
The available built-in formats are:

======== ==== ===== =============
 Format  Read Write Auto-identify
======== ==== ===== =============
    gse2  Yes   Yes            No
miniseed  Yes   Yes            No
     sac  Yes   Yes            No
======== ==== ===== =============

### `x0`

X-axis coordinate of the first data point

:type: `~astropy.units.Quantity` scalar


### `xcorr`

```python
xcorr(self, other: 'Any', *, maxlag: 'Optional[float]' = None, normalize: 'Optional[str]' = None, mode: 'str' = 'full', demean: 'bool' = True) -> "'TimeSeriesSignalMixin'"
```


Compute time-domain cross-correlation between two TimeSeries.


### `xindex`

Positions of the data on the x-axis

:type: `~astropy.units.Quantity` array


### `xspan`

X-axis [low, high) segment encompassed by these data

:type: `~gwpy.segments.Segment`


### `xunit`

Unit of x-axis index

:type: `~astropy.units.Unit`


### `zip`

```python
zip(self)
```

Zip the `xindex` and `value` arrays of this `Series`

Returns
-------
stacked : 2-d `numpy.ndarray`
    The array formed by stacking the the `xindex` and `value` of this
    series

Examples
--------
>>> a = Series([0, 2, 4, 6, 8], xindex=[-5, -4, -3, -2, -1])
>>> a.zip()
array([[-5.,  0.],
       [-4.,  2.],
       [-3.,  4.],
       [-2.,  6.],
       [-1.,  8.]])



### `zpk`

```python
zpk(self, zeros, poles, gain, analog=True, **kwargs)
```

Filter this `TimeSeries` by applying a zero-pole-gain filter

Parameters
----------
zeros : `array-like`
    list of zero frequencies (in Hertz)

poles : `array-like`
    list of pole frequencies (in Hertz)

gain : `float`
    DC gain of filter

analog : `bool`, optional
    type of ZPK being applied, if `analog=True` all parameters
    will be converted in the Z-domain for digital filtering

Returns
-------
timeseries : `TimeSeries`
    the filtered version of the input data

See also
--------
TimeSeries.filter
    for details on how a digital ZPK-format filter is applied

Examples
--------
To apply a zpk filter with file poles at 100 Hz, and five zeros at
1 Hz (giving an overall DC gain of 1e-10)::

>>> data2 = data.zpk([100]*5, [1]*5, 1e-10)



### `granger_causality`

```python
granger_causality(self, other, maxlag=10, test='ssr_ftest', verbose=False)
```

Check if 'other' Granger-causes 'self'.

Null Hypothesis: The past values of 'other' do NOT help in predicting 'self'.

Parameters
----------
other : `TimeSeries`
    The potential cause series.
maxlag : `int`, optional
    Maximum lag to check.
test : `str`, optional
    Statistical test to use ('ssr_ftest', 'ssr_chi2test', etc.).
verbose : `bool`, optional
    Whether to print verbose output.

Returns
-------
float
    The minimum p-value across all lags up to maxlag. 
    A small p-value (e.g., < 0.05) indicates Granger causality.


### `skewness`

```python
skewness(self, nan_policy='propagate')
```

Compute the skewness of the time series data.

Skewness is a measure of the asymmetry of the probability distribution 
of a real-valued random variable about its mean.

Parameters
----------
nan_policy : `str`, optional
    'propagate', 'raise', 'omit'.

Returns
-------
float
    The skewness.

# FrequencySeries

**Inherits from:** RegularityMixin, FittingMixin, StatisticalMethodsMixin, FrequencySeries

Light wrapper of gwpy's FrequencySeries for compatibility and future extension.

## Methods

### `DictClass`

```python
DictClass(*args: 'Any', **kwargs: 'Any')
```

Ordered mapping of `FrequencySeries` objects keyed by label.

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

### `angle`

```python
angle(self, unwrap: 'bool' = False) -> "'FrequencySeries'"
```

Alias for `phase(unwrap=unwrap)`.

### `append`

```python
append(self, other, inplace=True, pad=None, gap=None, resize=True)
```

Connect another series onto the end of the current one.

Parameters
----------
other : `Series`
    another series of the same type to connect to this one

inplace : `bool`, optional
    perform operation in-place, modifying current series,
    otherwise copy data and return new series, default: `True`

    .. warning::

       `inplace` append bypasses the reference check in
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

    If ``pad`` is given and is not `None`, the default is ``'pad'``,
    otherwise ``'raise'``. If ``gap='pad'`` is given, the default
    for ``pad`` is ``0``.

resize : `bool`, optional
    resize this array to accommodate new data, otherwise shift the
    old data to the left (potentially falling off the start) and
    put the new data in at the end, default: `True`.

Returns
-------
series : `Series`
    a new series containing joined data sets


### `channel`

Instrumental channel associated with these data

:type: `~gwpy.detector.Channel`


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

### `crop`

```python
crop(self, start=None, end=None, copy=False)
```

Crop this series to the given x-axis extent.

Parameters
----------
start : `float`, optional
    Lower limit of x-axis to crop to, defaults to
    :attr:`~Series.x0`.

end : `float`, optional
    Upper limit of x-axis to crop to, defaults to series end.

copy : `bool`, optional
    Copy the input data to fresh memory,
    otherwise return a view (default).

Returns
-------
series : `Series`
    A new series with a sub-set of the input data.

Notes
-----
If either ``start`` or ``end`` are outside of the original
`Series` span, warnings will be printed and the limits will
be restricted to the :attr:`~Series.xspan`.


### `degree`

```python
degree(self, unwrap: 'bool' = False) -> "'FrequencySeries'"
```


Calculate the phase of this FrequencySeries in degrees.

Parameters
----------
unwrap : `bool`, optional
    If `True`, unwrap the phase before converting to degrees.

Returns
-------
`FrequencySeries`
    The phase of the series, in degrees.


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


### `differentiate_time`

```python
differentiate_time(self) -> 'Any'
```


Apply time differentiation in frequency domain.

Multiplies by (2 * pi * i * f).
Converting Displacement -> Velocity -> Acceleration.

Returns
-------
`FrequencySeries`


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

### `dx`

X-axis sample separation

:type: `~astropy.units.Quantity` scalar


### `epoch`

GPS epoch associated with these data

:type: `~astropy.time.Time`


### `filter`

```python
filter(self, *filt, **kwargs)
```

Apply a filter to this `FrequencySeries`.

Parameters
----------
*filt : filter arguments
    1, 2, 3, or 4 arguments defining the filter to be applied,

        - an ``Nx1`` `~numpy.ndarray` of FIR coefficients
        - an ``Nx6`` `~numpy.ndarray` of SOS coefficients
        - ``(numerator, denominator)`` polynomials
        - ``(zeros, poles, gain)``
        - ``(A, B, C, D)`` 'state-space' representation

analog : `bool`, optional
    if `True`, filter definition will be converted from Hertz
    to Z-domain digital representation, default: `False`

inplace : `bool`, optional
    if `True`, this array will be overwritten with the filtered
    version, default: `False`

Returns
-------
result : `FrequencySeries`
    the filtered version of the input `FrequencySeries`,
    if ``inplace=True`` was given, this is just a reference to
    the modified input array

Raises
------
ValueError
    if ``filt`` arguments cannot be interpreted properly


### `filterba`

```python
filterba(self, *args, **kwargs)
```

Apply a [b, a] filter to this FrequencySeries.
Inherited from gwpy.


### `find_peaks`

```python
find_peaks(self, threshold: 'Optional[float]' = None, method: 'str' = 'amplitude', **kwargs: 'Any') -> 'Any'
```


Find peaks in the series.

Uses `SignalAnalysisMixin`.

Parameters
----------
threshold : `float`, optional
    Height threshold. If None, no threshold.
method : `str`, optional
    'amplitude', 'power', 'db', 'complex', 'abs'. Defines what metric to search on.
**kwargs
    Passed to `scipy.signal.find_peaks` (e.g. distance, prominence, width).

Returns
-------
peaks : `FrequencySeries`
    A new FrequencySeries containing only the peak values, indexed by their frequencies.
props : `dict`
    Dictionary of peak properties returned by `scipy.signal.find_peaks`.


### `fit`

```python
fit(self, model: 'Any', x_range: 'Optional[tuple[float, float]]' = None, sigma: 'Optional[Any]' = None, p0: 'Optional[dict[str, float]]' = None, limits: 'Optional[dict[str, tuple[float, float]]]' = None, fixed: 'Optional[Iterable[str]]' = None, **kwargs: 'Any') -> 'Any'
```

Fit the series data to a model.

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


### `from_control_frd`

```python
from_control_frd(frd: 'Any', *, frequency_unit: 'str' = 'Hz') -> 'Any'
```

Create from control.FRD.

### `from_cupy`

```python
from_cupy(array: 'Any', frequencies: 'Any', unit: 'Optional[Any]' = None) -> 'Any'
```

Create FrequencySeries from CuPy array.

### `from_hdf5_dataset`

```python
from_hdf5_dataset(group: 'Any', path: 'str') -> 'Any'
```

Read FrequencySeries from HDF5 dataset.

### `from_jax`

```python
from_jax(array: 'Any', frequencies: 'Any', unit: 'Optional[Any]' = None) -> 'Any'
```

Create FrequencySeries from JAX array.

### `from_lal`

```python
from_lal(lalfs, copy=True)
```

Generate a new `FrequencySeries` from a LAL `FrequencySeries`
of any type.


### `from_mne`

```python
from_mne(spectrum: 'Any', **kwargs: 'Any') -> 'Any'
```


Create FrequencySeries from MNE-Python Spectrum object.

Parameters
----------
spectrum : mne.time_frequency.Spectrum
    Input spectrum data.
**kwargs
    Additional arguments passed to constructor.

Returns
-------
FrequencySeries or FrequencySeriesDict


### `from_obspy`

```python
from_obspy(trace: 'Any', **kwargs: 'Any') -> 'Any'
```


Create FrequencySeries from Obspy Trace.

Parameters
----------
trace : obspy.Trace
    Input trace.
**kwargs
    Additional arguments.

Returns
-------
FrequencySeries


### `from_pandas`

```python
from_pandas(series: 'Any', **kwargs: 'Any') -> 'Any'
```

Create FrequencySeries from pandas.Series.

### `from_polars`

```python
from_polars(data: 'Any', frequencies: 'Optional[str]' = 'frequency', **kwargs: 'Any') -> 'Any'
```


Create a FrequencySeries from a polars.DataFrame or polars.Series.

Parameters
----------
data : polars.DataFrame or polars.Series
    Input data.
frequencies : str, optional
    If data is a DataFrame, name of the column to use as frequency.
**kwargs
    Additional arguments passed to frequency series constructor.

Returns
-------
FrequencySeries


### `from_pycbc`

```python
from_pycbc(fs, copy=True)
```

Convert a `pycbc.types.frequencyseries.FrequencySeries` into
a `FrequencySeries`

Parameters
----------
fs : `pycbc.types.frequencyseries.FrequencySeries`
    the input PyCBC `~pycbc.types.frequencyseries.FrequencySeries`
    array

copy : `bool`, optional, default: `True`
    if `True`, copy these data to a new array

Returns
-------
spectrum : `FrequencySeries`
    a GWpy version of the input frequency series


### `from_pyspeckit`

```python
from_pyspeckit(spectrum, **kwargs)
```


Create FrequencySeries from pyspeckit.Spectrum.

Parameters
----------
spectrum : pyspeckit.Spectrum
    Input spectrum.

Returns
-------
FrequencySeries


### `from_quantities`

```python
from_quantities(q: 'Any', frequencies: 'Any') -> 'Any'
```


Create FrequencySeries from quantities.Quantity.

Parameters
----------
q : quantities.Quantity
    Input data.
frequencies : array-like
    Frequencies corresponding to the data.

Returns
-------
FrequencySeries


### `from_root`

```python
from_root(obj: 'Any', return_error: 'bool' = False, **kwargs: 'Any') -> 'Any'
```


Create FrequencySeries from ROOT TGraph or TH1.


### `from_simpeg`

```python
from_simpeg(data_obj: 'Any', **kwargs: 'Any') -> 'Any'
```


Create FrequencySeries from SimPEG Data object.

Parameters
----------
data_obj : simpeg.data.Data
    Input SimPEG Data.

Returns
-------
FrequencySeries


### `from_specutils`

```python
from_specutils(spectrum, **kwargs)
```


Create FrequencySeries from specutils.Spectrum1D.

Parameters
----------
spectrum : specutils.Spectrum1D
    Input spectrum.

Returns
-------
FrequencySeries


### `from_tensorflow`

```python
from_tensorflow(tensor: 'Any', frequencies: 'Any', unit: 'Optional[Any]' = None) -> 'Any'
```

Create FrequencySeries from tensorflow.Tensor.

### `from_torch`

```python
from_torch(tensor: 'Any', frequencies: 'Any', unit: 'Optional[Any]' = None) -> 'Any'
```


Create FrequencySeries from torch.Tensor.

Parameters
----------
tensor : `torch.Tensor`
    Input tensor.
frequencies : `Array` or `Quantity`
    Frequency array matching the tensor size.
unit : `Unit` or `str`, optional
    Data unit.

Returns
-------
`FrequencySeries`


### `from_xarray`

```python
from_xarray(da: 'Any', **kwargs: 'Any') -> 'Any'
```

Create FrequencySeries from xarray.DataArray.

### `group_delay`

```python
group_delay(self) -> 'Any'
```


Calculate the group delay of the series.

Group delay is defined as -d(phase)/d(omega), where omega = 2 * pi * f.
It represents the time delay of the envelope of a signal at a given frequency.

Returns
-------
`FrequencySeries`
    A new FrequencySeries representing the group delay in seconds.


### `idct`

```python
idct(self, type: 'int' = 2, norm: 'str' = 'ortho', *, n: 'Optional[int]' = None) -> 'Any'
```

_No documentation available._

### `ifft`

```python
ifft(self, *, mode: 'str' = 'auto', trim: 'bool' = True, original_n: 'Optional[int]' = None, pad_left: 'Optional[int]' = None, pad_right: 'Optional[int]' = None, **kwargs: 'Any') -> 'Any'
```


Inverse FFT returning a gwexpy TimeSeries, supporting transient round-trip.

Parameters
----------
mode : {"auto", "gwpy", "transient"}
    auto: use transient restoration if `_gwex_fft_mode=="transient"` is detected, otherwise GWpy compatible.
trim : bool
    Whether to remove padding and trim to original length during transient mode.
original_n : int, optional
    Explicitly specify the length after restoration (takes priority).
pad_right, pad_left : int, optional
    Specify padding lengths for transient mode to override defaults.


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


### `integrate_time`

```python
integrate_time(self) -> 'Any'
```


Apply time integration in frequency domain.

Divides by (2 * pi * i * f).
Converting Acceleration -> Velocity -> Displacement.

Returns
-------
`FrequencySeries`


### `interpolate`

```python
interpolate(self, df)
```

Interpolate this `FrequencySeries` to a new resolution.

Parameters
----------
df : `float`
    desired frequency resolution of the interpolated `FrequencySeries`,
    in Hz

Returns
-------
out : `FrequencySeries`
    the interpolated version of the input `FrequencySeries`

See also
--------
numpy.interp
    for the underlying 1-D linear interpolation scheme


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

Return True if this series has a regular grid (constant spacing).

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

*(Inherited from `FrequencySeries`)*

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



*(Inherited from `FrequencySeries`)*

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

*(Inherited from `FrequencySeries`)*

### `name`

Name for this data set

:type: `str`


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


### `phase`

```python
phase(self, unwrap: 'bool' = False) -> "'FrequencySeries'"
```


Calculate the phase of this FrequencySeries.

Parameters
----------
unwrap : `bool`, optional
    If `True`, unwrap the phase to remove discontinuities.
    Default is `False`.

Returns
-------
`FrequencySeries`
    The phase of the series, in radians.


### `plot`

```python
plot(self, xscale='log', **kwargs)
```

Plot the data for this series

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


*(Inherited from `Series`)*

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


### `quadrature_sum`

```python
quadrature_sum(self, other: 'Any') -> 'Any'
```


Compute sqrt(self^2 + other^2) assuming checking independence.

Operates on magnitude. Phase information is lost (returns real).

Parameters
----------
other : `FrequencySeries`
    The other series to add.

Returns
-------
`FrequencySeries`
    Magnitude combined series.


### `read`

```python
read(source, *args, **kwargs)
```

Read data into a `FrequencySeries`

Arguments and keywords depend on the output format, see the
online documentation for full details for each format, the
parameters below are common to most formats.

Parameters
----------
source : `str`, `list`
    Source of data, any of the following:

    - `str` path of single data file,
    - `str` path of LAL-format cache file,
    - `list` of paths.

*args
    Other arguments are (in general) specific to the given
    ``format``.

format : `str`, optional
    Source format identifier. If not given, the format will be
    detected if possible. See below for list of acceptable
    formats.

**kwargs
    Other keywords are (in general) specific to the given ``format``.

Raises
------
IndexError
    if ``source`` is an empty list

Notes
-----
The available built-in formats are:

====== ==== ===== =============
Format Read Write Auto-identify
====== ==== ===== =============
   csv  Yes   Yes           Yes
  hdf5  Yes   Yes           Yes
ligolw  Yes    No            No
   txt  Yes   Yes           Yes
====== ==== ===== =============

### `rms`

```python
rms(self, axis=None, keepdims=False, ignore_nan=True)
```

_No documentation available._

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


### `smooth`

```python
smooth(self, width: 'Any', method: 'str' = 'amplitude', ignore_nan: 'bool' = True) -> 'Any'
```


Smooth the frequency series.

Parameters
----------
width : `int`
    Number of samples for the smoothing winow (e.g. rolling mean size).
method : `str`, optional
    Smoothing target:
    - 'amplitude': Smooth absolute value |X|, keep phase 'original' (or 0?).
                   Actually, strictly smoothing magnitude destroys phase coherence.
                   Returns REAL series (magnitude only).
    - 'power': Smooth power |X|^2. Returns REAL series.
    - 'complex': Smooth real and imaginary parts separately. Preserves complex.
    - 'db': Smooth dB values. Returns REAL series (in dB).
ignore_nan : `bool`, optional
    If True, ignore NaNs during smoothing. Default is True.

Returns
-------
`FrequencySeries`


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
step(self, **kwargs)
```

Create a step plot of this series
        

### `to_control_frd`

```python
to_control_frd(self, frequency_unit: 'str' = 'Hz') -> 'Any'
```

Convert to control.FRD.

### `to_cupy`

```python
to_cupy(self, dtype: 'Any' = None) -> 'Any'
```


Convert to CuPy array.

Returns
-------
`cupy.ndarray`


### `to_db`

```python
to_db(self, ref: 'Any' = 1.0, amplitude: 'bool' = True) -> "'FrequencySeries'"
```


Convert this series to decibels.

Parameters
----------
ref : `float` or `Quantity`, optional
    Reference value for 0 dB. Default is 1.0.
amplitude : `bool`, optional
    If `True` (default), treat data as amplitude (20 * log10).
    If `False`, treat data as power (10 * log10).

Returns
-------
`FrequencySeries`
    The series in dB.


### `to_hdf5_dataset`

```python
to_hdf5_dataset(self, group: 'Any', path: 'str', *, overwrite: 'bool' = False, compression: 'Optional[str]' = None, compression_opts: 'Any' = None) -> 'Any'
```

Write to HDF5 dataset within a group.

### `to_jax`

```python
to_jax(self, dtype: 'Any' = None) -> 'Any'
```


Convert to JAX array.

Returns
-------
`jax.Array`


### `to_lal`

```python
to_lal(self)
```

Convert this `FrequencySeries` into a LAL FrequencySeries.

Returns
-------
lalspec : `FrequencySeries`
    an XLAL-format FrequencySeries of a given type, e.g.
    `REAL8FrequencySeries`

Notes
-----
Currently, this function is unable to handle unit string
conversion.


### `to_mne`

```python
to_mne(self, info: 'Optional[Any]' = None) -> 'Any'
```


Convert to MNE-Python object.

Parameters
----------
info : mne.Info, optional
    MNE Info object.

Returns
-------
mne.time_frequency.SpectrumArray


### `to_obspy`

```python
to_obspy(self, **kwargs: 'Any') -> 'Any'
```


Convert to Obspy Trace.

Returns
-------
obspy.Trace


### `to_pandas`

```python
to_pandas(self, index: 'str' = 'frequency', *, name: 'Optional[str]' = None, copy: 'bool' = False) -> 'Any'
```

Convert to pandas.Series.

### `to_polars`

```python
to_polars(self, name: 'Optional[str]' = None, as_dataframe: 'bool' = True, frequencies: 'str' = 'frequency') -> 'Any'
```


Convert this series to a polars.DataFrame or polars.Series.

Parameters
----------
name : str, optional
    Name for the polars Series/Column.
as_dataframe : bool, default True
    If True, returns a DataFrame with a 'frequency' column.
    If False, returns a raw Series of values.
frequencies : str, default "frequency"
    Name of the frequency column (only if as_dataframe=True).

Returns
-------
polars.DataFrame or polars.Series


### `to_pycbc`

```python
to_pycbc(self, copy=True)
```

Convert this `FrequencySeries` into a
`~pycbc.types.frequencyseries.FrequencySeries`

Parameters
----------
copy : `bool`, optional, default: `True`
    if `True`, copy these data to a new array

Returns
-------
frequencyseries : `pycbc.types.frequencyseries.FrequencySeries`
    a PyCBC representation of this `FrequencySeries`


### `to_pyspeckit`

```python
to_pyspeckit(self, **kwargs)
```


Convert to pyspeckit.Spectrum.

Parameters
----------
**kwargs
    Arguments passed to pyspeckit.Spectrum constructor.

Returns
-------
pyspeckit.Spectrum


### `to_quantities`

```python
to_quantities(self, units: 'Optional[str]' = None) -> 'Any'
```


Convert to quantities.Quantity (Elephant/Neo compatible).

Parameters
----------
units : str or quantities.UnitQuantity, optional
    Target units.

Returns
-------
quantities.Quantity


### `to_simpeg`

```python
to_simpeg(self, location=None, rx_type='PointElectricField', orientation='x', **kwargs) -> 'Any'
```


Convert to SimPEG Data object.

Parameters
----------
location : array_like, optional
    Rx location (x, y, z). Default is [0, 0, 0].
rx_type : str, optional
    Receiver class name. Default "PointElectricField".
orientation : str, optional
    Receiver orientation ('x', 'y', 'z'). Default 'x'.

Returns
-------
simpeg.data.Data


### `to_specutils`

```python
to_specutils(self, **kwargs)
```


Convert to specutils.Spectrum1D.

Parameters
----------
**kwargs
    Arguments passed to Spectrum1D constructor.

Returns
-------
specutils.Spectrum1D


### `to_tensorflow`

```python
to_tensorflow(self, dtype: 'Any' = None) -> 'Any'
```


Convert to tensorflow.Tensor.

Returns
-------
`tensorflow.Tensor`


### `to_tgraph`

```python
to_tgraph(self, error: 'Optional[Any]' = None) -> 'Any'
```


Convert to ROOT TGraph or TGraphErrors.


### `to_th1d`

```python
to_th1d(self, error: 'Optional[Any]' = None) -> 'Any'
```


Convert to ROOT TH1D.


### `to_torch`

```python
to_torch(self, device: 'Optional[str]' = None, dtype: 'Any' = None, requires_grad: 'bool' = False, copy: 'bool' = False) -> 'Any'
```


Convert to torch.Tensor.

Parameters
----------
device : `str` or `torch.device`, optional
    Target device (e.g. 'cpu', 'cuda').
dtype : `torch.dtype`, optional
    Target data type. Defaults to preserving complex64/128 or float32/64.
requires_grad : `bool`, optional
    If `True`, enable gradient tracking.
copy : `bool`, optional
    If `True`, force a copy of the data.

Returns
-------
`torch.Tensor`


### `to_xarray`

```python
to_xarray(self, freq_coord: 'str' = 'Hz') -> 'Any'
```

Convert to xarray.DataArray.

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

### `unit`

The physical unit of these data

:type: `~astropy.units.UnitBase`


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
write(self, target, *args, **kwargs)
```

Write this `FrequencySeries` to a file

Arguments and keywords depend on the output format, see the
online documentation for full details for each format, the
parameters below are common to most formats.

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

====== ==== ===== =============
Format Read Write Auto-identify
====== ==== ===== =============
   csv  Yes   Yes           Yes
  hdf5  Yes   Yes            No
   txt  Yes   Yes           Yes
====== ==== ===== =============

### `x0`

X-axis coordinate of the first data point

:type: `~astropy.units.Quantity` scalar


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
zpk(self, zeros, poles, gain, analog=True)
```

Filter this `FrequencySeries` by applying a zero-pole-gain filter

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
spectrum : `FrequencySeries`
    the frequency-domain filtered version of the input data

See also
--------
FrequencySeries.filter
    for details on how a digital ZPK-format filter is applied

Examples
--------
To apply a zpk filter with file poles at 100 Hz, and five zeros at
1 Hz (giving an overall DC gain of 1e-10)::

    >>> data2 = data.zpk([100]*5, [1]*5, 1e-10)


## Numerical Semantics

The following behaviors are strictly enforced by tests to ensure physical correctness:

### Calculus Operations
* **DC Handling**: When integrating (e.g., `integrate_time()`), the DC component ($f=0$) is explicitly set to **0** to avoid singularities ($1/0$), rather than returning `inf` or `NaN`.
* **Unit Propagation**:
    * `differentiate_time()`: Multiplies unit by $Hz$.
    * `integrate_time()`: Multiplies unit by $s$.
    * `group_delay()`: Always returns unit in $s$.

### Transformations (IFFT)
* **Amplitude Preservation**: `ifft()` is designed to invert GWpy's `fft()` (one-sided output). It automatically applies a **0.5 scaling factor** to non-DC/Nyquist components to strictly preserve the amplitude of the original time series (Parseval's theorem).


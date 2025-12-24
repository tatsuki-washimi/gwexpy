# Array2D

**Inherits from:** AxisApiMixin, Array2D


2D Array with unified axis API.


## Methods

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

*(Inherited from `Series`)*

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


### `axes`

Tuple of AxisDescriptor objects for each dimension.

Returns
-------
tuple of AxisDescriptor
    Each descriptor contains the axis name and index values.


*(Inherited from `AxisApiMixin`)*

### `axis`

```python
axis(self, key: Union[int, str]) -> gwexpy.types.axis.AxisDescriptor
```

Get an axis descriptor by index or name.

Parameters
----------
key : int or str
    Axis index (0-based) or name.

Returns
-------
AxisDescriptor
    The requested axis descriptor.

Raises
------
KeyError
    If axis name not found.
TypeError
    If key is not int or str.


### `axis_names`

Names of all axes as a tuple of strings.

Returns
-------
tuple of str
    The name of each axis in order.


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


### `dy`

Y-axis sample separation

:type: `~astropy.units.Quantity` scalar


### `epoch`

GPS epoch associated with these data

:type: `~astropy.time.Time`


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


### `imshow`

```python
imshow(self, **kwargs)
```

Plot this array using matplotlib.axes.Axes.imshow.
Inherited from gwpy.


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


### `isel`

```python
isel(self, indexers=None, **kwargs)
```

Select by integer indices along specified axes.

Parameters
----------
indexers : dict, optional
    Mapping of axis name/index to integer index or slice.
**kwargs
    Additional indexers as keyword arguments.

Returns
-------
subset
    Sliced array.


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


### `pcolormesh`

```python
pcolormesh(self, **kwargs)
```

Plot this array using matplotlib.axes.Axes.pcolormesh.
Inherited from gwpy.


### `plot`

```python
plot(self, method='imshow', **kwargs)
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


### `read`

```python
read(source, *args, **kwargs)
```

Read data into a `Series`

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

Returns
-------
data : `Series`

Raises
------
IndexError
    if ``source`` is an empty list

Notes
-----

### `rename_axes`

```python
rename_axes(self, mapping: Dict[str, str], *, inplace=False)
```

Rename axes using a mapping of old names to new names.

Parameters
----------
mapping : dict
    Mapping from old axis names to new names.
inplace : bool, optional
    If True, modify in place. Otherwise return a copy.

Returns
-------
self or copy


### `sel`

```python
sel(self, indexers=None, *, method='nearest', **kwargs)
```

Select by coordinate values along specified axes.

Parameters
----------
indexers : dict, optional
    Mapping of axis name to coordinate value or slice.
method : str, optional
    Selection method: 'nearest' (default).
**kwargs
    Additional indexers as keyword arguments.

Returns
-------
subset
    Sliced array at nearest coordinate values.


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


### `step`

```python
step(self, **kwargs)
```

Create a step plot of this series
        

### `swapaxes`

```python
swapaxes(self, axis1, axis2)
```

a.swapaxes(axis1, axis2)

Return a view of the array with `axis1` and `axis2` interchanged.

Refer to `numpy.swapaxes` for full documentation.

See Also
--------
numpy.swapaxes : equivalent function

*(Inherited from `Array2D`)*

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

### `transpose`

```python
transpose(self, *axes)
```

Permute the dimensions of an array.

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
value_at(self, x, y)
```

Return the value of this `Series` at the given `(x, y)` coordinates

Parameters
----------
x : `float`, `~astropy.units.Quantity`
    the `xindex` value at which to search
x : `float`, `~astropy.units.Quantity`
    the `yindex` value at which to search

Returns
-------
z : `~astropy.units.Quantity`
    the value of this Series at the given coordinates


### `write`

```python
write(self, target, *args, **kwargs)
```

Write this `Series` to a file

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


### `y0`

Y-axis coordinate of the first data point

:type: `~astropy.units.Quantity` scalar


### `yindex`

Positions of the data on the y-axis

:type: `~astropy.units.Quantity` array


### `yspan`

Y-axis [low, high) segment encompassed by these data

:type: `~gwpy.segments.Segment`


### `yunit`

Unit of Y-axis index

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




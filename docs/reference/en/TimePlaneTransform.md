# TimePlaneTransform

**Inherits from:** object


A container for 3D data with a preferred (time, axis1, axis2) structure,
commonly produced by time-frequency or similar transforms.

This class wraps an Array3D to enforce semantic structure:
- Axis 0 is "time"
- Axis 1 and 2 are symmetric spatial/frequency dimensions.


## Methods

### `__init__`

```python
__init__(self, data3d, *, kind='custom', meta=None)
```


Initialize a TimePlaneTransform.

Parameters
----------
data3d : Array3D or tuple
    The underlying 3D data.
    Preferred: an `Array3D` instance.
    Supported tuple format: (value, time_axis, axis1, axis2, unit/metadata).
kind : str, optional
    A string describing the type of transform (e.g., "stlt", "bispectrum"). Default is "custom".
meta : dict, optional
    Additional metadata dictionary. Default is None (stored as empty dict).


### `at_sigma`

```python
at_sigma(self, sigma)
```


Extract a 2D plane (Spectrogram-like) at a specific sigma index (if axis1 is sigma)
or value.

This assumes axis 1 is sigma.


### `at_time`

```python
at_time(self, t, *, method='nearest')
```


Extract a Plane2D at the specific time `t`.

Parameters
----------
t : Quantity or float
    Time value. If float, assumed to be in the unit of the time axis.
method : str, optional
    "nearest" (default). Future versions may support interpolation.

Returns
-------
Plane2D


### `axes`


Return the 3 AxisDescriptors: (time, axis1, axis2).


### `axis1`

AxisDescriptor for the first symmetric axis (axis 1).

### `axis2`

AxisDescriptor for the second symmetric axis (axis 2).

### `kind`

String describing the type of transform (e.g., 'stlt', 'bispectrum').

### `meta`

Additional metadata dictionary.

### `ndim`

Number of dimensions (always 3).

### `plane`

```python
plane(self, drop_axis, drop_index, *, axis1=None, axis2=None)
```


Extract a 2D plane by slicing at a specific index along one axis.

Parameters
----------
drop_axis : int or str
    The axis to slice along (remove).
drop_index : int
    Integer index to select along `drop_axis`.
axis1 : str or int, optional
    The generic axis 1 for the resulting Plane2D.
axis2 : str or int, optional
    The generic axis 2 for the resulting Plane2D.

Returns
-------
Plane2D


### `shape`

Shape of the 3D data array (time, axis1, axis2).

### `times`

Coordinate array of the time axis (axis 0).

### `to_array3d`

```python
to_array3d(self)
```

Return the underlying Array3D object (advanced usage).

### `unit`

Physical unit of the data values.

### `value`

The underlying data values as numpy array.


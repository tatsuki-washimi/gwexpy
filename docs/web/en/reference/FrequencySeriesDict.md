# FrequencySeriesDict

**Inherits from:** FrequencySeriesBaseDict

Ordered mapping of `FrequencySeries` objects keyed by label.

## Methods

### `__init__`

```python
__init__(self, *args: 'Any', **kwargs: 'Any')
```

Initialize self.  See help(type(self)) for accurate signature.

*(Inherited from `OrderedDict`)*

### `EntryClass`

```python
EntryClass(data, unit=None, f0=None, df=None, frequencies=None, name=None, epoch=None, channel=None, **kwargs)
```

Light wrapper of gwpy's FrequencySeries for compatibility and future extension.

### `angle`

```python
angle(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```

Alias for phase(). Returns a new FrequencySeriesDict.

### `append`

```python
append(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```


Append to each FrequencySeries in the dict (in-place).
Returns self.


### `apply_response`

```python
apply_response(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```


Apply response to each FrequencySeries.
Returns a new FrequencySeriesDict.


### `copy`

```python
copy(self) -> "'FrequencySeriesBaseDict[_FS]'"
```

od.copy() -> a shallow copy of od

*(Inherited from `OrderedDict`)*

### `crop`

```python
crop(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```


Crop each FrequencySeries in the dict.
In-place operation (GWpy-compatible). Returns self.


### `degree`

```python
degree(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```


Compute phase (in degrees) of each FrequencySeries.
Returns a new FrequencySeriesDict.


### `differentiate_time`

```python
differentiate_time(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```


Apply time differentiation to each item.
Returns a new FrequencySeriesDict.


### `filter`

```python
filter(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```


Apply filter to each FrequencySeries.
Returns a new FrequencySeriesDict.


### `group_delay`

```python
group_delay(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```


Compute group delay of each item.
Returns a new FrequencySeriesDict.


### `ifft`

```python
ifft(self, *args, **kwargs)
```


Compute IFFT of each FrequencySeries.
Returns a TimeSeriesDict.


### `integrate_time`

```python
integrate_time(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```


Apply time integration to each item.
Returns a new FrequencySeriesDict.


### `interpolate`

```python
interpolate(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```


Interpolate each FrequencySeries in the dict.
Returns a new FrequencySeriesDict.


### `pad`

```python
pad(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```


Pad each FrequencySeries in the dict.
Returns a new FrequencySeriesDict.


### `phase`

```python
phase(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```


Compute phase of each FrequencySeries.
Returns a new FrequencySeriesDict.


### `plot`

```python
plot(self, label: 'str' = 'key', method: 'str' = 'plot', figsize: 'Optional[Any]' = None, **kwargs: 'Any')
```


Plot data.

Parameters
----------
label : str, optional
    labelling method, one of

    - ``'key'``: use dictionary key (default)
    - ``'name'``: use ``name`` attribute of each item
method : str, optional
    method of :class:`~gwpy.plot.Plot` to call, default: ``'plot'``
figsize : tuple, optional
    (width, height) tuple in inches
**kwargs
    other keyword arguments passed to the plot method


### `plot_all`

```python
plot_all(self, *args: 'Any', **kwargs: 'Any')
```

Alias for plot(). Plots all series in the dict.

### `prepend`

```python
prepend(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```


Prepend to each FrequencySeries in the dict (in-place).
Returns self.


### `read`

```python
read(source, *args, **kwargs)
```

_No documentation available._

### `setdefault`

```python
setdefault(self, key: 'str', default: 'Optional[_FS]' = None) -> '_FS'
```

Insert key with a value of default if key is not in the dictionary.

Return the value for key if key is in the dictionary, else default.

*(Inherited from `OrderedDict`)*

### `smooth`

```python
smooth(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```


Smooth each FrequencySeries.
Returns a new FrequencySeriesDict.


### `span`

Frequency extent across all elements (based on xspan).

### `to_control_frd`

```python
to_control_frd(self, *args, **kwargs) -> 'dict'
```


Convert each item to control.FRD.
Returns a dict of FRD objects.


### `to_cupy`

```python
to_cupy(self, *args, **kwargs) -> 'dict'
```


Convert each item to cupy.ndarray.
Returns a dict of Arrays.


### `to_db`

```python
to_db(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```


Convert each FrequencySeries to dB.
Returns a new FrequencySeriesDict.


### `to_jax`

```python
to_jax(self, *args, **kwargs) -> 'dict'
```


Convert each item to jax.Array.
Returns a dict of Arrays.


### `to_matrix`

```python
to_matrix(self)
```

Convert this FrequencySeriesDict to a FrequencySeriesMatrix (Nx1).

### `to_pandas`

```python
to_pandas(self, **kwargs)
```


Convert to pandas.DataFrame.
Keys become columns.


### `to_tensorflow`

```python
to_tensorflow(self, *args, **kwargs) -> 'dict'
```


Convert each item to tensorflow.Tensor.
Returns a dict of Tensors.


### `to_tmultigraph`

```python
to_tmultigraph(self, name: 'Optional[str]' = None) -> 'Any'
```

Convert to ROOT TMultiGraph.

### `to_torch`

```python
to_torch(self, *args, **kwargs) -> 'dict'
```


Convert each item to torch.Tensor.
Returns a dict of Tensors.


### `to_xarray`

```python
to_xarray(self)
```


Convert to xarray.Dataset.
Keys become data variables.


### `write`

```python
write(self, target: 'str', *args: 'Any', **kwargs: 'Any') -> 'Any'
```

Write dict to file (HDF5, ROOT, etc.).

For HDF5 output you can choose a layout (default is GWpy-compatible dataset-per-entry).

```python
fsd.write("out.h5", format="hdf5")               # GWpy-compatible (default)
fsd.write("out.h5", format="hdf5", layout="group")  # legacy group-per-entry
```

HDF5 dataset names (for GWpy `path=`):
- Keys are sanitized to be HDF5-friendly (e.g. `H1:ASD` -> `H1_ASD`).
- If multiple keys sanitize to the same name, a suffix like `__1` is added.
- The original keys are stored in file attributes, and `gwexpy` restores them on read.

.. warning::
   Never unpickle data from untrusted sources. ``pickle``/``shelve`` can execute
   arbitrary code on load.

Pickle portability note: pickled gwexpy `FrequencySeriesDict` unpickles as a built-in
`dict` of GWpy `FrequencySeries` (gwexpy not required on the loading side).

### `zpk`

```python
zpk(self, *args, **kwargs) -> "'FrequencySeriesDict'"
```


Apply ZPK filter to each FrequencySeries.
Returns a new FrequencySeriesDict.

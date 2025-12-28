# FrequencySeriesList

**Inherits from:** FrequencySeriesBaseList

List of `FrequencySeries` objects.

## Methods

### `__init__`

```python
__init__(self, *items: '_FS')
```

Initialize self.  See help(type(self)) for accurate signature.

*(Inherited from `list`)*

### `EntryClass`

```python
EntryClass(data, unit=None, f0=None, df=None, frequencies=None, name=None, epoch=None, channel=None, **kwargs)
```

Light wrapper of gwpy's FrequencySeries for compatibility and future extension.

### `angle`

```python
angle(self, *args, **kwargs) -> "'FrequencySeriesList'"
```

Alias for phase(). Returns a new FrequencySeriesList.

### `append`

```python
append(self, item: '_FS')
```

Append object to the end of the list.

*(Inherited from `list`)*

### `apply_response`

```python
apply_response(self, *args, **kwargs) -> "'FrequencySeriesList'"
```


Apply response to each FrequencySeries in the list.
Returns a new FrequencySeriesList.


### `copy`

```python
copy(self) -> "'FrequencySeriesBaseList[_FS]'"
```

Return a shallow copy of the list.

*(Inherited from `list`)*

### `crop`

```python
crop(self, *args, **kwargs) -> "'FrequencySeriesList'"
```


Crop each FrequencySeries in the list.
Returns a new FrequencySeriesList.


### `degree`

```python
degree(self, *args, **kwargs) -> "'FrequencySeriesList'"
```


Compute phase (in degrees) of each FrequencySeries.
Returns a new FrequencySeriesList.


### `differentiate_time`

```python
differentiate_time(self, *args, **kwargs) -> "'FrequencySeriesList'"
```


Apply time differentiation to each item.
Returns a new FrequencySeriesList.


### `extend`

```python
extend(self, items: 'Iterable[_FS]') -> 'None'
```

Extend list by appending elements from the iterable.

*(Inherited from `list`)*

### `filter`

```python
filter(self, *args, **kwargs) -> "'FrequencySeriesList'"
```


Apply filter to each FrequencySeries in the list.
Returns a new FrequencySeriesList.


### `group_delay`

```python
group_delay(self, *args, **kwargs) -> "'FrequencySeriesList'"
```


Compute group delay of each item.
Returns a new FrequencySeriesList.


### `ifft`

```python
ifft(self, *args, **kwargs)
```


Compute IFFT of each FrequencySeries.
Returns a TimeSeriesList.


### `insert`

```python
insert(self, index: 'int', item: '_FS') -> 'None'
```

Insert object before index.

*(Inherited from `list`)*

### `integrate_time`

```python
integrate_time(self, *args, **kwargs) -> "'FrequencySeriesList'"
```


Apply time integration to each item.
Returns a new FrequencySeriesList.


### `interpolate`

```python
interpolate(self, *args, **kwargs) -> "'FrequencySeriesList'"
```


Interpolate each FrequencySeries in the list.
Returns a new FrequencySeriesList.


### `pad`

```python
pad(self, *args, **kwargs) -> "'FrequencySeriesList'"
```


Pad each FrequencySeries in the list.
Returns a new FrequencySeriesList.


### `phase`

```python
phase(self, *args, **kwargs) -> "'FrequencySeriesList'"
```


Compute phase of each FrequencySeries.
Returns a new FrequencySeriesList.


### `plot`

```python
plot(self, **kwargs: 'Any')
```

Plot all series. Delegates to gwpy.plot.Plot.

### `plot_all`

```python
plot_all(self, *args: 'Any', **kwargs: 'Any')
```

Alias for plot(). Plots all series.

### `segments`

Frequency spans of each element (xspan).

### `smooth`

```python
smooth(self, *args, **kwargs) -> "'FrequencySeriesList'"
```


Smooth each FrequencySeries.
Returns a new FrequencySeriesList.


### `to_control_frd`

```python
to_control_frd(self, *args, **kwargs) -> 'list'
```


Convert each item to control.FRD.
Returns a list of FRD objects.


### `to_cupy`

```python
to_cupy(self, *args, **kwargs) -> 'list'
```


Convert each item to cupy.ndarray.
Returns a list of Arrays.


### `to_db`

```python
to_db(self, *args, **kwargs) -> "'FrequencySeriesList'"
```


Convert each FrequencySeries to dB.
Returns a new FrequencySeriesList.


### `to_jax`

```python
to_jax(self, *args, **kwargs) -> 'list'
```


Convert each item to jax.Array.
Returns a list of Arrays.


### `to_pandas`

```python
to_pandas(self, **kwargs)
```


Convert to pandas.DataFrame.
Columns are named by channel name or index.


### `to_tensorflow`

```python
to_tensorflow(self, *args, **kwargs) -> 'list'
```


Convert each item to tensorflow.Tensor.
Returns a list of Tensors.


### `to_tmultigraph`

```python
to_tmultigraph(self, name: 'Optional[str]' = None) -> 'Any'
```

Convert to ROOT TMultiGraph.

### `to_torch`

```python
to_torch(self, *args, **kwargs) -> 'list'
```


Convert each item to torch.Tensor.
Returns a list of Tensors.


### `to_xarray`

```python
to_xarray(self)
```


Convert to xarray.DataArray.
Concatenates along a new dimension 'channel'.


### `write`

```python
write(self, target: 'str', *args: 'Any', **kwargs: 'Any') -> 'Any'
```

Write list to file (HDF5, ROOT, etc.).

### `zpk`

```python
zpk(self, *args, **kwargs) -> "'FrequencySeriesList'"
```


Apply ZPK filter to each FrequencySeries in the list.
Returns a new FrequencySeriesList.



# Array4D

**Inherits from:** Array, AxisApiMixin, StatisticalMethodsMixin, GwpyArray

4D Array with explicit axis management.

Array4D extends the base `Array` class to provide explicit management of 4 axes, each with a name and a coordinate index (1D Quantity).

## Methods

### `isel`

```python
isel(self, indexers=None, **kwargs)
```

Select by integer indices along specified axes.

Inherited from `AxisApiMixin`. For Array4D, specifying an integer index for a dimension will reduce the dimensionality (returning a lower-dimensional array or Quantity), unlike `ScalarField` which always maintains 4 dimensions.

### `sel`

```python
sel(self, indexers=None, method='nearest', **kwargs)
```

Select by coordinate values along specified axes.

### `transpose`

```python
transpose(self, *axes)
```

Permute the dimensions of the array and update axis metadata.

## Properties

| Property | Description |
|----------|-------------|
| `axes` | Tuple of AxisDescriptor objects for each dimension |
| `axis_names` | Names of all axes |
| `unit` | Physical unit of these data |

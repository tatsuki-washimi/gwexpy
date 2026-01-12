# StandardizeTransform

**Inherits from:** Transform


Standardize TimeSeries/Matrix objects with optional robust scaling.


## Methods

### `__init__`

```python
__init__(self, method: str = 'zscore', ddof: int = 0, robust: bool = False, axis: str = 'time', *, multivariate: bool = False, align: str = 'intersection')
```

Initialize self.  See help(type(self)) for accurate signature.

*(Inherited from `Transform`)*

### `fit`

```python
fit(self, x)
```

Fit the transform to the data. Returns self.

*(Inherited from `Transform`)*

### `fit_transform`

```python
fit_transform(self, x)
```

Fit and transform in one step.

### `inverse_transform`

```python
inverse_transform(self, y)
```

Reverse the transform. Not all transforms support this.

*(Inherited from `Transform`)*

### `transform`

```python
transform(self, x)
```

Apply the transform to data. Must be implemented by subclasses.

*(Inherited from `Transform`)*


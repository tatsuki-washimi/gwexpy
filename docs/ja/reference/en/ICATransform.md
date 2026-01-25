# ICATransform

**Inherits from:** Transform


ICA wrapper using existing decomposition helpers.


## Methods

### `__init__`

```python
__init__(self, n_components: Optional[int] = None, *, multivariate: bool = True, align: str = 'intersection', **kwargs)
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


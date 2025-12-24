# Transform

**Inherits from:** object


Minimal transform interface for TimeSeries-like objects.


## Methods

### `fit`

```python
fit(self, x)
```

Fit the transform to the data. Returns self.

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

### `transform`

```python
transform(self, x)
```

Apply the transform to data. Must be implemented by subclasses.


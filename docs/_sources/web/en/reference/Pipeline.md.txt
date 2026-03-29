# Pipeline

**Inherits from:** object


Sequentially apply a list of transforms.


## Methods

### `__init__`

```python
__init__(self, steps: Sequence[Tuple[str, gwexpy.timeseries.pipeline.Transform]])
```

Initialize pipeline with named transform steps.

Parameters
----------
steps : list of (name, Transform) tuples
    Sequence of transforms to apply.


### `fit`

```python
fit(self, x)
```

Fit all transforms in sequence.

### `fit_transform`

```python
fit_transform(self, x)
```

Fit and transform in one step.

### `inverse_transform`

```python
inverse_transform(self, y, *, strict: bool = True)
```

Apply inverse transforms in reverse order.

Parameters
----------
y : data
    Transformed data.
strict : bool, optional
    If True, raise error if any step doesn't support inverse.


### `transform`

```python
transform(self, x)
```

Apply all transforms in sequence.


# Pipeline

<!-- reference-summary:start -->

## What it is

Use `Pipeline` to chain reusable preprocessing transforms and apply them deterministically to time-series-like data.

## Representative Signatures

```python
Pipeline(steps=[("impute", ImputeTransform()), ("standardize", StandardizeTransform())])
Pipeline.fit_transform(x)
```

## Minimal Example

```python
from gwexpy.timeseries import Pipeline, ImputeTransform, StandardizeTransform

pipeline = Pipeline([("impute", ImputeTransform()), ("standardize", StandardizeTransform())])
out = pipeline.fit_transform(ts_matrix)
```

## Related Theory

- [Validated Algorithms](../user_guide/validated_algorithms.md)
- [FFT_Conventions](FFT_Conventions.md)

## Related Tutorials

- [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_en.md)
- [Tutorial Index](../user_guide/tutorials/index.rst)
- [Getting Started](../user_guide/getting_started.md)

## API Reference

The detailed generated API continues below on this page.

<!-- reference-summary:end -->


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


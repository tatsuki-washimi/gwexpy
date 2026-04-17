# StandardizeTransform

<!-- reference-summary:start -->

**Stability:** Stable

## What it is

Use `StandardizeTransform` as a ready-made preprocessing step inside direct analysis code or a `Pipeline`.

## Representative Signatures

```python
StandardizeTransform(method="zscore", robust=False, axis="time")
StandardizeTransform.inverse_transform(y)
```

## Minimal Example

```python
from gwexpy.timeseries import StandardizeTransform

standardized = StandardizeTransform().fit_transform(ts_matrix)
```

## Related Theory

- [Validated Algorithms](../user_guide/validated_algorithms.md)
- [Numerical Stability](../user_guide/numerical_stability.md)

## Related Tutorials

- [ML Preprocessing Methods](../user_guide/tutorials/ml_preprocessing_methods.md)
- [ML Preprocessing Case Study](../user_guide/tutorials/case_ml_preprocessing.ipynb)

## API Reference

The detailed generated API continues below on this page.

<!-- reference-summary:end -->


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

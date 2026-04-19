# Transform

<!-- reference-summary:start -->

**Stability:** Stable

## What it is

Use `Transform` as the base contract for reusable preprocessing steps that implement `fit`, `transform`, and optional inversion.

## Representative Signatures

```python
Transform.fit(x)
Transform.transform(x)
```

## Minimal Example

```python
from gwexpy.timeseries import Transform

# Implement fit/transform in a subclass before using it inside Pipeline.
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

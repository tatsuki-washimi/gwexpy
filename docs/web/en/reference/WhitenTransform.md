# WhitenTransform

<!-- reference-summary:start -->

**Stability:** Stable

## What it is

Use `WhitenTransform` as a ready-made preprocessing step inside direct analysis code or a `Pipeline`.

## Representative Signatures

```python
WhitenTransform(method="pca", eps="auto", n_components=None)
WhitenTransform.transform(x)
```

## Minimal Example

```python
from gwexpy.timeseries import WhitenTransform

whitened = WhitenTransform(eps="auto").fit_transform(ts)
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


Whitening using PCA or ZCA on TimeSeriesMatrix-like data.


## Methods

### `__init__`

```python
__init__(self, method: str = 'pca', eps: float | Literal['auto'] | None = 'auto', n_components: Optional[int] = None, *, multivariate: bool = True, align: str = 'intersection')
```

Initialize self.  See help(type(self)) for accurate signature.

``eps="auto"`` (or ``None``) derives variance-relative regularization from
the fitted data. Explicit values must be finite and non-negative. ``eps=0``
disables this ridge term and may be unsuitable for rank-deficient input.

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

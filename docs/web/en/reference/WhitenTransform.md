# WhitenTransform

<!-- reference-summary:start -->

**Stability:** Stable

## What it is

Use `WhitenTransform` as a ready-made preprocessing step inside direct analysis code or a `Pipeline`.

## Representative Signatures

```python
WhitenTransform(fftlength=1.0, overlap=0.5, **kwargs)
WhitenTransform.transform(x)
```

## Minimal Example

```python
from gwexpy.timeseries import WhitenTransform

whitened = WhitenTransform(fftlength=1.0, overlap=0.5).fit_transform(ts)
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


**Inherits from:** Transform


Whitening using PCA or ZCA on TimeSeriesMatrix-like data.


## Methods

### `__init__`

```python
__init__(self, method: str = 'pca', eps: float = 1e-12, n_components: Optional[int] = None, *, multivariate: bool = True, align: str = 'intersection')
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


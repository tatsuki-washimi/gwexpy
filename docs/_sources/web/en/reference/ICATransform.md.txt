# ICATransform

<!-- reference-summary:start -->

**Stability:** Stable

## What it is

Use `ICATransform` as a ready-made preprocessing step inside direct analysis code or a `Pipeline`.

## Representative Signatures

```python
ICATransform(n_components=None, random_state=None, align="intersection")
ICATransform.fit_transform(x)
```

## Minimal Example

```python
from gwexpy.timeseries import ICATransform

components = ICATransform(n_components=2).fit_transform(ts_matrix)
```

## Related Theory

- [Validated Algorithms](../user_guide/validated_algorithms.md)
- [Numerical Stability](../user_guide/numerical_stability.md)

## Related Tutorials

- [ML Preprocessing Methods](../user_guide/tutorials/ml_preprocessing_methods.md)
- [BRUCO ICA Denoising](../user_guide/tutorials/case_bruco_ica_denoising.ipynb)

## API Reference

The detailed generated API continues below on this page.

<!-- reference-summary:end -->


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

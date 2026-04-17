# ImputeTransform

<!-- reference-summary:start -->

## What it is

Use `ImputeTransform` as a ready-made preprocessing step inside direct analysis code or a `Pipeline`.

## Representative Signatures

```python
ImputeTransform(method="interpolate", **kwargs)
ImputeTransform.transform(x)
```

## Minimal Example

```python
from gwexpy.timeseries import ImputeTransform

filled = ImputeTransform(method="interpolate").fit_transform(ts)
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


Impute missing values using existing lower-level helpers.


## Methods

### `__init__`

```python
__init__(self, method: str = 'interpolate', **kwargs)
```

Initialize self.  See help(type(self)) for accurate signature.

*(Inherited from `Transform`)*

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

*(Inherited from `Transform`)*


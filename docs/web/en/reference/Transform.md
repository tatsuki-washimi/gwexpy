# Transform

<!-- reference-summary:start -->

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
- [FFT_Conventions](FFT_Conventions.md)

## Related Tutorials

- [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_en.md)
- [Tutorial Index](../user_guide/tutorials/index.rst)
- [Getting Started](../user_guide/getting_started.md)

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


# Plot

**Inherits from:** `gwpy.plot.Plot`

## What it is

`Plot` is the GWexpy plotting entry point for single series, matrices, and
spectrogram grids. It keeps GWpy-compatible figure behavior while adding
automatic expansion of `SeriesMatrix`-like inputs, layout helpers for grouped
subplots, adaptive decimation for dense overlays, and colorbar handling for
spectrogram-style data.

## Representative Signatures

```python
Plot(*args, separate=None, geometry=None, monitor=None, decimate_threshold=50000, decimate_points=10000, **kwargs)
plot_mmm(median, min_s, max_s, ax=None, **kwargs)
```

## Minimal Example

```python
from gwexpy.plot import Plot

fig = Plot(ts_matrix, separate=True, figsize=(10, 6))
_ = fig.plot_mmm(median_series, min_series, max_series, alpha_fill=0.15)
```

## GWexpy-specific Behavior

- `SeriesMatrix` and `SpectrogramMatrix` inputs are expanded into subplot grids automatically.
- List and dict inputs inherit labels so legend names stay aligned with channel metadata.
- Spectrogram-like inputs can receive shared colorbar placement without manual `matplotlib` wiring.
- Large overlays are decimated automatically when `decimate_threshold` is exceeded.

## Related Theory

- [FFT_Conventions](FFT_Conventions.md)
- [Validated Algorithms](../user_guide/validated_algorithms.md)

## Related Tutorials

- [Tutorial Index](../user_guide/tutorials/index.rst)
- [Getting Started](../user_guide/getting_started.md)

## API Reference

See the generated API reference below for inherited plotting methods and the
GWexpy constructor signature.

```{eval-rst}
.. currentmodule:: gwexpy.plot

.. autoclass:: Plot
   :members: __init__, plot_mmm, show
   :show-inheritance:
```

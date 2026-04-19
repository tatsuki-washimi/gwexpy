# SpectrogramList

<!-- reference-summary:start -->

**Stability:** Stable

## What it is

Use `SpectrogramList` to keep multiple `Spectrogram` objects together while preserving channel labels and metadata.

## Representative Signatures

```python
SpectrogramList(data: list[Spectrogram])
SpectrogramList.to_matrix()
```

## Minimal Example

```python
from gwexpy.spectrogram import Spectrogram, SpectrogramList
import numpy as np

lst = SpectrogramList([Spectrogram(np.ones((8, 16)), dt=1.0, df=1.0)])
mat = lst.to_matrix()
```

## Related Theory

- [FFT_Conventions](FFT_Conventions.md)
- [Spectrogram](Spectrogram.md)
- [SpectrogramMatrix](SpectrogramMatrix.md)

## Related Tutorials

- [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_en.md)
- [Spectrogram Tutorial](../user_guide/tutorials/intro_spectrogram.ipynb)
- [Segment Visualization](../user_guide/tutorials/segment_visualization.ipynb)
- [Glitch Analysis](../user_guide/tutorials/case_glitch_analysis.ipynb)

## API Reference

The detailed generated API continues below on this page.

<!-- reference-summary:end -->


**Inherits from:** PhaseMethodsMixin, UserList


List of Spectrogram objects.
Reference: similar to TimeSeriesList but for 2D Spectrograms.

## Physical Context

Use `SpectrogramList` for several time-frequency maps that should stay separate but be processed together: repeated runs, multiple sensors, or several preprocessing branches of the same event.

- each item can have different provenance even when you batch-plot them
- the container does not enforce matching color semantics, normalization, or binning

## Common Misreadings

1. comparing brightness across items without matching scaling and units
2. assuming all entries share the same `dt`/`df` because they live in one list
3. treating stacked visual summaries as evidence of alignment without checking metadata

## Where to go next

- per-map interpretation: [Spectrogram](Spectrogram.md)
- aligned collection analysis: [SpectrogramMatrix](SpectrogramMatrix.md)
- practical workflows: [Segment Visualization](../user_guide/tutorials/segment_visualization.ipynb), [Glitch Analysis](../user_guide/tutorials/case_glitch_analysis.ipynb)

:::{note}
Spectrogram objects can be very large in memory.
Use `inplace=True` where possible to avoid deep copies.


:::
## Methods

### `__init__`

```python
__init__(self, initlist=None)
```

Initialize self.  See help(type(self)) for accurate signature.

*(Inherited from `PhaseMethodsMixin`)*

### `angle`

```python
angle(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any
```

Alias for `phase(unwrap=unwrap, deg=deg)`.

### `append`

```python
append(self, item)
```

S.append(value) -- append value to the end of the sequence

*(Inherited from `MutableSequence`)*

### `bootstrap_asd`

```python
bootstrap_asd(self, *args, **kwargs)
```

Estimate robust ASD from each spectrogram in the list (returns FrequencySeriesList).

### `crop`

```python
crop(self, t0, t1, inplace=False)
```

Crop each spectrogram.

### `crop_frequencies`

```python
crop_frequencies(self, f0, f1, inplace=False)
```

Crop frequencies.

### `degree`

```python
degree(self, unwrap: 'bool' = False) -> "'SpectrogramList'"
```

Compute phase (in degrees) of each spectrogram.

### `extend`

```python
extend(self, other)
```

S.extend(iterable) -- extend sequence by appending elements from the iterable

*(Inherited from `MutableSequence`)*

### `interpolate`

```python
interpolate(self, dt, df, inplace=False)
```

Interpolate each spectrogram.

### `phase`

```python
phase(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any
```


Calculate the phase of the data.

Parameters
----------
unwrap : `bool`, optional
    If `True`, unwrap the phase to remove discontinuities.
    Default is `False`.
deg : `bool`, optional
    If `True`, return the phase in degrees.
    Default is `False` (radians).
**kwargs
    Additional arguments passed to the underlying calculation.

Returns
-------
`Series` or `Matrix` or `Collection`
    The phase of the data.


### `plot`

```python
plot(self, **kwargs)
```

Plot all spectrograms stacked vertically.

### `plot_summary`

```python
plot_summary(self, **kwargs)
```


Plot List as side-by-side Spectrograms and percentile summaries.


### `radian`

```python
radian(self, unwrap: 'bool' = False) -> "'SpectrogramList'"
```

Compute phase (in radians) of each spectrogram.

### `read`

```python
read(self, source, *args, **kwargs)
```

Read spectrograms into the list from HDF5.

### `rebin`

```python
rebin(self, dt, df, inplace=False)
```

Rebin each spectrogram.

### `to_cupy`

```python
to_cupy(self, *args, **kwargs) -> 'list'
```

Convert each item to cupy.ndarray. Returns a list.

### `to_dask`

```python
to_dask(self, *args, **kwargs) -> 'list'
```

Convert each item to dask.array. Returns a list.

### `to_jax`

```python
to_jax(self, *args, **kwargs) -> 'list'
```

Convert each item to jax.Array. Returns a list.

### `to_matrix`

```python
to_matrix(self)
```

Convert to SpectrogramMatrix (N, Time, Freq).

### `to_tensorflow`

```python
to_tensorflow(self, *args, **kwargs) -> 'list'
```

Convert each item to tensorflow.Tensor. Returns a list.

### `to_torch`

```python
to_torch(self, *args, **kwargs) -> 'list'
```

Convert each item to torch.Tensor. Returns a list.

### `write`

```python
write(self, target, *args, **kwargs)
```

Write list to file.

For HDF5 output you can choose a layout (default is GWpy-compatible dataset-per-entry).

```python
sgl.write("out.h5", format="hdf5")               # GWpy-compatible (default)
sgl.write("out.h5", format="hdf5", layout="group")  # legacy group-per-entry
```

:::{warning}
Never unpickle data from untrusted sources. ``pickle``/``shelve`` can execute
arbitrary code on load.

:::
Pickle portability note: pickled gwexpy `SpectrogramList` unpickles as a built-in
`list` of GWpy `Spectrogram` (gwexpy not required on the loading side).

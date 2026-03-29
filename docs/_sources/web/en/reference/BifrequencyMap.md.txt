# BifrequencyMap

**Inherits from:** `FrequencySeriesMatrix`

Map representing the relationship between two frequency axes (Frequency 1 -> Frequency 2).
Typically used to represent transfer functions, coupling functions, or scattering matrices where energy is transferred from an input frequency (`frequency1`) to an output frequency (`frequency2`).

## Components

- **Frequency 1 (`frequency1`)**: The input or source frequency axis (columns).
- **Frequency 2 (`frequency2`)**: The output or target frequency axis (rows).
- **Value**: The coupling strength or transfer function scalar at each (f2, f1) point.

## Methods

### `from_points`

```python
from_points(cls, data, f2, f1, **kwargs)
```

Create a `BifrequencyMap` from a 2D array and two frequency axes.

Parameters
----------
data : array-like
    2D array of shape (len(f2), len(f1)).
f2 : array-like
    Frequency axis 2 (rows, output).
f1 : array-like
    Frequency axis 1 (columns, input).
**kwargs
    Additional arguments passed to the constructor (name, unit, etc.).

Returns
-------
BifrequencyMap

### `propagate`

```python
propagate(self, input_spectrum, interpolate=True, fill_value=0)
```

Propagate an input spectrum through the map to calculate the output spectrum.
Performs a matrix multiplication: $S_{\text{out}} = M \cdot S_{\text{in}}$.

Parameters
----------
input_spectrum : `FrequencySeries`
    The input spectrum $S_{\text{in}}(f_1)$.
interpolate : bool, optional
    If True, interpolates the input spectrum to match the map's `frequency1` axis.
fill_value : float, optional
    Value to use for interpolation outside the range.

Returns
-------
`FrequencySeries`
    The projected output spectrum $S_{\text{out}}(f_2)$.

### `convolute`

```python
convolute(self, input_spectrum, interpolate=True, fill_value=0)
```

Convolutes the map with an input spectrum (integration along f1).
Calculates: $S_{\text{out}}(f_2) = \int M(f_2, f_1) S_{\text{in}}(f_1) df_1$

Parameters
----------
input_spectrum : `FrequencySeries`
    Input spectrum.
interpolate : bool, optional
    If True, interpolates input spectrum.

Returns
-------
`FrequencySeries`
    Output spectrum with units adjusted by frequency integration.

### `diagonal`

```python
diagonal(self, method='mean', bins=None, absolute=False, **kwargs)
```

Calculates statistics along the diagonal axis ($f_2 - f_1$).

Parameters
----------
method : str, optional
    Statistical method: 'mean', 'median', 'max', 'min', 'std', 'rms', 'percentile'.
    Ignores NaNs by default.
bins : int or array-like, optional
    Number of bins. If None, automatically determined from resolution.
absolute : bool, optional
    If True, calculates statistics along $|f_2 - f_1|$.
**kwargs
    Additional arguments (e.g., `percentile` value).

Returns
-------
`FrequencySeries`
    The statistic as a function of frequency difference.

### `get_slice`

```python
get_slice(self, at, axis='f1', xaxis='remaining')
```

Extracts a slice of the map at a specific frequency.

Parameters
----------
at : float
    Frequency value to extract.
axis : str, optional
    Axis to fix: 'f1' or 'f2'.
xaxis : str, optional
    Definition of the x-axis for the result: 'remaining', 'diff', 'abs_diff', etc.

Returns
-------
`FrequencySeries`
    The extracted 1D slice.

### `plot`

```python
plot(self, **kwargs)
```

Plot the map as a 2D image (spectrogram-like).

### `plot_lines`

```python
plot_lines(self, xaxis='f1', color='f2', num_lines=None, ax=None, cmap=None, **kwargs)
```

Plot the map as a set of 1D lines.

Parameters
----------
xaxis : str, optional
    X-axis for lines: 'f1', 'f2', 'diff', etc.
color : str, optional
    Parameter for coloring/slicing: 'f1', 'f2'.
num_lines : int, optional
    Max number of lines to plot.
ax : matplotlib Axes, optional
    Axes to plot on.
cmap : colormap, optional
    Colormap for lines.

Returns
-------
matplotlib Axes

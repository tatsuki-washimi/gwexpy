# Field4D

**Inherits from:** Array4D, AxisApiMixin, StatisticalMethodsMixin, Array

4D Field with domain states and FFT operations.

Field4D represents physical fields that can exist in different domains (time/frequency for axis 0, real/k-space for spatial axes 1-3).

**Key feature**: All indexing operations return a Field4D, maintaining 4D structure. Integer indices result in axes with length 1 rather than being dropped.

## Methods

### `fft_time`

```python
fft_time(self, nfft=None)
```

Compute FFT along time axis (axis 0).

This method applies the same normalization as GWpy's ``TimeSeries.fft()``: rfft / nfft, with DC-excluded bins multiplied by 2 (Nyquist bin is not doubled when nfft is even).

Parameters
----------
nfft : int, optional
    Length of the FFT. If None, uses the length of axis 0.

Returns
-------
Field4D
    Transformed field with axis0_domain='frequency'.

Notes
-----
- Requires a real-valued, regularly spaced time axis with length >= 2.
- Preserves the time-axis origin for later reconstruction in ``ifft_time``.

### `ifft_time`

```python
ifft_time(self, nout=None)
```

Compute inverse FFT along frequency axis (axis 0).

This method applies the inverse normalization of ``fft_time()`` / GWpy's ``FrequencySeries.ifft()``.

Parameters
----------
nout : int, optional
    Length of the output time series. If None, computed as ``(n_freq - 1) * 2``.

Returns
-------
Field4D
    Transformed field with axis0_domain='time'.

Notes
-----
- Requires a regularly spaced frequency axis with length >= 2.
- Restores the time-axis origin if it was preserved by ``fft_time``.

### `fft_space`

```python
fft_space(self, axes=None, n=None)
```

Compute FFT along spatial axes.

This method uses two-sided FFT (numpy.fft.fftn) and produces angular wavenumber (k = 2π·fftfreq).

Parameters
----------
axes : iterable of str, optional
    Axis names to transform (e.g., ['x', 'y']). If None, transforms all spatial axes in 'real' domain.
n : tuple of int, optional
    FFT lengths for each axis.

Returns
-------
Field4D
    Transformed field with specified axes in 'k' domain.

Notes
-----
- Requires spatial axes to be regularly spaced, strictly monotonic, and length >= 2.
- Descending spatial axes yield a k-axis with flipped sign to preserve direction.

### `ifft_space`

```python
ifft_space(self, axes=None, n=None)
```

Compute inverse FFT along k-space axes.

Parameters
----------
axes : iterable of str, optional
    Axis names to transform (e.g., ['kx', 'ky']). If None, transforms all spatial axes in 'k' domain.
n : tuple of int, optional
    Output lengths for each axis.

Returns
-------
Field4D
    Transformed field with specified axes in 'real' domain.

Notes
-----
- Requires k-space axes with length >= 2.
- Descending k-axes reconstruct descending real-space axes.

### `wavelength`

```python
wavelength(self, axis)
```

Compute wavelength from wavenumber axis.

Parameters
----------
axis : str or int
    The k-domain axis name or index.

Returns
-------
`~astropy.units.Quantity`
    Wavelength values (λ = 2π / |k|). k=0 returns inf.

## Properties

| Property | Description |
|----------|-------------|
| `axis0_domain` | Domain of axis 0: 'time' or 'frequency' |
| `space_domains` | Mapping of spatial axis names to domains ('real' or 'k') |
| `axis_names` | Names of all axes |
| `unit` | Physical unit of these data |
| `axes` | Tuple of AxisDescriptor objects for each dimension |

## Slicing Behavior

Field4D overrides `__getitem__` to always return a Field4D. 
If you specify an integer index for an axis, that axis is kept with length 1 instead of being dropped.

```python
>>> field.shape
(100, 32, 32, 32)
>>> field[0].shape
(1, 32, 32, 32)
```

# ScalarField

**Inherits from:** FieldBase, Array4D, AxisApiMixin, StatisticalMethodsMixin, Array

4D scalar field with explicit axis domains and FFT operations.

ScalarField represents physical fields that can exist in different domains (time/frequency for axis 0, real/k-space for spatial axes 1-3). Shape is invariant `(axis0, x, y, z)`; slicing keeps all axes (length-1 for indexed axes) and preserves domain/unit metadata.

**Key features**
- Explicit domains: `axis0_domain in {time, frequency}`, spatial domains in `{real, k}` per axis.
- Unit/domain consistency is validated at construction and after FFTs.
- Indexing preserves 4D structure (axes never dropped).

## Methods (selected)

### `fft_time`

```python
fft_time(self, nfft=None)
```

Compute FFT along time axis (axis 0), applying the same normalization as GWpy's `TimeSeries.fft()`: rfft / nfft, with non-DC/non-Nyquist bins doubled. Domain is updated to `frequency`.

### `ifft_time`

```python
ifft_time(self, nout=None)
```

Inverse of `fft_time`, restoring `axis0_domain='time'` and reconstructing the time axis (length `nout` by default to two-sided real length).

### `fft_space`

```python
fft_space(self, axes=None, n=None)
```

Compute signed two-sided FFT along spatial axes (subset allowed). Transformed axes flip domain `real -> k` and axis names `x->kx`, etc. Non-transformed axes remain unchanged.

### `ifft_space`

```python
ifft_space(self, axes=None, n=None)
```

Inverse spatial FFT for axes currently in `k` domain, restoring `real` domain and spatial axis names.

## Collections

Use `gwexpy.fields.FieldList` and `FieldDict` for batch operations on ScalarField objects.

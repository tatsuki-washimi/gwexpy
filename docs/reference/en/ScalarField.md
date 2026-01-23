# ScalarField

**Inherits from:** `FieldBase`, `Array4D`, `AxisApiMixin`, `StatisticalMethodsMixin`, `GwpyArray`

4D scalar field with explicit axis domains and FFT operations.

ScalarField represents physical fields that can exist in different domains (time/frequency for axis 0, real/k-space for spatial axes 1-3). Shape is invariant `(axis0, x, y, z)`; slicing keeps all axes (length-1 for indexed axes) and preserves domain/unit metadata.

## Key features

- **Explicit domains**: `axis0_domain in {time, frequency}`, spatial domains in `{real, k}` per axis.
- **Physics-aware slicing**: Indexing preserves 4D structure (axes are never dropped, keeping metadata alive).
- **Unit/domain consistency**: Validated at construction and after FFT transitions.
- **Signal Processing**: Built-in support for PSD, coherence, and cross-correlation.

## Methods

### FFT Operations

#### `fft_time(nfft=None)`

Compute FFT along time axis (axis 0). Applies GWpy-style normalization: `rfft / nfft`, with non-DC/non-Nyquist bins doubled. Domain is updated to `frequency`.

#### `ifft_time(nout=None)`

Inverse of `fft_time`. Restores `axis0_domain='time'` and reconstructs the time axis (using stored `_axis0_offset`).

#### `fft_space(axes=None, n=None)`

Compute signed two-sided FFT along spatial axes. Transformed axes flip domain `real -> k` and axis names `x -> kx`, etc. Wavenumber is computed as $k = 2\pi/\lambda$.

#### `ifft_space(axes=None, n=None)`

Inverse spatial FFT for axes in `k` domain, restoring `real` domain.

### Signal Processing (on instance)

#### `filter(*args, **kwargs)`

Apply a digital filter along the time axis (axis 0). Accepts filters designed with `gwpy.signal.filter_design`. By default, uses `filtfilt` for zero-phase distortion.

#### `resample(rate, **kwargs)`

Resample the field along the time axis to a new sampling rate. Uses `scipy.signal.resample` internally and preserves all axis metadata and units.

### Signal Processing (High-level API)

These functions are available in `gwexpy.fields`:

- `compute_psd(field, point_or_region, ...)`: Compute Welch PSD at spatial locations.
- `coherence_map(field, ref_point, ...)`: Mapping coherence between a reference and a slice.
- `compute_xcorr(field, point_a, point_b, ...)`: Cross-correlation between two points.
- `time_delay_map(field, ref_point, ...)`: Estimated time delay map.

### Demo Data Generation

- `make_demo_scalar_field(...)`: General purpose 4D field generation.
- `make_propagating_gaussian(...)`: Propagating wave packet.
- `make_sinusoidal_wave(...)`: Plane wave simulation.

## Collections

Use `gwexpy.fields.FieldList` and `FieldDict` for batch operations on ScalarField objects.

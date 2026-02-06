# Migration Guide: nperseg/noverlap → fftlength/overlap

## Summary

gwexpy v0.1.0b2 unified all spectral analysis APIs to use **time-based parameters**:
- **Old API**: `nperseg`, `noverlap` (sample counts)
- **New API**: `fftlength`, `overlap` (seconds)

This change aligns gwexpy with GWpy conventions and improves API consistency across the library.

---

## Why This Change?

### Problems with Sample-Count Parameters

The old `nperseg`/`noverlap` parameters had several issues:

1. **Sample Rate Dependency**: The same parameter values meant different time windows at different sample rates
   - `nperseg=256` at 256 Hz → 1.0 second
   - `nperseg=256` at 4096 Hz → 0.0625 seconds (62.5 ms)

2. **Inconsistency with GWpy**: GWpy uses time-based parameters throughout its API
   - Mixing gwexpy and GWpy code required constant conversions

3. **User Confusion**: Users had to manually calculate sample counts from desired time windows

### Benefits of Time-Based Parameters

- **Intuitive**: Specify FFT length directly in seconds (e.g., `fftlength=1.0`)
- **Sample Rate Independent**: Same parameter values work across different sample rates
- **GWpy Compatible**: Seamless integration with GWpy workflows
- **Unit Support**: Accepts `astropy.units.Quantity` (e.g., `fftlength=1.0*u.s`)

---

## Affected Functions

The following functions now use `fftlength` and `overlap` instead of `nperseg` and `noverlap`:

### High-Level Analysis
- `gwexpy.spectral.bootstrap_spectrogram()`
- `gwexpy.fitting.fit_bootstrap_spectrum()`
- `gwexpy.spectrogram.Spectrogram.bootstrap()`
- `gwexpy.spectrogram.Spectrogram.bootstrap_asd()`

### Field Analysis
- `gwexpy.fields.signal.spectral_density()`
- `gwexpy.fields.signal.compute_psd()`
- `gwexpy.fields.signal.freq_space_map()`
- `gwexpy.fields.signal.coherence_map()`

### Matrix Methods (Internal)
- `gwexpy.timeseries.TimeSeriesMatrix._vectorized_psd()`
- `gwexpy.timeseries.TimeSeriesMatrix._vectorized_csd()`
- `gwexpy.timeseries.TimeSeriesMatrix._vectorized_coherence()`

---

## Migration Examples

### Example 1: fit_bootstrap_spectrum

**Before (v0.1.0b1)**:
```python
from gwexpy.fitting import fit_bootstrap_spectrum

def power_law(f, A, alpha):
    return A * f**alpha

# Old API: sample counts
result = fit_bootstrap_spectrum(
    timeseries,
    model_fn=power_law,
    nperseg=256,      # 256 samples
    noverlap=128,     # 128 samples
    n_boot=500,
)
```

**After (v0.1.0b2+)**:
```python
from gwexpy.fitting import fit_bootstrap_spectrum

def power_law(f, A, alpha):
    return A * f**alpha

# New API: time-based (seconds)
# Assuming sample_rate = 256 Hz:
# nperseg=256 → fftlength = 256/256 = 1.0 seconds
# noverlap=128 → overlap = 128/256 = 0.5 seconds

result = fit_bootstrap_spectrum(
    timeseries,
    model_fn=power_law,
    fftlength=1.0,    # 1.0 seconds
    overlap=0.5,      # 0.5 seconds
    n_boot=500,
)
```

### Example 2: Spectrogram.bootstrap_asd

**Before**:
```python
# Assuming sample_rate = 64 Hz
psd = spectrogram.bootstrap_asd(
    n_boot=500,
    nperseg=256,      # 256 samples = 4.0 seconds
    noverlap=192,     # 192 samples = 3.0 seconds
    window='hann'
)
```

**After**:
```python
psd = spectrogram.bootstrap_asd(
    n_boot=500,
    fftlength=4.0,    # 4.0 seconds (directly)
    overlap=3.0,      # 3.0 seconds (directly)
    window='hann'
)
```

### Example 3: Using astropy.units

```python
from astropy import units as u

# You can now use Quantity objects
result = fit_bootstrap_spectrum(
    timeseries,
    model_fn=power_law,
    fftlength=1.0 * u.s,    # Explicit unit
    overlap=500 * u.ms,     # 500 milliseconds
    n_boot=500,
)
```

---

## Converting Old Code

### Conversion Formula

If you have old code with `nperseg` and `noverlap`:

```python
# Given old parameters and sample rate
nperseg_old = 256
noverlap_old = 128
sample_rate = 256  # Hz

# Convert to new parameters
fftlength_new = nperseg_old / sample_rate      # 1.0 seconds
overlap_new = noverlap_old / sample_rate        # 0.5 seconds
```

### Automated Conversion Script

For large codebases, you can use this script to identify old API usage:

```bash
# Find all occurrences of nperseg/noverlap in Python files
grep -rn "nperseg\|noverlap" --include="*.py" your_project/

# Find in Jupyter notebooks
grep -rn "nperseg\|noverlap" --include="*.ipynb" your_project/
```

---

## Error Handling

### TypeError with Deprecated Parameters

Using deprecated `nperseg` or `noverlap` parameters will raise `TypeError`:

```python
>>> result = fit_bootstrap_spectrum(ts, model_fn, nperseg=256)
TypeError: nperseg is removed from the public API.
Specify fftlength in seconds instead.

>>> result = fit_bootstrap_spectrum(ts, model_fn, noverlap=128)
TypeError: noverlap is removed from the public API.
Specify overlap in seconds instead.
```

**No deprecation period**: The old parameters are immediately rejected. This ensures clean migration and prevents silent bugs.

### Common Migration Errors

1. **Forgetting to convert sample counts**:
   ```python
   # ❌ Wrong: using sample counts with new API
   result = fit_bootstrap_spectrum(ts, model_fn, fftlength=256)  # Treats as 256 seconds!

   # ✓ Correct: convert to seconds
   fftlength = 256 / ts.sample_rate.value  # Convert to seconds
   result = fit_bootstrap_spectrum(ts, model_fn, fftlength=fftlength)
   ```

2. **Unit mismatch**:
   ```python
   # ❌ Wrong: frequency unit instead of time
   from astropy import units as u
   result = fit_bootstrap_spectrum(ts, model_fn, fftlength=1.0*u.Hz)
   # ValueError: fftlength: expected a time-like Quantity; got unit 'Hz'.

   # ✓ Correct: time unit
   result = fit_bootstrap_spectrum(ts, model_fn, fftlength=1.0*u.s)
   ```

---

## Default Values

### Old API Defaults (v0.1.0b1)
- `nperseg=16` (samples)
- `noverlap=None` (auto-calculated based on window)

### New API Defaults (v0.1.0b2+)
- `fftlength=None` (auto-calculated based on data length and desired resolution)
- `overlap=None` (window-dependent, typically 50% for Hann window)

**Note**: Default behavior is now fully aligned with GWpy conventions.

---

## Compatibility Notes

### scipy.signal Functions

If you need to use scipy directly, you must still use sample counts:

```python
import scipy.signal

# scipy.signal.stft still requires sample counts
fs = ts.sample_rate.to('Hz').value
nperseg_scipy = int(fftlength * fs)
noverlap_scipy = int(overlap * fs)

f, t, Zxx = scipy.signal.stft(
    ts.value,
    fs=fs,
    nperseg=nperseg_scipy,
    noverlap=noverlap_scipy
)
```

However, we recommend using gwexpy wrappers instead of calling scipy directly.

### Backward Compatibility

**There is no backward compatibility**. Old code using `nperseg`/`noverlap` will fail immediately with clear error messages.

This design choice prevents silent bugs and encourages complete migration.

---

## Additional Resources

- CHANGELOG.md (repository root) - Full API change details
- [Technical Report](https://github.com/gwexpy/gwexpy/blob/main/docs/developers/reports/api_unification_spectral_params_20260206.md) - Implementation details
- [gwexpy.fitting API Reference](../reference/fitting.md)
- [gwexpy.spectral API Reference](../reference/Spectral.md)

---

## Getting Help

If you encounter issues during migration:

1. Check error messages - they provide clear guidance
2. Review the examples above
3. Report issues at: https://github.com/gwexpy/gwexpy/issues
4. Contact developers via Slack/Discord (if available)

---

**Last Updated**: 2026-02-06
**Applies to**: gwexpy v0.1.0b2 and later

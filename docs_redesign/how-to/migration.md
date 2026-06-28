# Migrating to GWexpy v0.1.1

This guide summarizes the breaking changes introduced in `v0.1.1` and provides migration paths for existing code.

## 1. Lorentzian Peak Parameters

The `fwhm` (Full Width at Half Maximum) parameter in `gwexpy.noise.peaks.lorentzian_line()` has been renamed to `gamma` (Half Width at Half Maximum) to better align with the underlying mathematical implementation and typical spectral modeling conventions.

### How to Migrate
If you were using `fwhm`, simply divide your value by 2 and pass it as `gamma`.

**Old (v0.1.0):**
```python
from gwexpy.noise.peaks import lorentzian_line
line = lorentzian_line(f0=100, amplitude=1e-21, fwhm=10)
```

**New (v0.1.1):**
```python
from gwexpy.noise.peaks import lorentzian_line
# Note: gamma = fwhm / 2
line = lorentzian_line(f0=100, amplitude=1e-21, gamma=5)
```

## 2. Spectral Fitting (Generalized Least Squares)

The `stride` parameter has been removed from `fit_bootstrap_spectrum()` as it was an unsupported keyword argument that caused errors in the underlying `scipy.signal.periodogram` calls.

### How to Migrate
Simply remove the `stride` keyword argument from your `fit_bootstrap_spectrum` calls. The function now correctly manages data segmentation through `fftlength` and `overlap`.

**Old (v0.1.0):**
```python
from gwexpy.fitting import fit_bootstrap_spectrum
results = fit_bootstrap_spectrum(series, fftlength=4, overlap=2, stride=4) # Error!
```

**New (v0.1.1):**
```python
from gwexpy.fitting import fit_bootstrap_spectrum
# stride is no longer required or supported
results = fit_bootstrap_spectrum(series, fftlength=4, overlap=2)
```

## 3. SegmentTable Lazy Loading

The `loader` argument in `SegmentTable.add_series_column()` now expects a callable that takes a single `segment` argument (`loader(segment)`), rather than a factory that takes an index (`loader(i)`).

### How to Migrate
Update your loader functions to accept the `segment` (span) of the row directly.

**Old (v0.1.0):**
```python
def my_factory(i):
    return lambda: load_data(st.row(i)['span'])
st.add_series_column("data", loader=my_factory)
```

**New (v0.1.1):**
```python
def my_loader(segment):
    return load_data(segment)
st.add_series_column("data", loader=my_loader)
```

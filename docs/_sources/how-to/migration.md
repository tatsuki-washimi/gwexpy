# Migrating Between GWexpy Versions

This guide summarizes breaking changes between GWexpy releases and provides migration paths for existing code.

The project-wide [GWpy compatibility policy](../explanation/gwpy_compatibility_policy.md)
governs default behavior of APIs corresponding to GWpy. Release-specific notes
below describe changes within that contract.

## v0.2.0

### Lazy registration

A plain `import gwexpy` no longer eagerly registers constructors or I/O
handlers. Supported public I/O entry points register the handlers they need on
demand. Call `gwexpy.register_all()` only when an application deliberately
needs the complete constructor and I/O surface registered up front.

### Removed developer proxy imports

The developer-only imports `gwexpy.utils.shell`, `gwexpy.utils.sphinx`,
`gwexpy.utils.sphinx.ex2rst`, and `gwexpy.utils.sphinx.zenodo` have been
removed. Replace shell helpers with `subprocess` or `shutil.which`, and use
maintained documentation or Zenodo tooling directly.

### SDB format name

The undocumented `sqlite` and `sqlite3` format aliases, including their GUI
fallback extensions, have been removed. Rename archives to `.sdb` and use
`format="sdb"` for direct I/O.

### SeriesMatrix arithmetic

The v0.2.0 B0 contract rejects dimensional raw-`ndarray` addition and
subtraction with `SpectrogramMatrix` atomically with `TypeError`. Do not rely
on implicit metadata-dropping arithmetic; convert inputs to an explicitly
supported representation first.

## v0.1.1

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

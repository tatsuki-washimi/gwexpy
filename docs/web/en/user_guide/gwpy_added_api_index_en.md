# GWpy Difference API Index

This page is a **difference-oriented lookup** for what GWexpy adds relative to GWpy.  
It is not a full API inventory. For the complete API surface, use the [API Reference](../reference/index.rst).

If you want the migration entry point first, go back to the [Migration Guide for GWpy Users](gwexpy_for_gwpy_users_en.md).

## Start with categories

| Category | Check this when you want to... | Representative items | Entry point |
| --- | --- | --- | --- |
| Multi-channel | process many channels together | `to_matrix()`, `TimeSeriesMatrix`, `FrequencySeriesMatrix` | [Matrix Tutorial](tutorials/matrix_timeseries.ipynb) |
| Added methods | pull external function calls back onto the data object | `.find_peaks()`, `.fit()`, `.hht()`, `.arima()` | [Added Methods](#added-methods) |
| Field API | work with spacetime-like or 4D data structures | `ScalarField`, `FieldList`, `FieldDict`, `fft_space()` | [Field API](#field-api) |
| Sharing / compatibility | understand result-sharing behavior | Transparent Pickle | [Sharing--Compatibility](#sharing--compatibility) |

## Detailed Index

### Multi-channel

| Representative API | Stability | What it adds relative to GWpy | Details |
| --- | --- | --- | --- |
| `TimeSeriesDict.to_matrix()` -> `TimeSeriesMatrix` | Stable | converts channel collections into a batch-analysis container for spectral or statistical workflows | [Matrix Tutorial](tutorials/matrix_timeseries.ipynb), [TimeSeriesDict](../reference/TimeSeriesDict.md), [TimeSeriesMatrix](../reference/TimeSeriesMatrix.md) |
| `FrequencySeriesDict.to_matrix()` -> `FrequencySeriesMatrix` | Stable | turns collections of frequency-series objects into a container suited for pairwise and batch analysis | [FrequencySeriesDict](../reference/FrequencySeriesDict.md), [FrequencySeriesMatrix](../reference/FrequencySeriesMatrix.md) |
| `SpectrogramDict.to_matrix()` / `SpectrogramList.to_matrix()` -> `SpectrogramMatrix` | Stable | lets time-frequency collections move into a matrix-style container for downstream processing | [SpectrogramDict](../reference/SpectrogramDict.md), [SpectrogramList](../reference/SpectrogramList.md), [SpectrogramMatrix](../reference/SpectrogramMatrix.md) |

### Added Methods

| Representative API | Stability | What it adds relative to GWpy | Details |
| --- | --- | --- | --- |
| `.find_peaks()` | Stable | lets you run peak detection directly on the data object instead of dropping to arrays first | [Frequency Series Tutorial](tutorials/intro_frequencyseries.ipynb), [TimeSeries](../reference/TimeSeries.md), [FrequencySeries](../reference/FrequencySeries.md) |
| `.fit()` | Stable | starts fitting workflows directly from the data object | [Fitting](tutorials/advanced_fitting.ipynb), [Fitting Reference](../reference/fitting.md) |
| `.hht()` | Experimental | exposes Hilbert-Huang Transform analysis as an object method | [HHT](tutorials/advanced_hht.ipynb), [TimeSeries](../reference/TimeSeries.md) |
| `.arima()` | Experimental | exposes time-series modelling and forecasting as an object method | [ARIMA](tutorials/advanced_arima.ipynb), [TimeSeries](../reference/TimeSeries.md) |

### Field API

| Representative API | Stability | What it adds relative to GWpy | Details |
| --- | --- | --- | --- |
| `ScalarField` | Experimental | introduces a metadata-aware 4D field container with time and spatial axes | [Field API Intro](tutorials/field_scalar_intro.ipynb), [ScalarField](../reference/ScalarField.md) |
| `ScalarField.fft_space()` | Experimental | performs spatial-domain transforms while staying inside the Field-object model | [Field API Intro](tutorials/field_scalar_intro.ipynb), [ScalarField](../reference/ScalarField.md) |
| `FieldList` / `FieldDict` | Experimental | groups multiple `ScalarField` objects for batch-style processing and shared validation | [Field API Intro](tutorials/field_scalar_intro.ipynb), [FieldList](../reference/FieldList.md), [FieldDict](../reference/FieldDict.md) |

### Sharing / Compatibility

| Representative API / behavior | Stability | What it adds relative to GWpy | Details |
| --- | --- | --- | --- |
| Transparent Pickle | Stable | allows a GWexpy-produced object to be restored as a GWpy base object on the receiving side even without GWexpy | [Migration Guide for GWpy Users](gwexpy_for_gwpy_users_en.md), [Installation Guide](installation.md) |

## What this page does not duplicate

- For **direct I/O formats**, use the [File I/O Supported Formats Guide](io_formats.md).
- For **external-library conversion lists**, use the [Interop / Conversion Guide](interop.md).
- For the **full API surface**, use the [API Reference](../reference/index.rst).

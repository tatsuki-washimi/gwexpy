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

| API Kind | Representative API | Stability | What it adds relative to GWpy | Details |
| --- | --- | --- | --- | --- |
| Conversion method | `TimeSeriesDict.to_matrix()` -> `TimeSeriesMatrix` | Stable | converts channel collections into a batch-analysis container for spectral or statistical workflows | [Matrix Tutorial](tutorials/matrix_timeseries.ipynb), [TimeSeriesDict](../reference/TimeSeriesDict.md), [TimeSeriesMatrix](../reference/TimeSeriesMatrix.md) |
| Conversion method | `FrequencySeriesDict.to_matrix()` -> `FrequencySeriesMatrix` | Stable | turns collections of frequency-series objects into a container suited for pairwise and batch analysis | [FrequencySeriesDict](../reference/FrequencySeriesDict.md), [FrequencySeriesMatrix](../reference/FrequencySeriesMatrix.md) |
| Conversion method | `SpectrogramDict.to_matrix()` / `SpectrogramList.to_matrix()` -> `SpectrogramMatrix` | Stable | lets time-frequency collections move into a matrix-style container for downstream processing | [SpectrogramDict](../reference/SpectrogramDict.md), [SpectrogramList](../reference/SpectrogramList.md), [SpectrogramMatrix](../reference/SpectrogramMatrix.md) |

#### Family-specific `to_matrix()` contract (current stable behavior)

`to_matrix()` is intentionally family-specific today. Use this table as the
public contract for what conversion checks and metadata guarantees are applied.

| Family | Collection entry points | Axis policy | Resampling / tolerance | Unit policy | Label / round-trip notes |
| --- | --- | --- | --- | --- | --- |
| `TimeSeries` | `TimeSeriesDict.to_matrix()` and `TimeSeriesList.to_matrix()` | Aligns onto a common time grid through `align_timeseries_collection()`; exact axis equality is not required when alignment succeeds. | `align="intersection"` default; forwards alignment kwargs (including `method` and `tolerance`) to the alignment helper. | Source `TimeSeries.unit` is not carried into per-cell matrix metadata by this conversion path; reconstructed elements are dimensionless unless metadata is set separately. | Dict keys are written into element names; generated row keys (`row0`, `row1`, ...) are used for round-trip dict keys. |
| `FrequencySeries` | `FrequencySeriesDict.to_matrix()` | Checks sample length equality only; frequency coordinate values are taken from the first element. | No resampling and no tolerance parameter. | Per-element `unit` / `name` / `channel` are preserved in matrix metadata and restored by round trip. | Dict keys are preserved as row keys; single output column is `value`. |
| `Spectrogram` | `SpectrogramDict.to_matrix()` and `SpectrogramList.to_matrix()` | Requires equal shape plus equal time/frequency axes after conversion to the first axis unit. | No resampling and no tolerance parameter; comparisons are exact after unit conversion. | Per-element `unit` / `name` / `channel` are preserved; global matrix unit is set only when all elements share the same unit. | Dict keys are preserved as row keys; list rows are generated (`batch0`, `batch1`, ...). |
| Fields | No SeriesMatrix `to_matrix()` collection API | Uses `FieldList` / `FieldDict` validation rules instead of SeriesMatrix conversion. | Field validation has its own axis tolerance checks. | Units stay on field objects; `to_array()` returns raw arrays. | Not a SeriesMatrix round-trip path. |

### Added Methods

| API Kind | Representative API | Stability | What it adds relative to GWpy | Details |
| --- | --- | --- | --- | --- |
| Instance method | `.find_peaks()` | Stable | lets you run peak detection directly on the data object instead of dropping to arrays first | [Frequency Series Tutorial](tutorials/intro_frequencyseries.ipynb), [TimeSeries](../reference/TimeSeries.md), [FrequencySeries](../reference/FrequencySeries.md) |
| Instance method | `.fit()` | Stable | starts fitting workflows directly from the data object | [Fitting](tutorials/advanced_fitting.ipynb), [Fitting Reference](../reference/fitting.md) |
| Instance method | `.hht()` | Experimental | exposes Hilbert-Huang Transform analysis as an object method | [HHT](tutorials/advanced_hht.ipynb), [TimeSeries](../reference/TimeSeries.md) |
| Instance method | `.arima()` | Experimental | exposes time-series modelling and forecasting as an object method | [ARIMA](tutorials/advanced_arima.ipynb), [TimeSeries](../reference/TimeSeries.md) |

### Field API

| API Kind | Representative API | Stability | What it adds relative to GWpy | Details |
| --- | --- | --- | --- | --- |
| Class | `ScalarField` | Experimental | introduces a metadata-aware 4D field container with time and spatial axes | [Field API Intro](tutorials/field_scalar_intro.ipynb), [ScalarField](../reference/ScalarField.md) |
| Instance method | `ScalarField.fft_space()` | Experimental | performs spatial-domain transforms while staying inside the Field-object model | [Field API Intro](tutorials/field_scalar_intro.ipynb), [ScalarField](../reference/ScalarField.md) |
| Collection | `FieldList` / `FieldDict` | Experimental | groups multiple `ScalarField` objects for batch-style processing and shared validation | [Field API Intro](tutorials/field_scalar_intro.ipynb), [FieldList](../reference/FieldList.md), [FieldDict](../reference/FieldDict.md) |

### Sharing / Compatibility

| API Kind | Representative API / behavior | Stability | What it adds relative to GWpy | Details |
| --- | --- | --- | --- | --- |
| Behavior | Transparent Pickle | Stable | allows a GWexpy-produced object to be restored as a GWpy base object on the receiving side even without GWexpy | [Migration Guide for GWpy Users](gwexpy_for_gwpy_users_en.md), [Installation Guide](installation.md) |

## What this page does not duplicate

- For **direct I/O formats**, use the [File I/O Supported Formats Guide](io_formats.md).
- For **external-library conversion lists**, use the [Interop / Conversion Guide](interop.md).
- For the **full API surface**, use the [API Reference](../reference/index.rst).

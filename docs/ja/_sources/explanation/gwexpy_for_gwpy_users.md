# GWexpy for GWpy Users

This page is the **entry point for moving from GWpy to GWexpy**.  
It is not intended to be a full API catalog. The goal is to understand, quickly, what still works as-is and where the high-value differences start.

If you want a **difference-oriented index of added APIs**, see [GWpy Difference API Index](../reference/gwpy_added_api.md).  
If you want the full API surface regardless of GWpy compatibility, use the [API Reference](../reference/index.md).

## What to lock in first

- **Single-channel workflows are often still familiar.**
  For `TimeSeries` / `FrequencySeries` / `Spectrogram`, many basic read, plot, and spectral-analysis paths can be tried first by swapping imports to `gwexpy`.
- **The biggest shift is multi-channel processing.**
  Instead of manually looping over `TimeSeriesDict`, you can convert it with `to_matrix()` and treat the channel set as a batch-analysis container.
- **The Field API is a GWexpy-only extension.**
  `ScalarField` and `FieldList` / `FieldDict` cover spatially indexed data and batch operations across multiple fields, which GWpy does not provide as a standard layer.
- **Some external-library calls move onto the data object itself.**
  Representative examples are `.find_peaks()`, `.fit()`, `.hht()`, and `.arima()`.
- **Direct I/O and interop have their own guides.**
  Treat [File I/O Supported Formats Guide](../how-to/io_formats.md) as the source of truth for `read(..., format=...)` / `write(..., format=...)`, and [Interop / Conversion Guide](../how-to/interop.md) as the source of truth for `to_*()` / `from_*()` conversions.

## Where migration usually pays off first

| Goal | First difference to check | Where to go next |
| --- | --- | --- |
| Analyse many channels together | `TimeSeriesDict.to_matrix()` -> `TimeSeriesMatrix` | [Matrix Tutorial](../tutorials/matrix_timeseries) |
| Work with spatially indexed data or multiple fields together | `ScalarField`, `FieldList`, `FieldDict` | [Field API Intro](../how-to/containers/field_scalar_intro), [GWpy Difference API Index](../reference/gwpy_added_api.md) |
| Reduce direct SciPy / Statsmodels plumbing | added object-level APIs | [GWpy Difference API Index](../reference/gwpy_added_api.md) |
| Move existing single-channel code quickly | swap imports first, then add difference APIs only where needed | [Quickstart](../tutorials/quickstart.md) |
| Understand result-sharing behavior | Transparent Pickle compatibility | [GWpy Difference API Index](../reference/gwpy_added_api.md) |

## Recipe 1: Move manual `TimeSeriesDict` loops toward `TimeSeriesMatrix`

In GWpy-style code, pairwise comparisons or per-channel spectral logic often become explicit loops.  
In GWexpy, `to_matrix()` is the main entry point for treating the whole channel set as a batch-analysis object.

### GWpy style

```python
from gwpy.timeseries import TimeSeriesDict

tsd = TimeSeriesDict.read(cache, channels)
reference = tsd["H1:STRAIN"]

csd = {}
for name, ts in tsd.items():
    if name == "H1:STRAIN":
        continue
    csd[name] = ts.csd(reference, fftlength=4)
```

### GWexpy style

```python
from gwexpy.timeseries import TimeSeriesDict

tsd = TimeSeriesDict.read(cache, channels)
matrix = tsd.to_matrix()
reference = tsd["H1:STRAIN"]

csm = matrix.csd(reference, fftlength=4)
csm.plot().show()
```

This difference matters when:

- you want fewer manual loops and more object-level batch operations
- you want multi-channel analysis to stay in `TimeSeriesMatrix` / `FrequencySeriesMatrix`

Related pages:

- [Matrix Tutorial](../tutorials/matrix_timeseries)
- [GWpy Difference API Index](../reference/gwpy_added_api.md)
- [TimeSeriesDict Reference](../reference/api/timeseries)
- [TimeSeriesMatrix Reference](../reference/api/timeseries)

## Recipe 2: Pull external function calls back onto the data object

In GWpy-based code, the natural pattern is often “extract arrays, then call SciPy or Statsmodels directly.”  
In GWexpy, part of that workflow is exposed as methods on the data object itself.

### GWpy style

```python
import numpy as np
from scipy.signal import find_peaks
from gwpy.frequencyseries import FrequencySeries

spec = FrequencySeries(...)
peaks, props = find_peaks(np.asarray(spec.value), height=0.2)
```

### GWexpy style

```python
from gwexpy.frequencyseries import FrequencySeries

spec = FrequencySeries(...)
peaks, props = spec.find_peaks(threshold=0.2)
```

The same direction applies to other added APIs in `gwexpy`, including:

- `.fit()` for object-level fitting workflows
- `.hht()` for Hilbert-Huang Transform analysis
- `.arima()` for time-series modelling and forecasting

Related pages:

- [GWpy Difference API Index](../reference/gwpy_added_api.md)
- [Frequency Series Tutorial](../tutorials/intro_frequencyseries)
- [Fitting](../tutorials/intro_fitting)
- [HHT](../how-to/spectral/advanced_hht)
- [ARIMA](../how-to/fitting/advanced_arima)

## Recipe 3: Single-channel code often does not need a full rewrite

If you already know GWpy’s base classes, you do not need to redesign everything up front.  
In many cases, the practical migration path is: change imports first, then adopt GWexpy-specific APIs only where they help.

### GWpy style

```python
from gwpy.timeseries import TimeSeries

ts = TimeSeries.read("data.gwf", "H1:STRAIN")
asd = ts.asd(fftlength=4)
asd.plot().show()
```

### GWexpy style

```python
from gwexpy.timeseries import TimeSeries

ts = TimeSeries.read("data.gwf", "H1:STRAIN")
asd = ts.asd(fftlength=4)
asd.plot().show()
```

Practical reading of this difference:

- start by carrying over existing single-channel workflows
- add GWexpy-only APIs later, where multi-channel or higher-level analysis benefits matter

Related pages:

- [Quickstart](../tutorials/quickstart.md)
- [Tutorial Index](../tutorials/index.md)
- [API Reference](../reference/index.md)

## Recipe 4: Group `ScalarField` objects with `FieldList` / `FieldDict`

Typical GWpy migration paths focus on series-like objects, and GWpy does not provide a standard container layer for field-like data with shared spatial metadata.
In GWexpy, you can keep a single field in `ScalarField` and manage multiple related fields through `FieldList` / `FieldDict`.

### GWpy style

```python
import numpy as np

field_a = np.random.randn(8, 3, 3, 3)
field_b = np.random.randn(8, 3, 3, 3)

fields = {"before": field_a, "after": field_b}

# axis metadata and units must be tracked separately
```

### GWexpy style

```python
import numpy as np
from gwexpy.fields import ScalarField, FieldDict

fields = FieldDict(
    {
        "before": ScalarField(np.random.randn(8, 3, 3, 3)),
        "after": ScalarField(np.random.randn(8, 3, 3, 3)),
    },
    validate=True,
)

fft_fields = fields.fft_space_all()
```

This difference matters when:

- you want multiple fields to share unit/axis/domain checks instead of managing metadata by hand
- you want spatial transforms or selections to stay at the collection level rather than writing field-by-field loops

Related pages:

- [Field API Intro](../how-to/containers/field_scalar_intro)
- [GWpy Difference API Index](../reference/gwpy_added_api.md)
- [ScalarField Reference](../reference/api/fields)
- [FieldList Reference](../reference/api/fields)
- [FieldDict Reference](../reference/api/fields)

## Recipe 5: Read Pickle sharing in terms of compatibility

GWexpy is designed with result sharing in mind, including cases where the receiver does not have GWexpy installed.  
The point here is not generic Pickle safety advice, but what a GWpy user can expect operationally.

### GWpy style

```python
import pickle
from gwpy.timeseries import TimeSeries

ts = TimeSeries(...)

with open("result.pkl", "wb") as f:
    pickle.dump(ts, f)

# sharing assumes the receiving side can read the same object type
```

### GWexpy style

```python
import pickle
from gwexpy.timeseries import TimeSeries

ts = TimeSeries(...)

with open("result.pkl", "wb") as f:
    pickle.dump(ts, f)

# the receiving side can restore it as a GWpy base object even without GWexpy
```

:::{important}
Only load Pickle data from trusted sources.
:::

Related pages:

- [GWpy Difference API Index](../reference/gwpy_added_api.md)
- [Installation Guide](../tutorials/installation.md)

## Recipe 6: Bridge to python-control

In GWpy-based code, using `control.bode()` or similar functions requires manually extracting a `FrequencySeries` value as a numpy array and passing it to `control.frd()`.  
In GWexpy, `to_control_frd()` handles the conversion in one call, removing that boilerplate.

### GWpy style

```python
import control
import numpy as np
from gwpy.frequencyseries import FrequencySeries

spec = FrequencySeries.read("data.hdf5", "H1:STRAIN_ASD")

# must manually extract numpy arrays before building the FRD object
omega = 2 * np.pi * np.asarray(spec.frequencies.value)
frd_sys = control.frd(np.asarray(spec.value), omega)

# draw Bode plot
control.bode(frd_sys)
```

### GWexpy style

```python
import control
from gwexpy.frequencyseries import FrequencySeries

spec = FrequencySeries.read("data.hdf5", "H1:STRAIN_ASD")

# one-line conversion and round-trip
frd_sys = spec.to_control_frd()                      # FrequencySeries → control.FrequencyResponseData
control.bode(frd_sys)                                # pass directly to Bode / Nichols / Nyquist

spec2 = FrequencySeries.from_control_frd(frd_sys)    # control.FRD → FrequencySeries
```

This difference matters when:

- you want to treat an ASD / PSD spectrum directly as a control-system FRD model
- you want to skip manual numpy extraction before passing data to `control.bode()` / `control.nichols()` / `control.nyquist()`
- you want to restore a processed FRD back to a `FrequencySeries` and feed it into an existing plot or statistics pipeline
- you want to convert each channel in a multi-channel spectrum with `FrequencySeriesDict.to_control_frd()`

Related pages:

- [Interop / Conversion Guide](../how-to/interop.md)
- [python-control API Reference](../reference/api/gwexpy.interop.control_)
- [Active Damping Tutorial](../how-to/case-studies/case_active_damping)
- [Frequency Series Tutorial](../tutorials/intro_frequencyseries)

## Where GWexpy sits relative to other GWpy-adjacent packages

GWexpy is not the only package that extends GWpy, and moving to it does not mean leaving the
rest of the ecosystem behind. spicypy adds signal-processing and control-systems methods,
GWDama organises data into HDF5 datasets for later analysis, and the detector-characterization
tools build operator-facing workflows on top of GWpy. GWexpy occupies the container, I/O, and
analysis-primitive layer between GWpy and those workflows, so the usual pattern is to combine
them rather than to choose one.

For the full comparison, including which projects GWexpy bridges to and which it deliberately
leaves out of scope, see
[Where GWexpy Sits in the GW Python Ecosystem](ecosystem).

## Treat direct I/O and external-library conversion as separate guides

This page intentionally does not duplicate the I/O-format list or the external-library conversion list.  
Use the dedicated guides below as the source of truth:

- Direct I/O: [File I/O Supported Formats Guide](../how-to/io_formats.md)
- External-library conversion: [Interop / Conversion Guide](../how-to/interop.md)

## Next Steps

- [GWpy Difference API Index](../reference/gwpy_added_api.md) - look up added APIs from a difference-oriented view
- [Field API Intro](../how-to/containers/field_scalar_intro) - see how `ScalarField`, `FieldList`, and `FieldDict` fit together
- [Tutorial Index](../tutorials/index.md) - move from migration recipes into worked examples
- [File I/O Supported Formats Guide](../how-to/io_formats.md) - check supported read/write formats
- [Interop / Conversion Guide](../how-to/interop.md) - check bridges to external libraries
- [API Reference](../reference/index.md) - inspect the full API surface

# From scientific Python to GWexpy

Turn NumPy arrays into objects that carry sampling, start time, and units, then analyze a dictionary of channels with one collection method.
Prerequisites: NumPy arrays, Python dictionaries, and basic plotting; no GWpy knowledge is assumed.
First complete [Installation](installation.md). The examples use core dependencies and synthetic data.
Study goal: 10–15 minutes. The small examples are intended to execute within seconds on a laptop, depending on the environment.

Run the Python blocks below in order in a notebook, or place them in a single `.py` file and run it with Python.
They save figures in the current working directory.

## Before: an array and separate metadata

A NumPy array stores samples. A spectral calculation also needs the sample rate, while plot labels and time selection need units and a time origin.
Here those pieces are separate variables:

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

fs = 512
t0 = 1400000000
unit = "V"
time = np.arange(16 * fs) / fs
rng = np.random.default_rng(10)
values = np.sin(2 * np.pi * 40 * time) + rng.normal(0, 0.3, len(time))

frequency, psd = welch(
    values, fs=fs, nperseg=2 * fs, noverlap=fs, window="hann",
    detrend="constant", scaling="density", average="mean",
)
fig, ax = plt.subplots()
ax.loglog(frequency[1:], np.sqrt(psd[1:]))
ax.set(xlabel="Frequency [Hz]", ylabel=r"ASD [V/$\sqrt{\mathrm{Hz}}$]")
fig.savefig("numpy-asd.png")
plt.close(fig)
```

`welch` returns a power spectral density; its square root is the ASD plotted here.
The code labels the plot explicitly because `values` itself carries no unit.

## After: a TimeSeries

Wrap the same samples with the metadata needed to interpret them:

```python
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

series = TimeSeries(
    values, sample_rate=fs, t0=t0, unit=unit, name="Sensor A"
)
spectrum = series.asd(fftlength=2, overlap=1, window="hann", method="welch")
plot = spectrum.plot(xlim=(1, 256))
plot.savefig("gwexpy-asd.png")
print(series.sample_rate, series.t0, series.unit)
print(spectrum.df, spectrum.unit)
```

The input array is the same. `series` now stores `sample_rate`, `t0`, and `unit`, and the returned spectrum carries its frequency axis and ASD unit.
The printed frequency spacing is 0.5 Hz and the spectral unit is V per square root Hz.
GWexpy's `fftlength` and `overlap` are durations in seconds; the corresponding SciPy `nperseg` and `noverlap` above are sample counts.

`t0` gives the GPS timestamp of the first sample.
For example, `series.crop(t0 + 2, t0 + 6)` selects four seconds beginning two seconds after that timestamp.
The sample at the right boundary is excluded.

## Before: a dictionary and a spectral loop

A multi-channel NumPy workflow often keeps arrays in a dictionary and repeats the spectral calculation:

```python
rng_b = np.random.default_rng(20)
arrays = {
    "Sensor A": values,
    "Sensor B": np.sin(2 * np.pi * 40 * time)
    + rng_b.normal(0, 0.8, len(time)),
}
numpy_spectra = {}
for name, data in arrays.items():
    f, power = welch(
        data, fs=fs, nperseg=2 * fs, noverlap=fs, window="hann",
        detrend="constant", scaling="density", average="mean",
    )
    numpy_spectra[name] = (f, np.sqrt(power))
```

## After: a TimeSeriesDict

Attach metadata once when constructing each channel, then apply the spectral settings to the collection:

```python
channels = TimeSeriesDict({
    name: TimeSeries(data, sample_rate=fs, t0=t0, unit=unit, name=name)
    for name, data in arrays.items()
})
spectra = channels.asd(fftlength=2, overlap=1, window="hann", method="welch")
plot = spectra.plot(xlim=(1, 256))
plot.gca().legend()
plot.savefig("channels-asd.png")
first_channel = channels["Sensor A"]
plain_values = first_channel.value
```

`channels["Sensor A"]` selects a `TimeSeries`; `spectra["Sensor A"]` selects its `FrequencySeries`.
The ASD figure should show the shared 40 Hz tone and the higher noise floor in Sensor B.
All channels in this example have the same sample rate, start time, duration, and unit.
When wrapping real arrays, supply the actual metadata for each channel.

`.value` exposes the sample array for a function that expects NumPy.
Pass time and unit information separately when that external function needs it.
For conversions to other scientific objects, use the [interoperability guide](../how-to/interop.md).

## Continue from the collection

Use [TimeSeries basics](intro_timeseries.ipynb) for creation, metadata, time selection, and plotting, or [TimeSeriesMatrix basics](matrix_timeseries.ipynb) when you want a matrix representation of aligned channels.
For a file-based workflow with saved settings, follow [Commissioner workflow](commissioner.md).

## Further reading

- [Interoperability capabilities](../reference/interop_capabilities.md)
- [API reference](../reference/index.md)
- [Start Here](getting_started.md)

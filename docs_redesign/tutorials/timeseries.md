---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# TimeSeries: Basics

```{note}
This page is rendered as a **notebook** by `myst-nb`. Use the launch (rocket)
button at the top-right to open it in Colab or Binder, or the download button
to grab the source. Execution is turned off for this prototype, so the outputs
below are illustrative.
```

`gwexpy` extends GWpy's `TimeSeries` with signal processing, statistics and a
large interop surface — while staying compatible with the GWpy you already know.

## 1. Environment setup

We import the libraries and synthesise two toy channels: a stable calibration
line and a drifting chirp.

```{code-cell} ipython3
import numpy as np
from gwexpy.timeseries import TimeSeries
from gwexpy.noise.wave import sine, gaussian, chirp, exponential

fs, duration = 100, 5.0

# Sensor 1: a stable 10 Hz line plus measurement noise.
ts1 = sine(duration=duration, sample_rate=fs, frequency=10, amplitude=1.0)
ts1 += gaussian(duration=duration, sample_rate=fs, std=0.2)
ts1.name = "Sensor 1"

# Sensor 2: a chirp with a growing envelope.
ts2 = chirp(duration=duration, sample_rate=fs, f0=5, f1=25, t1=duration)
ts2 *= exponential(duration=duration, sample_rate=fs, tau=2.0,
                   decay=False, amplitude=0.2)
ts2.name = "Chirp"
```

## 2. Hilbert transform & envelope

```{code-cell} ipython3
analytic = ts2.hilbert()
envelope = ts2.envelope()
envelope.plot()
```

```{code-cell} ipython3
:tags: [remove-input]

print("AnalyticSignal(name='Chirp', length=500, dtype=complex128)")
```

## 3. Instantaneous frequency

The phase slope of the analytic signal tracks the drifting line in time.

```{code-cell} ipython3
freq = ts2.instantaneous_frequency()
freq.plot(ylabel="Frequency [Hz]")
```

## 4. Lock-in demodulation

Averaging against a phase-matched reference suppresses broadband noise while
preserving the coherent 10 Hz component.

```{code-cell} ipython3
amp, phase = ts1.lock_in(f0=10, stride=0.1)
amp.plot(ylabel="Amplitude")
```

## 5. ARIMA forecasting

```{code-cell} ipython3
model = ts1.fit_arima(order=(1, 0, 0))
forecast, conf = model.forecast(steps=30)
forecast.plot()
```

## 6. Interoperability

Conversions to and from the wider data-science ecosystem are one call each.

```{code-cell} ipython3
df = ts1.to_pandas(index="datetime")     # -> pandas.Series
xr = ts1.to_xarray()                      # -> xarray.DataArray
tt = ts1.to_torch()                       # -> torch.Tensor
print(type(df), type(xr), type(tt))
```

```{code-cell} ipython3
:tags: [remove-input]

print("<class 'pandas.core.series.Series'> "
      "<class 'xarray.core.dataarray.DataArray'> "
      "<class 'torch.Tensor'>")
```

## Summary

You met the core `TimeSeries` enhancements: Hilbert-based signal processing,
demodulation, ARIMA, and frictionless interop. Next, explore
{doc}`../reference/containers` for the multi-dimensional data model.

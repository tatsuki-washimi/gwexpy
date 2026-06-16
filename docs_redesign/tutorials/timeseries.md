# Your first time-series matrix

In this lesson you will create a multi-channel time series, inspect it, and
compute a power spectral density. It should take about ten minutes.

## What you will build

A small `TimeSeriesMatrix` holding three synthetic channels, then a one-line
PSD estimate over all of them.

## Step 1 — Create some data

GWexpy containers wrap a 2-D array where the first axis is the channel and the
second axis is time.

```python
import numpy as np
from gwexpy_demo import TimeSeriesMatrix

rng = np.random.default_rng(0)
data = rng.normal(size=(3, 1024))     # 3 channels, 1024 samples
tsm = TimeSeriesMatrix(data, sample_rate=512.0)
```

## Step 2 — Inspect it

Every container exposes a few descriptive properties.

```python
print(tsm.n_channels)   # 3
print(tsm.n_samples)    # 1024
print(tsm.duration)     # 2.0  (seconds)
```

```{note}
`duration` is derived from `n_samples / sample_rate`. The container never
stores it directly, so it always stays consistent with the data.
```

## Step 3 — Estimate a PSD

The `psd` method returns one spectrum per channel.

```python
spectra = tsm.psd(nperseg=256)
print(spectra.shape)    # (3, 129)
```

## Step 4 — Combine channels

You can sum channels in quadrature to get a quick combined trace.

```python
from gwexpy_demo import combine_channels

combined = combine_channels(tsm)
print(combined.shape)   # (1024,)
```

## Recap

You created a `TimeSeriesMatrix`, read its descriptive properties, computed a
per-channel PSD, and combined channels. Next, explore the
{doc}`how-to guides <../how-to/index>` for task-focused recipes.

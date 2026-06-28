# Your first time-series matrix

In this lesson you will load a multi-channel segment into a
`TimeSeriesMatrix`, inspect it, and produce a frequency-domain view. By the
end you will understand the core container that the rest of GWexpy builds on.

:::{admonition} What you will learn
:class: tip

- How to construct and read a `TimeSeriesMatrix`
- How channels and the shared time axis work
- How to transform to the frequency domain with `.fft()`
:::

## Step 1 — Create a matrix

A `TimeSeriesMatrix` holds several synchronous channels that share one time
axis. Start by building a small one by hand:

```python
import numpy as np
from gwexpy.timeseries import TimeSeriesMatrix

t = np.arange(0, 1, 1 / 4096)
data = np.vstack([np.sin(2 * np.pi * 60 * t), np.cos(2 * np.pi * 60 * t)])

tsm = TimeSeriesMatrix(data, sample_rate=4096.0, channels=["sine", "cosine"])
print(tsm.n_channels)  # -> 2
```

## Step 2 — Inspect the channels

Every channel shares the sample rate and length, so operations apply across
the whole matrix at once:

```python
print(tsm.sample_rate)   # 4096.0
print(tsm.channels)      # ['sine', 'cosine']
```

## Step 3 — Move to the frequency domain

The `.fft()` method returns a `FrequencySeriesMatrix` with the same channel
layout:

```python
fsm = tsm.fft()
coh = fsm.coherence(fsm)
```

:::{admonition} Next steps
:class: seealso

You now have the core workflow. Continue with the
[noise-budget case study](../how-to/case-studies/noise_budget.md), or browse
the [container reference](../reference/index.md).
:::

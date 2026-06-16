# How to build a noise budget

A noise budget decomposes a measured spectrum into the individual sources that
contribute to it, so you can see which one limits your sensitivity.

## Goal

Given a `TimeSeriesMatrix` of witness channels and a target strain channel,
estimate each witness's projected contribution.

## Steps

1. Load the witness channels and the target.

   ```python
   import numpy as np
   from gwexpy_demo import TimeSeriesMatrix

   witnesses = TimeSeriesMatrix(np.random.normal(size=(4, 4096)),
                                sample_rate=1024.0)
   ```

2. Estimate per-channel spectra.

   ```python
   spectra = witnesses.psd(nperseg=512)
   ```

3. Project each witness onto the target and sum the contributions.

   ```python
   from gwexpy_demo import combine_channels

   total = combine_channels(witnesses)
   ```

4. Plot each projection against the measured strain to read off the limiting
   source.

```{tip}
Keep the same `nperseg` for every channel so the spectra share a frequency
grid and can be compared directly.
```

## Result

You now have a stacked set of projections that, summed, should approximate the
measured spectrum. Gaps between the sum and the measurement point to noise
sources you have not yet witnessed.

```{image} ../../_static/images/case_noise_budget_thumb.png
:alt: Example noise budget
:width: 70%
:align: center
```

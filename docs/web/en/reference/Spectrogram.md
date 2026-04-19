# Spectrogram

<!-- reference-summary:start -->

**Stability:** Stable

## What it is

Use `Spectrogram` for one time-frequency map with GWexpy analysis, plotting, and conversion helpers.

## Representative Signatures

```python
Spectrogram(data, t0=None, dt=None, f0=None, df=None, ...)
Spectrogram.percentile(q, axis="time")
```

## Minimal Example

```python
from gwexpy.spectrogram import Spectrogram
import numpy as np

sgm = Spectrogram(np.random.randn(16, 32), dt=1.0, df=1.0)
med = sgm.percentile(50, axis="time")
```

## Related Theory

- [FFT_Conventions](FFT_Conventions.md)
- [Spectrogram Tutorial](../user_guide/tutorials/intro_spectrogram.ipynb)
- [Time-Frequency Comparison Guide](../user_guide/tutorials/time_frequency_comparison.md)

## Related Tutorials

- [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_en.md)
- [Spectrogram Tutorial](../user_guide/tutorials/intro_spectrogram.ipynb)
- [Glitch Analysis](../user_guide/tutorials/case_glitch_analysis.ipynb)
- [HHT Analysis](../user_guide/tutorials/advanced_hht.ipynb)

## API Reference

The detailed generated API continues below on this page.

<!-- reference-summary:end -->


**Inherits from:** [`gwpy.spectrogram.Spectrogram`](https://gwpy.readthedocs.io/en/latest/api/gwpy.spectrogram.Spectrogram/)

Extended Spectrogram with gwexpy analysis and visualization helpers.

See {doc}`api/spectrogram` for the API reference.

## Physical Context

`Spectrogram` represents a **single-channel time-frequency map**. Use it when you need both time and frequency structure at once: nonstationary noise, short bursts, drifting spectral lines, control-line turn-on, glitch evolution, or chirp-like features.

- **Meaning of the two axes**: `t0` / `dt` define time bins, and `f0` / `df` define frequency bins. One pixel means "the strength of one frequency interval within one time window."
- **Dependence on upstream transforms**: a `Spectrogram` usually comes from `TimeSeries.spectrogram()`, `spectrogram2()`, `q_transform()`, or `hht(..., output="spectrogram")`. Window length, overlap, and transform family therefore directly affect the visual structure.
- **Unit semantics**: pixel values may represent power, ASD-like magnitude, normalized intensity, or phase-like quantities. You have to fix what the color scale encodes before drawing physical conclusions.

## Analysis Notes

### Time-frequency resolution is always a trade-off

Spectrograms are useful because they localize features in time and frequency, but they do not give arbitrarily high resolution in both at once.

- shorter windows improve time localization but blur frequency structure
- longer windows sharpen narrow lines but smear short transients
- Q transforms and HHT expose the trade-off differently from standard STFT

### Color intensity is not automatically physical amplitude

Plots often apply log scaling, normalization, clipping, or percentile summaries. Brightness alone is not enough to interpret physical strength.

- check whether the plot uses log normalization or dB scaling
- align the color scale before comparing two figures
- keep percentile/bootstrap summaries distinct from raw instantaneous power

### Common misreadings

1. treating pixel width as the exact physical event duration
2. comparing spectrograms with different scales as if the colormap were equivalent
3. reading transform-induced smearing as frequency drift
4. confusing percentile/bootstrap outputs with raw per-bin intensity

## Where to go next

- upstream time-domain context: [TimeSeries](TimeSeries.md)
- time-frequency trade-offs: [Time-Frequency Comparison Guide](../user_guide/tutorials/time_frequency_comparison.md)
- interactive method comparison: [Interactive Time-Frequency Comparison](../user_guide/tutorials/time_frequency_analysis_comparison.ipynb)
- glitch workflow: [Glitch Analysis](../user_guide/tutorials/case_glitch_analysis.ipynb)
- HHT comparison: [HHT Analysis](../user_guide/tutorials/advanced_hht.ipynb)

## Pickle / shelve portability

:::{warning}
Never unpickle data from untrusted sources. ``pickle``/``shelve`` can execute
arbitrary code on load.

:::
gwexpy pickling prioritizes portability: unpickling returns **GWpy types**
so that loading does not require gwexpy to be installed.

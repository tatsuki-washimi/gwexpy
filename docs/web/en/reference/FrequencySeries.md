# FrequencySeries

<!-- reference-summary:start -->

**Stability:** Stable

## What it is

Use `FrequencySeries` for one frequency-domain spectrum with GWexpy fitting, statistics, filtering, and plotting extensions.

## Representative Signatures

```python
FrequencySeries(data, unit=None, f0=None, df=None, frequencies=None, ...)
FrequencySeries.ifft(...)
```

## Minimal Example

```python
from gwexpy.frequencyseries import FrequencySeries
import numpy as np

fs = FrequencySeries(np.ones(128), df=1.0, unit="V / Hz")
phase = fs.phase()
```

## Related Theory

- [FFT_Conventions](FFT_Conventions.md)
- [FrequencySeries Tutorial](../user_guide/tutorials/intro_frequencyseries.ipynb)
- [Transfer Function Measurement](../user_guide/tutorials/case_transfer_function.ipynb)

## Related Tutorials

- [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_en.md)
- [FrequencySeries Tutorial](../user_guide/tutorials/intro_frequencyseries.ipynb)
- [Noise Budgeting](../user_guide/tutorials/case_noise_budget.ipynb)
- [Advanced Fitting](../user_guide/tutorials/advanced_fitting.ipynb)

## API Reference

The detailed generated API continues below on this page.

<!-- reference-summary:end -->


**Inherits from:** [`gwpy.frequencyseries.FrequencySeries`](https://gwpy.readthedocs.io/en/stable/reference/gwpy.frequencyseries.FrequencySeries/)

Extended FrequencySeries with gwexpy analysis and interop features.

See {doc}`api/frequencyseries` for the API reference.

## Physical Context

`FrequencySeries` represents a **single-channel frequency-domain quantity**. Use it for complex FFT outputs, PSD/ASD estimates, transfer functions, response functions, or any spectrum where each bin corresponds to one frequency interval.

- **Relation to time-domain data**: many `FrequencySeries` objects come from `TimeSeries.fft()`, `psd()`, `asd()`, or `csd()`. That means `df`, window length, and averaging assumptions are inherited from upstream time-domain processing.
- **Unit semantics**: the unit depends on what the spectrum represents. ASD may be `strain / sqrt(Hz)`, PSD may be `strain2 / Hz`, while response functions may carry ratios such as `m / V` or `count / N`.
- **Complex-valued meaning**: `real`, `imag`, `phase()`, and `unwrap_phase()` are not just numerical helpers. They are often the quantities used to interpret delay, resonance, causality, and control-loop phase margin.

## Analysis Notes

### Fix what kind of spectrum you have first

The same container can hold an amplitude spectrum, PSD, ASD, or a complex transfer function. The interpretation changes with that choice.

- use ASD/PSD language when comparing noise floors
- use transfer/response-function language when comparing input-output relations
- if you want to return to a waveform with `ifft()`, check the FFT convention and Hermitian assumptions first

### FFT conventions and normalization still matter

Because the object is already in the frequency domain, normalization mistakes are easy to hide. Before comparing with another code path or a paper figure, check:

1. one-sided vs two-sided spectrum
2. whether window correction is included
3. amplitude quantity vs power quantity
4. what reference is used before converting to dB

Use [FFT_Conventions](FFT_Conventions.md) as the baseline.

### Common misreadings

1. comparing ASD and PSD as if they were the same observable
2. adding or subtracting `to_db()` values as though they were linear amplitudes
3. reading wrapped phase as a physical jump
4. interpreting linewidth or resonance Q without checking `df` and averaging setup

## Where to go next

- frequency-domain conventions: [FFT_Conventions](FFT_Conventions.md)
- time-domain entry point: [TimeSeries](TimeSeries.md)
- fitting and response estimation: [Advanced Fitting](../user_guide/tutorials/advanced_fitting.ipynb)
- practical workflow: [Transfer Function Measurement](../user_guide/tutorials/case_transfer_function.ipynb)
- noise-floor workflow: [Noise Budgeting](../user_guide/tutorials/case_noise_budget.ipynb)

## Pickle / shelve portability

:::{warning}
Never unpickle data from untrusted sources. ``pickle``/``shelve`` can execute
arbitrary code on load.

:::
gwexpy pickling prioritizes portability: unpickling returns **GWpy types**
so that loading does not require gwexpy to be installed.

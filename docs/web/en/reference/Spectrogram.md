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

- [Physics Models](../user_guide/physics_models.md)
- [Validated Algorithms](../user_guide/validated_algorithms.md)
- [FFT_Conventions](FFT_Conventions.md)

## Related Tutorials

- [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_en.md)
- [Tutorial Index](../user_guide/tutorials/index.rst)
- [Getting Started](../user_guide/getting_started.md)

## API Reference

The detailed generated API continues below on this page.

<!-- reference-summary:end -->


**Inherits from:** [`gwpy.spectrogram.Spectrogram`](https://gwpy.readthedocs.io/en/latest/api/gwpy.spectrogram.Spectrogram/)

Extended Spectrogram with gwexpy analysis and visualization helpers.

See {doc}`api/spectrogram` for the API reference.

## Pickle / shelve portability

:::{warning}
Never unpickle data from untrusted sources. ``pickle``/``shelve`` can execute
arbitrary code on load.

:::
gwexpy pickling prioritizes portability: unpickling returns **GWpy types**
so that loading does not require gwexpy to be installed.

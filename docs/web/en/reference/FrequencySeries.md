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


**Inherits from:** [`gwpy.frequencyseries.FrequencySeries`](https://gwpy.readthedocs.io/en/latest/api/gwpy.frequencyseries.FrequencySeries/)

Extended FrequencySeries with gwexpy analysis and interop features.

See {doc}`api/frequencyseries` for the API reference.

## Pickle / shelve portability

:::{warning}
Never unpickle data from untrusted sources. ``pickle``/``shelve`` can execute
arbitrary code on load.

:::
gwexpy pickling prioritizes portability: unpickling returns **GWpy types**
so that loading does not require gwexpy to be installed.

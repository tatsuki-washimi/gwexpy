---
myst:
  html_meta:
    description: "Understand GWexpy numerical stability behavior, including whitening eps handling, safe log plotting, NaN and Inf failure modes, and when users should tune parameters."
---

# Numerical Stability and Precision

:::{note}
**Who should read this page?**
Refer to this detailed guide when:
- You see "holes" or "unusual colors" in your plots caused by `NaN` or `Inf`.
- You are working with a dynamic range that makes a numerical parameter or plotted scale hard to choose.
- You want to deeply understand the numerical behavior of algorithms and tune parameters like `eps` or `tol`.
:::

`gwexpy` provides selected stabilization utilities for operations such as whitening and logarithmic conversion. Their behavior and limits depend on the API used.

(numerical-stability-en-tldr)=
## TL;DR

- For normal analysis, start by trusting the default `gwexpy` settings.
- Do not add manual offsets such as `+ 1e-20` before plotting unless you have a concrete reason.
- Tune parameters only when you actually observe `NaN` / `Inf`, work with extreme amplitudes, or need algorithm-level validation.

(numerical-stability-en-impact)=
## Impact of Stabilization (Before & After)

A comparison between standard methods (simple `log10` or fixed `eps`) and `gwexpy`'s robust numerical stabilization algorithms.

![Numerical stabilization comparison: Noisy visualization with NaN/Inf artifacts (Left) vs. Clean gravitational-wave signal (Right)](../_static/images/numerical_stability_comparison.png)

| Item | Standard path | GWexpy path |
| :--- | :--- | :--- |
| **Zero Values** | `log10(0)` produces `-inf`, which a caller must handle | `safe_log_scale()` applies a data-dependent floor before converting to dB |
| **Small denominators** | A fixed `eps` can be unsuitable for the data scale | The documented whitening entry points accept `eps="auto"` |
| **Non-finite input** | A numerical operation can be undefined | Individual APIs validate or reject input according to their documented contract |

---

(numerical-stability-en-methods)=
## Core Stabilization Methods and APIs

| Method | Target API | Issues Resolved | Configuration Hint |
| :--- | :--- | :--- | :--- |
| **Adaptive Whitening** | `TimeSeriesMatrix.whiten_channels()`, `whiten_matrix()`, `gwexpy.signal.preprocessing.whiten()` | Small denominators | Use `eps="auto"` where that parameter is supported |
| **Safe Log Conversion** | `gwexpy.numerics.safe_log_scale()` | `-inf` from zero values during dB conversion | Set `dynamic_range_db` for the intended display range |
| **Data-scaled epsilon** | `gwexpy.numerics.safe_epsilon()` | Selecting an epsilon from the data scale | Specify `rel_tol` and `abs_tol` when the defaults do not match the operation |

---

(numerical-stability-en-examples)=
## Detailed Explanations and Examples

### 1. Adaptive Whitening

**Goal:** Avoid signal loss caused by a fixed `eps`.
**Input:** A `TimeSeries` containing very small amplitudes.
**Output:** A whitened series with automatic scaling.

Standard whitening often uses a fixed normalization parameter (`eps`) to prevent division by zero. If this value is too large, micro-signals are lost.

#### ❌ Bad Example: Fixed eps causing signal loss
```python
# A fixed eps of 1e-12 rounds a 1e-21 signal to zero
whitened = data / (asd + 1e-12) 
```

#### ✅ Good Example: GWexpy's `eps="auto"`
`gwexpy` dynamically scales `eps` relative to the data range and uses a `SAFE_FLOOR` (1e-50) for singularities. The adaptive `eps` is available on the channel-whitening entry points: `TimeSeriesMatrix.whiten_channels()`, the functional `whiten_matrix()`, and `gwexpy.signal.preprocessing.whiten()`.

```python
from gwexpy.timeseries import TimeSeriesMatrix
import numpy as np

tsm = TimeSeriesMatrix(np.random.randn(3, 1, 1000) * 1e-21, sample_rate=1024)
whitened, model = tsm.whiten_channels(eps="auto")  # adaptive eps (returns matrix + model)
```

### 2. Safe Logarithmic Scaling (Safe Log)

**Goal:** Prevent `-inf` values and broken plots when zeros are present.
**Input:** ASD/PSD-like data with zeros or very quiet regions.
**Output:** A stable visualization with a dynamic floor.

Prevents `-inf` values when visualizing spectrograms or PSDs containing zeros or quiet regions.

#### ❌ Bad Example: Numerical errors via manual conversion
```python
asd_db = 10 * np.log10(asd)  # Zeros become -inf, breaking the plot
```

#### ✅ Good Example: Explicit safe-log conversion
`safe_log_scale()` calculates a floor from the finite data maximum and the requested dynamic range, then returns decibels.

```python
from gwexpy.numerics import safe_log_scale

asd_db = safe_log_scale(asd.value, dynamic_range_db=120.0)
```

---

(numerical-stability-en-recommendations)=
## Recommendations for Users

- **Use an explicit floor for dB conversion**: Call `safe_log_scale()` when converting data that may contain zeros or quiet regions.
- **Choose whitening parameters at the API boundary**: Use `eps="auto"` only on the whitening entry points that document it, then inspect the result for the scientific application.
- **Check the API contract**: Validation, finite-value handling, and tolerances are implemented per operation rather than by a single global policy.

## Next to Read

- [Signal Processing API Reference](../reference/api/signal)
- [Validated Algorithms](validated_algorithms.md)
- [API Reference](../reference/index.md) — Entry point to all API pages
- [Prerequisites and Conventions](prerequisites_and_conventions.md) — Shared FFT and numerical assumptions across the docs

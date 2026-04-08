# Numerical Stability and Precision

> [!NOTE]
> **Who should read this page?**
> Standard analysis in `gwexpy` works out-of-the-box with high stability. Refer to this detailed guide only if:
> - You see "holes" or "unusual colors" in your plots caused by `NaN` or `Inf`.
> - You are working with extremely small signals (below $10^{-23}$) or huge signals (above 1) simultaneously.
> - You want to deeply understand the numerical behavior of algorithms and tune parameters like `eps` or `tol`.

`gwexpy` is designed to handle data with an extremely wide dynamic range without numerical breakdown.

## Impact of Stabilization (Before & After)

A comparison between standard methods (simple `log10` or fixed `eps`) and `gwexpy`'s robust numerical stabilization algorithms.

![Numerical stabilization comparison: Noisy visualization with NaN/Inf artifacts (Left) vs. Clean gravitational-wave signal (Right)](../../../_static/images/numerical_stability_comparison.png)

| Item | Standard Implementation (Before) | GWexpy (After) |
| :--- | :--- | :--- |
| **Zero Values** | `log10(0)` produces `-inf`, causing blank holes in plots | **:term:`Safe Log`** automatically sets an optimal floor based on the max value |
| **Micro-signals** | Rounded to zero by fixed `eps=1e-12`, causing signal loss | **:term:`Adaptive Whitening`** (`eps="auto"`) maintains signal sensitivity |
| **Failures** | `NaN` propagates, making the entire dataset uncomputable | Pre/Post-computation validation protects data integrity |

---

## Core Stabilization Methods and APIs

| Method | Target API | Issues Resolved | Configuration Hint |
| :--- | :--- | :--- | :--- |
| **:term:`Adaptive Whitening`** | `.whiten()` | Zero-division / Signal loss | `eps="auto"` is recommended |
| **:term:`Safe Log`** | `.plot()`, `.spectrogram()` | `-inf` holes in plots | Adjustable via `dynamic_range` |
| **Internal Standardization** | `ica_fit()` | Non-convergence | Works regardless of input amplitude |
| **Relative Tolerance** | Various | Early termination | Auto-scales `tol` based on variance |

---

## Detailed Explanations and Examples

### 1. :term:`Adaptive Whitening`

Standard whitening often uses a fixed normalization parameter (`eps`) to prevent division by zero. If this value is too large, micro-signals are lost.

#### ❌ Bad Example: Fixed eps causing signal loss
```python
# A fixed eps of 1e-12 rounds a 1e-21 signal to zero
whitened = data / (asd + 1e-12) 
```

#### ✅ Good Example: GWexpy's `eps="auto"`
`gwexpy` dynamically scales `eps` relative to the data range and uses a `SAFE_FLOOR` (1e-50) for singularities.

```python
from gwexpy.timeseries import TimeSeries
import numpy as np

data = TimeSeries(np.random.randn(1000) * 1e-21, sample_rate=1024)
whitened = data.whiten(eps="auto")  # Automatically applies appropriate scaling
```

### 2. Safe Logarithmic Scaling (:term:`Safe Log`)

Prevents `-inf` values when visualizing spectrograms or PSDs containing zeros or quiet regions.

#### ❌ Bad Example: Numerical errors via manual conversion
```python
asd_db = 10 * np.log10(asd)  # Zeros become -inf, breaking the plot
```

#### ✅ Good Example: Automatic dynamic floor
`gwexpy` calculates a safe floor based on the maximum value in the data.

```python
asd = data.asd()
plot = asd.plot()  # Safe Log is applied internally for a clean visualization
```

---

## Recommendations for Users

- **Avoid Manual Offsets**: Do not add arbitrary small values like `data + 1e-20` before plotting. `gwexpy` handles this internally.
- **Trust the Defaults**: Default parameters for `whiten()` and `ica_fit()` are tuned for numerical safety.
- **Check Warnings**: `gwexpy` issues informative warnings with suggested fixes for truly unstable operations.

## Related Documents

- {doc}`../reference/api/signal` — Signal processing API reference
- {doc}`validated_algorithms` — List of validated algorithms
- {doc}`glossary` — Definitions for :term:`NaN/Inf propagation` and more

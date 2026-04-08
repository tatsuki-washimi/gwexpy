# Numerical Stability and Precision

`gwexpy` is designed to handle data with an extremely wide dynamic range, which is common in gravitational-wave data analysis.
While gravitational-wave strain signals are typically on the order of $10^{-21}$, intermediate processing may involve values close to 1.

To ensure scientific accuracy and prevent numerical artifacts such as signal loss or :term:`NaN/Inf propagation`, `gwexpy` implements robust numerical stabilization strategies.

## TL;DR: Why Numerical Stability Matters

- **Prevention of Death Floats (NaN/Inf)**: Avoids computational failures caused by division by zero or log of zero.
- **Micro-signal Protection**: Prevents gravitational-wave signals ($10^{-21}$) from being rounded to zero by rigid `eps` parameters.
- **Visual Improvement**: Automatically adjusts dynamic range during plotting to reveal signal details.

### Impact of Stabilization on Visualization

![Spectral stabilization comparison showing typical NaN artifacts vs clean GWexpy output](/home/washimi/.gemini/antigravity/brain/389da455-0c02-483f-928d-e8f3db2746b8/spectral_stabilization_comparison_1775634367572.png)

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

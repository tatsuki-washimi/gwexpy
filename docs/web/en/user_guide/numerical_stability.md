---
myst:
  html_meta:
    description: "Understand GWexpy numerical stability behavior, including whitening eps handling, safe log plotting, NaN and Inf failure modes, and when users should tune parameters."
---

# Numerical Stability and Precision

**Page role:** Guide

:::{note}
**Who should read this page?**
Standard analysis in `gwexpy` works out-of-the-box with high stability. Refer to this detailed guide only if:
- You see "holes" or "unusual colors" in your plots caused by `NaN` or `Inf`.
- You are working with extremely small signals (below $10^{-23}$) or huge signals (above 1) simultaneously.
- You want to deeply understand the numerical behavior of algorithms and tune parameters like `eps` or `tol`.
:::

`gwexpy` is designed to handle data with an extremely wide dynamic range without numerical breakdown.

**Search hints:** `numerical stability`, `NaN`, `Inf`, `whiten`, `eps`, `safe log`, `tol`

## At a Glance

| Item | Details |
| --- | --- |
| **Page Role** | Guide |
| **Audience** | Users seeing `NaN` / `Inf` issues, or users working with extremely small and large amplitudes in the same workflow |
| **Prerequisites** | Basic familiarity with FFTs, ASD/PSD plots, and whitening |
| **Use Cases** | Diagnose broken plots, understand when to tune `eps` or `tol`, or review GWexpy's stabilization design |
| **Search Keywords** | numerical stability, `NaN`, `Inf`, `whiten`, `eps`, safe log, `tol` |

## On This Page

- [TL;DR](#tldr)
- [Impact of Stabilization (Before & After)](#impact-of-stabilization-before--after)
- [Core Stabilization Methods and APIs](#core-stabilization-methods-and-apis)
- [Detailed Explanations and Examples](#detailed-explanations-and-examples)
- [Recommendations for Users](#recommendations-for-users)

(numerical-stability-en-tldr)=
## TL;DR

- For normal analysis, start by trusting the default `gwexpy` settings.
- Do not add manual offsets such as `+ 1e-20` before plotting unless you have a concrete reason.
- Tune parameters only when you actually observe `NaN` / `Inf`, work with extreme amplitudes, or need algorithm-level validation.

(numerical-stability-en-impact)=
## Impact of Stabilization (Before & After)

A comparison between standard methods (simple `log10` or fixed `eps`) and `gwexpy`'s robust numerical stabilization algorithms.

![Numerical stabilization comparison: Noisy visualization with NaN/Inf artifacts (Left) vs. Clean gravitational-wave signal (Right)](../../../_static/images/numerical_stability_comparison.png)

| Item | Standard path | GWexpy path |
| :--- | :--- | :--- |
| **Zero Values** | `log10(0)` produces `-inf`, causing blank holes in plots | **Safe Log** automatically sets an optimal floor based on the max value |
| **Micro-signals** | Rounded to zero by fixed `eps=1e-12`, causing signal loss | **Adaptive Whitening** (`eps="auto"`) maintains signal sensitivity |
| **Failures** | `NaN` propagates, making the entire dataset uncomputable | Pre/Post-computation validation protects data integrity |

---

(numerical-stability-en-methods)=
## Core Stabilization Methods and APIs

| Method | Target API | Issues Resolved | Configuration Hint |
| :--- | :--- | :--- | :--- |
| **Adaptive Whitening** | `.whiten()` | Zero-division / Signal loss | `eps="auto"` is recommended |
| **Safe Log** | `.plot()`, `.spectrogram()` | `-inf` holes in plots | Adjustable via `dynamic_range` |
| **Internal Standardization** | `ica_fit()` | Non-convergence | Works regardless of input amplitude |
| **Relative Tolerance** | Various | Early termination | Auto-scales `tol` based on variance |

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
`gwexpy` dynamically scales `eps` relative to the data range and uses a `SAFE_FLOOR` (1e-50) for singularities.

```python
from gwexpy.timeseries import TimeSeries
import numpy as np

data = TimeSeries(np.random.randn(1000) * 1e-21, sample_rate=1024)
whitened = data.whiten(eps="auto")  # Automatically applies appropriate scaling
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

#### ✅ Good Example: Automatic dynamic floor
`gwexpy` calculates a safe floor based on the maximum value in the data.

```python
asd = data.asd()
plot = asd.plot()  # Safe Log is applied internally for a clean visualization
```

---

(numerical-stability-en-recommendations)=
## Recommendations for Users

- **Avoid Manual Offsets**: Do not add arbitrary small values like `data + 1e-20` before plotting. `gwexpy` handles this internally.
- **Trust the Defaults**: Default parameters for `whiten()` and `ica_fit()` are tuned for numerical safety.
- **Check Warnings**: `gwexpy` issues informative warnings with suggested fixes for truly unstable operations.

## Next to Read

- [Signal Processing API Reference](../reference/api/signal.rst)
- [Validated Algorithms](validated_algorithms.md)
- [Glossary](glossary.rst) — Definitions for `NaN/Inf propagation` and more
- [Prerequisites and Conventions](prerequisites_and_conventions.md) — Shared FFT and numerical assumptions across the docs

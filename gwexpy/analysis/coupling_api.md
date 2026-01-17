# Coupling Threshold API Reference

This document describes the thresholding strategies used in Coupling Function (CF) analysis within `gwexpy.analysis.coupling`. These classes determine whether an injection (witness) signal has successfully "excited" a target channel or the witness channel itself, distinguishing it from background noise.

## Overview

In coupling function analysis, we compare the Power Spectral Density (PSD) of an "injection" period ($P_{inj}$) against a "background" period ($P_{bkg}$). A **Threshold Strategy** defines the criteria for identifying significant excess power.

Different physical and statistical assumptions require different strategies. Choosing the wrong strategy can lead to false positives (overestimating coupling) or false negatives (missing actual coupling).

## Threshold Strategy Summary

| Class | Statistical Assumption | Required Inputs | Robust to Non-Gaussian? | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **`RatioThreshold`** | None (Scalar ratio) | PSDs only | Medium | Simple physical screening; fast computation. |
| **`SigmaThreshold`** | **Gaussian** (Normal) Distribution | PSDs + `n_avg` | **Low** | Well-behaved, stationary noise; statistical significance testing. |
| **`PercentileThreshold`** | **Empirical** Distribution | **Raw Background Time Series** | **High** | Real-world data with glitches, outliers, or non-stationary noise. |

---

## Class Details

### 1. `RatioThreshold`

**Logic**: $P_{inj} > \text{ratio} \times P_{bkg}$

A simple, heuristic approach thatchecks if the injection power is simply $N$ times larger than the background mean. It does not account for the variance of the background.

* **Pros**: Extremely fast; easy to interpret physically ("power doubled").
* **Cons**: Does not provide statistical confidence levels (e.g., "3 sigma").

### 2. `SigmaThreshold`

**Logic**: $P_{inj} > P_{bkg} + \sigma \times \text{std}(P_{bkg})$

This is a **statistical significance test**. It assumes that the background noise power in each frequency bin follows a Gaussian distribution. It uses the number of averages ($N_{avg}$) from the spectral estimation (e.g., Welch's method) to estimate the standard error of the mean.

$$ \text{Threshold} = \mu_{bkg} \left( 1 + \frac{\sigma}{\sqrt{N_{avg}}} \right) $$

* **Assumption**: Background is stationary and Gaussian.
* **Pros**: Provides a standard statistical metric (e.g., $3\sigma$ confidence).
* **Cons**: **Unreliable** if the noise is non-Gaussian (e.g., glitchy) or if $N_{avg}$ is small. It can underestimate the noise floor variance in these cases.

### 3. `PercentileThreshold`

**Logic**: $P_{inj} > \text{factor} \times \text{Percentile}(P_{bkg}(t))$

This strategy computes the **empirical distribution** of the background power at each frequency bin by analyzing the background time series segment-by-segment (spectrogram).

* **Assumption**: The background distribution over time represents the noise variability.
* **Pros**: **Robust**. Effectively handles outliers, glitches, and non-Gaussian tails.
* **Cons**: Slower (requires computing a full spectrogram for the background).

---

## Usage Examples

### Basic Usage with `CouplingFunctionAnalysis`

```python
from gwexpy.analysis import CouplingFunctionAnalysis
from gwexpy.analysis.coupling import RatioThreshold, SigmaThreshold, PercentileThreshold

# 1. Simple Ratio Threshold
# "Injection must be 2x the background power"
thresh_ratio = RatioThreshold(ratio=2.0)

# 2. Sigma Threshold (Gaussian Assumption)
# "Injection must be 5 sigma above the background mean"
# valid only for clean, stationary data.
thresh_sigma = SigmaThreshold(sigma=5.0)

# 3. Percentile Threshold (Robust)
# "Injection must be higher than the 95th percentile of background segments"
# Robust against glitches.
thresh_perc = PercentileThreshold(percentile=95, factor=1.0)

# Apply in analysis
analysis = CouplingFunctionAnalysis()
results = analysis.compute(
    data_inj=inj_data,
    data_bkg=bkg_data,
    fftlength=1.0,
    threshold_witness=thresh_sigma,   # Check if witness is excited (statistically)
    threshold_target=thresh_perc      # Check target with robust threshold
)
```

## Statistical Warnings

* **Do not treat `SigmaThreshold` as a physical upper limit.** It is a significance test. If you set `sigma=3`, you are asking "is this signal 3 standard deviations away from the noise mean?", not "is this the maximum possible coupling?".
* **Check `n_avg`.** `SigmaThreshold` relies on the $1/\sqrt{N}$ reduction of variance. If your data segments are short ($N_{avg} \approx 1$), the Gaussian approximation fails.
* **Glitches.** If your background data contains glitches (transients), `SigmaThreshold` will likely produce false positives because the mean/std are not representative of the tails. Use `PercentileThreshold`.

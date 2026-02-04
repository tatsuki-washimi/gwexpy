# Getting Started

This page provides a **structured learning path** for GWexpy users. Whether you're new to time series analysis or migrating from GWpy, you'll find tailored recommendations for your background.

**What you'll learn:**

- Installation and setup
- Core data structures (TimeSeries, FrequencySeries, Spectrogram)
- Multi-channel analysis and matrix containers
- Advanced signal processing techniques
- Real-world application examples

**Time estimate:** 2-3 hours for beginners, 30-60 minutes for GWpy users

:::{tip}
If you're in a hurry, start with [Quickstart](quickstart.md) for a 5-minute overview.
:::

:::{admonition} About This Page
:class: note

This page provides a **detailed learning roadmap**. If you haven't visited the [documentation homepage](../index) yet, start there to understand GWexpy's overall capabilities.
:::

## Quick Example

Here's the most basic GWexpy workflow:

```python
from gwexpy.timeseries import TimeSeries
import numpy as np

# Create a time series
ts = TimeSeries(np.random.randn(1000), sample_rate=100, t0=0)

# Plot it
plot = ts.plot()
plot.show()
```

For more examples, continue to [Quickstart](quickstart.md).

## Prerequisites

- Basic Python 3.9+ knowledge
- NumPy fundamentals (array operations)
- (Optional) GWpy experience

## Learning Path

### 1. Installation

First, install GWexpy by following [installation](installation.md).

### 2. Quickstart

Learn the basics with [quickstart](quickstart.md).

### 3. Basic Data Structures (Recommended Order)

**For Beginners**

1. [intro_timeseries](tutorials/intro_timeseries.ipynb) - Time series basics
2. [intro_frequencyseries](tutorials/intro_frequencyseries.ipynb) - Frequency series basics
3. [intro_spectrogram](tutorials/intro_spectrogram.ipynb) - Spectrogram basics
4. [intro_plotting](tutorials/intro_plotting.ipynb) - Plotting features

**For GWpy Users**

- Check the GWpy compatibility guide for migration information

### 4. Advanced Topics

**Multi-channel & Matrix Containers**

- [matrix_timeseries](tutorials/matrix_timeseries.ipynb) - Time series matrices
- [matrix_frequencyseries](tutorials/matrix_frequencyseries.ipynb) - Frequency series matrices

**High-dimensional Fields (Field API)**

- [field_scalar_intro](tutorials/field_scalar_intro.ipynb) - ScalarField introduction
- [scalarfield_slicing](scalarfield_slicing.md) - Slicing guide (Important)

**Advanced Signal Processing**

- [advanced_fitting](tutorials/advanced_fitting.ipynb) - Fitting methods
- [advanced_peak_detection](tutorials/advanced_peak_detection.ipynb) - Peak detection
- [advanced_hht](tutorials/advanced_hht.ipynb) - Hilbert-Huang Transform
- [advanced_arima](tutorials/advanced_arima.ipynb) - ARIMA models
- [advanced_correlation](tutorials/advanced_correlation.ipynb) - Correlation analysis

### 5. Practical Examples

Browse the [{doc}`Examples Gallery <../examples/index>`] for real-world applications:

- [case_noise_budget](tutorials/case_noise_budget.ipynb) - Noise budget analysis
- [case_transfer_function](tutorials/case_transfer_function.ipynb) - Transfer function calculation
- [case_active_damping](tutorials/case_active_damping.ipynb) - Active damping

## Next Steps

- [{doc}`Examples Gallery <../examples/index>`] - Visual examples and case studies
- Complete tutorial list: [{doc}`tutorials/index <tutorials/index>`]
- API Reference: [{doc}`Reference <../reference/index>`]
- [validated_algorithms](validated_algorithms.md) - Algorithm validation report

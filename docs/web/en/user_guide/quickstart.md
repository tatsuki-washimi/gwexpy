# Quickstart

Get your first analysis plot with GWexpy as quickly as possible.

## At a Glance

| Item | Details |
| --- | --- |
| **Page Role** | Guide |
| **Audience** | First-time users who want one working plot quickly, and GWpy users checking the basic entry point |
| **Prerequisites** | Python 3.11+, basic `pip` usage, and basic NumPy familiarity |
| **Use Cases** | Confirm installation, run a 3-line example, or decide what to read next |
| **Search Keywords** | quickstart, first plot, `TimeSeries`, CSD, minimal example |

## On This Page

- [Quick Install](#quick-install)
- [3-line Quickstart](#3-line-quickstart)
- [30-min Hands-on (Interactive Tutorial)](#30-min-hands-on-interactive-tutorial)
- [Multi-channel Analysis Example](#multi-channel-analysis-example)
- [Next Steps](#next-steps)

## Quick Install

Since GWexpy is currently in development, please install it directly from GitHub:

```bash
pip install git+https://github.com/tatsuki-washimi/gwexpy.git
```

## 3-line Quickstart

GWexpy's `TimeSeries` can be created directly from NumPy arrays and features built-in plotting capabilities.

- Purpose: display a first plot from a random time series
- Input: a 4096-sample NumPy array, sample rate 4096 Hz, and `t0=0`
- Output: a `TimeSeries` object and a rendered plot

```python
import numpy as np
from gwexpy.timeseries import TimeSeries

ts = TimeSeries(np.random.randn(4096), sample_rate=4096.0, t0=0)
ts.plot().show()
```

## 30-min Hands-on (Interactive Tutorial)

For a more practical workflow, we recommend the following tutorial. You can run it immediately on Google Colab.

### 🧪 GWexpy Basic Hands-on

[See the tutorial index](tutorials/index.rst)

Experience everything from loading data to frequency analysis (ASD/CSD) and matrix manipulation using the latest Field API.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsuki-washimi/gwexpy/blob/main/docs/web/en/user_guide/tutorials/intro_timeseries.ipynb)

## Core Concepts

The two pillars to mastering GWexpy.

* **TimeSeries / FrequencySeries**: 
  Basic classes for handling single-channel data. High compatibility with GWpy allows you to run existing code as is.
* **TimeSeriesMatrix / ScalarField**:
  New APIs for handling multi-channel (matrix format) or multi-dimensional data. Large-scale analysis with over 100 channels can be handled safely with a single line of code.

## Multi-channel Analysis Example

An example of calculating the cross-spectral density (CSD) between multiple channels.

- Purpose: convert multiple channels into a `TimeSeriesMatrix` and compute cross-spectral density
- Input: a two-channel `TimeSeriesDict` and `fftlength=1`
- Output: a `csd` object and a plotted CSD result

```python
import numpy as np
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

# Create data
tsd = TimeSeriesDict({
    "H1:STRAIN": TimeSeries(np.random.randn(4096 * 4), sample_rate=4096, t0=0),
    "L1:STRAIN": TimeSeries(np.random.randn(4096 * 4), sample_rate=4096, t0=0),
})

# Convert to a matrix and calculate Cross Spectral Density (CSD)
csd = tsd.to_matrix().csd(fftlength=1)
csd.plot().show()
```

This example converts a `TimeSeriesDict` into a `TimeSeriesMatrix` and stores the resulting cross-spectral density in `csd`.

## Need Help?

If you encounter errors or plots do not appear, check the [Troubleshooting Guide](troubleshooting.md).

## Next Steps

* [Installation Guide](installation.md) - Setting up your environment.
* [Getting Started](getting_started.md) - Systematic learning roadmap.
* [Prerequisites and Conventions](prerequisites_and_conventions.md) - Review FFT, GPS time, and compatibility assumptions first.
* [Migration from GWpy](gwexpy_for_gwpy_users_en.md) - Difference guide for existing users.

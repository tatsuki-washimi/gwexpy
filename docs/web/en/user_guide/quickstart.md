---
myst:
  html_meta:
    description: "Start GWexpy quickly with a PyPI install command, a 3-line first plot, and pointers to the next tutorials and migration guides."
---

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

- [Quick Install](#en-quickstart-install-command)
- [3-line Quickstart](#en-quick-demo)
- [30-min Hands-on (Interactive Tutorial)](#30-min-hands-on-interactive-tutorial)
- [Multi-channel Analysis Example](#multi-channel-analysis-example)
- [Next Steps](#next-to-read)

(en-quickstart-install-command)=
## Quick Install

Install the core GWexpy package from PyPI:

- Purpose: install the released core package as quickly as possible
- Input: Python 3.11+ with `pip`
- Output: a local GWexpy installation ready for the first example

```bash
pip install gwexpy
```

If you need a Conda-managed environment, GW binary dependencies such as NDS2 / FrameLIB, or optional tools such as `pygmt`, start with the [Installation Guide](installation.md) instead of adding those packages ad hoc after this quick install. Installing the PyPI package inside a Conda environment is supported; the native conda-forge package is still in staged-recipes review, so the public `conda install -c conda-forge gwexpy` command is not documented as available yet.

(en-quick-demo)=
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
  New APIs for handling multi-channel (matrix format) or multi-dimensional field-like data. Large-scale analysis with over 100 channels can be handled safely with a single line of code.

## Multi-channel Analysis Example

Minimal example for calculating cross-spectral density (CSD) between two channels.

- Purpose: compute cross-spectral density from two `TimeSeries` channels
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

# Calculate Cross Spectral Density (CSD) between the two channels
csd = tsd["H1:STRAIN"].csd(tsd["L1:STRAIN"], fftlength=1)
csd.plot().show()
```

This example selects the two channels from the `TimeSeriesDict` and stores the resulting cross-spectral density in `csd`.

## Need Help?

If you encounter errors or plots do not appear, check the [Troubleshooting Guide](troubleshooting.md). If the problem is really an environment setup issue, return to the [Installation Guide](installation.md) for the Conda-first workflow.

<a id="next-to-read"></a>
<a id="next-steps"></a>

## Next to Read

* [Installation Guide](installation.md) - Setting up your environment.
* [Troubleshooting Guide](troubleshooting.md) - Reverse-lookup fixes by symptom.
* [Getting Started](getting_started.md) - Systematic learning roadmap.
* [Prerequisites and Conventions](prerequisites_and_conventions.md) - Review FFT, GPS time, and compatibility assumptions first.
* [Migration from GWpy](gwexpy_for_gwpy_users_en.md) - Difference guide for existing users.

# Quickstart

Get your first analysis plot with GWexpy as quickly as possible.

## Quick Install

Since GWexpy is currently in development, please install it directly from GitHub:

```bash
pip install git+https://github.com/tatsuki-washimi/gwexpy.git
```

## 3-line Quickstart

GWexpy's `TimeSeries` can be created directly from NumPy arrays and features built-in plotting capabilities.

```python
import numpy as np
from gwexpy.timeseries import TimeSeries

ts = TimeSeries(np.random.randn(4096), sample_rate=4096.0, t0=0)
ts.plot().show()
```

## 30-min Hands-on (Interactive Tutorial)

For a more practical workflow, we recommend the following tutorial. You can run it immediately on Google Colab.

.. grid:: 1
    :gutter: 3

    .. grid-item-card:: 🧪 GWexpy Basic Hands-on
        :link: tutorials/index
        :link-type: doc

        Experience everything from loading data to frequency analysis (ASD/CSD) and matrix manipulation using the latest ScalarField API.
        ^^^
        .. image:: https://colab.research.google.com/assets/colab-badge.svg
           :target: https://colab.research.google.com/github/gwexpy/gwexpy/blob/main/docs/web/en/user_guide/tutorials/intro_timeseries.ipynb
           :alt: Open In Colab

## Core Concepts

The two pillars to mastering GWexpy.

* **TimeSeries / FrequencySeries**: 
  Basic classes for handling single-channel data. High compatibility with GWpy allows you to run existing code as is.
* **TimeSeriesMatrix / ScalarField**:
  New APIs for handling multi-channel (matrix format) or multi-dimensional data. Large-scale analysis with over 100 channels can be handled safely with a single line of code.

## Multi-channel Analysis Example

An example of calculating the cross-spectral density (CSD) between multiple channels.

```python
import numpy as np
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

# Create data
tsd = TimeSeriesDict({
    "H1:STRAIN": TimeSeries(np.random.randn(4096 * 4), sample_rate=4096, t0=0),
    "L1:STRAIN": TimeSeries(np.random.randn(4096 * 4), sample_rate=4096, t0=0),
})

# Convert to matrix and calculate Cross Spectral Density (CSD)
asd = tsd.to_matrix().asd(fftlength=1)
asd.plot().show()
```

## Need Help?

If you encounter errors or plots do not appear, check the :doc:`Troubleshooting Guide <troubleshooting>`.

## Next Steps

* :doc:`Installation Guide <installation>` - Setting up your environment.
* :doc:`Getting Started <getting_started>` - Systematic learning roadmap.
* :doc:`Migration from GWpy <gwexpy_for_gwpy_users_en>` - Difference guide for existing users.

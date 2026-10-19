# Graphical User Interface (Standalone pyaggui Project)

## Overview

**Status:** Extracted to standalone project ([pyaggui](https://github.com/tatsuki-washimi/pyaggui))

The legacy in-tree GUI implementation (`gwexpy.gui`) has been removed from GWexpy and transitioned to an independent repository: **pyaggui**.

GWexpy focuses strictly on scientific gravitational-wave analysis, physical data models, and high-performance digital signal processing algorithms. Interactive GUI tools and applications are maintained in standalone companion projects.

## Companion GUI Projects

- **pyaggui**: Standalone interactive diagnostic GUI and plotting application for gravitational-wave data exploration.
  - Repository: [https://github.com/tatsuki-washimi/pyaggui](https://github.com/tatsuki-washimi/pyaggui)
  - Features: Interactive multi-channel time-series inspection, spectral accumulation, spectrogram visualization, and NDS streaming.
- **GWexpy Studio**: Next-generation desktop analytical studio (independent application project).

## Installation and Usage

The `pyaggui` GitHub repository is currently private, and no public PyPI
distribution is available. Repository access is required to install it. For
users with repository access, its README documents this source installation
path; `pip install pyaggui` is not the documented path:

```bash
git clone https://github.com/tatsuki-washimi/pyaggui.git
cd pyaggui
python -m pip install .
pyaggui
```

Please refer to the [pyaggui repository](https://github.com/tatsuki-washimi/pyaggui) for installation requirements, supported desktop environments, and issue reporting.

# Graphical User Interface (GUI, Experimental)

## Overview

**Status:** Experimental, source/development track

GWexpy includes a **PyQt5-based GUI** for interactive data exploration and visualization. However, the GUI should currently be treated as an **experimental, prototype-stage interface**, not as a finalized end-user product. For reproducible and fully supported workflows, the **Python API** remains the primary interface.

The GUI app is **not part of the first PyPI supported surface**. The first PyPI release focuses on the Python library API; GUI stabilization is tracked separately as post-release work.

## Installation

For source checkout or development use, install the GUI dependencies with the `gui` extra:

```bash
pip install "gwexpy[gui] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
```

This installs additional dependencies:
- `PyQt5` - GUI framework
- `pyqtgraph` / `qtpy` - interactive plotting and Qt abstraction
- `sounddevice` - audio-related GUI features

## Launching the GUI

```bash
python -m gwexpy.gui
```

The first PyPI release does not install a `gwexpy.gui` console script.

### Programmatically

```python
from gwexpy.gui.pyaggui import MainWindow
import sys
from PyQt5 import QtWidgets

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
```

## Supported File Formats

The GUI supports loading data in the following formats:

- **GBD** (GRAPHTEC binary)
- **HDF5** (HDF5-based time series)
- **FITS** (Flexible Image Transport System)
- **MiniSEED** (seismic data)
- **Text/CSV** (comma/space-separated)

Open files via:
1. **File → Open** or press `Ctrl+O`
2. Select the data file from your filesystem
3. The data will be loaded and displayed in the main plotting area

## Features

### Data Visualization
- Interactive time series plots
- Spectrogram generation
- Frequency domain analysis
- Multi-channel support

### Data Inspection
- View metadata (sampling rate, units, duration)
- Zoom and pan controls
- Cursor position indicator

### Export
- Save figures as images (PNG, PDF, etc.)
- Export processed data

## Known Limitations

- The GUI is still a prototype-stage feature. Behavior, supported workflows, and screen layout may change without the same compatibility expectations as the core Python API.
- The GUI is optimized for single-file analysis. For batch processing, use the Python API.
- Memory usage increases with file size. For large datasets, consider using the API with streaming options.
- Some advanced analysis features (e.g., matched filtering, machine learning pipelines) are not available in the GUI. Use the Python API for these workflows.

## Troubleshooting

### "ModuleNotFoundError: No module named 'PyQt5'"

Ensure that the GUI extra was installed:

```bash
pip install "gwexpy[gui] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
```

### GUI does not launch

Check that your system has a display (X11 on Linux, native on macOS/Windows):

```bash
export DISPLAY=:0  # Linux/WSL (if needed)
python -m gwexpy.gui
```

### File does not load

Verify that the file exists and is in a supported format. Check the console output for error messages:

```bash
python -m gwexpy.gui 2>&1 | head -20
```

## See Also

- {doc}`Python API Documentation <../index>` - For programmatic data analysis
- {doc}`Tutorials <tutorials/index>` - Interactive examples for learning GWexpy

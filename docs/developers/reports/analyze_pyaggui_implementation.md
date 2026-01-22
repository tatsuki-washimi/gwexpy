# pyaggui Implementation Analysis Report

**Created:** 2026-01-22
**Target:** `gwexpy/gui/pyaggui.py` and related files
**Author:** Antigravity Agent

---

# 1. Overview

`pyaggui` is an application aimed at reproducing and extending the GUI functionality of LIGO Diagnostic Test Tools (DTT), particularly `diaggui`, on Python (`gwexpy`).
This report summarizes the UI structure, functionality, and simulation logic implementation status based on source code analysis, using `gwexpy/gui/pyaggui.py` as the entry point.

# 2. Application Structure

## 2.1 Entry Point (`gwexpy/gui/pyaggui.py`)

* **Role**: Application startup point.
* **Implementation Details**:
  * Argument parsing using `argparse` (allows specifying data file to open at startup).
  * Initialization of `QtWidgets.QApplication` and starting the main loop.
  * Instantiation and display of `MainWindow`.
  * Includes fallback processing between relative and absolute imports, designed to be executable both as a package and as a standalone script.

## 2.2 Main Window (`gwexpy/gui/ui/main_window.py`)

* **Base Class**: `QtWidgets.QMainWindow`
* **Layout**:
  * Top: Menu bar (File, Edit, Measurement, Plot, Window, Help)
  * Center: Tab switching structure using `QTabWidget`.
    * **Input**: Data source selection (various settings).
    * **Measurement**: Measurement parameters and channel selection.
    * **Excitation (Simulation)**: Excitation/simulation signal settings.
    * **Result**: Graph display and analysis result confirmation (main view).
  * Bottom: Status bar and control buttons (Start, Pause, Resume, Abort).

### Implementation Status of Key Features

1. **Data Source Management**:
    * **NDS/NDS2**: Has data acquisition logic from online servers in cooperation with the `NDSDataCache` class. Asynchronous background acquisition of channel lists via `preload_nds_channels` is implemented.
    * **Simulation**: Simulation mode using internal `SignalGenerator`.
    * **PC Audio**: Support for microphone input (checkbox implementation available).
2. **Graph Synchronization (X-Link Logic)**:
    * `update_x_link_logic`: Dynamically controls X-axis synchronization between two graph panels. Advanced logic that switches link enable/disable based on graph type (time axis/frequency axis) and auto-range settings is implemented.
3. **Data Streaming and Updates**:
    * Achieves drawing updates every 50ms through `start_animation` / `update_graphs` loop.
    * Uses `SpectralAccumulator` for data accumulation.
    * Includes logic to generate and inject simulation (Excitation) signals in real-time, not just acquired data.

# 3. UI Component Details

## 3.1 Tab Configuration (`gwexpy/gui/ui/tabs.py`)

Detailed configuration screens that mimic the DTT UI are constructed.

1. **Input Tab**:
    * Forms for NDS server/port settings, GPS time synchronization (using `astropy.time`), LiDaX settings, etc.
2. **Measurement Tab**:
    * Fourier analysis settings: `start_freq`, `stop_freq`, `bw` (Bandwidth), `averages`, `window` (Hann, Flattop, etc.), `overlap`.
    * Channel selection: Bank switching every 16 channels, Active state management via checkboxes.
    * Integration with Channel Browser (`ChannelBrowserDialog`).
3. **Excitation Tab**:
    * 4 signal generation panels. Each panel allows configuration of waveform type, frequency, amplitude, offset, phase, and injection destination channel.
    * `Waveform` types: Sine, Square, Ramp, Noise (Gauss/Uniform), Sweep (Linear/Log), Impulse, etc.
4. **Result Tab**:
    * Graph area split into top and bottom (`pg.PlotWidget`).
    * The `GraphPanel` (detailed settings) that should be placed on the right side of the screen is implemented as being stored in the left panel of the `QSplitter`.

## 3.2 Graph Panel (`gwexpy/gui/ui/graph_panel.py`)

Custom widget for configuring detailed settings per graph. Controls `pyqtgraph` functionality from the UI.

* **Traces Tab**: Selection and style (color, line type, symbol, bar) settings for 8 traces (Channel A/B).
* **Range Tab**: Log/Lin (logarithmic/linear), auto/manual range switching for X-axis/Y-axis respectively.
* **Units Tab**: Display unit settings (Hz, s, m, V, etc.). Complex number data display format (Magnitude, Phase, dB, Real, Imag) conversion function is also implemented.
* **Cursor Tab**: Coordinate display and delta measurement function for 2 cursors (vertical, horizontal, cross). Snap function (attraction to data points) is also implemented.
* **Style Tab**: Graph title, font, margin settings.

# 4. Simulation Logic (`gwexpy/gui/excitation/generator.py`)

* **Role**: Generates waveform data corresponding to a specified time array based on `GeneratorParams`.
* **Implemented Waveforms**:
  * **Sine, Square, Ramp, Triangle**: Basic waveforms.
  * **Impulse**: Pulse train (Duty cycle control).
  * **Noise (Gauss / Uniform)**: With filtering function (Butterworth SOS). Automatically applies bandpass/highpass/lowpass depending on parameters.
  * **Sweep (Linear / Log)**: Sweep signal using `scipy.signal.chirp`.
* **Features**:
  * Maintains filter internal state (`zi`) between frames through state retention (`filter_states`), achieving continuous filtering output (corresponding to NDS chunk processing).

# 5. Analysis Summary

* **Completeness**: High. Most of DTT's main GUI functionality (parameter settings, multiple tabs, detailed graph control) has already been re-implemented in Python/Qt.
* **DTT Compatibility**:
  * Has XML data import logic (`open_file`), indicating awareness of loading existing DTT configuration files.
  * Terminology (Start Freq, BW, Averages, etc.) also conforms to DTT.
* **Extensibility**:
  * In addition to NDS connections, features that are not in DTT (or are difficult to use), such as simulation mode and PC audio input, are integrated.
  * Based on `pyqtgraph`, so rendering performance for large amounts of data is also considered.

**Conclusion**: `pyaggui` has gone beyond a mere prototype and is in a state where it can function as the foundation for a practical diagnostic tool. Future challenges will be complete integration with the actual DTT kernel logic (advanced analysis calculations in the `Result` tab, etc.) and edge case bug fixes.

# gwexpy GUI (pyaggui)

> [!WARNING]
> This GUI module is **under active development**. The API and UI may change without notice as improvements are made.

## Overview

`pyaggui` is a GUI tool modeled after the LIGO DTT (Diagnostic Test Tools) `diaggui`. It enables real-time visualization, analysis of time-series data, and browsing of saved file data using the `gwexpy` library.

## Features

### 1. Data Sources

- **NDS / NDS2 (Online)**: Retrieve and display data from NDS servers (e.g., KAGRA NDS).
  - **Channel Browser**: Browse channel lists on the server and select measurement targets.
  - **PC Audio**: Utilize PC microphone input or speaker output as data sources.
- **Excitation (Simulation)**: Use various waveform generators for signal injection into online data or standalone simulation signal generation.
- **FILE (File Load)**: Load and display data from local files (DTT XML, GWF, HDF5, CSV, etc.).

### 2. Channel Handling

Based on the design philosophy of `diaggui`, measurement targets and display targets are managed separately.

- **Measurement Tab**:
  - Select channels for data acquisition and set them to `Active`.
  - Supports adding channels from the Channel Browser and restoring configurations from loaded files.
- **Results Tab**:
  - **Display Selection**: Choose which channels to plot on the graph from those marked as `Active` in the Measurement tab.
  - **Dual Graphs**: Features two independent graph panels, allowing simultaneous display of different analyses (Time Series, ASD, Coherence, Spectrogram, etc.).

### 3. Analysis & Plotting

- **Analysis Types**: Time Series, ASD, CSD, Coherence, Squared Coherence, Transfer Function, Spectrogram.
- **Rich Display Options**:
  - **Style**: Customize line types, symbols, colors, and axis scales (Linear/Log).
  - **Legend / Cursor**: Display legends and read values using a cursor with a snap-to-data-point feature.
  - **Display**: Toggle options such as dB display, phase display, and unwrapped phase.

### 4. File Support

Utilizing the powerful I/O capabilities of `gwexpy`, the GUI supports standard LIGO/KAGRA formats as well as various instrument-specific and general data formats.

| Format | Extension | Data Types | Remarks |
| :--- | :--- | :--- | :--- |
| **DTT XML** | `.xml` | TS, ASD, CSD, COH, TF, **Measurement State** | **Recommended**. Fully restores measurement settings and analysis results. |
| **LIGO LW XML** | `.xml` | TS, ASD, CSD, COH, TF | General LIGO LW format (not specific to DTT). |
| **GW Frame** | `.gwf` | TS | Standard format for gravitational-wave observation data. |
| **HDF5** | `.h5`, `.hdf5` | TS, ASD, CSD, COH, TF, Spectrogram | Hierarchical general-purpose data format. |
| **MiniSEED / SAC** | `.mseed`, `.sac` | TS | Format used for seismometers (requires ObsPy). |
| **WAV Audio** | `.wav` | TS | Audio data. Useful for analyzing microphone recordings. |
| **NI TDMS** | `.tdms` | TS | Data format for **National Instruments** equipment. |
| **Graphtec GBD** | `.gbd` | TS | Binary format for **Graphtec** data loggers (analog converted via range; statuses handled for `Alarm/Logic`). |
| **Metronix ATS** | `.ats` | TS | Data format for **Metronix ADU** equipment. |
| **Text / CSV** | `.txt`, `.csv`, `.dat` | TS | Comma or space-separated text data. |
| **Others** | `.npy`, `.mat`, `.fits`, `.pkl`, `.ffl`, `.sdb` | TS | Support for NumPy, MATLAB, FFL, and other common scientific formats. |

> [!NOTE]
> - **TS**: Time Series, **ASD**: Amplitude Spectral Density
> - **CSD**: Cross Spectral Density, **COH**: Coherence, **TF**: Transfer Function
> - For files other than DTT XML, saving/restoring channel settings (e.g., which channels are Active) is not supported.

## Comparison with Existing Tools

- **ndscope** (Developed by LIGO, used by KAGRA):
  - `ndscope` primarily handles time-series waveforms or trends, whereas this tool also supports spectral analysis and spectrograms.
- **diaggui** (LIGO DTT, used by KAGRA):
  - This tool reproduces the operational feel of `diaggui` but runs on modern environments including Apple Silicon (ARM) and includes features not present in `diaggui`, such as spectrogram display.

## How to Run

```bash
cd gwexpy/gui
python pyaggui.py [filename]
```

## Dependencies

- PyQt5
- pyqtgraph
- qtpy
- sounddevice (required for PC Audio)
- nds2-client (required for NDS connection)
- gwexpy (depends on numpy, scipy, astropy, gwpy, etc.)

## Directory Structure

- `pyaggui.py`: Main entry point.
- `ui/`: Definitions for the main window, tabs, graph panels, and UI components.
- `loaders/`: Loaders for various file formats and XML parsing logic.
- `nds/`: Management of NDS/PC Audio communication and data buffering.
- `excitation/`: Signal generation engine and parameter management.
- `plotting/`: Plotting utilities.

## Future Work

- **Robust NDS Connection**: Improve timeout handling under specific network conditions.
- **Code Integration**: Generalize XML parsing logic from `gui/loaders/` into `gwexpy/io/` (partially complete).
- **Feature Expansion**: Direct loading of Foton filters and implementation of advanced statistical analysis features.

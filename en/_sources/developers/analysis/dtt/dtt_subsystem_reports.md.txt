# DTT Subsystem Detailed Analysis Report

**Created:** 2026-01-22
**Target:** `gwexpy/gui/reference-dtt/dtt-master/src/dtt/`
**Author:** Antigravity Agent

---

# Section 1: GUI Logic Detailed Analysis

## 1.1 Overview

This section summarizes the results of a detailed code analysis of the DTT (Diagnostic Test Tools) GUI implementation, focusing specifically on the main window structure, event handling, data binding, and rendering logic.

## 1.2 Window Structure and Layout

The application's main window is defined by the `DiagMainWindow` class (inheriting from ROOT's `TGMainFrame`).

* **Menu Bar (`TGMenuBar`)**:
  * Standard configuration including File, Edit, Measurement, Plot, Window, Help, etc.
* **Main Control Area**:
  * Functions are divided into tabs using `DiagTabControl`.
  * **Configuration Tab**: For parameter settings such as Input, Measurement, Excitation.
  * **Display Tab**: Contains `TLGMultiPad` for result display.
* **Button Bar**:
  * Button group for test execution control (Start, Pause, Resume, Abort).
* **Status Bar (`TGStatusBar`)**:
  * Displays program status, heartbeat, and progress.

## 1.3 Widget Class Hierarchy

Based on the ROOT framework, custom classes with the `TLG` prefix are extended to meet LIGO-specific requirements.

* **Containers**:
  * `TGMainFrame` -> `DiagMainWindow`
  * `TGCompositeFrame` -> `TLGPad`, `TLGMultiPad`, `DiagTabControl`
* **Custom Controls** (`src/dtt/gui/dttgui/`):
  * **`TLGTextEntry` / `TLGNumericEntry`**: Input fields with validation, unit input, increment/decrement buttons.
  * **`TLGChannelBox`**: Hierarchical combo box for LIGO channel selection (Site -> IFO -> System).
* **Drawing Widgets**:
  * **`TLGPad`**: Wraps `TRootEmbeddedCanvas`, managing a single drawing area for graphs (`TGraph`, `TH1`), axes, legends, and option panels.
  * **`TLGMultiPad`**: Container that arranges and manages multiple `TLGPad` instances in a grid layout (e.g., 2x2).

## 1.4 Event Handling Patterns

Adopts a standard event-driven model using ROOT's message maps and dispatch functions.

* **Message Processing**:
  * `ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)` functions as the central processing hub.
  * Dispatches to handlers such as `ProcessButton`, `ProcessMenu` based on message types like `kC_COMMAND` (button, menu), `kC_STATUS`, `kC_NOTIFY`.
* **Timer Processing**:
  * **Heartbeat (`fHeartbeat`)**: 100ms interval timer. Checks the notification queue (`fNotifyMsgs`) from the backend and triggers UI updates (such as test completion notifications).
  * **Watchdog (`fXExitTimer`)**: Monitors X11 display connection.
* **Backend Communication**:
  * The GUI does not perform calculations directly. It follows a client-server (or separated kernel) model structure, sending commands ("run", "save", etc.) through the `basic_commandline` interface and retrieving state.

## 1.5 Data Binding and Flow

Data flow is based on a polling and command-response model.

* **Configuration**:
  * The `TestParam_t` structure holds all test settings.
  * The `TransferParameters` method synchronizes values between GUI widgets and the structure.
* **Data Discovery Process**:
  * `AddDataFromIndex` queries the `basic_commandline` for an index to discover available data (time series, PSD, transfer functions, etc.).
  * Parses text-based hierarchical structures (e.g., `Result.IndexEntry[1] = ...`) for analysis.
* **Data Descriptors**:
  * **`DiagDataDescriptor`**: Accessor for actual data. Calls `cmd->getData` when drawing is needed, performing lazy loading.
  * **`PlotDescriptor`**: Associates data (`BasicDataDescriptor`) with metadata and calibration information.

## 1.6 Rendering Logic

The drawing system is separated from the main window by the `PlotSet` container.

1. **Repository (`PlotSet`)**:
    * Central registry managing all available plot data. Organized by graph type and channel name.
2. **Update Cycle**:
    * When the heartbeat timer detects "new test results", `UpdatePlot` fires.
    * Scans for new data and adds to `PlotSet`.
3. **Rendering**:
    * `fPlot->Update()` is executed, iterating through registered `TLGPad` instances.
    * If there are data updates, `TLGPad` retrieves array data via `DiagDataDescriptor` and updates ROOT objects (`TGraph`).
    * Performs complex number data transformation based on user display settings (Mag, Phase, dB, etc.) before rendering.

## 1.7 Auxiliary Functions and Dialogs

### Menu System (`mainmenu.cc`, `TLGMainMenu`)

* **Configuration**: The `TLGMainMenu` class handles menu bar construction and initial callback processing.
* **Functions**:
  * File operations (`New`, `Open`, `Save`, `Import`, `Export`, `Print`)
  * Window operations (`Zoom`, `Layout`, `Active`)
  * External tool launch (`Launch` menu): `mainmenu.cc` also serves as a launcher function that uses the `launch_client` class to start external processes such as `dataviewer`, `foton`.

### Export and Save (`TLGExport`, `TLGSave`)

* **Formats**: Primarily supports LIGO's proprietary **XSIL (XML)** format and generic text formats.
* **Logic**:
  * `TLGExport::ExportToFileXML`: Scans data in `PlotSet` and generates XML tag structure (`<XSIL>`).
  * **Data Conversion (`DoConversion`)**: Converts raw data (complex numbers, etc.) to user-specified formats (amplitude, dB, phase, real/imaginary parts) before writing to file.

---

# Section 2: Kernel Logic Analysis

## 2.1 Overview

The `src/dtt/diag/` directory contains the core "diagnostic kernel" of the DTT application. This kernel is responsible for executing measurement tests (such as FFT, Swept Sine, Time Series), managing excitation signals, synchronizing with data acquisition (NDS/DAQS), and performing real-time signal processing and analysis.

## 2.2 Architecture & Core Components

### Class Hierarchy

* **`diagtest` (Abstract Base Class)**: Defines the universal interface for any diagnostic test. It encapsulates the test environment (`diagStorage`, `dataBroker`, `excitationManager`) and defines lifecycle methods (`begin`, `setup`, `end`) and the main analysis trigger.
* **`stdtest` (Standard Test)**: Inherits from `diagtest`. Implements the common logic found in most tests:
  * **Channel Management**: Manages lists of Stimulus (`stimulus`) and Measurement (`measurementchannel`) channels.
  * **Measurement Scheduling**: Defines `addMeasurements` and `newMeasPoint` to generate a sequence of `interval`s and `syncpointer`s.
  * **Synchronization**: Implements `syncAction`, which is called when data for a specific interval is available, triggering the `analyze` method.
* **Concrete Signal Processing Tests**: `ffttest`, `sweptsine`, `timeseries`, `sineresponse`.

### Execution Engine: `standardsupervisory`

* **Modes**: Supports Real-Time (`runRT`) and Off-Line (`runOL`) execution.
* **Loop**: It runs a main loop that drives the test.
  * Calls `test->setup()` to initialize the measurement schedule.
  * Enters a wait loop (`syncWait` or `syncRead`).
  * Upon data availability, it triggers the callback mechanism in `stdtest` which performs the analysis.
  * Handles asynchronous events like Pause, Resume, and Abort.

### Synchronization: `testsync`

* **`syncpointer`**: Represents a future point in time or a data condition that the test waits for.
* **`interval`**: Defines a specific time window `[t0, t0 + duration]` for which data is required.

## 2.3 Test Implementations & Signal Processing

### FFT Test (`ffttest`)

* **Setup**: Calculates parameters (`calcTimes`) like bandwidth, window type, overlap, and averaging.
* **Execution**:
  * **`calcMeasurements`**: Sets up excitation (random noise, periodic) and measurement intervals.
  * **`analyze`**: Allocates temporary storage (`tmpresult`) and iterates through channels.
    * **`fft`**: Computes Power Spectral Density (PSD).
    * **`cross`**: Computes Cross-Spectral Density (CSD) and Coherence.

### Swept Sine Test (`sweptsine`)

* **Setup**: Generates a list of sweep points (`sweeppoints`). Supports Linear, Logarithmic, and User-defined sweeps.
* **Signal Processing**:
  * **`sinedet` (Sine Detection)**: Extracts the specific frequency component from the time series data.
  * **`transfn`**: Computes the Transfer Function `H = Out/In` and Coherence.

### Triggered Time Series (`timeseries`)

* **Setup**: Configures pre-trigger and post-trigger durations.
* **Processing**: Supports simple averaging or summation of time traces across multiple triggers.

## 2.4 Data Flow Summary

1. **Configuration**: GUI parameters (`TestParam`) are read by `readParam` in the specific test class.
2. **Scheduling**: `standardsupervisory` calls `test->setup()`.
3. **Acquisition**: `dataBroker` (external) fetches data corresponding to these intervals.
4. **Wait**: `standardsupervisory` blocks on `syncWait`.
5. **Callback**: When data arrives, `stdtest::syncAction` is invoked.
6. **Analysis**: `syncAction` calls the virtual `analyze` method.
7. **Processing**: `analyze` calls helper methods (`fft`, `sinedet`).
8. **Publication**: Results are stored in `diagStorage`.

---

# Section 3: Data Acquisition Logic Analysis

## 3.1 Overview

The "Data Acquisition" logic in DTT is decoupled into two distinct layers:

1. **Channel Metadata Layer (`src/dtt/daq/`)**: Handles channel name resolution, attribute lookup (sample rate, units, calibration), and site-specific prefixes.
2. **Data Transport Layer (`src/dtt/storage/`)**: Defines the abstract `dataBroker` interface for connecting to NDS/DAQS servers, requesting time-series data, and managing data streams.

## 3.2 Channel Metadata Layer (`src/dtt/daq/`)

### Core C API: `gdschannel`

* **File**: `gdschannel.h`, `gdschannel.c`
* **Struct `gdsChnInfo_t`**: The core data structure defining a channel.
  * **Identity**: `chName` (max 60 chars), `chNum`, `dcuId`, `ifoId`.
  * **Properties**: `dataRate` (Hz), `dataType` (int16, float, complex, etc.), `unit`, `chGroup` (Fast/Slow).
  * **Calibration**: `gain`, `slope`, `offset`.
* **Key Functions**:
  * `gdsChannelInfo(name, info)`: Fills the struct for a given channel name.
  * `gdsChannelList(ifo, query, ...)`: Returns a list of channels matching criteria.

### C++ Abstraction: `testchn`

* **Class `channelHandler`**: A utility class to manage site and interferometer prefixes.
  * **Prefix Management**: Stores Default/Mandatory Site (H, L) and IFO (0, 1, 2) identifiers.
  * **Name Expansion**: `channelName()` expands short names (e.g., "ASC-X_TR") to full names (e.g., "H1:ASC-X_TR").

## 3.3 Data Transport Interface (`src/dtt/storage/`)

### Abstract Data Broker: `dataBroker`

* **Key Responsibilities**:
  * **Connection**: `connect()`, `reconnect()`.
  * **Subscription**: `add(channel)` to build a request list.
  * **Request**: `set(start, duration)` (Offline/Archive) or `set(start, active)` (Real-time).
  * **Flow Control**: `clear()`, `reset()`, `stop()`.
* **Error Handling**: Defines specific exceptions for `NoDataError` and `DataOnTapeError`.

## 3.4 Porting Implications for `gwexpy`

* **Legacy C API**: The `gdschannel` C struct and API should be replaced by NDS2-Client Python bindings.
* **Structure Preservation**: The separation of "Name Resolution" from "Data Fetching" is a good pattern.
* **Name Expansion**: The logic for handling "H1:", "L1:" prefixes is essential for UX.

---

# Section 4: Foton (Filter Online Tool) Logic Analysis

## 4.1 Overview

Foton (Filter Online Tool) is the primary logic for designing, visualizing, and exporting digital filters (IIR) for the LIGO real-time systems.

## 4.2 Architecture

### Directory Structure

* **`src/dtt/foton/`**: Contains the `main` entry point (`foton.cc`) and the application wrapper.
* **`src/dtt/filterwiz/`**: Contains the core business logic and GUI implementation.

### Key Classes

* **`TLGFilterWizard` (Logic + GUI)**: The main window controller.
* **`FilterFile` (Data Model)**: Represents the entire filter configuration file.
* **`FilterModule`**: Represents a named collection of filters.
* **`FilterSection`**: Represents a single filter stage (e.g., "Boost" or "Notch").
* **`IIRSos` (Signal Processing)**: Represents a single Biquad (Second Order Section).

## 4.3 Core Logic Flow

### Loading a Filter File

1. **Entry**: `foton.cc` parses arguments.
2. **`TLGFilterWizard::Setup`**: Initializes the window and triggers file loading.
3. **`FilterFile::read`**: Parses the text file line-by-line.

### Designing a Filter

1. **User Input**: The user enters a design string (e.g., `zpk(...)`, `reso(...)`).
2. **Parsing**: The string is parsed into poles and zeros.
3. **Computation**: The z-domain coefficients are computed.
4. **Validation**: Foton checks if the filter is stable.

### Saving

1. **`FilterFile::write`**: Iterates over all `FilterModule`s and writes the `# MODULES` header and SOS coefficients.

## 4.4 Porting Implications for `gwexpy`

* **Parsing Logic**: The parsing of the `.txt` filter file format must be ported to Python.
* **Filter Design**: Python's `scipy.signal` (`zpk2sos`, `bilinear`) provides 90% of the math needed.
* **Stability Checks**: `gwexpy` should implement checks similar to `check_poles`.

---

# Section 5: AWG GUI Logic Analysis

## 5.1 Overview

This section summarizes the analysis results of the GUI implementation for waveform generation (Excitation/AWG) in the LIGO diagnostic tool suite. The analysis covers both the standalone waveform generation tool **AWG GUI** (`awggui.cc`) and the **DTT GUI Excitation Tab** (`diagctrl.cc`), which generates waveforms as part of diagnostic tests.

## 5.2 Detailed Analysis of AWG GUI (`awggui.cc`)

The AWG GUI is a standalone application (or independent window) for controlling and outputting arbitrary waveform signals in real-time.

### Classes and Data Structures

* **Main Class:** `AwgMainFrame` (inherits from `TGMainFrame`)
* **Data Structure:** `struct awg` (instantiated as global variable `awgCmd`)
* **Configuration Storage:** `struct configuration`

### Backend Communication Logic

The AWG GUI communicates directly with the backend (AWG server, `awgtpman`, etc.) using API functions defined in `awgapi.h`.

1. **Channel Acquisition (`ReadChannel`)**:
    * `awgSetChannel(const char* channelName)`: Obtains a slot number.
    * `awgRemoveChannel(int slotNum)`: Releases channel assignment.

2. **Command Transmission (`HandleButtons` - "Set/Run")**:
    * **Command Example:** `set <slotNum> sine <freq> <amp> <offset> <phase>`
    * `awgcmdline(const char* command)`: Sends and executes the command.

### Key Features

* **Waveform Types:** Sine, Square, Ramp, Triangle, Offset (DC), Uniform (Noise), Normal (Gaussian Noise), Arbitrary, Sweep.
* **Filter:** Can dynamically load the `foton` tool to generate and apply filter coefficients.
* **Control:** Gain adjustment, ramp time setting, stop.

## 5.3 Detailed Analysis of DTT GUI Excitation Tab (`diagctrl.cc`)

The Excitation tab in DTT GUI is an interface for configuring stimulus signals applied to the system under test when executing diagnostic tests such as transfer function measurements.

### Backend Communication Logic

DTT adopts a "batch configuration at test execution" model.

1. **Parameter Setting (`DiagMainWindow::WriteParam`)**:
    * Uses the `fCmdLine->putVar(...)` method to update backend variables.

2. **Execution**:
    * After all parameters are set, `fCmdLine->parse("run")` is executed.

## 5.4 Comparison and Summary

| Feature | AWG GUI (`awggui.cc`) | DTT GUI Excitation (`diagctrl.cc`) |
| :--- | :--- | :--- |
| **Primary Purpose** | Manual signal output at arbitrary timing and waveform | Stimulus signal configuration for diagnostic tests (FFT/SweptSine, etc.) |
| **Communication Timing** | Immediate transmission on user operation | Batch transmission at test start |
| **Communication API** | `awgcmdline()` (direct text command transmission) | `fCmdLine->putVar()` (variable setting) -> `run` |
| **Channel Management** | Dynamically acquires slots with `awgSetChannel` | Managed as configuration variables in `basic_commandline` |
| **Flexibility** | High (overlay with Add, immediate stop, etc.) | Subordinate to test sequence |

**Conclusion:**
The AWG GUI performs **procedural, immediate execution** control, while the DTT GUI takes a **declarative, batch execution** approach.

---

*This report provides an understanding of the implementation details of each DTT subsystem.*

# DTT (Diagnostic Test Tools) Comprehensive Analysis Report

**Created:** 2026-01-22
**Target:** `gwexpy/gui/reference-dtt/dtt-master/`
**Author:** Antigravity Agent

---

# Part 1: Documentation & UI Audit (Phase 1)

## 1.1 Executive Summary

The legacy Diagnostic Test Tool (DTT) is a C++ application built upon the CERN ROOT framework. It follows a client-server architecture where `diaggui` handles the user interface and parameter configuration, while the heavy lifting (signal processing, data acquisition) is performed by a diagnostic kernel (`diagd` or internal classes).

## 1.2 Documentation Review

Since the PDF contents (`T990013-v3`, `G000079-00`) are binary specifications, we inferred their role from the project structure and markdown summaries:

- **T990013 (Diagnostics Test Software)**: Defines the core requirements for LIGO diagnostics, including FFT performance, Swept Sine accuracy, and real-time data access.
- **G000079**: Likely the user manual or GUI specification, detailing the operational modes.

## 1.3 UI/UX Architecture Analysis

### Framework

- **Base Library**: ROOT (`TApplication`, `TGMainFrame`).
- **Custom Wrapper**: `TLG` (The LIGO GUI) classes in `src/dtt/gui/dttgui/`.
  - `TLGMainWindow`: Base window layout with Menu, Status, Plot area.
  - `TLGPlot`: Advanced plotting widget supporting Zoom, Log/Linear toggles, multiple traces.
  - `TLGChannelBox`: Hierarchical tree view for selecting NDS channels.

### Key Operational Modes (Tabs)

The screenshots and source code structure (`src/dtt/diag/`) indicate four primary measurement modes:

1. **Fourier Tools (FFT)**:
    - computes Power Spectral Density (PSD), Cross Spectral Density (CSD), Coherence.
    - Parameters: Start/Stop Freq, BW, Window (Hann/Flat-top), Averages.
2. **Swept Sine**:
    - Measures Transfer Functions (Bode Plots) by injecting a sine wave and sweeping frequency.
    - Critical for servo characterization.
3. **Time Series**:
    - Oscilloscope-like view of raw channel data.
4. **Excitation**:
    - Configuration of output signals (Gaussian Noise, Sine, Swept Sine) for system identification.

---

# Part 2: Core Code Analysis (Phase 2)

## 2.1 FFT-Based Measurement Logic (`ffttools.cc`)

The `ffttest` class is the engine for all FFT-based measurements (PSD, CSD, Coherence).

### Class Hierarchy

```
diagtest (abstract base)
  └── stdtest (common measurement logic: channels, stimuli, averages)
        └── ffttest (FFT-based: PSD, CSD, Coherence)
        └── sweptsine (Swept sine: Transfer Function)
        └── sineresponse (Single frequency)
        └── timeseries (Oscilloscope mode)
```

### Key Parameters (`diagnames.h`)

| C++ Parameter | Default | gwexpy Equivalent |
|---|---|---|
| `fftStartFrequency` | 0 | `fmin` |
| `fftStopFrequency` | 900 | `fmax` |
| `fftBW` | 1 Hz | `df` (resolution, `1 / segment_duration`) |
| `fftOverlap` | 0.5 | `overlap` (Welch overlap fraction) |
| `fftWindow` | 1 (Hann) | `window` (scipy/gwpy window name) |
| `fftRemoveDC` | false | `detrend='constant'` |
| `fftAverages` | 10 | `n_avg` |
| `fftAverageType` | 0 (Linear) | `method='median'` or `'mean'` |

### Core FFT Workflow (`ffttest::analyze`)

1. **Allocate Temporary Storage (`tmps`):** For each channel, allocate `fftPoints` complex floats for intermediate FFT results.
2. **Iterate `ffttest::fft()` per Channel:**
   - Get time-series data from `gdsDataObject`.
   - Apply window (`psGen(PS_INIT_ALL, ...)`).
   - Compute FFT (`psGen(PS_TAKE_FFT, ...)`).
   - **Zoom FFT:** If `fZoom > 0`, data undergoes complex heterodyne transformation (shifting center frequency to baseband), followed by decimation (`decimate2`) and standard FFT.
   - **Convert to PSD (`fftToPs`):** Apply power normalization.
   - **Averaging (`avg()`):** Linear (`AVG_LINEAR_SQUARE`) or Exponential (`AVG_EXPON_SQUARE`).
3. **Iterate `ffttest::cross()` per A-channel pair:**
   - Compute Cross-Spectral Density (`crossPower`).
   - Average complex CSD.
   - Compute Coherence (`coherenceCP`): `|CSD_AB|^2 / (PSD_AA * PSD_BB)`.

### Window Normalization

```c++
// calculate resolution (window) bandwidth
double windowNorm = sMean (fftPlan.window_coeff, fftPoints);
windowNorm *= windowNorm;
if (windowNorm > 0) {
   windowBW = BW / windowNorm;
}
```

- `BW` is the *resolution bandwidth* (1/segment_duration).
- `windowBW` is the *Equivalent Noise Bandwidth (ENBW)*, adjusted for window amplitude loss.
- **Critical for gwexpy:** `scipy.signal.welch` returns PSD already normalized for ENBW when `scaling='density'`. Ensure `gwexpy` uses consistent units.

## 2.2 Swept Sine Implementation

This is the most complex measurement mode and is **not currently implemented in gwexpy**.

### Core Concepts

- **Sweep Points (`sweeppoints`):** A list of `(freq, ampl, phase)` tuples.
- **Sweep Types:** Linear, Logarithmic, or from a file.
- **Per-Point Measurement:** At each frequency:
  1. Inject a single-frequency sine wave excitation.
  2. Wait for settling time.
  3. Acquire response for `measurementTime`.
  4. Compute sine amplitude and phase using `sinedet()` (demodulation).
  5. Store coefficients.
- **Transfer Function Calculation (`transfn()`):** `H = Y_response / X_excitation`.

### Parameters

| C++ Parameter | Default | Description |
|---|---|---|
| `ssStartFrequency` | 1 Hz | Start of sweep |
| `ssStopFrequency` | 1000 Hz | End of sweep |
| `ssNumberOfPoints` | 61 | Number of frequency points |
| `ssSweepType` | 1 (Log) | 0: Linear, 1: Log |
| `ssMeasurementTime[2]` | {0.1, 10} | Range for adaptive meas. time |
| `ssSettlingTime` | 0.25 | Settling time (as fraction of meas. time) |
| `ssHarmonicOrder` | 1 | For detecting harmonics |

---

# Part 3: Deep Dive Analysis

## 3.1 Swept Sine Detailed Algorithm

### Sweep Point Generation

```cpp
switch (sweepType) {
   case 0:  // Linear sweep
      for (int i = 0; i < nSweep; i++) {
         f = fStart + (double) i / (nSweep - 1.0) * (fStop - fStart);
         fPoints.push_back(sweeppoint(f, ampl));
      }
      break;
   case 1:  // Logarithmic sweep
      for (int i = 0; i < nSweep; i++) {
         f = fStart * power(fStop/fStart, (double) i / (nSweep - 1.0));
         fPoints.push_back(sweeppoint(f, ampl));
      }
      break;
}
```

### Synchronous Detection (`sinedet`)

At each frequency point, the excitation frequency component is extracted from the response signal.

**Estimated Logic of `sineAnalyze`:**

```
X[k] = (2/N) * Σ x[n] * exp(-j * 2π * f_target * n / fs)
```

This is equivalent to computing a specific frequency bin of the Discrete Fourier Transform and can be implemented using `scipy.signal` or `numpy`.

## 3.2 Zoom FFT (High-Resolution Spectrum)

With standard FFT, `df = fs / N`, making it difficult to simultaneously achieve a wide frequency range and high resolution. Zoom FFT uses complex heterodyne transformation to analyze a specific narrow frequency band with high resolution.

### Algorithm

1. **Heterodyne (Frequency Shift)**: `x_shifted[n] = x[n] * exp(-j * 2π * f_zoom * n / fs)`
2. **Decimation (Downsampling)**: Reduce sampling rate so that Nyquist frequency matches the new bandwidth
3. **FFT Execution**: Perform FFT on the shifted & decimated data
4. **Data Rotation**: Restore the frequency axis to its original position

### Implementation Plan for gwexpy

```python
def zoom_fft(ts: TimeSeries, f_center: float, bandwidth: float, 
             df: float = None, window: str = 'hann') -> FrequencySeries:
    from scipy.signal import decimate
    import numpy as np
    
    # Step 1: Heterodyne (shift to baseband)
    t = ts.times.value
    x_shifted = ts.value * np.exp(-2j * np.pi * f_center * t)
    
    # Step 2: Low-pass filter and decimate
    decim_factor = int(ts.sample_rate.value / bandwidth)
    x_decimated = decimate(x_shifted, decim_factor, ftype='fir')
    
    # Step 3: FFT
    n_fft = int(bandwidth / df) if df else 1024
    spectrum = np.fft.fft(x_decimated[:n_fft] * get_window(window, n_fft))
    spectrum = np.fft.fftshift(spectrum)
    
    # Step 4: Build frequency axis
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=decim_factor/ts.sample_rate.value))
    freqs += f_center
    
    return FrequencySeries(spectrum, frequencies=freqs)
```

## 3.3 Burst Noise Measurement

In standard FFT measurements, the excitation signal is applied continuously, but in Burst Noise mode, the excitation signal is intermittent (gated), with quiet periods to capture the response ringing.

### Timing Diagram

```
|<-- pitch -->|<-- pitch -->|<-- pitch -->|
[RAMP][==EXCITATION==][RAMP][QUIET][MEASURE]
```

---

# Part 4: Architecture Analysis

## 4.1 `stdtest` Base Class

`stdtest` is the common base class for all measurement types in DTT (FFT, Swept Sine, Sine Response, Time Series).

### Key Inner Classes and Data Types

```cpp
class stimulus {
   std::string name;          // Excitation channel name
   bool isReadback;           // Readback usage flag
   AWG_WaveType waveform;     // Waveform type (Sine, Square, Noise, etc.)
   double freq, ampl, offs, phas;  // Frequency, amplitude, offset, phase
};

class measurementchannel {
   std::string name;          // Channel name
   gdsChnInfo_t info;         // Channel information
   partitionlist partitions;  // Data partitions
};
```

### Virtual Methods: Measurement Lifecycle

```
begin() → setup() → [syncAction() → analyze()] * N → end()
         ↓
         calcTimes()
         calcMeasurements()
         startMeasurements()
```

## 4.2 AWG (Arbitrary Waveform Generator) API

### Key Functions

| Function | Description |
|---|---|
| `awg_client()` | Initialize AWG client interface |
| `awgSetChannel(name)` | Associate channel name with slot |
| `awgAddWaveform(slot, comp, num)` | Add waveform component |
| `awgSetWaveform(slot, y, len)` | Download arbitrary waveform |
| `awgSendWaveform(slot, time, epoch, y, len)` | Send stream data |
| `awgStopWaveform(slot, terminate, time)` | Stop waveform (reset/freeze/phase-out) |
| `awgSetGain(slot, gain, time)` | Set overall gain (ramp capable) |
| `awgSetFilter(slot, y, len)` | Set IIR filter (SOS format) |

### Waveform Types (`AWG_WaveType`)

```cpp
enum AWG_WaveType {
   awgNone = 0,
   awgSine = 1,       // Sine wave
   awgSquare = 2,     // Square wave
   awgRamp = 3,       // Ramp wave
   awgTriangle = 4,   // Triangle wave
   awgImpulse = 5,    // Impulse
   awgConst = 6,      // Constant offset
   awgNoiseN = 7,     // Normal (Gaussian) noise
   awgNoiseU = 8,     // Uniform noise
   awgArb = 9,        // Arbitrary waveform
   awgStream = 10     // Stream waveform
};
```

## 4.3 NDS2 Data Input

`nds2input.hh` / `nds2input.cc` implements a client that retrieves real-time or archive data from NDS2 (Network Data Server v2).

### Key Classes

- **`nds2Manager`**: Manages NDS2 connections, adding/removing channels, starting/stopping data flow
- **`NDS2Connection`**: Single NDS2 connection instance, asynchronous data reading thread

---

# Part 5: Additional Modules Analysis

## 5.1 Foton - Filter Design Tool

Foton (Filter Online Tool) is a GUI tool for designing IIR filters used in LIGO/KAGRA.

### Relevance to gwexpy

```python
class FotonFilter:
    @classmethod
    def from_file(cls, path: str) -> 'FotonFilter':
        """Load filter from Foton .txt file."""
        ...
    
    def to_sos(self) -> np.ndarray:
        """Convert to Second Order Sections format for scipy.signal."""
        ...
```

## 5.2 StripChart - Real-Time Plotting

StripChart is a widget for plotting time-varying data in real-time.

## 5.3 DFM - Data Flow Manager

DFM (Data Flow Manager) is an abstraction layer that provides unified access to multiple data sources (NDS, File, Shared Memory, Tape).

### Data Service Types

```cpp
enum dataservicetype {
   st_Invalid = 0,
   st_LARS = 1,     // LARS/DFM server
   st_NDS = 2,      // NDS server (v1)
   st_SENDS = 3,    // NDS2 server
   st_File = 4,     // Local file system
   st_Tape = 5,     // Local tape drive/robot
   st_SM = 6,       // Online shared memory
   st_Func = 7      // User callback
};
```

---

# Part 6: External Dependencies & Python Integration

## 6.1 dttxml Package - DTT Data Access in Python

`dttxml` is a Python library for parsing XML files output by DTT (diaggui).

```python
from dttxml import DiagAccess

da = DiagAccess('measurement.xml')

# Get PSD (ASD)
asd = da.asd('K1:PEM-MIC_BOOTH_ENV_OUT_DQ')

# Get Transfer Function
tf = da.xfer('K1:SAS-ITMY_TM_OPLEV_SERVO_OUT', 'K1:SUS-ITMX_SUS_OUT')

# Get Coherence
coh = da.coherence('Channel1', 'Channel2')
```

## 6.2 Python Re-implementation of sineAnalyze

```python
import numpy as np
from scipy.signal import windows

def sine_analyze(data: np.ndarray, fs: float, freq: float,
                 window: str = 'hann', n_avg: int = 1,
                 t0: float = 0.0) -> complex:
    """
    Compute the complex amplitude at a specific frequency using lock-in detection.
    This is equivalent to DTT's sineAnalyze function.
    """
    n = len(data)
    segment_len = n // n_avg
    
    win = windows.get_window(window, segment_len)
    win_sum = np.sum(win)
    
    coeffs = []
    for i in range(n_avg):
        start_idx = i * segment_len
        segment = data[start_idx:start_idx + segment_len] * win
        
        t = np.arange(segment_len) / fs + t0 + start_idx / fs
        phase = 2 * np.pi * freq * t
        ref_cos = np.cos(phase)
        ref_sin = np.sin(phase)
        
        I = 2 * np.sum(segment * ref_cos) / win_sum
        Q = 2 * np.sum(segment * ref_sin) / win_sum
        
        coeffs.append(I + 1j * Q)
    
    return np.mean(coeffs)
```

---

# Part 7: Gap Analysis & Recommendations

## 7.1 Feature Gap Analysis

| DTT Feature | gwexpy Status | Required Work |
|---|---|---|
| **FFT/PSD** | ✅ `Spectrogram` / `FrequencySeries` | Minor: Verify ENBW normalization. |
| **CSD** | ✅ `FrequencySeries.csd()` | Minor: Verify phase convention. |
| **Coherence** | ✅ `coherence()` function | None. |
| **Swept Sine** | ❌ Not Implemented | Major: Requires AWG API, sync demod. |
| **Zoom FFT** | ❌ Not Implemented | Medium: Complex heterodyne + decimate. |
| **Burst Noise** | ❌ Not Implemented | Medium: Gated excitation logic. |
| **Plot Save/Load** | Partial (HDF5) | Medium: JSON schema for GUI state. |
| **Linear/Exp Avg** | ✅ `method` param | None. |

## 7.2 Implementation Roadmap

| Phase | Goal | Effort Estimate |
|---|---|---|
| **Phase 1** | PSD/CSD/Coherence Normalization Verification | 2-4 hours |
| **Phase 2** | Zoom FFT Implementation | 1-2 days |
| **Phase 3** | MeasurementState JSON Schema | 1-2 days |
| **Phase 4** | BaseMeasurement Abstract Class Design | 2-3 days |
| **Phase 5** | Swept Sine Prototype (Simulation) | 1 week |
| **Phase 6** | AWG API Python Bindings | 1-2 weeks |
| **Phase 7** | NDS2 Real-Time Integration | 1 week |

## 7.3 Recommendations

1. **Priority 1: Verify PSD Normalization.** Write a test that compares `gwexpy.Spectrogram` PSD output against known DTT output for identical input signals and parameters.
2. **Priority 2: Design Swept Sine API.** Draft a class `SweptSineMeasurement` with methods `setup()`, `measure_point()`, `next_point()`, `get_transfer_function()`.
3. **Priority 3: Define Measurement State Schema.** Create a `MeasurementState` dataclass for JSON serialization that mirrors `PlotSet`.

---

# Part 8: Data Flow & Class Interaction Diagrams

## 8.1 Class Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      GUI (diagmain, diagctrl)                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   standardsupervisory                            │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│   │ gdsStorage  │  │ dataBroker  │  │ testExcitation (AWG)    │ │
│   └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘ │
└──────────│────────────────│──────────────────────│──────────────┘
           │                │                      │
           ▼                ▼                      ▼
┌──────────────────────────────────────────────────────────────────┐
│                         stdtest                                   │
│   ┌───────────┐  ┌───────────────┐  ┌───────────────────────┐    │
│   │ stimulus  │  │measurementchn │  │       interval        │    │
│   │  list     │  │     list      │  │        list           │    │
│   └───────────┘  └───────────────┘  └───────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
     ┌────────────┐      ┌────────────┐      ┌────────────┐
     │  ffttest   │      │ sweptsine  │      │ timeseries │
     └────────────┘      └────────────┘      └────────────┘
```

## 8.2 Data Flow Diagram

```
[NDS2/RTDD/File]
       │
       ▼
┌──────────────┐
│  dataBroker  │  ─────  Channel subscription/data reception
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ chnCallback  │  ─────  Data delivery via callback
└──────┬───────┘
       │
       ▼
┌──────────────┐              ┌──────────────┐
│  gdsStorage  │ ◄──────────► │  gdsDatum    │
│              │              │  (Data)      │
│  - Results   │              └──────────────┘
│  - Index     │
│  - Test      │              ┌──────────────┐
│  - References│ ◄──────────► │ gdsParameter │
│              │              │  (Settings)  │
└──────────────┘              └──────────────┘
       │
       ▼
┌──────────────┐
│   PlotSet    │  ─────  Plot data management
└──────┬───────┘
       │
       ▼
    [XML Save]
```

---

# Part 9: File Analysis Summary

## 9.1 Analyzed Files (31 items)

| Category | File | Lines | Status |
|---|---|---|---|
| **Core Tests** | ffttools.hh/.cc | 379/1272 | ✅ |
| | sweptsine.hh/.cc | 396/1356 | ✅ |
| | sineresponse.hh/.cc | 313/1123 | ✅ |
| | timeseries.hh/.cc | 313/958 | ✅ |
| **Base Classes** | stdtest.hh/.cc | 700/1645 | ✅ |
| | diagtest.hh/.cc | 313/... | ✅ |
| | diagnames.h/.c | 800/... | ✅ |
| **Storage** | diagdatum.hh/.cc | 1884/... | ✅ |
| | gdsdatum.hh/.cc | 2019/... | ✅ |
| | channelinput.hh/.cc | 819/... | ✅ |
| | databroker.hh/.cc | 425/... | ✅ |
| | nds2input.hh/.cc | 400/... | ✅ |
| **Sync/Control** | testsync.hh/.cc | 522/... | ✅ |
| | supervisory.hh/.cc | 277/... | ✅ |
| | stdsuper.hh/.cc | 242/... | ✅ |
| **AWG** | awgapi.h, awgtype.h | .../... | ✅ |
| **GUI** | diagmain.hh, diagctrl.hh | .../... | ✅ |
| **Tools** | foton.cc | 642 | ✅ |
| | StripChart.hh | ... | ✅ |
| | dfmtype.hh | 166 | ✅ |
| **Containers** | PlotSet.hh, DataDesc.hh | 597/800 | ✅ |

---

# Part 10: Conclusion

The complete source code analysis of the DTT repository has been finished.

**Key Findings:**

1. **Class Hierarchy**: `diagtest → stdtest → {ffttest, sweptsine, timeseries, sineresponse}`
2. **Data Structures**: Multi-dimensional data management based on `gdsDatum`
3. **Measurement Flow**: Parameter reading → Time calculation → Measurement setup → Callback analysis
4. **External Dependencies**: `gdsalgorithm.h` (libgds) is non-public, but algorithms can be re-implemented
5. **Python Integration**: DTT XML data can be loaded using the `dttxml` package

---

*This analysis provides a comprehensive understanding of the core architecture and signal processing pipeline of DTT.*

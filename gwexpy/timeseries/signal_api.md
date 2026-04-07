# Signal Processing API Reference

This document is the API reference for signal processing methods in `gwexpy.timeseries.TimeSeries`.

## Hilbert Transform Related

### Overview

The Hilbert transform is a method for generating an analytic signal from a real-valued signal. By using the analytic signal, instantaneous phase and instantaneous frequency can be calculated.

### Mathematical Definition

For a real-valued signal $x(t)$, the analytic signal $z(t)$ is defined as:

$$
z(t) = x(t) + i \cdot \mathcal{H}[x(t)]
$$

where $\mathcal{H}[x]$ is the Hilbert transform of $x$, defined by convolution with $1/(\pi t)$.

Instantaneous phase and instantaneous frequency are defined as:

$$
\phi(t) = \arg(z(t))
$$

$$
f(t) = \frac{1}{2\pi} \frac{d\phi}{dt}
$$

---

## `hilbert`

```python
TimeSeries.hilbert(
    pad: int | Quantity = 0,
    pad_mode: str = "reflect",
    pad_value: float = 0.0,
    nan_policy: Literal["raise", "propagate"] = "raise",
    copy: bool = True
) -> TimeSeries
```

### Description

Calculate the analytic signal using the Hilbert transform.

### Parameters

| Parameter | Type | Default | Description |
|-----------|-----|-----------|------|
| `pad` | int or Quantity | 0 | Number of samples (or time duration) to pad at each end. |
| `pad_mode` | str | "reflect" | Padding mode (e.g., 'reflect', 'constant', 'edge'). |
| `pad_value` | float | 0.0 | Value used in 'constant' mode. |
| `nan_policy` | str | "raise" | How to handle NaN/Inf. 'raise' for an exception, 'propagate' to pass them through. |
| `copy` | bool | True | Whether to return a copy if the input is complex. |

### Returns

A `TimeSeries` containing the complex analytic signal. Same length as the input.

### Exceptions

- `ValueError`: If the input contains NaN or infinity (when `nan_policy='raise'`).
- `ValueError`: If the sampling is irregular.

### Notes

⚠️ **Pre-processing is the user's responsibility**: Operations like demean, detrend, filtering, and windowing are not automatically applied. Users should apply them as needed beforehand.

⚠️ **Endpoint Artifacts**: The Hilbert transform can produce artifacts at the endpoints due to spectral leakage. Use the `pad` parameter or apply an appropriate window function.

---

## `instantaneous_phase`

```python
TimeSeries.instantaneous_phase(
    deg: bool = False,
    unwrap: bool = False,
    **kwargs
) -> TimeSeries
```

### Description

Calculate the instantaneous phase using the Hilbert transform.

### Parameters

| Parameter | Type | Default | Description |
|-----------|-----|-----------|------|
| `deg` | bool | False | If True, returns the phase in degrees. If False, returns in radians. |
| `unwrap` | bool | False | If True, removes phase discontinuities (unwrapping). |
| `**kwargs` | - | - | Options passed to `hilbert()`. |

### Returns

A `TimeSeries` containing the instantaneous phase. Units are 'rad' or 'deg'.

### Definition

```python
analytic = hilbert(x)
phase = np.angle(analytic)  # radians
if unwrap:
    phase = np.unwrap(phase, period=2*np.pi)  # period=360 if degrees
```

### Notes

- Endpoints are not automatically trimmed.
- Pre-processing (demean, detrend, etc.) is not automatically applied.

---

## `instantaneous_frequency`

```python
TimeSeries.instantaneous_frequency(
    unwrap: bool = True,
    smooth: int | Quantity | None = None,
    **kwargs
) -> TimeSeries
```

### Description

Calculate the instantaneous frequency using the Hilbert transform.

### Parameters

| Parameter | Type | Default | Description |
|-----------|-----|-----------|------|
| `unwrap` | bool | True | Whether to unwrap the phase before differentiation (recommended: True). |
| `smooth` | int, Quantity, None | None | Smoothing window width. None for no smoothing. |
| `**kwargs` | - | - | Options passed to `hilbert()`. |

### Returns

A `TimeSeries` containing the instantaneous frequency. Units are 'Hz'.

### Definition

```python
phase = instantaneous_phase(unwrap=True, deg=False)  # radians
dphi_dt = np.gradient(phase, dt)  # time differentiation
f_inst = dphi_dt / (2 * np.pi)  # convert to Hz
```

### Notes

- Endpoints are not automatically trimmed.
- Accuracy may decrease near the endpoints due to numerical differentiation and Hilbert transform artifacts.
- It is recommended to use only the central region (e.g., 10%-90%) for accuracy evaluation.

---

## Examples

### Basic Usage

```python
import numpy as np
from gwexpy.timeseries import TimeSeries

# Generate test signal
t = np.linspace(0, 10, 10000)
f0 = 5.0  # Hz
signal = np.cos(2 * np.pi * f0 * t)
ts = TimeSeries(signal, dt=0.001, unit='V')

# Pre-processing (User responsibility)
ts_processed = ts.detrend().taper()

# Hilbert transform
analytic = ts_processed.hilbert()
envelope = np.abs(analytic.value)

# Instantaneous phase
phase = ts_processed.instantaneous_phase(unwrap=True)

# Instantaneous frequency
f_inst = ts_processed.instantaneous_frequency()

# Evaluate frequency in the central region
n = len(f_inst.value)
central = f_inst.value[int(n*0.1):int(n*0.9)]
print(f"Median frequency: {np.median(central):.2f} Hz")  # ≈ 5.0 Hz
```

### Mitigating Endpoint Artifacts

```python
# Use padding
analytic = ts.hilbert(pad=100)

# Or apply a window function
ts_windowed = ts.taper(side='both')
analytic = ts_windowed.hilbert()
```

### Chirp Signal Analysis

```python
# Chirp signal with varying frequency
f_start, f_end = 10.0, 50.0
t = np.linspace(0, 5, 50000)
chirp_phase = 2 * np.pi * (f_start * t + (f_end - f_start) / (2 * 5) * t**2)
signal = np.cos(chirp_phase)
ts = TimeSeries(signal, dt=0.0001, unit='V')

# Track frequency changes using instantaneous frequency
f_inst = ts.instantaneous_frequency()
```

---

## `heterodyne`

```python
TimeSeries.heterodyne(
    phase: array_like | TimeSeries,
    stride: float | Quantity = 1.0,
    singlesided: bool = False
) -> TimeSeries
```

### Description

Heterodyne (demodulate) the signal using the specified phase series and average over each stride. This method uses the same algorithm and default behavior (`singlesided=False`) as GWpy's `TimeSeries.heterodyne()`.

### Parameters

| Parameter | Type | Default | Description |
|-----------|-----|-----------|------|
| `phase` | array_like | (Required) | Phase used for demodulation (radians). Must be the same length as the input signal. |
| `stride` | float or Quantity | 1.0 | Time step for averaging (seconds). Internally rounded to `int(stride * sample_rate)` samples. |
| `singlesided` | bool | False | If True, doubles the amplitude (real-signal convention). Complies with GWpy's default (False). |

### Returns

A `TimeSeries` containing the complex demodulated and averaged signal.

### Algorithm

1. Multiply by the complex oscillator $\exp(-i \cdot \text{phase})$.
2. Split into segments for each defined `stride` and calculate the complex mean for each segment.
3. Segment length is calculated as `int(stride * sample_rate)`.
4. Trailing samples that do not fill a complete segment are discarded (floor).
5. If `singlesided=True`, the result is doubled.

---

## `lock_in`

```python
TimeSeries.lock_in(
    f0: float | Quantity | None = None,
    *,
    phase: array_like | None = None,
    fdot: float | Quantity = 0.0,
    fddot: float | Quantity = 0.0,
    stride: float | Quantity | None = None,
    bandwidth: float | Quantity | None = None,
    singlesided: bool = True,
    output: Literal["amp_phase", "complex", "iq"] = "amp_phase",
    deg: bool = True,
    **kwargs
) -> TimeSeries | tuple
```

### Description

Perform demodulation and averaging using a lock-in amplifier approach. Supports two modes: averaging-based (`stride`) and filter-based (`bandwidth`).

### Parameters

| Parameter | Type | Default | Description |
|-----------|-----|-----------|------|
| `f0` | float | None | Center frequency (Hz) for fixed-frequency demodulation. |
| `phase` | array_like | None | Explicit phase array (radians). |
| `stride` | float | None | Averaging time (seconds). Required if `bandwidth` is not specified. |
| `bandwidth` | float | None | Corner frequency (Hz) of the low-pass filter. Enables filter-based mode when specified. |
| `singlesided` | bool | True | If True, doubles the amplitude (standard lock-in convention). **Note: Differs from `heterodyne` default (False).** |
| `output` | str | "amp_phase" | Output format ('amp_phase', 'complex', 'iq'). |
| `deg` | bool | True | Phase unit for `amp_phase` output (True for degrees, False for radians). |

### Important Notes

- If `bandwidth` is not specified, it internally use `heterodyne` for averaging.
- If `bandwidth` is specified, it internally use `baseband` for filtering.
- `stride` is only valid when `bandwidth` is not specified.
- Fractional samples are discarded, similar to `heterodyne`.

---

## Related Methods

- `envelope()`: Calculate the envelope (amplitude) using the Hilbert transform.
- `radian()`: Phase angle of a complex signal (no Hilbert).
- `degree()`: Phase angle in degrees (no Hilbert).
- `unwrap_phase()`: Alias for `instantaneous_phase(unwrap=True)`.
- `mix_down()`: Perform only frequency mixing (complex demodulation).
- `transfer_function()`: Estimate the transfer function.

---

## Baseband Demodulation

### Overview

The `baseband` method shifts the carrier frequency to baseband (DC) and optionally applies a low-pass filter and resampling.

Processing Chain:

```
mix_down(f0) → [lowpass(cutoff)] → [resample(output_rate)]
```

### Two Execution Modes

**Mode A (Explicit Analysis Bandwidth)**:

- `baseband(f0=fc, lowpass=cutoff, output_rate=None|...)`
- Apply a low-pass filter after mixing to define the analysis bandwidth.
- Optionally resample to reduce the data rate.

**Mode B (Downsampling Priority)**:

- `baseband(f0=fc, lowpass=None, output_rate=rate)`
- Skip explicit low-pass filtering and rely on the resampling anti-aliasing filter.
- Useful when avoiding double filtering.

---

## `baseband`

```python
TimeSeries.baseband(
    *,
    phase: array_like | None = None,
    f0: float | Quantity | None = None,
    fdot: float | Quantity = 0.0,
    fddot: float | Quantity = 0.0,
    phase_epoch: float | None = None,
    phase0: float = 0.0,
    lowpass: float | Quantity | None = None,
    lowpass_kwargs: dict | None = None,
    output_rate: float | Quantity | None = None,
    resample_kwargs: dict | None = None,
    singlesided: bool = False
) -> TimeSeries
```

### Description

Demodulates the `TimeSeries` to baseband by frequency shifting (heterodyning) and optionally applying a low-pass filter and resampling.

### Parameters

| Parameter | Type | Default | Description |
|-----------|-----|-----------|------|
| `phase` | array_like or None | None | Explicit phase array (radians) for mixing. |
| `f0` | float or Quantity | None | Center frequency for mixing (Hz). Must be 0 < f0 < Nyquist. |
| `fdot` | float or Quantity | 0.0 | Frequency derivative (Hz/s). |
| `fddot` | float or Quantity | 0.0 | Frequency second derivative (Hz/s²). |
| `phase_epoch` | float or None | None | Reference epoch for the phase model. |
| `phase0` | float | 0.0 | Initial phase offset (radians). |
| `lowpass` | float or Quantity or None | None | Corner frequency of the low-pass filter (Hz). |
| `lowpass_kwargs` | dict or None | None | Additional arguments passed to `lowpass()`. |
| `output_rate` | float or Quantity or None | None | Output sample rate (Hz). |
| `resample_kwargs` | dict or None | None | Additional arguments passed to `resample()`. |
| `singlesided` | bool | False | If True, doubles the amplitude (for real signals). |

### Returns

A `TimeSeries` containing the complex baseband signal.

### Exception Conditions

| Condition | Exception |
|------|------|
| `f0 <= 0` | `ValueError` |
| `f0 >= Nyquist` (for regular series) | `ValueError` |
| `lowpass <= 0` | `ValueError` |
| `lowpass >= Nyquist` | `ValueError` |
| `output_rate <= 0` | `ValueError` |
| Both `lowpass` and `output_rate` are None | `ValueError` |
| `lowpass >= output_rate/2` (exceeds new Nyquist) | `ValueError` |

### Notes

⚠️ **Pre-processing is the user's responsibility**: Demean, detrend, and filtering are not automatically applied. DC offsets or trends will affect the baseband results.

⚠️ **Relationship between lowpass and f0**: Generally `lowpass < f0` is recommended, but not enforced. To capture only modulation around the carrier, set lowpass smaller than the carrier frequency.

⚠️ **GWpy Compatibility**: Internal processing for lowpass and resampling is delegated to GWpy methods. Customization is possible via `lowpass_kwargs` and `resample_kwargs`.

---

## Baseband Examples

### Mode A: Specifying Low-pass

```python
import numpy as np
from gwexpy.timeseries import TimeSeries

# 100 Hz carrier signal
t = np.arange(0, 10, 0.001)  # 1000 Hz sampling
signal = np.cos(2 * np.pi * 100 * t)
ts = TimeSeries(signal, dt=0.001, unit='V')

# Pre-processing (recommended)
ts = ts.detrend()

# Demodulate to baseband (10 Hz analysis bandwidth)
z = ts.baseband(f0=100, lowpass=10)

# DC component becomes dominant
print(f"DC magnitude: {np.abs(np.mean(z.value)):.3f}")
```

### Mode B: Resampling Only

```python
# Rely on resampling anti-aliasing
z = ts.baseband(f0=100, lowpass=None, output_rate=50)

# Output sample rate becomes 50 Hz
print(f"Output rate: {z.sample_rate}")
```

### Specifying Both

```python
# Both low-pass and resample
z = ts.baseband(f0=100, lowpass=10, output_rate=50)

# lowpass must be < output_rate/2 (= 25 Hz)
```

### Passing GWpy kwargs

```python
# Customizing the low-pass filter
z = ts.baseband(
    f0=100,
    lowpass=10,
    lowpass_kwargs={"filtfilt": True}  # GWpy option
)

# Customizing resampling
z = ts.baseband(
    f0=100,
    lowpass=None,
    output_rate=50,
    resample_kwargs={"window": "hamming"}  # GWpy option
)
```

---

## `heterodyne` (GWpy Compatible)

```python
TimeSeries.heterodyne(
    phase: array_like,
    stride: float | Quantity = 1.0,
    singlesided: bool = False
) -> TimeSeries
```

### Description

Implements **exactly the same algorithm** as GWpy's `TimeSeries.heterodyne()`.
Heterodynes the input `TimeSeries` with a phase series and averages over a fixed stride.

### Parameters

| Parameter | Type | Default | Description |
|-----------|-----|-----------|------|
| `phase` | array_like | - | Phase array for mixing (radians). `len(phase) == len(self)` is required. |
| `stride` | float or Quantity | 1.0 | Averaging time step (seconds). Sample count is floored to `int(stride * sample_rate)`. |
| `singlesided` | bool | False | If True, doubles the amplitude (for real signals). Complies with GWpy's default (False). |

### Returns

A complex `TimeSeries`. `dt = stride`, with values representing the average amplitude and phase in each stride as `mag * exp(1j * phase)`.

### Exceptions

| Condition | Exception |
|------|------|
| `phase` is not array_like (`len()` fails) | `TypeError` |
| `len(phase) != len(self)` | `ValueError` |

### Algorithm (Same as GWpy)

```python
stridesamp = int(stride * sample_rate)  # floor truncation
nsteps = int(N // stridesamp)           # discard remaining samples

for step in range(nsteps):
    istart = stridesamp * step
    iend = istart + stridesamp          # exclusive end
    mixed = exp(-1j * phase[istart:iend]) * data[istart:iend]
    out[step] = 2 * mixed.mean() if singlesided else mixed.mean()

output.sample_rate = 1 / stride
```

### Examples

```python
import numpy as np
from gwexpy.timeseries import TimeSeries

# Generate sine wave
A, f0, phi0 = 2.5, 30.0, np.pi/4
sample_rate = 1024.0
duration = 10.0
n = int(duration * sample_rate)
t = np.arange(n) / sample_rate

data = A * np.cos(2 * np.pi * f0 * t + phi0)
ts = TimeSeries(data, dt=1/sample_rate, unit='V')

# Create phase array
phase = 2 * np.pi * f0 * t

# Heterodyne (using singlesided=True as an example)
het = ts.heterodyne(phase, stride=1.0)

# Expected values: A * exp(1j * phi0)
print(f"Amplitude: {np.mean(np.abs(het.value)):.3f}")  # ≈ 2.5
print(f"Phase: {np.mean(np.angle(het.value)):.3f}")    # ≈ 0.785 (π/4)
```

---

## `lock_in` (Lock-in Amplifier)

```python
TimeSeries.lock_in(
    f0: float | Quantity | None = None,
    *,
    phase: array_like | None = None,
    fdot: float | Quantity = 0.0,
    fddot: float | Quantity = 0.0,
    phase_epoch: float | None = None,
    phase0: float = 0.0,
    stride: float | Quantity | None = None,
    bandwidth: float | Quantity | None = None,
    singlesided: bool = True,
    output: str = "amp_phase",
    deg: bool = True,
    **kwargs
) -> TimeSeries | tuple
```

### Description

Executes lock-in amplification (demodulation + averaging or filtering).
There are two operating modes, selected by the `bandwidth` parameter.

### Operating Modes

**LPF Mode (when `bandwidth` is specified)**:

- Demodulation and low-pass filtering using `baseband(lowpass=bandwidth, ...)`.
- Specifying `stride` is **prohibited** (ValueError).

**Stride Averaging Mode (when `bandwidth` is not specified)**:

- Demodulation and fixed-stride averaging using `heterodyne(phase, stride, ...)`.
- `stride` is **required**.

### Parameters

| Parameter | Type | Default | Description |
|-----------|-----|-----------|------|
| `f0` | float or Quantity | None | Center frequency (Hz). **Mutually exclusive** with `phase`. |
| `phase` | array_like | None | Explicit phase array (rad). **Mutually exclusive** with `f0` parameters. |
| `fdot` | float or Quantity | 0.0 | Frequency derivative (Hz/s). |
| `fddot` | float or Quantity | 0.0 | Frequency second derivative (Hz/s²). |
| `phase_epoch` | float | None | Reference epoch for the phase model. |
| `phase0` | float | 0.0 | Initial phase offset (rad). |
| `stride` | float or Quantity | None | Averaging time step (seconds). **Mutually exclusive** with `bandwidth`. |
| `bandwidth` | float or Quantity | None | LPF bandwidth (Hz). **Mutually exclusive** with `stride`. |
| `singlesided` | bool | True | If True, doubles the amplitude. |
| `output` | str | "amp_phase" | Output format: `'complex'`, `'amp_phase'`, `'iq'`. |
| `deg` | bool | True | Whether to return phase in degrees when using `'amp_phase'`. |
| `**kwargs` | - | - | Arguments passed to `baseband()` in LPF mode. |

### Returns

| `output` | Returns |
|----------|--------|
| `'complex'` | Complex TimeSeries. |
| `'amp_phase'` | `(amplitude, phase)` tuple. |
| `'iq'` | `(I, Q)` tuple (real/imag components). |

### Exception Conditions

| Condition | Exception |
|------|------|
| Simultaneous specification of `phase` and any `f0` parameters. | `ValueError` |
| Neither `phase` nor `f0` specified. | `ValueError` |
| Simultaneous specification of `bandwidth` and `stride`. | `ValueError` |
| Neither `bandwidth` nor `stride` specified. | `ValueError` |
| Invalid value for `output`. | `ValueError` |

### Phase Specification Precedence Rules

The `phase` parameter takes **highest precedence**. If `phase` is specified,
`f0`/`fdot`/`fddot`/`phase_epoch`/`phase0` parameters cannot be set
(except for their default values). This prevents ambiguous configurations.

### Examples

**Stride Averaging Mode (Phase generated from f0):**

```python
# Demodulation at a fixed frequency
amp, phase = ts.lock_in(f0=100.0, stride=1.0, output='amp_phase')
```

**Stride Averaging Mode (Explicit phase):**

```python
# Demodulation with a custom phase array
phase_arr = 2 * np.pi * 100.0 * ts.times.value
result = ts.lock_in(phase=phase_arr, stride=1.0, output='complex')
```

**LPF Mode:**

```python
# Demodulation using a low-pass filter
amp, phase = ts.lock_in(f0=100.0, bandwidth=10.0, output='amp_phase')
```

**Tracking a Chirp Signal:**

```python
# Demodulation of a signal with varying frequency
result = ts.lock_in(f0=100.0, fdot=0.1, stride=1.0, output='complex')
```

---

## See Also

- `heterodyne()`: Phase heterodyne and stride averaging.
- `baseband()`: Baseband demodulation (LPF + resample).
- `mix_down()`: Mixing with a complex oscillator (low-level).
- `_build_phase_series()`: Internal helper (phase generation from f0 parameters).

---

# HHT (Hilbert-Huang Transform) API Reference

This section describes HHT-related methods in `gwexpy.timeseries.TimeSeries`.

---

## Overview

The Hilbert-Huang Transform (HHT) is a method for analyzing non-linear and non-stationary time-series data. It consists of the following two steps:

1.  **Empirical Mode Decomposition (EMD)**: Decomposes a signal into Intrinsic Mode Functions (IMFs) and a residual.
2.  **Hilbert Spectral Analysis (HSA)**: Applies the Hilbert transform to each IMF to calculate instantaneous amplitude and instantaneous frequency.

`gwexpy` provides individual methods for these steps (`emd`, `hilbert_analysis`) and a unified method (`hht`) to execute them together.

---

## `emd`

```python
TimeSeries.emd(
    method: str = "eemd",
    max_imf: int | None = None,
    sift_max_iter: int = 1000,
    stopping_criterion: Any = "default",
    eemd_noise_std: float = 0.2,
    eemd_trials: int = 100,
    random_state: int | None = None,
    return_residual: bool = True,
    eemd_parallel: bool | None = None,
    eemd_processes: int | None = None,
    eemd_noise_kind: str | None = None
) -> TimeSeriesDict
```

### Description

Decomposes time-series data into IMFs (Intrinsic Mode Functions) and a residual using EMD (Empirical Mode Decomposition) or EEMD (Ensemble EMD). Requires the PyEMD package.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | str | "eemd" | Decomposition method ('emd' or 'eemd'). |
| `max_imf` | int, None | None | Maximum number of IMFs to extract (None for all). |
| `sift_max_iter` | int | 1000 | Maximum iterations for a single sifting (PyEMD's `MAX_ITERATION`). |
| `stopping_criterion` | Any | "default" | Uses PyEMD default if `"default"` or `None`. Numerical values are used as `std_thr`. |
| `eemd_noise_std` | float | 0.2 | Standard deviation of added noise in EEMD (relative to signal std). |
| `eemd_trials` | int | 100 | Number of trials for EEMD. |
| `random_state` | int, None | None | Random seed. Required for reproducibility in EEMD (stochastic). Ignored for EMD (deterministic). |
| `return_residual` | bool | True | Whether to include the residual in the result. |
| `eemd_parallel` | bool, None | None | Whether to enable parallel processing for EEMD. |
| `eemd_processes` | int, None | None | Number of parallel processes for EEMD. |

### Returns

`TimeSeriesDict`: Keys are `'IMF1'`, `'IMF2'`, ..., `'residual'`.

### Notes

- **Reproducibility**: EEMD is a stochastic process. To ensure reproducibility, specify `random_state` or use PyEMD's `noise_seed()`. The EMD method is deterministic.
- **Residual**: Residual extraction may vary by PyEMD version, but this method handles it appropriately.
- **Stopping Criterion**: Numerical values passed to `stopping_criterion` are treated as `std_thr` for PyEMD.

### Examples

```python
# Recommended: execute everything via hht()
result = ts.hht(
    emd_method="eemd",
    emd_kwargs={
        "eemd_trials": 20,
        "random_state": 42,
        "sift_max_iter": 200,
        "stopping_criterion": 0.2,
    },
    hilbert_kwargs={"pad": 100, "if_smooth": 11},
    output="dict",
)

# Executing EMD directly
imfs = ts.emd(method="emd", sift_max_iter=200, stopping_criterion=0.2)
```

---

## `hilbert_analysis`

```python
TimeSeries.hilbert_analysis(
    unwrap_phase: bool = True,
    frequency_unit: str = "Hz",
    if_smooth: int | Quantity | None = None,
    **hilbert_kwargs
) -> dict[str, Any]
```

### Description

Performs Hilbert transform analysis to calculate the analytic signal, instantaneous amplitude (IA), instantaneous phase, and instantaneous frequency (IF).

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `unwrap_phase` | bool | True | Whether to perform phase unwrapping. |
| `frequency_unit` | str | "Hz" | Unit for instantaneous frequency. |
| `if_smooth` | int, Quantity, None | None | Smoothing window width for instantaneous frequency. Odd numbers are recommended. |
| `**hilbert_kwargs` | Any | - | Arguments passed to the internal `hilbert()` call (e.g., `pad`, `pad_mode`). |

### Returns

Dictionary format:
- `'analytic'`: Analytic signal (complex TimeSeries)
- `'amplitude'`: Instantaneous amplitude (IA)
- `'phase'`: Instantaneous phase
- `'frequency'`: Instantaneous frequency (IF)

### Notes

- **Endpoint Effects**: Artifacts occur due to the Hilbert transform and numerical differentiation. It is recommended to use `pad` or trim the edges after analysis.
- **Smoothing**: Instantaneous frequency is sensitive to noise. Stabilization can be achieved using a moving average via `if_smooth`.

---

## `hht`

```python
TimeSeries.hht(
    emd_method: str = "eemd",
    emd_kwargs: dict | None = None,
    hilbert_kwargs: dict | None = None,
    output: str = "dict",
    n_bins: int = 100,
    freq_bins: Any = None,
	fmin: float | None = None,
	fmax: float | None = None,
	weight: str = "ia2",
	if_policy: str = "drop",
	finite_only: bool = True
) -> Any
```

### Description

Executes the full HHT process (EMD → Hilbert). Returns the result in a dictionary or as a spectrogram (Hilbert Spectrum).

### Parameters

**Common Options**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output` | str | "dict" | Output format ('dict' or 'spectrogram'). |
| `emd_method` | str | "eemd" | EMD method. |
| `emd_kwargs` | dict | None | Arguments passed to `emd()`. |
| `hilbert_kwargs` | dict | None | Arguments passed to `hilbert_analysis()`. |

**Spectrogram Options (`output='spectrogram'`)**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_bins` | int | 100 | Number of frequency bins (ignored if `freq_bins` is specified). |
| `freq_bins` | array-like | None | Custom frequency bin edges (must be monotonically increasing). |
| `fmin`/`fmax` | float | None | Frequency range. |
| `weight` | str | "ia2" | Weighting scheme ('ia2': energy, 'ia': amplitude). |
| `if_policy` | str | "drop" | Treatment of out-of-range IF ('drop': ignore, 'clip': clamp to edge). |
| `finite_only` | bool | True | **Important**: Exclude NaN/Inf values after Hilbert analysis before binning. |

### Returns

- `output='dict'`: `TimeSeriesDict` containing the analysis results for each IMF.
- `output='spectrogram'`: `gwpy.spectrogram.Spectrogram` object (Hilbert Spectrum).

### Notes

- **Spectrogram**: Unlike standard STFT spectrograms, this is constructed as a histogram of instantaneous frequencies. `.hht(weight='ia2')` is the default, representing the energy spectrum (amplitude squared).
- **Validation**: `if_policy` must be 'drop' or 'clip'. `freq_bins` must be monotonically increasing.

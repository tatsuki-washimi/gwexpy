# Noise

Utilities for detector and environmental noise models, including ASD helpers
and time-domain synthesis.
The `gwexpy.noise` module is organized into two submodules:

- `gwexpy.noise.asd`: Functions that return Amplitude Spectral Density (`FrequencySeries`)
- `gwexpy.noise.wave`: Functions that return time-series waveforms (`TimeSeries`)

## gwexpy.noise.asd

Functions for generating Amplitude Spectral Density. All functions return `FrequencySeries`.

### Detector Noise Models

| Function | Description |
|----------|-------------|
| `from_pygwinc(ifo, *, quantity='strain', fmin=10, fmax=8192, df=1.0)` | Generate ASD from pyGWINC detector noise models (aLIGO, AdV, etc.) |
| `from_obspy(model, *, quantity='displacement', fmin=0.01, fmax=100)` | Generate ASD from ObsPy seismic noise models (NLNM, NHNM) |

### Colored Noise ASD

| Function | Description |
|----------|-------------|
| `power_law(exponent, amplitude=1.0, f_ref=1.0, frequencies=...)` | Generate power-law ASD (f^-exponent) |
| `white_noise(amplitude=1.0, frequencies=...)` | White noise ASD (flat spectrum) |
| `pink_noise(amplitude=1.0, frequencies=...)` | Pink noise ASD (1/f^0.5) |
| `red_noise(amplitude=1.0, frequencies=...)` | Red/Brownian noise ASD (1/f) |

### Geomagnetic Noise Models

| Function | Description |
|----------|-------------|
| `schumann_resonance(harmonics=8, frequencies=...)` | Schumann resonance model (~7.83 Hz and harmonics) |
| `geomagnetic_background(frequencies=...)` | Background geomagnetic noise model |

### Spectral Line Shapes

| Function | Description |
|----------|-------------|
| `lorentzian_line(center, width, amplitude=1.0, frequencies=...)` | Lorentzian line shape |
| `gaussian_line(center, sigma, amplitude=1.0, frequencies=...)` | Gaussian line shape |
| `voigt_line(center, sigma, gamma, amplitude=1.0, frequencies=...)` | Voigt profile (convolution of Gaussian and Lorentzian) |

## gwexpy.noise.wave

Functions for generating time-series waveforms. All functions return `TimeSeries`.

### Noise Generators

| Function | Description |
|----------|-------------|
| `gaussian(duration, sample_rate, std=1.0, mean=0.0, ...)` | Gaussian (normal) white noise |
| `uniform(duration, sample_rate, low=-1.0, high=1.0, ...)` | Uniform white noise |
| `colored(duration, sample_rate, exponent, amplitude=1.0, ...)` | Power-law colored noise |
| `white_noise(duration, sample_rate, amplitude=1.0, ...)` | White noise (exponent=0) |
| `pink_noise(duration, sample_rate, amplitude=1.0, ...)` | Pink noise (1/f^0.5 spectrum) |
| `red_noise(duration, sample_rate, amplitude=1.0, ...)` | Red/Brownian noise (1/f spectrum) |
| `from_asd(asd, duration, sample_rate, ...)` | Generate colored noise from ASD |

### Periodic Waveforms

| Function | Description |
|----------|-------------|
| `sine(duration, sample_rate, frequency, ...)` | Sine wave |
| `square(duration, sample_rate, frequency, duty=0.5, ...)` | Square wave |
| `sawtooth(duration, sample_rate, frequency, width=1.0, ...)` | Sawtooth wave |
| `triangle(duration, sample_rate, frequency, ...)` | Triangle wave |
| `chirp(duration, sample_rate, f0, f1, method='linear', ...)` | Swept-frequency cosine (chirp) |

### Transient Signals

| Function | Description |
|----------|-------------|
| `step(duration, sample_rate, t_step=0.0, amplitude=1.0, ...)` | Step (Heaviside) function |
| `impulse(duration, sample_rate, t_impulse=0.0, amplitude=1.0, ...)` | Impulse signal |
| `exponential(duration, sample_rate, tau, decay=True, ...)` | Exponential (decay/growth) |

## Examples

```python
from gwexpy.noise.wave import sine, gaussian, chirp, from_asd
from gwexpy.noise.asd import from_pygwinc, schumann_resonance

# Sine wave
wave = sine(duration=1.0, sample_rate=1024, frequency=10.0)

# Gaussian noise
noise = gaussian(duration=1.0, sample_rate=1024, std=0.1)

# Chirp (swept sine)
sweep = chirp(duration=1.0, sample_rate=1024, f0=10, f1=100)

# Detector strain ASD from pyGWINC
asd = from_pygwinc('aLIGO', quantity='strain', fmin=4.0, fmax=1024.0, df=0.01)
noise = from_asd(asd, duration=128, sample_rate=2048, t0=0)

# Schumann resonance model
sch_asd = schumann_resonance(harmonics=5)
```


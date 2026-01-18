# Noise

Utilities for detector and environmental noise models, including ASD helpers
and time-domain synthesis.
The `gwexpy.noise` module is organized into two submodules:

- `gwexpy.noise.asd`: Functions that return Amplitude Spectral Density (`FrequencySeries`)
- `gwexpy.noise.wave`: Functions that return time-series waveforms (`TimeSeries`)

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

### Examples

```python
from gwexpy.noise.wave import sine, gaussian, chirp, from_asd
from gwexpy.noise.asd import from_pygwinc

# Sine wave
wave = sine(duration=1.0, sample_rate=1024, frequency=10.0)

# Gaussian noise
noise = gaussian(duration=1.0, sample_rate=1024, std=0.1)

# Chirp (swept sine)
sweep = chirp(duration=1.0, sample_rate=1024, f0=10, f1=100)

# Noise from ASD
asd = from_pygwinc('aLIGO', fmin=4.0, fmax=1024.0, df=0.01)
noise = from_asd(asd, duration=128, sample_rate=2048, t0=0)
```

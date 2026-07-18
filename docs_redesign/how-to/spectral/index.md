# Spectral & time-frequency

Spectrogram conditioning, Hilbert-Huang, peak detection and method comparison.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Spectrogram: Normalization and Cleaning
:link: advanced_spectrogram_processing
:link-type: doc

Convert raw spectrogram power to SNR or relative units, then remove lines and artifacts with `clean()`.
:::

:::{grid-item-card} HHT: Analysis
:link: advanced_hht
:link-type: doc

Hilbert-Huang Transform for nonlinear, non-stationary signals and instantaneous frequency extraction.
:::

:::{grid-item-card} Peak Detection: Finding Peaks
:link: advanced_peak_detection
:link-type: doc

Detect peaks in `TimeSeries` / `FrequencySeries` with physical-unit constraints on distance, height, and width.
:::

:::{grid-item-card} Peak Tracking: Time Evolution
:link: advanced_peak_tracking
:link-type: doc

Follow a spectral line through time across a spectrogram, e.g. drifting power-line harmonics or violin modes.
:::

:::{grid-item-card} Time-Frequency Analysis: Interactive Comparison
:link: time_frequency_analysis_comparison
:link-type: doc

Compare STFT and Q-transform baselines against other time-frequency methods and see when each is the right choice.
:::

::::

```{toctree}
:hidden:

advanced_spectrogram_processing
advanced_hht
advanced_peak_detection
advanced_peak_tracking
time_frequency_analysis_comparison
```

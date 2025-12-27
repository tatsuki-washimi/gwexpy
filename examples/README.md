
# gwexpy Examples & Tutorials

This directory contains Jupyter notebooks demonstrating the features and applications of the `gwexpy` library. The files are organized with prefixes to distinguish between basic feature introductions and scenario-based case studies.

## 1. Feature Introductions (`intro_*.ipynb`)
These notebooks focus on introducing specific classes and methods within the `gwexpy` library.

- **[intro_TimeSeries.ipynb](intro_TimeSeries.ipynb)**: Basic `TimeSeries` operations.
- **[intro_FrequencySeries.ipynb](intro_FrequencySeries.ipynb)**: Basic `FrequencySeries` operations.
- **[intro_TimeSeriesMatrix.ipynb](intro_TimeSeriesMatrix.ipynb)**: Multi-channel time-series handling with `TimeSeriesMatrix`.
- **[intro_FrequencySeriesMatrix.ipynb](intro_FrequencySeriesMatrix.ipynb)**: Spectral matrices (CSD, Coherence) and `FrequencySeriesMatrix`.
- **[intro_Spectrogram-containers.ipynb](intro_Spectrogram-containers.ipynb)**: `SpectrogramDict` and `SpectrogramList`.
- **[intro_SpectrogramMatrix.ipynb](intro_SpectrogramMatrix.ipynb)**: 2D/3D spectrogram handling.
- **[intro_PeakDetection.ipynb](intro_PeakDetection.ipynb)**: Finding peaks with physical unit constraints (Hz, seconds).
- **[intro_SeriesMatrix-base.ipynb](intro_SeriesMatrix-base.ipynb)**: Core logic of the `SeriesMatrix` base class.
- **[intro_time-operations.ipynb](intro_time-operations.ipynb)**: Vectorized time conversion utilities.
- **[intro_Noise.ipynb](intro_Noise.ipynb)**: Noise generation and physics-based noise models.
- **[intro_Interop.ipynb](intro_Interop.ipynb)**: Data exchange with PyTorch, TensorFlow, JAX, Dask, Polars, etc.
- **[intro_Fitting.ipynb](intro_Fitting.ipynb)**: Parameter estimation using `iminuit` and MCMC.

## 2. Case Studies & Applications (`example_*.ipynb`)
These notebooks demonstrate how to combine multiple `gwexpy` features to solve practical data analysis problems.

- **[example_signal-extraction.ipynb](example_signal-extraction.ipynb)**: Extracting weak signals from noisy backgrounds using whitening and filtering.
- **[example_calibration.ipynb](example_calibration.ipynb)**: Converting raw digital counts to physical units (displacement) using transfer functions.
- **[example_trend-analysis.ipynb](example_trend-analysis.ipynb)**: Monitoring long-term stability and detecting impulsive glitches.
- **[example_active-damping.ipynb](example_active-damping.ipynb)**: Control system analysis and noise suppression verification.
- **[example_noise-budget.ipynb](example_noise-budget.ipynb)**: Multi-channel correlation analysis and noise projection.
- **[example_lockin-detection.ipynb](example_lockin-detection.ipynb)**: Demodulation and heterodyne analysis of carrier signals.
- **[example_wiener-filter.ipynb](example_wiener-filter.ipynb)**: Multi-component noise reduction using MIMO Wiener filter matrices.
- **[example_bootstrap-spectral.ipynb](example_bootstrap-spectral.ipynb)**: Robust spectral estimation using resampling techniques.

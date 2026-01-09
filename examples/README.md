
# gwexpy Examples & Tutorials

This directory contains Jupyter notebooks demonstrating the features and applications of the `gwexpy` library. The files are organized with prefixes to distinguish between basic feature introductions and scenario-based case studies.

## 1. Feature Introductions (`basic-new-methods/intro_*.ipynb`)
These notebooks focus on introducing specific classes and methods within the `gwexpy` library.

- **[intro_TimeSeries.ipynb](basic-new-methods/intro_TimeSeries.ipynb)**: Basic `TimeSeries` operations.
- **[intro_FrequencySeries.ipynb](basic-new-methods/intro_FrequencySeries.ipynb)**: Basic `FrequencySeries` operations.
- **[intro_TimeSeriesMatrix.ipynb](basic-new-methods/intro_TimeSeriesMatrix.ipynb)**: Multi-channel time-series handling with `TimeSeriesMatrix`.
- **[intro_FrequencySeriesMatrix.ipynb](basic-new-methods/intro_FrequencySeriesMatrix.ipynb)**: Spectral matrices (CSD, Coherence) and `FrequencySeriesMatrix`.
- **[intro_Spectrogram-containers.ipynb](basic-new-methods/intro_Spectrogram-containers.ipynb)**: `SpectrogramDict` and `SpectrogramList`.
- **[intro_SpectrogramMatrix.ipynb](basic-new-methods/intro_SpectrogramMatrix.ipynb)**: 2D/3D spectrogram handling.
- **[intro_PeakDetection.ipynb](basic-new-methods/intro_PeakDetection.ipynb)**: Finding peaks with physical unit constraints (Hz, seconds).
- **[intro_SeriesMatrix-base.ipynb](basic-new-methods/intro_SeriesMatrix-base.ipynb)**: Core logic of the `SeriesMatrix` base class.
- **[intro_time-operations.ipynb](basic-new-methods/intro_time-operations.ipynb)**: Vectorized time conversion utilities.
- **[intro_Noise.ipynb](basic-new-methods/intro_Noise.ipynb)**: Noise generation and physics-based noise models.
- **[intro_Interop.ipynb](basic-new-methods/intro_Interop.ipynb)**: Data exchange with PyTorch, TensorFlow, JAX, Dask, Polars, etc.
- **[intro_Fitting.ipynb](basic-new-methods/intro_Fitting.ipynb)**: Parameter estimation using `iminuit` and MCMC.

## 2. Case Studies & Applications (`case-studies/example_*.ipynb`)
These notebooks demonstrate how to combine multiple `gwexpy` features to solve practical data analysis problems.

- **[example_signal-extraction.ipynb](case-studies/example_signal-extraction.ipynb)**: Extracting weak signals from noisy backgrounds using whitening and filtering.
- **[example_calibration.ipynb](case-studies/example_calibration.ipynb)**: Converting raw digital counts to physical units (displacement) using transfer functions.
- **[example_trend-analysis.ipynb](case-studies/example_trend-analysis.ipynb)**: Monitoring long-term stability and detecting impulsive glitches.
- **[example_active-damping.ipynb](case-studies/example_active-damping.ipynb)**: Control system analysis and noise suppression verification.
- **[example_noise-budget.ipynb](case-studies/example_noise-budget.ipynb)**: Multi-channel correlation analysis and noise projection.
- **[example_lockin-detection.ipynb](case-studies/example_lockin-detection.ipynb)**: Demodulation and heterodyne analysis of carrier signals.
- **[example_wiener-filter.ipynb](case-studies/example_wiener-filter.ipynb)**: Multi-component noise reduction using MIMO Wiener filter matrices.
- **[example_bootstrap-spectral.ipynb](case-studies/example_bootstrap-spectral.ipynb)**: Robust spectral estimation using resampling techniques.

## 3. Specialized Analysis & Tutorials (`advanced-methods/tutorial_*.ipynb`)
These notebooks provide in-depth tutorials on specific advanced analysis techniques.

- **[tutorial_HHT_Analysis.ipynb](advanced-methods/tutorial_HHT_Analysis.ipynb)**: Hilbert-Huang Transform (EMD + Hilbert Spectral Analysis) for non-stationary signals.
- **[tutorial_ARIMA_Forecast.ipynb](advanced-methods/tutorial_ARIMA_Forecast.ipynb)**: Time-series forecasting and whitening using ARIMA models.
- **[tutorial_Correlation.ipynb](advanced-methods/tutorial_Correlation.ipynb)**: Advanced statistical features including Granger Causality and Distance Correlation.
- **[tutorial_ShortTimeLaplaceTransformation.ipynb](advanced-methods/tutorial_ShortTimeLaplaceTransformation.ipynb)**: Time-frequency analysis using the Short-Time Laplace Transform (STLT).
- **[tutorial_Bruco.ipynb](advanced-methods/tutorial_Bruco.ipynb)**: Brute-force coherence analysis (BruCo) for noise hunting.


# gwexpy Examples & Tutorials

This directory is the **canonical location for all user-runnable examples and tutorials**.
The files are organized with prefixes to distinguish between basic feature introductions and scenario-based case studies.

## Role of this directory vs. the documentation site

| Location | Purpose | Audience |
| --- | --- | --- |
| `examples/` **(here)** | Executable notebooks and scripts. Clone and run locally. | Users who want hands-on practice |
| `docs/web/*/user_guide/tutorials/` | Static read-only versions embedded in the Sphinx site. | Users reading the online documentation |

The notebooks in `examples/` are the **source of truth**. Selected notebooks are referenced or rendered by the Sphinx documentation, but the runnable originals always live here.

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
- **[intro_ScalarField.ipynb](basic-new-methods/intro_ScalarField.ipynb)**: Introduction to `ScalarField` — 4D physical field type added by gwexpy.
- **[plot_ScalarField.ipynb](basic-new-methods/plot_ScalarField.ipynb)**: Visualization of `ScalarField` data.

## 2. Case Studies & Applications (`case-studies/`)

These notebooks demonstrate how to combine multiple `gwexpy` features to solve practical data analysis problems.

- **[case_signal_extraction.ipynb](case-studies/case_signal_extraction.ipynb)**: Extracting weak signals from noisy backgrounds using whitening and filtering.
- **[case_trend_analysis.ipynb](case-studies/case_trend_analysis.ipynb)**: Monitoring long-term stability and detecting impulsive glitches.
- **[case_active_damping.ipynb](case-studies/case_active_damping.ipynb)**: Control system analysis and noise suppression verification.
- **[case_noise_budget.ipynb](case-studies/case_noise_budget.ipynb)**: Multi-channel correlation analysis and noise projection.
- **[case_lockin_detection.ipynb](case-studies/case_lockin_detection.ipynb)**: Demodulation and heterodyne analysis of carrier signals.
- **[case_wiener_filter.ipynb](case-studies/case_wiener_filter.ipynb)**: Multi-component noise reduction using MIMO Wiener filter matrices.
- **[case_bootstrap_spectral.ipynb](case-studies/case_bootstrap_spectral.ipynb)**: Robust spectral estimation using resampling techniques.
- **[case_coupling_analysis.ipynb](case-studies/case_coupling_analysis.ipynb)**: Coupling function estimation between sensor channels.
- **[case_response_analysis.ipynb](case-studies/case_response_analysis.ipynb)**: Frequency-domain response characterization of physical systems.
- **[case_transfer_function.ipynb](case-studies/case_transfer_function.ipynb)**: Transfer function estimation and Bode analysis.
- **[case_cagmon_noise_diagnostics.ipynb](case-studies/case_cagmon_noise_diagnostics.ipynb)**: CAGMon-based noise diagnostics pipeline for gravitational-wave detectors.
- **[case_rayleigh_gauch.ipynb](case-studies/case_rayleigh_gauch.ipynb)**: Rayleigh and GauCh statistic tutorial for non-Gaussian noise characterization.

## 3. Specialized Analysis & Tutorials (`advanced-methods/tutorial_*.ipynb`)

These notebooks provide in-depth tutorials on specific advanced analysis techniques.

- **[advanced_hht.ipynb](../docs/web/en/user_guide/tutorials/advanced_hht.ipynb)**: Hilbert-Huang Transform (EMD + Hilbert Spectral Analysis) for non-stationary signals (published in User Guide).
- **[tutorial_ARIMA_Forecast.ipynb](advanced-methods/tutorial_ARIMA_Forecast.ipynb)**: Time-series forecasting and whitening using ARIMA models.
- **[tutorial_Correlation.ipynb](advanced-methods/tutorial_Correlation.ipynb)**: Advanced statistical features including Granger Causality and Distance Correlation.
- **[tutorial_ShortTimeLaplaceTransformation.ipynb](advanced-methods/tutorial_ShortTimeLaplaceTransformation.ipynb)**: Time-frequency analysis using the Short-Time Laplace Transform (STLT).
- **[tutorial_Bruco.ipynb](advanced-methods/tutorial_Bruco.ipynb)**: Brute-force coherence analysis (BruCo) for noise hunting.
- **[tutorial_Control_00_DiscretizationBasics.ipynb](advanced-methods/tutorial_Control_00_DiscretizationBasics.ipynb)**: Discretization theory for control systems.
- **[tutorial_Control_01_Basics.ipynb](advanced-methods/tutorial_Control_01_Basics.ipynb)**: Control engineering basics with gwexpy.
- **[tutorial_Control_02_Modeling.ipynb](advanced-methods/tutorial_Control_02_Modeling.ipynb)**: System modeling for control design.
- **[tutorial_Control_03_Design.ipynb](advanced-methods/tutorial_Control_03_Design.ipynb)**: Control system design workflow.

## 4. Paper Figures (`paper-figures/`)

Executable [marimo](https://marimo.io/) scripts that reproduce the figures in the gwexpy paper.
These serve as both reproducibility artifacts and advanced usage examples.

- **[01_transfer_function_workflow.py](paper-figures/01_transfer_function_workflow.py)**: DTT XML import, transfer function estimation, and Bode plotting.
- **[02_coherence_ranking.py](paper-figures/02_coherence_ranking.py)**: Multichannel coherence ranking (synthetic data, CI-covered).
- **[03_gwosc_case_study.py](paper-figures/03_gwosc_case_study.py)**: Public GWOSC strain data analysis (network-dependent, optional).

```bash
# Run interactively
python -m marimo run examples/paper-figures/02_coherence_ranking.py

# Or as a plain script
python examples/paper-figures/02_coherence_ranking.py
```

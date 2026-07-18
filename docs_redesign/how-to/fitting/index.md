# Fitting & statistics

Curve fitting, ARIMA, decomposition, correlation and non-Gaussian statistics.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Fitting: Spectral Line Analysis
:link: advanced_fitting
:link-type: doc

Fit `TimeSeries` or `FrequencySeries` data with `iminuit`-backed least-squares fitting and error estimation.
:::

:::{grid-item-card} ARIMA: Time Series Forecasting
:link: advanced_arima
:link-type: doc

Model and forecast time series with AR, MA, ARMA, and ARIMA/SARIMAX methods added to `TimeSeries`.
:::

:::{grid-item-card} Decomposition Analysis: PCA, ICA, and Eigenmodes
:link: advanced_decomposition
:link-type: doc

Principal and Independent Component Analysis on `TimeSeriesMatrix` via `pca()` / `ica()`.
:::

:::{grid-item-card} Correlation Analysis: Statistical Methods
:link: advanced_correlation
:link-type: doc

Pearson, Kendall, and other correlation measures between `TimeSeries` objects for noise hunting and nonlinear coupling.
:::

:::{grid-item-card} Non-Gaussian Noise Analysis: Rayleigh and Gaussian-Chi
:link: rayleigh_gauch_tutorial
:link-type: doc

A non-Gaussian noise analysis toolkit based on Rayleigh and Gaussian-Chi statistics.
:::

::::

```{toctree}
:hidden:

advanced_fitting
advanced_arima
advanced_decomposition
advanced_correlation
rayleigh_gauch_tutorial
```

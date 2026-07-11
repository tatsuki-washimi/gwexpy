# Tutorials

Learning-oriented lessons. If you are new to GWexpy, start here and work
through the pages in order. Each tutorial is a guided, hands-on lesson that
builds a complete result.

## Start here

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Installation
:link: installation
:link-type: doc

Get GWexpy and its dependencies installed.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quickstart
:link: quickstart
:link-type: doc

The shortest path from import to a first result.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Getting started
:link: getting_started
:link-type: doc

A fuller orientation to the core concepts and workflow.
:::
::::

## Core lessons

Work through these in order to learn the primary containers and operations.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` TimeSeries basics
:link: intro_timeseries
:link-type: doc

Signal processing, spectral analysis and interoperability on a time series.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` FrequencySeries basics
:link: intro_frequencyseries
:link-type: doc

Work in the frequency domain with spectra and transfer functions.
:::

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Spectrogram basics
:link: intro_spectrogram
:link-type: doc

Build and read time-frequency representations.
:::

:::{grid-item-card} {octicon}`paintbrush;1.5em;sd-mr-1` Plotting basics
:link: intro_plotting
:link-type: doc

Make and customize publication-ready figures.
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Fitting basics
:link: intro_fitting
:link-type: doc

Fit models to data with iminuit-backed helpers.
:::

:::{grid-item-card} {octicon}`broadcast;1.5em;sd-mr-1` Noise generation basics
:link: intro_noise
:link-type: doc

Synthesize colored and physically motivated noise for tests.
:::

:::{grid-item-card} {octicon}`stack;1.5em;sd-mr-1` TimeSeriesMatrix basics
:link: matrix_timeseries
:link-type: doc

Handle multi-channel data with the matrix container.
:::
::::

```{toctree}
:hidden:
:caption: Start here

installation
quickstart
getting_started
```

```{toctree}
:hidden:
:caption: Core lessons

intro_timeseries
intro_frequencyseries
intro_spectrogram
intro_plotting
intro_fitting
intro_noise
matrix_timeseries
```

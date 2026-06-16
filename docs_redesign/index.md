---
sd_hide_title: true
---

# GWexpy

::::{div} sd-text-center sd-fs-3 sd-font-weight-bold sd-mb-1
GWexpy
::::

::::{div} sd-text-center sd-fs-5 sd-text-muted sd-mb-3
An interactive, notebook-first toolkit for gravitational-wave &
multi-dimensional data analysis — built on top of [GWpy](https://gwpy.github.io).
::::

::::{div} sd-text-center sd-mb-4
```{button-ref} tutorials/index
:ref-type: myst
:color: primary
:class: sd-px-4 sd-fs-5
Launch the tutorials
```
```{button-ref} reference/index
:ref-type: myst
:color: secondary
:outline:
:class: sd-px-4 sd-fs-5
Browse the API
```
::::

---

## Featured notebooks

A living, runnable book. Open any notebook below directly in Colab or Binder.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} TimeSeries: Basics
:img-top: _static/hero_plot.png
:link: tutorials/timeseries
:link-type: myst
:class-card: sd-shadow-sm

Hilbert transforms, demodulation, lock-in detection, ARIMA and seamless
interop with Pandas / Xarray / PyTorch.
+++
{bdg-primary}`tutorial` {bdg-secondary}`timeseries`
:::

:::{grid-item-card} Active damping
:img-top: _static/case_active_damping_thumb.png
:link: how-to/case-studies/index
:link-type: myst
:class-card: sd-shadow-sm

Design and validate a feedback damping loop for a mechanical resonance
using control-system modeling.
+++
{bdg-info}`case study` {bdg-secondary}`control`
:::

:::{grid-item-card} Transfer functions
:img-top: _static/case_transfer_function_thumb.png
:link: how-to/case-studies/index
:link-type: myst
:class-card: sd-shadow-sm

Measure and model multi-input transfer functions from coherent
excitation data with uncertainty bands.
+++
{bdg-info}`case study` {bdg-secondary}`spectral`
:::

:::{grid-item-card} Noise budget
:img-top: _static/case_noise_budget_thumb.png
:link: how-to/case-studies/index
:link-type: myst
:class-card: sd-shadow-sm

Decompose a measured ASD into contributing noise terms and check the
sum against the measurement.
+++
{bdg-info}`case study` {bdg-secondary}`noise`
:::

:::{grid-item-card} Containers reference
:img-top: _static/hero_plot.png
:link: reference/containers
:link-type: myst
:class-card: sd-shadow-sm

`TimeSeriesMatrix`, `FrequencySeriesMatrix`, `ScalarField` and friends —
the multi-dimensional data model.
+++
{bdg-success}`reference`
:::

:::{grid-item-card} Architecture
:img-top: _static/case_active_damping_thumb.png
:link: explanation/architecture
:link-type: myst
:class-card: sd-shadow-sm

How GWexpy layers analysis, containers and ~40 interop backends on a
GWpy-compatible core.
+++
{bdg-warning}`explanation`
:::

::::

---

## Find your way around

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Tutorials
:link: tutorials/index
:link-type: myst
Learning-oriented notebooks. Start here.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Case studies
:link: how-to/case-studies/index
:link-type: myst
Task-oriented, end-to-end analyses.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Reference
:link: reference/index
:link-type: myst
The API: containers and analysis.
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Explanation
:link: explanation/index
:link-type: myst
Design and architecture background.
:::

::::

```{toctree}
:hidden:
:caption: Book

Home <self>
tutorials/index
how-to/case-studies/index
reference/index
explanation/index
about/index
```

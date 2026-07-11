---
sd_hide_title: true
---

# GWexpy

::::{div} sd-text-center sd-my-5
:::{image} _static/images/hero_plot.png
:alt: GWexpy multi-dimensional spectral analysis
:width: 640px
:class: sd-mb-4 sd-shadow-lg sd-rounded-3
:::

:::{div} sd-fs-1 sd-font-weight-bolder
GWexpy
:::

```{rubric} Multi-dimensional time- and frequency-series analysis for gravitational-wave science.
```

GWexpy extends [GWpy](https://gwpy.github.io/) with matrix and field containers,
integrated noise-budget and modelling tools, and broad detector interoperability.

[Get started →](tutorials/index){.sd-btn .sd-btn-primary .sd-shadow-sm .sd-px-4 .sd-fs-5}
::::

---

## Find your path

The documentation follows the [Diátaxis](https://diataxis.fr/) framework.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Tutorials
:link: tutorials/index
:link-type: doc

Learning-oriented lessons that take you from zero to a working analysis.
+++
Start learning →
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` How-to guides
:link: how-to/index
:link-type: doc

Goal-oriented recipes and real-world case studies for specific problems.
+++
Solve a problem →
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Reference
:link: reference/index
:link-type: doc

Technical description of the containers, analysis API, and I/O formats.
+++
Look up the API →
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Explanation
:link: explanation/index
:link-type: doc

Discussion of the architecture and the ideas behind the design.
+++
Understand the design →
:::
::::

---

## Highlights

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Multidimensional fields
:link: how-to/containers/index
:link-type: doc

Work natively with `TimeSeriesMatrix`, `FrequencySeriesMatrix`, and
`Scalar`/`Vector`/`TensorField` containers.
:::

:::{grid-item-card} Integrated analysis
:link: how-to/case-studies/index
:link-type: doc

BrUCo noise budgets, ARIMA modelling, and fitting / MCMC pipelines in one place.
:::

:::{grid-item-card} Broad interop
:link: how-to/interop
:link-type: doc

~50 interoperability modules for converting data to and from external scientific libraries.
:::
::::

---

## Install and try it

```bash
pip install gwexpy
```

```python
from gwexpy.timeseries import TimeSeriesMatrix

# Load a multi-channel segment and project it to the frequency domain
tsm = TimeSeriesMatrix.read("data.gwf", channels=["X1:CH1", "X1:CH2"])
fsm = tsm.fft()

# Coherence of each channel against a reference channel
ref = tsm[0, 0]  # X1:CH1 as a TimeSeries
coh = tsm.coherence(ref, fftlength=4)
```

```{toctree}
:hidden:
:caption: Documentation

tutorials/index
how-to/index
reference/index
explanation/index
about/index
```

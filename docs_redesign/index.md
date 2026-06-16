---
sd_hide_title: true
---

# GWexpy

```{div} sd-text-center sd-pt-4 sd-pb-2
```

:::{div} sd-fs-2 sd-font-weight-bold sd-text-center
Multi-dimensional time-series analysis for gravitational-wave science
:::

:::{div} sd-fs-5 sd-text-muted sd-text-center sd-pb-3
A calm, focused extension of GWpy for matrices, fields, and the analysis
methods that work over them.
:::

```{button-ref} tutorials/index
:ref-type: doc
:color: primary
:class: sd-px-4 sd-fs-5 sd-rounded-pill sd-mx-auto sd-d-block sd-w-25

Get started
```

```{image} _static/images/hero_plot.png
:alt: Example GWexpy analysis plot
:width: 60%
:align: center
:class: sd-my-5 sd-rounded
```

---

## Where to go next

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`mortar-board` Tutorials
:link: tutorials/index
:link-type: doc

Guided, learning-oriented lessons. Start here if GWexpy is new to you.
:::

:::{grid-item-card} {octicon}`tools` How-to guides
:link: how-to/index
:link-type: doc

Goal-oriented recipes and real case studies for concrete problems.
:::

:::{grid-item-card} {octicon}`book` Reference
:link: reference/index
:link-type: doc

The container and analysis API, generated from the source.
:::

:::{grid-item-card} {octicon}`light-bulb` Explanation
:link: explanation/index
:link-type: doc

Background and design — why GWexpy is shaped the way it is.
:::

::::

---

## Install and try it

```console
$ pip install gwexpy
```

```python
import numpy as np
from gwexpy_demo import TimeSeriesMatrix

# A 3-channel, 1-second snippet sampled at 256 Hz.
data = np.random.normal(size=(3, 256))
tsm = TimeSeriesMatrix(data, sample_rate=256.0)

print(tsm.n_channels, tsm.duration)
psd = tsm.psd()
```

```{toctree}
:hidden:
:caption: Documentation

tutorials/index
how-to/index
reference/index
explanation/index
```

```{toctree}
:hidden:
:caption: Project

about/index
```

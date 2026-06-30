# How-to guides

Goal-oriented recipes. These pages assume you already know the basics
(see the [tutorials](../tutorials/index.md)) and answer the question
*"how do I accomplish X?"*.

## Data and formats

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`file-binary;1.5em;sd-mr-1` Reading & writing data
:link: io_formats
:link-type: doc

Supported file formats and direct `.read()` / `.write()` paths.
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Interop & conversion
:link: interop
:link-type: doc

Convert to and from external libraries with `to_*()` / `from_*()`.
:::

:::{grid-item-card} {octicon}`stack;1.5em;sd-mr-1` Slicing scalar fields
:link: scalarfield_slicing
:link-type: doc

Work with 4D field containers and extract slices.
:::

:::{grid-item-card} {octicon}`clock;1.5em;sd-mr-1` Time & GPS utilities
:link: time_utilities
:link-type: doc

Convert between GPS, UTC, and human-readable times.
:::
::::

## Tools and operations

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting
:link-type: doc

Diagnose and fix common problems.
:::

:::{grid-item-card} {octicon}`versions;1.5em;sd-mr-1` Migration
:link: migration
:link-type: doc

Upgrade notes and breaking changes between releases.
:::
::::

## Technique recipes

Task-focused notebooks grouped by analysis domain. Each card opens a small
gallery of worked recipes.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`stack;1.5em;sd-mr-1` Containers & matrices
:link: containers/index
:link-type: doc

Matrix and field containers that extend GWpy's series types.
:::

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Spectral & time-frequency
:link: spectral/index
:link-type: doc

Spectrograms, Hilbert-Huang, peak detection and method comparison.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Fitting & statistics
:link: fitting/index
:link-type: doc

Curve fitting, ARIMA, decomposition and non-Gaussian statistics.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Control & modeling
:link: control/index
:link-type: doc

Resonance, feedback, discretization and plant modeling.
:::

:::{grid-item-card} {octicon}`git-merge;1.5em;sd-mr-1` Coupling & correlation
:link: coupling/index
:link-type: doc

Track contributions across auxiliary channels and fields.
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Visualization & interop
:link: viz/index
:link-type: doc

Map plotting and conversion to/from external libraries.
:::
::::

## Case studies

::::{grid} 1 1 1 1
:gutter: 3

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Case-study gallery
:link: case-studies/index
:link-type: doc

A gallery of complete, real-world analyses you can adapt.
:::
::::

```{toctree}
:hidden:
:caption: Data and formats

io_formats
interop
scalarfield_slicing
time_utilities
```

```{toctree}
:hidden:
:caption: Tools and operations

troubleshooting
migration
```

```{toctree}
:hidden:
:caption: Technique recipes

containers/index
spectral/index
fitting/index
control/index
coupling/index
segments/index
viz/index
```

```{toctree}
:hidden:
:caption: Case studies

case-studies/index
```

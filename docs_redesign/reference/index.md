# Reference

Technical, information-oriented description of the GWexpy API. These pages are
generated directly from the package docstrings by `autosummary` and `autodoc`,
and describe the package used for this documentation build. Check the build information
against `gwexpy.__version__` in your own environment.

:::{note}
API reference bodies are generated from the source docstrings, which are
written in English. This holds on the Japanese build of this site as well
(only the surrounding navigation and page chrome are translated); translating
the docstrings themselves is out of scope for now.
:::

## Public API and support

Use documented public entry points. A generated member listing alone does not
establish a supported contract. Experimental and implementation-only paths keep
the status stated in their guide; pre-1.0 APIs are not universally stable.
See [Known limitations](../about/known_limitations.md),
[I/O capabilities](io_capabilities.md), and [Conversion capabilities](interop_capabilities.md)
for supported classes, dependencies, and metadata boundaries.

## API by domain

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`container;1.5em;sd-mr-1` Containers
:link: api/timeseries
:link-type: doc

`TimeSeries`, `FrequencySeries`, `Spectrogram`, their matrix and dictionary
forms, fields, tables, and histograms.
:::

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Analysis & signal processing
:link: api/spectral
:link-type: doc

Spectral estimation, filtering, coupling/statistics, fitting, noise models,
preprocessing, and segments.
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Interoperability
:link: api/interop
:link-type: doc

`to_*()` / `from_*()` bridges to external scientific libraries and data models.
:::

:::{grid-item-card} {octicon}`file-binary;1.5em;sd-mr-1` I/O
:link: api/io
:link-type: doc

Multi-format readers and writers across the GW data ecosystem.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Plotting & tools
:link: api/plot
:link-type: doc

Plotting helpers, time/GPS utilities, and compatibility shims.
:::

:::{grid-item-card} {octicon}`git-compare;1.5em;sd-mr-1` GWpy differences
:link: gwpy_added_api
:link-type: doc

What GWexpy adds on top of, or changes relative to, GWpy.
:::
::::

```{toctree}
:hidden:
:caption: Containers

api/timeseries
api/frequencyseries
api/spectrogram
api/matrix
api/fields
api/types
api/table
api/histogram
```

```{toctree}
:hidden:
:caption: Analysis & signal processing

api/spectral
api/signal
api/analysis
api/fitting
api/noise
api/preprocessing
api/segments
api/detector
api/astro
```

```{toctree}
:hidden:
:caption: Interoperability & I/O

api/interop
api/io
io_capabilities
interop_capabilities
```

```{toctree}
:hidden:
:caption: Plotting & tools

api/plot
api/time
```

```{toctree}
:hidden:
:caption: GWpy differences

gwpy_added_api
```

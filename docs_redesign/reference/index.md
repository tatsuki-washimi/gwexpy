# Reference

Technical, information-oriented description of the GWexpy API. These pages are
generated directly from the package docstrings by `autosummary` and `autodoc`,
so they always match the installed version.

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

`to_*()` / `from_*()` bridges to ~50 external libraries and data models.
:::

:::{grid-item-card} {octicon}`file-binary;1.5em;sd-mr-1` I/O
:link: api/io
:link-type: doc

Multi-format readers and writers across the GW data ecosystem.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Plotting & tools
:link: api/plot
:link-type: doc

Plotting helpers, time/GPS utilities, the CLI, the GUI, and compatibility
shims.
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

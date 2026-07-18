# Containers & matrices

Work with the matrix and field containers that extend GWpy's series types.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} FrequencySeriesMatrix: Matrix Basics
:link: matrix_frequencyseries
:link-type: doc

A 3D container (row x col x freq) for batches of `FrequencySeries` with frequency-domain-aware indexing.
:::

:::{grid-item-card} SpectrogramMatrix: Matrix Basics
:link: matrix_spectrogram
:link-type: doc

Handle multiple `Spectrogram` objects as a single 3D (batch, time, frequency) container.
:::

:::{grid-item-card} Field API: ScalarField Basics
:link: field_scalar_intro
:link-type: doc

Basic usage of `ScalarField` for 4D (time + 3D space) data, including indexing and metadata handling.
:::

:::{grid-item-card} Field API: Advanced Analysis Workflow
:link: field_advanced_workflow
:link-type: doc

A full analysis workflow built on `ScalarField`, going beyond the basics into applied field processing.
:::

:::{grid-item-card} Scalar Field Slicing Guide (Why 4D is Preserved)
:link: scalarfield_slicing
:link-type: doc

Why `ScalarField` indexing always keeps 4 dimensions instead of collapsing like NumPy, and when `squeeze()` is safe.
:::

:::{grid-item-card} Histogram: Basics
:link: intro_histogram
:link-type: doc

Create, visualize, rebin, and compute statistics on `gwexpy.histogram.Histogram` objects.
:::

::::

```{toctree}
:hidden:

matrix_frequencyseries
matrix_spectrogram
field_scalar_intro
field_advanced_workflow
scalarfield_slicing
intro_histogram
```

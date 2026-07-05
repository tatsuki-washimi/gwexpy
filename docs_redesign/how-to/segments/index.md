# Segments & tables

Build, visualize and analyze data-quality segments and tabular results.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} ASD Analysis: Pipeline
:link: segment_asd_pipeline
:link-type: doc

Batch ASD analysis across `SegmentTable` rows: crop data per segment, compute ASDs, and visualize the variation.
:::

:::{grid-item-card} Segment Analysis: Visualization
:link: segment_visualization
:link-type: doc

Compare spectra across multiple segments with `overlay_spectra()`.
:::

:::{grid-item-card} SegmentTable: Basics
:link: intro_segment_table
:link-type: doc

How GWpy `Segment` types extend into `SegmentTable` for building and managing segment lists.
:::

:::{grid-item-card} Segment Analysis: Basic Pipeline
:link: intro_table
:link-type: doc

Use `SegmentTable` to manage time-keyed data with lazy-loading, plus visualization and GravitySpy integration.
:::

::::

```{toctree}
:hidden:

segment_asd_pipeline
segment_visualization
intro_segment_table
intro_table
```

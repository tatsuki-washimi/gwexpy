# Monitoring and Event Workflows

Goal-oriented recipes for analyzing stored slow monitoring data, correlating transient event catalogs with multi-channel detector time series, and processing long continuous observations with bounded memory.

## Analysis Pathways

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Long-term Trends
:link: long_term_trend
:link-type: doc

Separate slow diurnal/linear baseline drift from short pulse transients and persistent state steps.
:::

:::{grid-item-card} {octicon}`clock;1.5em;sd-mr-1` Event Alignment
:link: event_catalog_timeseries
:link-type: doc

Refine discrete catalog event times against witness sensors and align multi-channel time windows.
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Chunked Processing
:link: chunked_long_data
:link-type: doc

Stream long time-series records in bounded blocks, maintaining global PSD continuity and atomic checkpoints.
:::
::::

## Workflow Comparison

| Workflow | Primary Input | Key Output | Required Extras | Missing Data / Boundary Handling |
|---|---|---|---|---|
| **Trend & State Analysis** | Multi-channel 1-minute slow records | Estimated baseline, pulse & step candidate tables | `core` | Valid contiguous runs, edge exclusion flags |
| **Catalog & Waveform Alignment** | Event timestamp catalog + time series | Refined event times, channel-aligned arrays | `core` | Half-open intervals, partial / gap / no_peak status |
| **Chunked Streaming** | Multi-channel file on disk (HDF5) | Streaming PSD, time-binned RMS, checkpoints | `core` | Bounded memory buffer, resume consistency |

## Recommended Progression

For post-run diagnostic investigations, we recommend starting with **Long-term Trends** to identify anomalous quiet periods and operational state transitions. When examining specific glitch or trigger times, follow **Event Alignment** to verify witness coupling across auxiliary channels. For multi-hour or multi-day datasets that exceed available RAM, use **Chunked Processing** to aggregate summary metrics without loading full waveforms.

## Related Resources

- [Segment Tables](../segments/intro_segment_table.ipynb): Construct and manipulate time segments.
- [Case Study: Segment Analysis](../case-studies/case_segment_analysis.ipynb): End-to-end data quality vetoing workflow.
- [Time & GPS Utilities](../time_utilities.md): Conversions between GPS, UTC, and sample index grids.

```{toctree}
:hidden:

long_term_trend
event_catalog_timeseries
chunked_long_data
```

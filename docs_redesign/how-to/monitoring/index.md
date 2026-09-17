# Monitoring and Event Workflows

Goal-oriented recipes for analyzing stored slow monitoring records, correlating transient event catalogs with multi-channel detector time series, and processing long continuous observations with bounded memory.

:::{note}
These tutorials cover **offline and post-run diagnostic workflows** on recorded data files. They are designed for reproducible batch analysis and do not implement an online, 24/7 real-time telemetry daemon.
:::

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

| Workflow | Primary Input | Key Output | Required Dependencies | Missing Data / Boundary Handling |
|---|---|---|---|---|
| **Trend & State Analysis** | Multi-channel 1-minute slow records | Estimated baseline, pulse & step candidate tables | `numpy`, `scipy`, `gwpy`, `pandas` | Valid contiguous runs, edge exclusion flags |
| **Catalog & Waveform Alignment** | Event timestamp catalog + time series | Refined event times, channel-aligned arrays | `numpy`, `scipy`, `gwpy`, `pandas` | Half-open intervals, partial / gap / no_peak status |
| **Chunked Streaming** | Multi-channel file on disk (HDF5) | Streaming PSD, time-binned RMS, checkpoints | `numpy`, `scipy`, `gwpy`, `h5py` | Bounded memory buffer, resume consistency |

## Recommended Progression

- **T1 to T2 Sequential Pathway**: In diagnostic investigations, start with **Long-term Trends (T1)** to isolate anomalous intervals, transient bursts, and operational state steps across slow monitor channels. The resulting candidate event timestamps feed directly into **Event Alignment (T2)**, where discrete triggers are refined against high-bandwidth witness channels and converted to structured `SegmentTable` objects.
- **Independent High-Throughput Pathway (T5)**: When analyzing multi-hour or multi-day datasets that exceed available workstation RAM, use **Chunked Processing (T5)** independently. It provides bounded-memory sequential streaming, stateful digital filtering (`sosfilt(..., zi=...)`), cumulative PSD estimation, and atomic checkpointing without loading the full recording into memory.

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

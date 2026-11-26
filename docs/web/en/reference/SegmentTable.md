# SegmentTable

<!-- reference-summary:start -->

**Stability:** Stable

## What it is

`SegmentTable` is a standalone container for time segments and associated
metadata. It is not a `pandas.DataFrame` subclass or a `gwpy.table.Table`
subclass.
It stores a pandas DataFrame internally, but its public object is the
`SegmentTable` container itself.

## Representative signatures

```text
SegmentTable.from_segments(segments: 'Sequence[Any]', **meta_columns: 'Sequence[Any]') -> 'SegmentTable'
SegmentTable.from_table(table: 'Any', span: 'str' = 'span') -> 'SegmentTable'
SegmentTable.read_csv(filepath: 'str', span_cols: 'tuple[str, str]' = ('start', 'end'), **kwargs: 'Any') -> 'SegmentTable'
SegmentTable.read(source: 'Any', format: 'Optional[str]' = None, **kwargs: 'Any') -> 'SegmentTable'
SegmentTable.write(target: 'Any', format: 'Optional[str]' = None, *, allow_pickle: 'bool' = False, **kwargs: 'Any') -> 'None'
SegmentTable.extract_stat(stat_fn: 'str | Callable[[Any], Any]', column: 'str', out_column: 'str | None' = None, *, parallel: 'bool | int' = False) -> 'SegmentTable'
SegmentTable.crop(column: 'str', out_col: 'Optional[str]' = None, span_col: 'str' = 'span', inplace: 'bool' = False, errors: 'str' = 'raise', *, parallel: 'bool | int' = False) -> 'SegmentTable'
SegmentTable.asd(column: 'str', out_col: 'Optional[str]' = None, *, errors: 'str' = 'raise', inplace: 'bool' = False, parallel: 'bool | int' = False, **kwargs: 'Any') -> 'SegmentTable'
SegmentTable.bandpass(column: 'str', flow: 'float', fhigh: 'float', out_col: 'Optional[str]' = None, *, errors: 'str' = 'raise', inplace: 'bool' = False, parallel: 'bool | int' = False, **kwargs: 'Any') -> 'SegmentTable'
SegmentTable.resample(column: 'str', rate: 'float', out_col: 'Optional[str]' = None, *, errors: 'str' = 'raise', inplace: 'bool' = False, parallel: 'bool | int' = False, **kwargs: 'Any') -> 'SegmentTable'
SegmentTable.groupby(keys: 'str | Sequence[str]') -> 'SegmentTableGroupBy'
SegmentTable.plot(column: 'Optional[str]' = None, *, row: 'Optional[int]' = None, mode: 'Optional[str]' = None, **kwargs: 'Any') -> 'Any'
SegmentTable.scatter(x: 'str', y: 'str', color: 'Optional[str]' = None, *, selection: 'Optional[Any]' = None, **kwargs: 'Any') -> 'Any'
SegmentTable.hist(column: 'str', *, bins: 'int' = 10, **kwargs: 'Any') -> 'Any'
SegmentTable.segments(*, y: 'Optional[str]' = None, color: 'Optional[str]' = None, **kwargs: 'Any') -> 'Any'
SegmentTable.overlay(column: 'str', rows: 'list[int]', *, separate: 'bool' = False, sharex: 'bool' = True, **kwargs: 'Any') -> 'Any'
SegmentTable.overlay_spectra(column: 'str', *, channel: 'Optional[str]' = None, rows: 'Optional[list[int]]' = None, color_by: 'Optional[str]' = None, sort_by: 'Optional[str]' = None, cmap: 'str' = 'viridis', alpha: 'float' = 0.7, linewidth: 'float' = 0.8, colorbar: 'bool' = True, colorbar_label: 'Optional[str]' = None, xscale: 'str' = 'log', yscale: 'str' = 'log', xlim: 'Optional[Any]' = None, ylim: 'Optional[Any]' = None, ax: 'Optional[Any]' = None) -> 'Any'
```

`read` is a format-dispatching reader supporting both CSV (delegating `.csv` to
`read_csv`) and HDF5 (`.h5`/`.hdf5`/`.hdf`, lazily restoring payload cells by
default and validating `gwexpy_schema` / `gwexpy_schema_version`).
Reading a pickled payload requires an explicit `allow_pickle=True` and should
be used only for files from a trusted source.
`write` persists the table to HDF5 (metadata, schema, cell statuses, payloads via
the payload serializer registry without pickle unless `allow_pickle=True`, and
provenance) or to CSV.
`extract_stat` returns a new table with one scalar per segment. Built-in aliases
are `mean`, `median`, `rms`, `min`, and `max`; a callable receives the whole
payload. Results are numeric scalar columns with physical units in column
metadata and a `<name>_status` column for
missing or masked source cells. `parallel=True` uses available CPU threads;
`parallel=N` uses N workers.
`crop`, `asd`, `bandpass`, and `resample` also accept `parallel=False` (default), `True` (available CPU
threads), or a positive worker count. Their row operations run concurrently
while output order and `errors=` status handling stay aligned with the input.
Only the requested payload column is loaded, plus `span_col` for `crop`.
`table.frame()` creates a deferred, sequential pipeline. `define(...)` and
`filter(...)` return new branches; `collect()`, `count()`, and `report()`
execute each branch afresh. The frame snapshots table structure and cell
state, while already loaded payload values are shared.
`explode_channels(column)` expands series dictionaries into one row per
channel, and `expand_bands(column, bands)` crops frequency series into named
bands using GWpy's endpoint semantics. Both require `load=True` to resolve
unloaded source payloads; source cell states remain unchanged.
`stack_spectra(column, groupby=None, statistic="median", interval=(0.16, 0.84), align="strict")`
returns a `SummaryTable` with one row per frequency and group, including units,
status counts, and per-frequency `n_bin`. `intersection` and `union` use exact
frequency matches; `interpolate` linearly interpolates onto the first valid
spectrum's grid in each group, without extrapolation. `statistic="percentile"`
accepts a `percentile` in the range 0–100.
`n_interpolated` counts interpolated contributions within `n_bin`; the
difference is the number of contributions at original frequency coordinates.
Frequency matches are exact after conversion to float64 Hz.
`bandpass` and `resample` apply the corresponding GWpy TimeSeries method to each
series, including each member of a TimeSeriesDict; their additional keyword
arguments are forwarded to that method.
For integer payloads, `median` preserves exact integral results. A large
half-integer median that binary64 cannot represent exactly raises `ValueError`;
use an exact-arithmetic workflow outside the scalar table column if a rational
result is required.
Exact integer results remain integers when other rows are missing or contain
floating-point results, including after an HDF5 round trip.

`groupby(keys).aggregate(...)` groups eager scalar metadata by one or more
metadata keys and returns a `SummaryTable` without a `span` column. Named
aggregates use `name=(source, method)`, where `method` is `count`, `mean`,
`median`, `min`, `max`, or `quantile(q)` from `gwexpy.table.segment`.
One call uses one source column so the always-present `n_total`, `n_valid`,
`n_missing`, `n_failed`, and `n_masked` counts have one meaning. By default,
invalid source rows are skipped; `skipna=False` raises instead. `min_count`
controls statistics but `count` always reports valid rows. Missing group keys
are rejected. Apply `extract_stat` first to group a payload-derived statistic.
`SummaryTable.to_pandas()` and `.show()` return copies; `.plot(x, y)` plots a
summary column; `.write(path)` supports CSV or unit-aware JSON snapshots.
Integer extrema and integral mean or quantile results remain exact even for
mixed signed and unsigned 64-bit values. Fractional integer statistics use
binary64 when its absolute rounding error is at most `1e-12`; otherwise they
raise `ValueError`. Thus ordinary decimal quantiles such as `0.1` remain
usable while large values cannot silently lose a fractional part.

## Minimal example

```python
import matplotlib

matplotlib.use("Agg")

from gwpy.segments import Segment
from gwexpy.table import SegmentTable

segments = SegmentTable.from_segments([Segment(0, 1), Segment(2, 3)])
plot = segments.segments()
import matplotlib.pyplot as plt

plt.close(plot)
```

## Span representations

The constructor, `from_segments`, and `from_table` require each `span` value
to be a `gwpy.segments.Segment` object. A plain `(start, end)` tuple or list
is not accepted by those APIs.

`read_csv` accepts these span forms:

- With no `span` column, numeric `start`/`end` columns (or the two names in
  `span_cols`) are converted to `Segment(float(start), float(end))`.
- An existing `gwpy.segments.Segment` value is kept as-is.
- A span string may be `(start, end)`, `Segment(start, end)`, or
  `[start ... end)`, with exactly two numeric endpoints.

## Plot helpers

The table methods `plot`, `scatter`, `hist`, `segments`, `overlay`, and
`overlay_spectra` delegate to the corresponding payload or table-level plot
helper. Plot methods return a plot object and do not call `show()` themselves.
`SegmentTable` does not define `step` or `bar` methods.

## Related tutorials

- [SegmentTable: Basics](../user_guide/tutorials/intro_segment_table.ipynb)
- [Segment ASD Pipeline](../user_guide/tutorials/segment_asd_pipeline.ipynb)
- [Segment Visualization](../user_guide/tutorials/segment_visualization.ipynb)

## API reference

.. currentmodule:: gwexpy.table

.. autoclass:: SegmentTable
   :members:

<!-- reference-summary:end -->

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
SegmentTable.read(filepath: 'str', span_cols: 'tuple[str, str]' = ('start', 'end'), **kwargs: 'Any') -> 'SegmentTable'
SegmentTable.plot(column: 'Optional[str]' = None, *, row: 'Optional[int]' = None, mode: 'Optional[str]' = None, **kwargs: 'Any') -> 'Any'
SegmentTable.scatter(x: 'str', y: 'str', color: 'Optional[str]' = None, *, selection: 'Optional[Any]' = None, **kwargs: 'Any') -> 'Any'
SegmentTable.hist(column: 'str', *, bins: 'int' = 10, **kwargs: 'Any') -> 'Any'
SegmentTable.segments(*, y: 'Optional[str]' = None, color: 'Optional[str]' = None, **kwargs: 'Any') -> 'Any'
SegmentTable.overlay(column: 'str', rows: 'list[int]', *, separate: 'bool' = False, sharex: 'bool' = True, **kwargs: 'Any') -> 'Any'
SegmentTable.overlay_spectra(column: 'str', *, channel: 'Optional[str]' = None, rows: 'Optional[list[int]]' = None, color_by: 'Optional[str]' = None, sort_by: 'Optional[str]' = None, cmap: 'str' = 'viridis', alpha: 'float' = 0.7, linewidth: 'float' = 0.8, colorbar: 'bool' = True, colorbar_label: 'Optional[str]' = None, xscale: 'str' = 'log', yscale: 'str' = 'log', xlim: 'Optional[Any]' = None, ylim: 'Optional[Any]' = None, ax: 'Optional[Any]' = None) -> 'Any'
```

`read` is an alias for `read_csv`: the very same classmethod object, with the
same signature, so it reads CSV and nothing else.
`SegmentTable` has no `write` method.

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

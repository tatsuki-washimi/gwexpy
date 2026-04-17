# SegmentTable

<!-- reference-summary:start -->

**Stability:** Stable

## What it is

Use `SegmentTable` to store or manipulate time segments together with metadata needed for data-quality and analysis windows.

## Representative Signatures

```python
SegmentTable.from_segments(segments, **kwargs)
SegmentTable.plot(**kwargs)
```

## Minimal Example

```python
from gwexpy.table import SegmentTable

segments = SegmentTable.from_segments([(0, 1), (2, 3)])
plot = segments.plot()
```

## Related Theory

- [Validated Algorithms](../user_guide/validated_algorithms.md)

## Related Tutorials

- [Tutorial Index](../user_guide/tutorials/index.rst)
- [Getting Started](../user_guide/getting_started.md)

## API Reference

The detailed generated API continues below on this page.

<!-- reference-summary:end -->


**Inherits from:** [GWpy table documentation](https://gwpy.readthedocs.io/en/latest/table/)

A container for time segments and associated metadata, extending the standard GWpy/Astropy Table functionality.

## Overview

`SegmentTable` is designed to store lists of segments (start and end times) along with arbitrary columns of metadata (e.g., flags, process IDs, or amplitude values). It provides specialized methods for segment intersection, union, and plotting.

## Methods

### `read`

```python
read(source, format=None, **kwargs)
```

Read a SegmentTable from a file.

### `write`

```python
write(target, format=None, **kwargs)
```

Write the SegmentTable to a file.

### `plot`

```python
plot(**kwargs)
```

Visualize the segments in the table.

## API Reference

.. currentmodule:: gwexpy.table

.. autoclass:: SegmentTable
   :members:
   :show-inheritance:

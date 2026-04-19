# SegmentCell

<!-- reference-summary:start -->

**Stability:** Stable

## What it is

Use `SegmentCell` for a single metadata-aware time interval, especially when it will live inside a `SegmentTable`.

## Representative Signatures

```python
SegmentCell(start, end)
SegmentCell.duration
```

## Minimal Example

```python
from gwexpy.table import SegmentCell

segment = SegmentCell(0, 10)
```

## Related Theory

- [Prerequisites and Conventions](../user_guide/prerequisites_and_conventions.md)
- [Validated Algorithms](../user_guide/validated_algorithms.md)

## Related Tutorials

- [SegmentTable: Basics](../user_guide/tutorials/intro_segment_table.ipynb)
- [Segment Analysis Case Study](../user_guide/tutorials/case_segment_analysis.ipynb)

## API Reference

The detailed generated API continues below on this page.

<!-- reference-summary:end -->


**Inherits from:** [`gwpy.segments.Segment`](https://gwpy.readthedocs.io/en/latest/api/gwpy.segments.Segment/)

Represents a single time segment (start, end) with optional metadata.

## Overview

`SegmentCell` extends the basic interval functionalities and can be stored within a `SegmentTable` or as individual metadata-aware intervals.

## API Reference

.. currentmodule:: gwexpy.table

.. autoclass:: SegmentCell
   :members:
   :show-inheritance:

# SegmentCell

<!-- reference-summary:start -->

## What it is

Use `SegmentCell` to store or manipulate time segments together with metadata needed for data-quality and analysis windows.

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

- [Validated Algorithms](../user_guide/validated_algorithms.md)

## Related Tutorials

- [Tutorial Index](../user_guide/tutorials/index.rst)
- [Getting Started](../user_guide/getting_started.md)

## API Reference

The detailed generated API continues below on this page.

<!-- reference-summary:end -->


**Inherits from:** `gwpy.segments.Segment`

Represents a single time segment (start, end) with optional metadata.

## Overview

`SegmentCell` extends the basic interval functionalities and can be stored within a `SegmentTable` or as individual metadata-aware intervals.

## API Reference

.. currentmodule:: gwexpy.table

.. autoclass:: SegmentCell
   :members:
   :show-inheritance:

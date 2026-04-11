# SegmentTable

**Inherits from:** `gwpy.table.Table`

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

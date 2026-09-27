"""Inspect public matrix read metadata for instrument formats."""

import json
from pathlib import Path

from gwexpy.timeseries import TimeSeriesMatrix

root = Path(__file__).resolve().parent / "fixtures"
for fmt, filename, kwargs in (
    ("tdms", "TDMS-TIME-VALID-START-NOROOT.tdms", {}),
    ("gbd", "GBD-CONTROL-001.gbd", {"timezone": "UTC"}),
):
    matrix = TimeSeriesMatrix.read(root / fmt / filename, format=fmt, **kwargs)
    cells = [[repr(matrix.meta[i, j]) for j in range(matrix.shape[1])]
             for i in range(matrix.shape[0])]
    print(json.dumps({"format": fmt, "shape": matrix.shape,
                      "meta": repr(getattr(matrix, "meta", None)),
                      "cells": cells,
                      "row_names": repr(getattr(matrix, "row_names", None)),
                      "col_names": repr(getattr(matrix, "col_names", None))}, default=str))

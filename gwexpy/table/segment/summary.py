"""Non-segment summary rows returned by eager SegmentTable aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u


def _json_scalar(value: Any) -> Any:
    """Convert a primitive summary scalar to strict JSON form."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("SummaryTable cannot write non-finite JSON values")
        return value
    raise TypeError(f"SummaryTable JSON does not support {type(value).__name__}")


class SummaryTable:
    """Eager grouped summary with no requirement for a ``span`` column.

    ``to_pandas`` and ``show`` return copies. ``write`` supports CSV for simple
    exchange and JSON for a snapshot that also records column units.
    """

    def __init__(self, frame: pd.DataFrame, *, units: dict[str, Any] | None = None):
        self._frame = frame.copy(deep=True).reset_index(drop=True)
        self._units = dict(units or {})

    @property
    def units(self) -> dict[str, u.UnitBase]:
        """Return a copy of column unit metadata."""
        return {name: u.Unit(unit) for name, unit in self._units.items()}

    def __len__(self) -> int:
        """Return the number of summary groups."""
        return len(self._frame)

    def to_pandas(self) -> pd.DataFrame:
        """Return a detached DataFrame copy of the summary rows."""
        return self._frame.copy(deep=True)

    def show(self, max_rows: int = 20) -> pd.DataFrame:
        """Return the first *max_rows* rows for notebook or terminal display."""
        if max_rows < 0:
            raise ValueError("max_rows must be nonnegative")
        return self._frame.head(max_rows).copy(deep=True)

    def plot(self, x: str, y: str, *, kind: str = "bar", **kwargs: Any) -> Any:
        """Plot one numeric summary column against a group key."""
        if x not in self._frame or y not in self._frame:
            raise KeyError(f"Unknown SummaryTable plot column: {x!r} or {y!r}")
        return self._frame.plot(x=x, y=y, kind=kind, **kwargs)

    def write(self, target: str | Path, *, format: str | None = None) -> None:
        """Write CSV or a unit-aware JSON snapshot, inferred from suffix."""
        path = Path(target)
        fmt = format or path.suffix.lstrip(".").lower()
        if fmt == "csv":
            self._frame.to_csv(path, index=False)
            return
        if fmt == "json":
            columns = list(self._frame.columns)
            doc = {
                "schema": "gwexpy.summary_table",
                "schema_version": 1,
                "columns": columns,
                "units": {name: str(unit) for name, unit in self.units.items()},
                "rows": [
                    [_json_scalar(value) for value in row]
                    for row in self._frame.itertuples(index=False, name=None)
                ],
            }
            path.write_text(json.dumps(doc, allow_nan=False, indent=2) + "\n")
            return
        raise ValueError("SummaryTable.write supports format='csv' or 'json'")

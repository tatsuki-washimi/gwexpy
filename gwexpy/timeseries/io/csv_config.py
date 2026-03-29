"""Configuration schema for the enhanced CSV reader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a single CSV column.

    Parameters
    ----------
    name : str
        Channel/column name used in the output TimeSeriesDict.
    column_index : int
        Zero-based column index in the CSV file.
    unit : str or None
        Astropy-compatible unit string (e.g. ``"m/s^2"``, ``"Pa"``).
    role : {'data', 'time', 'time_component', 'skip'}
        Role of this column:

        - ``'data'``: Data channel to include in output.
        - ``'time'``: Single column containing the full timestamp.
        - ``'time_component'``: One part of a multi-column timestamp.
        - ``'skip'``: Ignore this column.
    time_component : str or None
        For ``role='time_component'``: which component this column represents.
        One of ``'year'``, ``'month'``, ``'day'``, ``'hour'``, ``'minute'``,
        ``'second'``.
    scale_factor : float
        Multiplicative factor applied to the raw values. Default 1.0.
    """

    name: str
    column_index: int
    unit: str | None = None
    role: str = "data"
    time_component: str | None = None
    scale_factor: float = 1.0


@dataclass(frozen=True)
class CSVFormatConfig:
    """Configuration for reading a CSV file with flexible column mapping.

    Parameters
    ----------
    columns : list of ColumnSpec
        Column definitions.
    delimiter : str
        Field separator. Default ``","``
    skip_rows : int or None
        Number of rows to skip at the start. ``None`` = auto-detect.
    header_row : int or None
        Row number (0-based, after *skip_rows*) containing column headers.
        ``None`` = auto-detect or no header.
    encoding : str
        File encoding. Default ``"utf-8"``.
    timezone : str or None
        Timezone for timestamps (e.g. ``"Asia/Tokyo"``, ``"+09:00"``).
    sample_rate : float or None
        Expected uniform sample rate in Hz. If ``None``, infer from data.
    resample_method : str or None
        Method to resample non-uniform data to uniform spacing:
        ``"interpolate"`` or ``"asfreq"``. ``None`` = no resampling.
    comment_char : str
        Character(s) marking comment lines. Default ``"#"``.
    """

    columns: list[ColumnSpec] = field(default_factory=list)
    delimiter: str = ","
    skip_rows: int | None = None
    header_row: int | None = None
    encoding: str = "utf-8"
    timezone: str | None = None
    sample_rate: float | None = None
    resample_method: str | None = None
    comment_char: str = "#"

    @classmethod
    def from_yaml(cls, path: str | Path) -> CSVFormatConfig:
        """Load configuration from a YAML file.

        The YAML structure should have top-level keys ``format`` and
        ``columns``.  See the module docstring for an example.
        """
        import yaml  # lazy import — yaml is optional

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_json(cls, path: str | Path) -> CSVFormatConfig:
        """Load configuration from a JSON file."""
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CSVFormatConfig:
        """Create a config from a plain dict (e.g. parsed YAML/JSON)."""
        fmt = d.get("format", {})
        cols_raw = d.get("columns", [])

        columns = [
            ColumnSpec(
                name=c["name"],
                column_index=c.get("index", c.get("column_index", 0)),
                unit=c.get("unit"),
                role=c.get("role", "data"),
                time_component=c.get("time_component"),
                scale_factor=c.get("scale_factor", 1.0),
            )
            for c in cols_raw
        ]

        return cls(
            columns=columns,
            delimiter=fmt.get("delimiter", ","),
            skip_rows=fmt.get("skip_rows"),
            header_row=fmt.get("header_row"),
            encoding=fmt.get("encoding", "utf-8"),
            timezone=fmt.get("timezone"),
            sample_rate=fmt.get("sample_rate"),
            resample_method=fmt.get("resample_method"),
            comment_char=fmt.get("comment_char", "#"),
        )

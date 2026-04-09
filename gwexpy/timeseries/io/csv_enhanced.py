"""Enhanced CSV reader with flexible column mapping and timestamp reconstruction.

This module provides a configurable CSV reader that can handle instrument-
specific formats (ADX3, custom loggers, etc.) through YAML/JSON configuration
files rather than hard-coded logic.
"""
from __future__ import annotations

import csv
import datetime as _dt
import io
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u

from gwexpy.io.utils import datetime_to_gps, filter_by_channels, parse_timezone

from .csv_config import CSVFormatConfig


def _detect_skip_rows(
    lines: list[str], delimiter: str, comment_char: str
) -> int:
    """Heuristic to detect how many header/comment rows to skip."""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(comment_char):
            continue
        # Try to parse as numeric
        parts = stripped.split(delimiter)
        numeric_count = 0
        for p in parts:
            try:
                float(p.strip())
                numeric_count += 1
            except ValueError:
                pass
        if numeric_count > len(parts) / 2:
            return i
    return 0


def _detect_delimiter(sample: str) -> str:
    """Detect CSV delimiter from a sample string."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;| ")
        return dialect.delimiter
    except csv.Error:
        return ","


def _reconstruct_timestamps(
    raw_data: np.ndarray,
    time_components: dict[str, int],
    timezone: _dt.tzinfo,
) -> np.ndarray:
    """Build GPS timestamps from separate year/month/day/hour/min/sec columns.

    Parameters
    ----------
    raw_data : ndarray, shape (N, ncols)
        Raw CSV data as floats.
    time_components : dict
        Mapping from component name to column index.
    timezone : tzinfo
        Timezone to apply.

    Returns
    -------
    gps_times : ndarray, shape (N,)
        GPS timestamps.

    """
    nrows = raw_data.shape[0]
    gps = np.empty(nrows, dtype=float)

    # Extract component arrays
    years = raw_data[:, time_components["year"]].astype(int)
    months = raw_data[:, time_components["month"]].astype(int)
    days = raw_data[:, time_components["day"]].astype(int)
    hours = raw_data[:, time_components["hour"]].astype(int) if "hour" in time_components else np.zeros(nrows, dtype=int)
    minutes = raw_data[:, time_components["minute"]].astype(int) if "minute" in time_components else np.zeros(nrows, dtype=int)

    if "second" in time_components:
        sec_raw = raw_data[:, time_components["second"]]
        secs = sec_raw.astype(int)
        microsecs = ((sec_raw - secs) * 1e6).astype(int)
    else:
        secs = np.zeros(nrows, dtype=int)
        microsecs = np.zeros(nrows, dtype=int)

    for i in range(nrows):
        # Validate component ranges before constructing datetime
        if not (1 <= months[i] <= 12):
            raise ValueError(
                f"Row {i}: month value {months[i]} is out of range [1, 12]"
            )
        if not (1 <= days[i] <= 31):
            raise ValueError(
                f"Row {i}: day value {days[i]} is out of range [1, 31]"
            )
        if not (0 <= hours[i] <= 23):
            raise ValueError(
                f"Row {i}: hour value {hours[i]} is out of range [0, 23]"
            )
        if not (0 <= minutes[i] <= 59):
            raise ValueError(
                f"Row {i}: minute value {minutes[i]} is out of range [0, 59]"
            )
        if not (0 <= secs[i] <= 59):
            raise ValueError(
                f"Row {i}: second value {secs[i]} is out of range [0, 59]"
            )
        try:
            tz = timezone if timezone is not None else _dt.UTC
            dt = _dt.datetime(
                years[i], months[i], days[i],
                hours[i], minutes[i], secs[i], microsecs[i],
                tzinfo=tz,
            )
        except ValueError as exc:
            raise ValueError(
                f"Row {i}: invalid datetime components "
                f"({years[i]}-{months[i]:02d}-{days[i]:02d} "
                f"{hours[i]:02d}:{minutes[i]:02d}:{secs[i]:02d})"
            ) from exc
        gps[i] = datetime_to_gps(dt)

    return gps


def _resample_uniform(
    times: np.ndarray,
    values: np.ndarray,
    sample_rate: float,
    method: str = "interpolate",
) -> tuple[np.ndarray, np.ndarray]:
    """Resample non-uniform data to a uniform grid.

    Parameters
    ----------
    times : ndarray
        GPS timestamps.
    values : ndarray
        Data values.
    sample_rate : float
        Target sample rate in Hz.
    method : str
        ``"interpolate"`` uses scipy interp1d, ``"asfreq"`` uses nearest.

    Returns
    -------
    new_times, new_values : ndarray
        Uniformly sampled arrays.

    """
    dt = 1.0 / sample_rate
    t_start = times[0]
    t_end = times[-1]
    n_samples = max(1, round((t_end - t_start) / dt) + 1)
    new_times = np.linspace(t_start, t_end, n_samples)

    if method == "interpolate":
        from scipy.interpolate import interp1d

        f = interp1d(times, values, kind="linear", bounds_error=False, fill_value=np.nan)
        new_values = f(new_times)
    elif method == "asfreq":
        # Nearest-neighbor resampling
        indices = np.searchsorted(times, new_times, side="left")
        indices = np.clip(indices, 0, len(values) - 1)
        new_values = values[indices]
    else:
        raise ValueError(
            f"Unknown resample method: {method!r}. Choose 'interpolate' or 'asfreq'."
        )

    return new_times, new_values


def read_timeseriesdict_csv(
    source: str | Path,
    config: CSVFormatConfig | str | Path | dict[str, Any] | None = None,
    *,
    channels: list[str] | None = None,
    timezone: str | None = None,
    resample: float | None = None,
    resample_method: str = "interpolate",
    **kwargs: Any,
) -> Any:
    """Read CSV/ASCII data with flexible column mapping.

    Parameters
    ----------
    source : str or Path
        Path to a CSV file.
    config : CSVFormatConfig, str, Path, dict, or None
        Column mapping configuration. Can be:

        - :class:`CSVFormatConfig` object
        - Path to a YAML (``.yaml``/``.yml``) or JSON (``.json``) config file
        - ``dict`` with config keys
        - ``None`` for auto-detection mode (simple numeric CSV assumed)
    channels : list of str, optional
        Subset of channel names to read.
    timezone : str, optional
        Timezone override (e.g. ``"Asia/Tokyo"``).  Overrides the config
        timezone if both are given.
    resample : float, optional
        Target sample rate in Hz for resampling non-uniform data.
        Overrides config.sample_rate if both are given.
    resample_method : str
        Resampling method: ``"interpolate"`` or ``"asfreq"``.

    """
    from .. import TimeSeriesDict

    # --- Resolve config ---
    if config is None:
        cfg = CSVFormatConfig()
    elif isinstance(config, CSVFormatConfig):
        cfg = config
    elif isinstance(config, dict):
        cfg = CSVFormatConfig.from_dict(config)
    elif isinstance(config, (str, Path)):
        p = Path(config)
        if p.suffix in (".yaml", ".yml"):
            cfg = CSVFormatConfig.from_yaml(p)
        else:
            cfg = CSVFormatConfig.from_json(p)
    else:
        raise TypeError(f"Unsupported config type: {type(config)}")

    # Override timezone/resample from function args
    tz_str = timezone or cfg.timezone
    target_rate = resample or cfg.sample_rate
    resample_meth = resample_method or cfg.resample_method or "interpolate"

    # --- Read raw file ---
    if hasattr(source, "read"):
        # Handle file-like objects (strings, buffers, etc.)
        text = source.read()
        if isinstance(text, bytes):
            text = text.decode(cfg.encoding or "utf-8")
    else:
        # Handle paths
        source = Path(source)
        text = source.read_text(encoding=cfg.encoding)
    lines = text.splitlines()

    # Auto-detect delimiter if config is default
    delimiter = cfg.delimiter
    if cfg.columns and delimiter == ",":
        pass  # trust config
    elif not cfg.columns:
        # Auto-detect from first data lines
        sample = "\n".join(lines[:20])
        delimiter = _detect_delimiter(sample)

    # Determine rows to skip
    skip = cfg.skip_rows
    if skip is None:
        skip = _detect_skip_rows(lines, delimiter, cfg.comment_char)

    # Parse data lines
    data_lines = []
    for line in lines[skip:]:
        stripped = line.strip()
        if not stripped or stripped.startswith(cfg.comment_char):
            continue
        data_lines.append(stripped)

    if not data_lines:
        return TimeSeriesDict()

    # Parse into float array
    reader = csv.reader(io.StringIO("\n".join(data_lines)), delimiter=delimiter)
    rows = []
    for row in reader:
        try:
            rows.append([float(v.strip()) for v in row if v.strip()])
        except ValueError:
            continue  # skip non-numeric rows

    if not rows:
        return TimeSeriesDict()

    raw = np.array(rows)

    # --- Column mapping ---
    if cfg.columns:
        # Use explicit config
        time_columns: dict[str, int] = {}
        time_col_index: int | None = None
        data_columns: list[tuple[str, int, str | None, float]] = []

        for col in cfg.columns:
            if col.role == "time_component":
                if col.time_component:
                    time_columns[col.time_component] = col.column_index
            elif col.role == "time":
                time_col_index = col.column_index
            elif col.role == "data":
                data_columns.append(
                    (col.name, col.column_index, col.unit, col.scale_factor)
                )
            # skip role is ignored

        # Build timestamps
        if time_columns:
            if tz_str is None:
                raise ValueError(
                    "timezone is required when using time_component columns"
                )
            tz = parse_timezone(tz_str)
            gps_times = _reconstruct_timestamps(raw, time_columns, tz)
        elif time_col_index is not None:
            gps_times = raw[:, time_col_index]
        else:
            # No time info — use sample indices
            if target_rate:
                gps_times = np.arange(raw.shape[0]) / target_rate
            else:
                gps_times = np.arange(raw.shape[0], dtype=float)
    else:
        # Auto-detect: first column = time, rest = data
        gps_times = raw[:, 0]
        data_columns = [
            (f"ch{i}", i, None, 1.0) for i in range(1, raw.shape[1])
        ]

    # --- Build TimeSeriesDict ---
    result: dict[str, Any] = {}
    from .. import TimeSeries

    for name, col_idx, unit_str, scale in data_columns:
        values = raw[:, col_idx] * scale

        # Resample if requested
        if target_rate and resample_meth:
            # Check if data is already uniform
            dt_diff = np.diff(gps_times)
            expected_dt = 1.0 / target_rate
            is_uniform = np.allclose(dt_diff, expected_dt, rtol=0.05, atol=1e-6)

            if not is_uniform and len(gps_times) > 1:
                ts_times, values = _resample_uniform(
                    gps_times, values, target_rate, resample_meth
                )
            else:
                ts_times = gps_times
        else:
            ts_times = gps_times

        # Infer sample rate
        if target_rate:
            dt_val = 1.0 / target_rate
        elif len(ts_times) > 1:
            dt_val = float(np.median(np.diff(ts_times)))
        else:
            dt_val = 1.0

        ts = TimeSeries(
            values,
            t0=float(ts_times[0]),
            dt=dt_val,
            unit=u.Unit(unit_str) if unit_str else u.dimensionless_unscaled,
            name=name,
        )
        result[name] = ts

    tsd = TimeSeriesDict(filter_by_channels(result, channels))
    return tsd


# --- Format registration ---
# Wrapped in try/except so importing this module in isolation (e.g. tests)
# does not fail if the registration infrastructure is unavailable.
try:
    from ._registration import register_timeseries_format  # noqa: E402

    register_timeseries_format(
        "csv",
        reader_dict=read_timeseriesdict_csv,
        extension="csv",
    )
except (ImportError, AttributeError):  # pragma: no cover
    pass

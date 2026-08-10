"""Enhanced CSV reader with flexible column mapping and timestamp reconstruction.

This module provides a configurable CSV reader that can handle instrument-
specific formats (ADX3, custom loggers, etc.) through YAML/JSON configuration
files rather than hard-coded logic.
"""

from __future__ import annotations

import csv
import datetime as _dt
import io
import math
import warnings
from decimal import Decimal, InvalidOperation
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.time import Time

from gwexpy.io.utils import (
    _consume_warning_state,
    _make_warning_state,
    _parse_timezone_for_format,
    _validate_regular_timestamps,
    filter_by_channels,
)

from .csv_config import CSVFormatConfig

_CSV_TIMEZONE_WARNING = (
    "timezone is ignored for CSV numeric/index time routes because their "
    "timestamps already define the time semantics"
)


def _record_or_warn_timezone_ignored(marker: list[bool] | None) -> None:
    if marker is None:
        warnings.warn(_CSV_TIMEZONE_WARNING, UserWarning, stacklevel=3)
    else:
        marker[0] = True


def _validate_and_warn_timezone_ignored(
    timezone: Any,
    marker: list[bool] | None,
) -> None:
    _parse_timezone_for_format("csv", timezone)
    _record_or_warn_timezone_ignored(marker)


def _validate_source_sample_rate(value: Any) -> float | None:
    """Return a finite, positive declared source sample rate."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("CSV source sample rate must be finite and positive")
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("CSV source sample rate must be finite and positive") from exc
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("CSV source sample rate must be finite and positive")
    return rate


def _parse_comment_metadata(
    lines: list[str],
    comment_char: str,
) -> dict[str, str]:
    """Parse simple ``key=value`` metadata from leading comment lines."""
    metadata: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith(comment_char):
            break
        body = stripped[len(comment_char) :].strip()
        if not body or body.startswith("gwexpy.timeseries.csv"):
            continue
        if "=" not in body:
            continue
        key, value = body.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _detect_skip_rows(lines: list[str], delimiter: str, comment_char: str) -> int:
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


def _is_float_token(value: str) -> bool:
    """Return whether a CSV token is numeric, including non-finite floats."""
    try:
        float(value.strip())
    except ValueError:
        return False
    return True


def _reconstruct_timestamps(
    raw_data: np.ndarray,
    raw_tokens: list[list[str]],
    time_components: dict[str, int],
    timezone: _dt.tzinfo,
) -> tuple[float, np.ndarray, list[Decimal]]:
    """Build an origin and exact relative timestamps from time components.

    Parameters
    ----------
    raw_data : ndarray, shape (N, ncols)
        Raw CSV data as floats.
    raw_tokens : list of list of str
        Original CSV tokens, retained for exact fractional seconds.
    time_components : dict
        Mapping from component name to column index.
    timezone : tzinfo
        Timezone to apply.

    Returns
    -------
    time_origin, relative_times, canonical_times
        GPS origin, float relative seconds, and exact UTC canonical instants.

    """
    nrows = raw_data.shape[0]
    for component, column_index in time_components.items():
        for row_index, value in enumerate(raw_data[:, column_index]):
            if not np.isfinite(value):
                raise ValueError(
                    "CSV timestamp component "
                    f"'{component}' at row {row_index} is non-finite"
                )

    # Extract component arrays
    years = raw_data[:, time_components["year"]].astype(int)
    months = raw_data[:, time_components["month"]].astype(int)
    days = raw_data[:, time_components["day"]].astype(int)
    hours = (
        raw_data[:, time_components["hour"]].astype(int)
        if "hour" in time_components
        else np.zeros(nrows, dtype=int)
    )
    minutes = (
        raw_data[:, time_components["minute"]].astype(int)
        if "minute" in time_components
        else np.zeros(nrows, dtype=int)
    )

    second_values = (
        [Decimal(row[time_components["second"]]) for row in raw_tokens]
        if "second" in time_components
        else [Decimal("0")] * nrows
    )
    canonical_times: list[Decimal] = []
    gps_origin = 0.0
    unix_epoch = _dt.datetime(1970, 1, 1, tzinfo=_dt.UTC)

    for i in range(nrows):
        second_value = second_values[i]
        if not second_value.is_finite():
            raise ValueError(f"CSV timestamp second at row {i} is non-finite")
        second = int(second_value)
        fractional_second = second_value - second
        # Validate component ranges before constructing datetime
        if not (1 <= months[i] <= 12):
            raise ValueError(
                f"Row {i}: month value {months[i]} is out of range [1, 12]"
            )
        if not (1 <= days[i] <= 31):
            raise ValueError(f"Row {i}: day value {days[i]} is out of range [1, 31]")
        if not (0 <= hours[i] <= 23):
            raise ValueError(f"Row {i}: hour value {hours[i]} is out of range [0, 23]")
        if not (0 <= minutes[i] <= 59):
            raise ValueError(
                f"Row {i}: minute value {minutes[i]} is out of range [0, 59]"
            )
        if not (Decimal("0") <= second_value < Decimal("60")):
            raise ValueError(
                f"Row {i}: second value {second_value} is out of range [0, 60)"
            )
        try:
            tz = timezone if timezone is not None else _dt.UTC
            whole_second = _dt.datetime(
                years[i],
                months[i],
                days[i],
                hours[i],
                minutes[i],
                second,
                tzinfo=tz,
            )
        except ValueError as exc:
            raise ValueError(
                f"Row {i}: invalid datetime components "
                f"({years[i]}-{months[i]:02d}-{days[i]:02d} "
                f"{hours[i]:02d}:{minutes[i]:02d}:{second:02d})"
            ) from exc
        utc_whole_second = whole_second.astimezone(_dt.UTC)
        elapsed = utc_whole_second - unix_epoch
        canonical_times.append(
            Decimal(elapsed.days * 86400 + elapsed.seconds) + fractional_second
        )
        if i == 0:
            gps_origin = float(Time(utc_whole_second).gps) + float(fractional_second)

    relative_times = np.asarray(
        [float(value - canonical_times[0]) for value in canonical_times],
        dtype=float,
    )
    return gps_origin, relative_times, canonical_times


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

        f = interp1d(
            times, values, kind="linear", bounds_error=False, fill_value=np.nan
        )
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
    source : str, Path, or list of str/Path
        Path to a CSV file, or a list of paths.  When a list is given,
        channels found in several files are concatenated along the time
        axis and channels unique to one file are merged in.
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
        Target sample rate in Hz. The reader validates the regular source grid
        before applying this target; resampling never repairs missing records.
        ``config.sample_rate`` declares the source cadence; ``resample`` is a
        separate target cadence applied only after source-grid validation.
    resample_method : str
        Resampling method: ``"interpolate"`` or ``"asfreq"``.
    **kwargs
        Additional keyword arguments reserved for compatibility with I/O dispatch.
        ``start`` and ``end`` are honoured by cropping the result rather than
        ignored, matching GWpy's own ASCII reader (issue #611).

    """
    from gwexpy.io.time_selection import apply_time_selection, pop_time_selection

    from .. import TimeSeriesDict
    from ._multi import expand_multi_source, read_multi_dict

    start, end = pop_time_selection(kwargs)
    timezone_warning_marker = _consume_warning_state(
        kwargs,
        "_timezone_warning_state",
        "_timezone_warning_marker",
    )

    multi = expand_multi_source(source)
    if multi is not None:
        top_level_marker = [False]
        merged = read_multi_dict(
            partial(
                read_timeseriesdict_csv,
                _timezone_warning_state=_make_warning_state(top_level_marker),
            ),
            multi,
            "csv",
            config=config,
            channels=channels,
            timezone=timezone,
            resample=resample,
            resample_method=resample_method,
            **kwargs,
        )
        if top_level_marker[0]:
            _record_or_warn_timezone_ignored(timezone_warning_marker)
        return apply_time_selection(merged, start, end)

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
    tz_str = timezone if timezone is not None else cfg.timezone
    source_rate = _validate_source_sample_rate(cfg.sample_rate)
    target_rate = resample
    resample_meth = resample_method or cfg.resample_method or "interpolate"
    if tz_str is not None:
        # Validate before any source-dependent early return. Route-specific
        # warning/localization still happens only after the route is known.
        _parse_timezone_for_format("csv", tz_str)
    if tz_str is None and any(col.role == "time_component" for col in cfg.columns):
        raise ValueError("timezone is required when using time_component columns")

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
    metadata = _parse_comment_metadata(lines, cfg.comment_char)

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
    data_lines: list[tuple[int, str]] = []
    for line_number, line in enumerate(lines[skip:], start=skip + 1):
        stripped = line.strip()
        if not stripped or stripped.startswith(cfg.comment_char):
            continue
        data_lines.append((line_number, stripped))

    if not data_lines:
        return TimeSeriesDict()

    # With no numeric row, auto-detection cannot distinguish a sole textual
    # header from data.  Preserve the established header-only empty result;
    # any non-numeric row after a detected header still fails below.
    if not cfg.columns and len(data_lines) == 1:
        only_line = data_lines[0][1]
        only_row = next(csv.reader(io.StringIO(only_line), delimiter=delimiter))
        if all(not _is_float_token(value) for value in only_row):
            return TimeSeriesDict()

    # Parse into float array
    rows = []
    raw_tokens: list[list[str]] = []
    for line_number, line in data_lines:
        row = next(csv.reader(io.StringIO(line), delimiter=delimiter))
        try:
            tokens = [value.strip() for value in row]
            rows.append([float(v) for v in tokens])
            raw_tokens.append(tokens)
        except ValueError as exc:
            raise ValueError(
                f"CSV line {line_number} contains non-numeric data"
            ) from exc

    if not rows:
        return TimeSeriesDict()

    raw = np.array(rows)

    # --- Column mapping ---
    time_origin = 0.0
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
            tz = _parse_timezone_for_format("csv", tz_str)
            time_origin, gps_times, exact_times = _reconstruct_timestamps(
                raw, raw_tokens, time_columns, tz
            )
            source_dt = _validate_regular_timestamps(
                exact_times,
                source="CSV",
                expected_dt=(Decimal("1") / Decimal(str(source_rate)))
                if source_rate is not None
                else None,
            )
        elif time_col_index is not None:
            if tz_str is not None:
                _validate_and_warn_timezone_ignored(
                    tz_str,
                    timezone_warning_marker,
                )
            try:
                exact_times = [Decimal(row[time_col_index]) for row in raw_tokens]
            except (InvalidOperation, IndexError) as exc:
                raise ValueError(
                    "CSV numeric time column contains an invalid token"
                ) from exc
            source_dt = _validate_regular_timestamps(
                exact_times,
                source="CSV",
                expected_dt=(Decimal("1") / Decimal(str(source_rate)))
                if source_rate is not None
                else None,
            )
            time_origin = float(exact_times[0])
            gps_times = np.asarray(
                [float(value - exact_times[0]) for value in exact_times], dtype=float
            )
        else:
            # No time info — use sample indices
            if tz_str is not None:
                _validate_and_warn_timezone_ignored(
                    tz_str,
                    timezone_warning_marker,
                )
            if source_rate:
                gps_times = np.arange(raw.shape[0]) / source_rate
            else:
                gps_times = np.arange(raw.shape[0], dtype=float)
    else:
        # Auto-detect: first column = time, rest = data
        if tz_str is not None:
            _validate_and_warn_timezone_ignored(
                tz_str,
                timezone_warning_marker,
            )
        try:
            exact_times = [Decimal(row[0]) for row in raw_tokens]
        except (InvalidOperation, IndexError) as exc:
            raise ValueError(
                "CSV numeric time column contains an invalid token"
            ) from exc
        source_dt = _validate_regular_timestamps(
            exact_times,
            source="CSV",
            expected_dt=(Decimal("1") / Decimal(str(source_rate)))
            if source_rate is not None
            else None,
        )
        time_origin = float(exact_times[0])
        gps_times = np.asarray(
            [float(value - exact_times[0]) for value in exact_times], dtype=float
        )
        if raw.shape[1] == 2:
            data_columns = [
                (metadata.get("name", "ch1"), 1, metadata.get("unit"), 1.0),
            ]
        else:
            data_columns = [(f"ch{i}", i, None, 1.0) for i in range(1, raw.shape[1])]

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
        elif "source_dt" in locals():
            dt_val = source_dt
            if source_rate is None and len(ts_times) > 1:
                # The validated cadence is the exact Decimal median, while
                # TimeSeries needs a float interval.  For an inferred numeric
                # CSV grid, use the endpoint average to avoid accumulating
                # serialized-float roundoff into crop's sample index.
                dt_val = float((ts_times[-1] - ts_times[0]) / (len(ts_times) - 1))
        elif len(ts_times) > 1:
            # The median of the diffs is robust to a gappy or irregular time
            # column, but on a uniform decimal grid every diff carries the
            # ~1-ulp noise of two float parses, and that noise is enough to
            # make Series.crop's floor((t - t0)/dt) land one sample early —
            # which gwpy 4's registry coverage check then escalates to a
            # ValueError on a fully in-span bounded read (issue #611 review).
            # The end-to-end average spreads the same parse noise over N-1
            # samples, so when the two estimators agree the grid is uniform
            # and the average is the more exact dt; when they disagree the
            # column has gaps and the median remains the safer choice.
            dt_val = float(np.median(np.diff(ts_times)))
            span_dt = float((ts_times[-1] - ts_times[0]) / (len(ts_times) - 1))
            if dt_val and abs(span_dt - dt_val) <= 1e-12 * abs(dt_val):
                dt_val = span_dt
        else:
            dt_val = 1.0

        ts = TimeSeries(
            values,
            t0=time_origin + float(ts_times[0]),
            dt=dt_val,
            unit=u.Unit(unit_str) if unit_str else u.dimensionless_unscaled,
            name=name,
        )
        result[name] = ts

    tsd = TimeSeriesDict(filter_by_channels(result, channels))
    return apply_time_selection(tsd, start, end)


def read_timeseries_csv(
    source: str | Path,
    **kwargs: Any,
) -> Any:
    """Read a single ``TimeSeries`` from a CSV source."""
    tsd = read_timeseriesdict_csv(source, **kwargs)
    if not tsd:
        raise ValueError(f"No time-series data found in CSV source: {source}")
    return next(iter(tsd.values()))


def write_timeseries_csv(
    ts: Any,
    target: str | Path | Any,
    *,
    delimiter: str = ",",
    **kwargs: Any,
) -> str | Path | Any:
    """Write a single ``TimeSeries`` to CSV with minimal metadata comments."""
    del kwargs

    rows = [
        f"{float(t):.18e}{delimiter}{float(v):.18e}"
        for t, v in zip(ts.times.value, ts.value, strict=False)
    ]
    header = [
        "# gwexpy.timeseries.csv v1",
        f"# name={ts.name}" if ts.name else "",
        f"# unit={ts.unit}" if str(ts.unit) else "",
        f"# t0={float(ts.t0.value):.18e}",
        f"# dt={float(ts.dt.value):.18e}",
    ]
    content = "\n".join(line for line in header if line) + "\n" + "\n".join(rows) + "\n"

    if hasattr(target, "write"):
        target.write(content)
        return target

    path = Path(target)
    path.write_text(content, encoding="utf-8")
    return target


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

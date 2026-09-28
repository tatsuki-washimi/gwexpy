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
from contextlib import AbstractContextManager, nullcontext
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, InvalidOperation
from functools import partial
from itertools import chain, islice
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.time import Time
from gwpy.io.registry import default_registry as io_registry
from gwpy.io.registry import identify_factory
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.io.utils import (
    _consume_warning_state,
    _localize_naive_datetime,
    _make_warning_state,
    _parse_timezone_for_format,
    _validate_float_time_axis,
    _validate_regular_timestamps,
    filter_by_channels,
)

from .csv_config import CSVFormatConfig

_CSV_TIMEZONE_WARNING = (
    "timezone is ignored for CSV numeric/index time routes because their "
    "timestamps already define the time semantics"
)
_GPS_NANOSECOND = Decimal("1e-9")
_MAX_RESAMPLED_VALUES = 10_000_000
_RESAMPLE_METHODS = frozenset({"interpolate", "asfreq"})
_RESAMPLE_BUDGET_SENTINEL = object()
_MAX_CSV_MATRIX_CHUNK_BYTES = 64 * 1024 * 1024


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


def _consume_resample_budget_state(kwargs: dict[str, Any]) -> list[int] | None:
    """Consume trusted shared budget state for one top-level multi-file read."""
    state = kwargs.pop("_resample_budget_state", None)
    if (
        isinstance(state, tuple)
        and len(state) == 2
        and state[0] is _RESAMPLE_BUDGET_SENTINEL
        and isinstance(state[1], list)
        and len(state[1]) == 1
        and isinstance(state[1][0], int)
        and not isinstance(state[1][0], bool)
        and state[1][0] >= 0
    ):
        return state[1]
    return None


def _parse_decimal_time_column(
    raw_tokens: list[str],
    line_numbers: list[int],
) -> list[Decimal]:
    """Parse one numeric CSV time column with physical-line diagnostics."""
    times: list[Decimal] = []
    for token, line_number in zip(raw_tokens, line_numbers, strict=True):
        try:
            value = Decimal(token)
        except InvalidOperation as exc:
            raise ValueError(
                f"CSV line {line_number}: timestamp is non-numeric"
            ) from exc
        if not value.is_finite():
            raise ValueError(f"CSV line {line_number}: timestamp is non-finite")
        times.append(value)
    return times


def _validate_sample_rate(value: Any, *, role: str) -> float | None:
    """Return a finite, positive CSV sample rate for *role*."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"CSV {role} sample rate must be finite and positive")
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CSV {role} sample rate must be finite and positive") from exc
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"CSV {role} sample rate must be finite and positive")
    interval = 1.0 / rate
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError(
            f"CSV {role} sample rate must yield a finite positive interval"
        )
    return rate


def _validate_source_sample_rate(value: Any) -> float | None:
    """Return a finite, positive declared source sample rate."""
    return _validate_sample_rate(value, role="source")


def _validate_target_sample_rate(value: Any) -> float | None:
    """Return a finite, positive requested target sample rate."""
    return _validate_sample_rate(value, role="target")


def _validate_resample_method(value: Any) -> str:
    """Return a supported CSV resampling method."""
    if not isinstance(value, str) or value not in _RESAMPLE_METHODS:
        raise ValueError(
            f"Unknown resample method: {value!r}. Choose 'interpolate' or 'asfreq'."
        )
    return value


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
        parts = next(csv.reader(io.StringIO(stripped), delimiter=delimiter))
        numeric_count = 0
        for p in parts:
            try:
                float(p.strip())
                numeric_count += 1
            except ValueError:
                pass
        # A row containing any numeric token is potentially data. Treat it as
        # the first data row so a malformed timestamp alongside numeric sample
        # values is reported instead of being silently discarded as a header.
        first_token = parts[0].strip().casefold() if parts else ""
        if numeric_count or first_token == "nat":
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
    raw_tokens: dict[int, list[str]],
    line_numbers: list[int],
    time_components: dict[str, int],
    timezone: _dt.tzinfo,
) -> tuple[float, np.ndarray, list[Decimal]]:
    """Build an origin and exact relative timestamps from time components.

    Parameters
    ----------
    raw_tokens : dict of int to list of str
        Original timestamp tokens, retained for exact fractional seconds.
    line_numbers : list of int
        Physical one-based CSV line numbers corresponding to the rows.
    time_components : dict
        Mapping from component name to column index.
    timezone : tzinfo
        Timezone to apply.

    Returns
    -------
    time_origin, relative_times, canonical_times
        GPS origin, float relative seconds, and continuous GPS canonical
        instants.

    """
    nrows = len(line_numbers)
    component_values: dict[str, list[Decimal]] = {}
    for component, column_index in time_components.items():
        values = []
        for row_index, token in enumerate(raw_tokens[column_index]):
            value = Decimal(token)
            line_number = line_numbers[row_index]
            if not value.is_finite():
                raise ValueError(
                    f"CSV line {line_number}: timestamp component "
                    f"'{component}' is non-finite"
                )
            if component != "second" and value != value.to_integral_value():
                raise ValueError(
                    f"CSV line {line_number}: timestamp component "
                    f"'{component}' must be an integer, got {token!r}"
                )
            values.append(value)
        component_values[component] = values

    # Extract component arrays
    years = [int(value) for value in component_values["year"]]
    months = [int(value) for value in component_values["month"]]
    days = [int(value) for value in component_values["day"]]
    hours = [int(value) for value in component_values.get("hour", [Decimal(0)] * nrows)]
    minutes = [
        int(value) for value in component_values.get("minute", [Decimal(0)] * nrows)
    ]
    second_values = component_values.get("second", [Decimal(0)] * nrows)
    canonical_times: list[Decimal] = []
    gps_origin = 0.0

    for i in range(nrows):
        line_number = line_numbers[i]
        second_value = second_values[i]
        second = int(second_value)
        fractional_second = second_value - second
        # Validate component ranges before constructing datetime
        if not (1 <= years[i] <= 9999):
            raise ValueError(
                f"CSV line {line_number}: year value {years[i]} "
                "is out of range [1, 9999]"
            )
        if not (1 <= months[i] <= 12):
            raise ValueError(
                f"CSV line {line_number}: month value {months[i]} "
                "is out of range [1, 12]"
            )
        if not (1 <= days[i] <= 31):
            raise ValueError(
                f"CSV line {line_number}: day value {days[i]} is out of range [1, 31]"
            )
        if not (0 <= hours[i] <= 23):
            raise ValueError(
                f"CSV line {line_number}: hour value {hours[i]} is out of range [0, 23]"
            )
        if not (0 <= minutes[i] <= 59):
            raise ValueError(
                f"CSV line {line_number}: minute value {minutes[i]} "
                "is out of range [0, 59]"
            )
        if not (Decimal("0") <= second_value < Decimal("60")):
            raise ValueError(
                f"CSV line {line_number}: second value {second_value} "
                "is out of range [0, 60)"
            )
        try:
            naive_whole_second = _dt.datetime(
                years[i],
                months[i],
                days[i],
                hours[i],
                minutes[i],
                second,
            )
        except ValueError as exc:
            raise ValueError(
                f"CSV line {line_number}: invalid datetime components "
                f"({years[i]}-{months[i]:02d}-{days[i]:02d} "
                f"{hours[i]:02d}:{minutes[i]:02d}:{second:02d})"
            ) from exc
        tz = timezone if timezone is not None else _dt.UTC
        try:
            whole_second = _localize_naive_datetime(naive_whole_second, tz)
        except ValueError as exc:
            raise ValueError(f"CSV line {line_number}: {exc}") from exc
        utc_whole_second = whole_second.astimezone(_dt.UTC)
        # Astropy's split-JD conversion can leave picosecond-scale Decimal
        # residue. Quantize that below the supported source nanosecond
        # resolution before restoring the source token's fractional component.
        # Do not round to an integer: pre-1972 UTC rubber seconds have a real
        # fractional TAI/GPS offset even at a whole civil second.
        whole_gps = (
            Time(utc_whole_second)
            .to_value("gps", "decimal")
            .quantize(_GPS_NANOSECOND, rounding=ROUND_HALF_EVEN)
        )
        canonical_time = whole_gps + fractional_second
        canonical_times.append(canonical_time)
        if i == 0:
            gps_origin = float(canonical_time)

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
    *,
    max_samples: int | None = None,
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
    max_samples : int, optional
        Maximum number of output samples to allocate for this value array.
        Defaults to the per-source resampling safety budget.

    Returns
    -------
    new_times, new_values : ndarray
        Uniformly sampled arrays.

    """
    validated_method = _validate_resample_method(method)
    validated_rate = _validate_target_sample_rate(sample_rate)
    if validated_rate is None:  # pragma: no cover - required by the signature
        raise ValueError("CSV target sample rate must be finite and positive")
    dt = 1.0 / validated_rate
    source_times = np.asarray(times, dtype=float)
    source_values = np.asarray(values)
    if source_times.ndim != 1 or source_values.ndim != 1:
        raise ValueError("CSV resampling requires one-dimensional time and value data")
    if not len(source_times) or len(source_values) != len(source_times):
        raise ValueError("CSV resampling requires non-empty time-aligned values")
    if not np.all(np.isfinite(source_times)):
        raise ValueError("CSV resampling requires finite source timestamps")

    sample_limit = _MAX_RESAMPLED_VALUES if max_samples is None else max_samples
    if isinstance(sample_limit, (bool, np.bool_)) or not isinstance(
        sample_limit, (int, np.integer)
    ):
        raise ValueError("CSV resampling sample limit must be a positive integer")
    sample_limit = int(sample_limit)
    if sample_limit <= 0:
        raise ValueError("CSV resampling sample limit must be a positive integer")

    t_start = float(source_times[0])
    t_end = float(source_times[-1])
    span = t_end - t_start
    if not math.isfinite(span) or span < 0:
        raise ValueError("CSV resampling requires a finite non-negative time span")
    interval_ratio = Decimal(str(span)) * Decimal(str(validated_rate))
    if interval_ratio >= Decimal(sample_limit):
        raise ValueError(
            "CSV resampled output exceeds the "
            f"{sample_limit}-value safety limit for one source read"
        )
    interval_count = max(
        0,
        int(interval_ratio.to_integral_value(rounding=ROUND_FLOOR)),
    )
    sample_count = interval_count + 1
    if sample_count > sample_limit:
        raise ValueError(
            "CSV resampled output exceeds the "
            f"{sample_limit}-value safety limit for one source read"
        )
    new_times = t_start + np.arange(sample_count, dtype=float) * dt
    interpolation_times = np.clip(new_times, t_start, t_end)

    if validated_method == "interpolate":
        from scipy.interpolate import interp1d

        f = interp1d(
            source_times,
            source_values,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        new_values = f(interpolation_times)
    else:
        # Nearest-neighbor resampling
        right = np.clip(
            np.searchsorted(source_times, interpolation_times, side="left"),
            0,
            len(source_values) - 1,
        )
        left = np.clip(right - 1, 0, len(source_values) - 1)
        use_left = np.abs(interpolation_times - source_times[left]) <= np.abs(
            source_times[right] - interpolation_times
        )
        indices = np.where(use_left, left, right)
        new_values = source_values[indices]

    return new_times, new_values


def _convert_numeric_chunk(
    rows: list[list[str]], line_numbers: list[int], width: int
) -> np.ndarray:
    """Convert a bounded set of already tokenized rows in one NumPy call.

    Python's ``float`` accepts a few spellings that ``fromstring`` does not.
    Fall back only for those uncommon chunks or to locate the first bad row.
    """
    numeric_text = ",".join(",".join(row) for row in rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        values = np.fromstring(numeric_text, sep=",", dtype=np.float64)
    if values.size == len(rows) * width:
        return values.reshape(len(rows), width)
    converted: list[list[float]] = []
    for row, line_number in zip(rows, line_numbers, strict=True):
        try:
            converted.append([float(token) for token in row])
        except ValueError as exc:
            raise ValueError(
                f"CSV line {line_number} contains non-numeric data"
            ) from exc
    return np.asarray(converted, dtype=np.float64)


def _read_numeric_rows(
    source: Any,
    cfg: CSVFormatConfig,
    *,
    channels: list[str] | None,
    start: Any,
    end: Any,
) -> tuple[
    dict[str, str],
    np.ndarray | None,
    dict[int, np.ndarray],
    dict[int, list[str]],
    list[int],
    int,
]:
    """Stream CSV records, retaining only required values and time lexemes."""
    stream_context: AbstractContextManager[Any]
    if hasattr(source, "read"):
        # Non-iterable streams retain the old read() contract. Ordinary file
        # objects, including StringIO, take the streaming branch.
        stream_context = nullcontext(source)
    else:
        stream_context = Path(source).open(encoding=cfg.encoding, newline=None)

    with stream_context as stream:
        if hasattr(stream, "__iter__"):
            source_lines = iter(stream)
        else:
            contents = stream.read()
            if isinstance(contents, bytes):
                contents = contents.decode(cfg.encoding or "utf-8")
            source_lines = iter(contents.splitlines())
        first_lines = list(islice(source_lines, 20))
        delimiter = cfg.delimiter
        if not cfg.columns:
            sample = "\n".join(
                line.decode(cfg.encoding or "utf-8")
                if isinstance(line, bytes)
                else line
                for line in first_lines
            )
            delimiter = _detect_delimiter(sample)

        metadata: dict[str, str] = {}
        metadata_active = True
        skip = cfg.skip_rows
        detected_data = skip is not None
        first_candidate: tuple[int, str] | None = None
        first_candidate_count = 0
        expected_width: int | None = None
        pending_textual_line: int | None = None
        required_width = max(
            (col.column_index + 1 for col in cfg.columns if col.role != "skip"),
            default=0,
        )
        time_indices = (
            {
                col.column_index
                for col in cfg.columns
                if col.role in {"time", "time_component"}
            }
            if cfg.columns
            else {0}
        )
        time_tokens: dict[int, list[str]] = {index: [] for index in time_indices}
        line_numbers: list[int] = []
        matrix_chunks: list[np.ndarray] = []
        chunk_rows: list[list[str]] = []
        chunk_lines: list[int] = []
        chunk_text_bytes = 0
        selected_values: dict[int, list[float]] = {}
        bulk = channels is None and start is None and end is None and not cfg.columns

        def flush_chunk() -> None:
            nonlocal chunk_text_bytes
            if chunk_rows:
                assert expected_width is not None
                matrix_chunks.append(
                    _convert_numeric_chunk(chunk_rows, chunk_lines, expected_width)
                )
                chunk_rows.clear()
                chunk_lines.clear()
                chunk_text_bytes = 0

        for line_number, source_line in enumerate(
            chain(first_lines, source_lines), start=1
        ):
            if isinstance(source_line, bytes):
                source_line = source_line.decode(cfg.encoding or "utf-8")
            stripped = source_line.strip()
            if metadata_active:
                if stripped and not stripped.startswith(cfg.comment_char):
                    metadata_active = False
                else:
                    metadata.update(
                        _parse_comment_metadata([source_line], cfg.comment_char)
                    )
            if not detected_data:
                if stripped and not stripped.startswith(cfg.comment_char):
                    if first_candidate is None:
                        first_candidate = (line_number, stripped)
                    first_candidate_count += 1
                    parts = next(csv.reader(io.StringIO(stripped), delimiter=delimiter))
                    first_token = parts[0].strip().casefold() if parts else ""
                    if (
                        any(_is_float_token(token) for token in parts)
                        or first_token == "nat"
                    ):
                        detected_data = True
                    else:
                        continue
                else:
                    continue
            if skip is not None and line_number <= skip:
                continue
            if not stripped or stripped.startswith(cfg.comment_char):
                continue
            tokens = [
                value.strip()
                for value in next(
                    csv.reader(io.StringIO(stripped), delimiter=delimiter)
                )
            ]
            if pending_textual_line is not None:
                raise ValueError(
                    f"CSV line {pending_textual_line} contains non-numeric data"
                )
            if (
                not cfg.columns
                and expected_width is None
                and all(not _is_float_token(token) for token in tokens)
            ):
                pending_textual_line = line_number
                continue
            width = len(tokens)
            if expected_width is None:
                expected_width = width
                if not bulk:
                    if cfg.columns:
                        selected_values = {
                            col.column_index: []
                            for col in cfg.columns
                            if col.role == "data"
                            and (channels is None or col.name in channels)
                        }
                    else:
                        selected_values = {
                            index: []
                            for index in range(1, width)
                            if channels is None
                            or (
                                metadata.get("name", "ch1")
                                if width == 2
                                else f"ch{index}"
                            )
                            in channels
                        }
            elif width != expected_width:
                flush_chunk()
                raise ValueError(
                    f"CSV line {line_number} has {width} columns; expected {expected_width}"
                )
            if width < required_width:
                flush_chunk()
                raise ValueError(
                    f"CSV line {line_number} has {width} columns; configured "
                    f"columns require at least {required_width}"
                )
            if bulk:
                row_bytes = sum(len(token) for token in tokens) + width
                if chunk_rows and (
                    (len(chunk_rows) + 1) * width * 8 > _MAX_CSV_MATRIX_CHUNK_BYTES
                    or chunk_text_bytes + row_bytes > _MAX_CSV_MATRIX_CHUNK_BYTES
                ):
                    flush_chunk()
                chunk_rows.append(tokens)
                chunk_lines.append(line_number)
                chunk_text_bytes += row_bytes
            else:
                for index, token in enumerate(tokens):
                    try:
                        value = float(token)
                    except ValueError as exc:
                        raise ValueError(
                            f"CSV line {line_number} contains non-numeric data"
                        ) from exc
                    if index in selected_values:
                        selected_values[index].append(value)
            line_numbers.append(line_number)
            for index in time_tokens:
                time_tokens[index].append(tokens[index])

        if not detected_data and first_candidate is not None:
            # Auto-detection's no-numeric case starts at line one. Its sole
            # textual row is treated as a header-only empty source.
            candidate_line, candidate = first_candidate
            candidate_tokens = next(
                csv.reader(io.StringIO(candidate), delimiter=delimiter)
            )
            if (
                not cfg.columns
                and first_candidate_count == 1
                and all(not _is_float_token(token) for token in candidate_tokens)
            ):
                return metadata, None, {}, {}, [], 0
            raise ValueError(f"CSV line {candidate_line} contains non-numeric data")

        if pending_textual_line is not None:
            return metadata, None, {}, {}, [], 0

        flush_chunk()
        matrix = np.concatenate(matrix_chunks) if matrix_chunks else None
        selected = {
            index: np.asarray(values, dtype=np.float64)
            for index, values in selected_values.items()
        }
        return (
            metadata,
            matrix,
            selected,
            time_tokens,
            line_numbers,
            expected_width or 0,
        )


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
    resample_budget_state = _consume_resample_budget_state(kwargs)

    multi = expand_multi_source(source)
    if multi is not None:
        top_level_marker = [False]
        top_level_budget = (
            resample_budget_state
            if resample_budget_state is not None
            else [_MAX_RESAMPLED_VALUES]
        )
        merged = read_multi_dict(
            partial(
                read_timeseriesdict_csv,
                _timezone_warning_state=_make_warning_state(top_level_marker),
                _resample_budget_state=(
                    _RESAMPLE_BUDGET_SENTINEL,
                    top_level_budget,
                ),
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
    target_rate = _validate_target_sample_rate(resample)
    resample_meth = resample_method or cfg.resample_method or "interpolate"
    if target_rate is not None:
        resample_meth = _validate_resample_method(resample_meth)
    if tz_str is not None:
        # Validate before any source-dependent early return. Route-specific
        # warning/localization still happens only after the route is known.
        _parse_timezone_for_format("csv", tz_str)
    if tz_str is None and any(col.role == "time_component" for col in cfg.columns):
        raise ValueError("timezone is required when using time_component columns")

    metadata, raw, selected_values, raw_tokens, row_line_numbers, width = (
        _read_numeric_rows(
            source,
            cfg,
            channels=channels,
            start=start,
            end=end,
        )
    )
    if not row_line_numbers:
        return TimeSeriesDict()
    row_count = len(row_line_numbers)

    # --- Column mapping ---
    exact_time_origin = Decimal("0")
    has_serialized_time_axis = False
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
            _gps_origin, gps_times, exact_times = _reconstruct_timestamps(
                raw_tokens, row_line_numbers, time_columns, tz
            )
            exact_time_origin = exact_times[0]
            has_serialized_time_axis = True
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
            exact_times = _parse_decimal_time_column(
                raw_tokens[time_col_index],
                row_line_numbers,
            )
            source_dt = _validate_regular_timestamps(
                exact_times,
                source="CSV",
                expected_dt=(Decimal("1") / Decimal(str(source_rate)))
                if source_rate is not None
                else None,
            )
            exact_time_origin = exact_times[0]
            has_serialized_time_axis = True
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
                source_dt = 1.0 / source_rate
                gps_times = np.arange(row_count) / source_rate
            else:
                gps_times = np.arange(row_count, dtype=float)
    else:
        # Auto-detect: first column = time, rest = data
        if tz_str is not None:
            _validate_and_warn_timezone_ignored(
                tz_str,
                timezone_warning_marker,
            )
        exact_times = _parse_decimal_time_column(raw_tokens[0], row_line_numbers)
        source_dt = _validate_regular_timestamps(
            exact_times,
            source="CSV",
            expected_dt=(Decimal("1") / Decimal(str(source_rate)))
            if source_rate is not None
            else None,
        )
        exact_time_origin = exact_times[0]
        has_serialized_time_axis = True
        gps_times = np.asarray(
            [float(value - exact_times[0]) for value in exact_times], dtype=float
        )
        if width == 2:
            data_columns = [
                (metadata.get("name", "ch1"), 1, metadata.get("unit"), 1.0),
            ]
        else:
            data_columns = [(f"ch{i}", i, None, 1.0) for i in range(1, width)]

    if has_serialized_time_axis:
        _validate_float_time_axis(
            Decimal("0"),
            source_dt,
            sample_count=len(gps_times),
            source="CSV source-relative",
        )
        if not np.all(np.isfinite(gps_times)) or (
            len(gps_times) > 1 and not np.all(gps_times[1:] > gps_times[:-1])
        ):
            raise ValueError(
                "CSV source-relative absolute time axis is not representable"
            )

    if channels is not None:
        wanted_channels = set(channels)
        data_columns = [
            column for column in data_columns if column[0] in wanted_channels
        ]

    # --- Build TimeSeriesDict ---
    result: dict[str, Any] = {}
    from .. import TimeSeries

    if target_rate is not None and data_columns:
        channel_count = len(data_columns)
        available_budget = (
            resample_budget_state[0]
            if resample_budget_state is not None
            else _MAX_RESAMPLED_VALUES
        )
        if channel_count > available_budget:
            raise ValueError(
                "CSV resampled output exceeds the "
                f"{available_budget}-value safety limit for one source read"
            )
        max_samples_per_channel = available_budget // channel_count
    else:
        max_samples_per_channel = _MAX_RESAMPLED_VALUES

    for name, col_idx, unit_str, scale in data_columns:
        values = (
            raw[:, col_idx] if raw is not None else selected_values[col_idx]
        ) * scale

        # Resample if requested
        if target_rate is not None and len(gps_times) > 1:
            ts_times, values = _resample_uniform(
                gps_times,
                values,
                target_rate,
                resample_meth,
                max_samples=max_samples_per_channel,
            )
        else:
            ts_times = gps_times

        # Infer sample rate
        if target_rate is not None:
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

        exact_series_t0 = exact_time_origin + Decimal(str(float(ts_times[0])))
        series_t0, series_dt = _validate_float_time_axis(
            exact_series_t0,
            dt_val,
            sample_count=len(values),
            source="CSV",
        )
        ts = TimeSeries(
            values,
            t0=series_t0,
            dt=series_dt,
            unit=u.Unit(unit_str) if unit_str else u.dimensionless_unscaled,
            name=name,
        )
        result[name] = ts

    if target_rate is not None and resample_budget_state is not None:
        produced_values = sum(len(series) for series in result.values())
        if produced_values > resample_budget_state[0]:  # pragma: no cover
            raise AssertionError("CSV resample budget was exceeded after validation")
        resample_budget_state[0] -= produced_values

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

    header = [
        "# gwexpy.timeseries.csv v1",
        f"# name={ts.name}" if ts.name else "",
        f"# unit={ts.unit}" if str(ts.unit) else "",
        f"# t0={float(ts.t0.value):.18e}",
        f"# dt={float(ts.dt.value):.18e}",
    ]
    stream_context = (
        nullcontext(target)
        if hasattr(target, "write")
        else Path(target).open("w", encoding="utf-8", newline=None)
    )
    with stream_context as stream:
        for line in header:
            if line:
                stream.write(line + "\n")
        wrote_row = False
        for timestamp, value in zip(ts.times.value, ts.value, strict=False):
            stream.write(f"{float(timestamp):.18e}{delimiter}{float(value):.18e}\n")
            wrote_row = True
        if not wrote_row:
            stream.write("\n")
    return target


# --- Format registration ---
# Wrapped in try/except so importing this module in isolation (e.g. tests)
# does not fail if the registration infrastructure is unavailable.
try:
    from ._registration import register_timeseries_format  # noqa: E402

    _native_timeseries_csv_reader = io_registry.get_reader("csv", GwpyTimeSeries)
    _native_timeseries_csv_writer = io_registry.get_writer("csv", GwpyTimeSeries)

    def _identify_enhanced_csv(
        _origin: str,
        filepath: str | Path | None,
        _fileobj: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> bool:
        return filepath is not None and str(filepath).lower().endswith(".csv")

    register_timeseries_format(
        "csv",
        reader_dict=read_timeseriesdict_csv,
        reader_single=_native_timeseries_csv_reader,
        writer_single=_native_timeseries_csv_writer,
        identifier_dict=_identify_enhanced_csv,
        identifier_single=identify_factory("csv"),
        extension="csv",
    )

    # ``register_timeseries_format`` shares the single-series identifier with
    # matrices by default.  CSV matrices keep the enhanced, case-insensitive
    # route while the exact TimeSeries surface uses GWpy's native identifier.
    from .. import TimeSeriesMatrix  # noqa: E402

    io_registry.register_identifier(
        "csv", TimeSeriesMatrix, _identify_enhanced_csv, force=True
    )
except (ImportError, AttributeError):  # pragma: no cover
    pass

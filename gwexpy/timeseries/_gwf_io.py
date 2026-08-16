from __future__ import annotations

import copy
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io.registry import IORegistryError
from gwpy.io.registry import default_registry as io_registry
from gwpy.time import to_gps

from gwexpy.io.utils import _pad_gwf_series_to_span, check_pad_dtype_compatible

# Match gwpy's Series.is_contiguous() tolerance (gwpy/types/series.py) so the
# per-channel gap check below fires exactly when gwpy's own gap detection
# would materialize `merge_pad` values for that channel — no more, no less.
_GWF_GAP_TOL = 2**-18

_GWF_FORMATS = frozenset(
    {
        "gwf",
        "gwf.framecpp",
        "gwf.framel",
        "gwf.lalframe",
    }
)
_GWF_BACKENDS = {
    "gwf": None,
    "gwf.framecpp": "frameCPP",
    "gwf.framel": "framel",
    "gwf.lalframe": "lalframe",
}
_GWF_BACKEND_HINTS = {
    "gwf": "gwpy",
    None: "gwpy",
    "frameCPP": "frameCPP",
    "framel": "framel",
    "lalframe": "lalframe",
}
_GWF_ALIAS_TO_CANONICAL = {
    "frame": "gwf",
    "framecpp": "gwf.framecpp",
    "framel": "gwf.framel",
    "lalframe": "gwf.lalframe",
}
_GWF_REGISTRY_SYNCED = False


def _safe_get_reader(format_name: str, cls: type[Any]) -> Any | None:
    try:
        return io_registry.get_reader(format_name, cls)
    except IORegistryError:
        # No reader registered for this format/class (e.g. optional backend missing).
        return None
    except Exception as exc:
        # This runs at import time; warn rather than crash on unexpected failures.
        warnings.warn(
            f"GWF alias registration skipped for {format_name!r}: {exc}",
            stacklevel=2,
        )
        return None


def _safe_get_writer(format_name: str, cls: type[Any]) -> Any | None:
    try:
        return io_registry.get_writer(format_name, cls)
    except IORegistryError:
        # No writer registered for this format/class (e.g. optional backend missing).
        return None
    except Exception as exc:
        # This runs at import time; warn rather than crash on unexpected failures.
        warnings.warn(
            f"GWF alias registration skipped for {format_name!r}: {exc}",
            stacklevel=2,
        )
        return None


def _read_timeseriesmatrix_gwf(*args: Any, **kwargs: Any) -> Any:
    from .matrix import TimeSeriesMatrix

    return TimeSeriesMatrix.read(*args, **kwargs)


def _sync_gwf_registry_aliases() -> None:
    """Register gwf alias formats and matrix adapters in the astropy I/O registry."""
    global _GWF_REGISTRY_SYNCED
    if _GWF_REGISTRY_SYNCED:
        return

    try:
        from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
    except ImportError as exc:
        # Circular import during package bootstrap; warn so the skip is visible.
        warnings.warn(
            f"GWF alias registration skipped: {exc}",
            stacklevel=2,
        )
        return

    canonical_formats = ("gwf", "gwf.framecpp", "gwf.framel", "gwf.lalframe")
    alias_formats = tuple(_GWF_ALIAS_TO_CANONICAL.keys())
    all_formats = set(canonical_formats + alias_formats)

    for fmt in all_formats:
        io_registry.register_reader(
            fmt, TimeSeriesMatrix, _read_timeseriesmatrix_gwf, force=True
        )

    for alias, canonical in _GWF_ALIAS_TO_CANONICAL.items():
        canonical_reader = _safe_get_reader(canonical, TimeSeries)
        canonical_dict_reader = _safe_get_reader(canonical, TimeSeriesDict)
        canonical_writer = _safe_get_writer(canonical, TimeSeries)
        canonical_dict_writer = _safe_get_writer(canonical, TimeSeriesDict)

        if canonical_reader is not None:
            io_registry.register_reader(alias, TimeSeries, canonical_reader, force=True)
        if canonical_dict_reader is not None:
            io_registry.register_reader(
                alias, TimeSeriesDict, canonical_dict_reader, force=True
            )
        if canonical_writer is not None:
            io_registry.register_writer(alias, TimeSeries, canonical_writer, force=True)
        if canonical_dict_writer is not None:
            io_registry.register_writer(
                alias, TimeSeriesDict, canonical_dict_writer, force=True
            )

        canonical_matrix_reader = _safe_get_reader(canonical, TimeSeriesMatrix)
        canonical_matrix_writer = _safe_get_writer(canonical, TimeSeriesMatrix)
        if canonical_matrix_reader is not None:
            io_registry.register_reader(
                alias, TimeSeriesMatrix, canonical_matrix_reader, force=True
            )
        if canonical_matrix_writer is not None:
            io_registry.register_writer(
                alias, TimeSeriesMatrix, canonical_matrix_writer, force=True
            )

    _GWF_REGISTRY_SYNCED = True


def _format_gwf_import_error(fmt: str, exc: Exception) -> ImportError:
    backend = _GWF_BACKENDS.get(fmt, None)
    package = _GWF_BACKEND_HINTS.get(backend, _GWF_BACKEND_HINTS.get(None))
    error = ImportError(
        f"Missing optional dependency for {fmt!r} GWF reader (backend hint: {backend!r}, "
        f"package={package!r}). Install the optional dependency or use format 'gwf' when available."
    )
    error.__cause__ = exc
    return error


def _normalize_gwf_format(fmt: str) -> str | None:
    """Normalize user-facing format aliases used by GWF readers."""
    value = fmt.strip().lower()
    if not value:
        return None
    if value == "frame":
        return "gwf"
    if value.startswith("gwf."):
        if value == "gwf":
            return "gwf"
        suffix = value[4:]
        if suffix in {"framecpp", "framel", "lalframe"}:
            return value
        return None
    if value in {"framecpp", "framel", "lalframe"}:
        return f"gwf.{value}"
    if value in _GWF_FORMATS:
        return value
    return None


def _resolve_gwf_format(source: Any, fmt: Any) -> str | None:
    """Resolve GWF format from explicit format, then by extension fallback."""
    if fmt is not None:
        if isinstance(fmt, str):
            return _normalize_gwf_format(fmt)
        return None

    if isinstance(source, (list, tuple)):
        if not source:
            return None
        if all(
            _normalize_path_suffix(value).suffix.lower() == ".gwf" for value in source
        ):
            return "gwf"
        return None

    try:
        path = Path(source)
    except TypeError:
        return None

    if path.suffix.lower() == ".gwf":
        return "gwf"
    return None


def _normalize_path_suffix(source: Any) -> Path:
    try:
        return Path(source)
    except TypeError:
        return Path()


def _source_for_gwf_channel_listing(source: Any) -> Any:
    """Return a single source suitable for GWF channel-name discovery."""
    if isinstance(source, (list, tuple)) and source:
        return source[0]
    return source


def _normalize_gwf_read_limit(value: Any | None) -> Any | None:
    """Normalize a GWF read boundary to GWpy's GPS representation."""
    if value is None:
        return None
    return to_gps(value)


def _normalize_gwf_gap_options(pad: Any, gap: Any) -> tuple[Any, Any]:
    """Return GWpy-compatible append gap mode and pad value."""
    merge_gap = gap if gap is not None else ("pad" if pad is not None else "raise")
    # Missing samples must read back as "no data" (NaN), not a valid-looking
    # 0.0. This mirrors SeriesMatrix.append's np.nan default (gwexpy/types/
    # series_matrix_analysis.py, #443) which the GWF read path had not yet
    # adopted; see #481.
    merge_pad = np.nan if merge_gap == "pad" and pad is None else pad
    return merge_gap, merge_pad


_GWF_PARALLEL_WORKER_CAP = 8


class _GWFParallelContractError(TypeError):
    """Signal a public GWF parallel-read contract violation."""


def _normalize_gwf_parallel_kwargs(
    gwf_kwargs: dict[str, Any], *, number_of_spans: int = 1
) -> tuple[bool, int]:
    """Return requested-parallel state and effective worker count."""
    has_parallel = "parallel" in gwf_kwargs
    has_nproc = "nproc" in gwf_kwargs
    if has_parallel and has_nproc:
        raise _GWFParallelContractError(
            "Specify either 'parallel' or 'nproc', not both."
        )
    if not has_parallel and not has_nproc:
        return False, 1

    option = "parallel" if has_parallel else "nproc"
    value = gwf_kwargs.pop(option)
    if value is None:
        return False, 1

    if isinstance(value, (bool, np.bool_)):
        if not bool(value):
            return False, 1
        return True, min(
            os.cpu_count() or 1,
            number_of_spans,
            _GWF_PARALLEL_WORKER_CAP,
        )
    if not isinstance(value, Integral):
        raise _GWFParallelContractError(f"{option} must be an integer, bool, or None")

    count = int(value)
    if count <= 0:
        raise ValueError(f"{option} must be a positive integer")
    if count == 1:
        return False, 1
    if count > _GWF_PARALLEL_WORKER_CAP:
        raise ValueError(f"{option} must be at most {_GWF_PARALLEL_WORKER_CAP}")
    return True, count


def _consume_gwf_parallel_kwargs(
    gwf_kwargs: dict[str, Any], *, number_of_spans: int = 1
) -> int:
    """Consume and normalize the GWF parallel worker option."""
    return _normalize_gwf_parallel_kwargs(gwf_kwargs, number_of_spans=number_of_spans)[
        1
    ]


def _resolve_gwf_path_span(
    source: Any, channels: list[str], backend: str | None
) -> tuple[Any, Any]:
    """Resolve the data span of one filesystem GWF path before reading it."""
    from gwpy.io.cache import file_segment
    from gwpy.io.gwf.core import data_segments

    try:
        filename_span = file_segment(source)
    except (AttributeError, TypeError, ValueError):
        filename_span = None
    if filename_span is not None:
        start, end = filename_span
        if _gwf_time_to_ns(start) >= _gwf_time_to_ns(end):
            raise ValueError(f"Invalid GWF frame span for {source!r}")
        return start, end

    try:
        segments = []
        for channel in channels:
            segments.extend(
                data_segments([source], str(channel), warn=False, backend=backend)
            )
    except ValueError:
        raise
    except (AttributeError, OSError, TypeError) as exc:
        raise ValueError(f"Could not resolve GWF frame span for {source!r}") from exc

    if not segments:
        raise ValueError(f"Could not resolve GWF frame span for {source!r}")

    start = min(segment[0] for segment in segments)
    end = max(segment[1] for segment in segments)
    if _gwf_time_to_ns(start) >= _gwf_time_to_ns(end):
        raise ValueError(f"Invalid GWF frame span for {source!r}")
    return start, end


def _gwf_time_to_ns(value: Any) -> int:
    """Convert a GWpy/LIGO or numeric time value to integer nanoseconds."""
    ns = getattr(value, "ns", None)
    if callable(ns):
        return int(ns())
    return int(round(float(value) * 1_000_000_000))


def _gwf_span_sort_key(span: tuple[Any, Any], input_index: int) -> tuple[int, int, int]:
    """Return the deterministic merge key required for parallel GWF parts."""
    return (_gwf_time_to_ns(span[0]), _gwf_time_to_ns(span[1]), input_index)


def _read_gwf_timeseriesdict_worker(
    source: Any,
    channels: list[str],
    start: Any | None,
    end: Any | None,
    backend: str | None,
    dict_class: type[Any],
    series_class: type[Any],
    read_kwargs: dict[str, Any],
) -> Any:
    """Read one GWF path in a spawn-compatible worker process."""
    from gwpy.timeseries.io.gwf.core import read_timeseriesdict

    return _coerce_gwf_timeseriesdict(
        read_timeseriesdict(
            source,
            channels,
            start=start,
            end=end,
            backend=backend,
            series_class=series_class,
            **read_kwargs,
        ),
        dict_class,
        series_class,
    )


def _is_filesystem_path(source: Any) -> bool:
    """Return whether a source is a filesystem path accepted by GWF readers."""
    return isinstance(source, (str, bytes, os.PathLike))


def _coerce_gwf_series(series: Any, series_class: type[Any]) -> Any:
    """Rebuild one worker result as the requested GWexpy-compatible class."""
    if isinstance(series, series_class):
        return series

    kwargs: dict[str, Any] = {
        "unit": getattr(series, "unit", None),
        "name": getattr(series, "name", None),
        "channel": getattr(series, "channel", None),
    }
    times = getattr(series, "times", None)
    if times is not None:
        kwargs["times"] = times
    else:
        kwargs["t0"] = getattr(series, "t0", None)
        kwargs["dt"] = getattr(series, "dt", None)

    result = series_class(np.asarray(series.value), **kwargs)
    for attribute in ("_gwex_t0_gps_ns", "_gwex_t0_gps_precision"):
        if hasattr(series, attribute):
            setattr(result, attribute, copy.deepcopy(getattr(series, attribute)))
    provenance = getattr(series, "_gwexpy_io", None)
    if isinstance(provenance, dict):
        result._gwexpy_io = copy.deepcopy(provenance)
    return result


def _coerce_gwf_timeseriesdict(
    source: Any, dict_class: type[Any], series_class: type[Any]
) -> Any:
    """Rebuild a worker collection and entries after the GWpy pickle boundary."""
    result = dict_class()
    entry_provenance = getattr(source, "_gwexpy_entry_io", {})
    for key, series in source.items():
        rebuilt = _coerce_gwf_series(series, series_class)
        if (
            isinstance(entry_provenance, dict)
            and key in entry_provenance
            and getattr(rebuilt, "_gwexpy_io", None) is None
        ):
            rebuilt._gwexpy_io = copy.deepcopy(entry_provenance[key])
        result[key] = rebuilt
    provenance = getattr(source, "_gwexpy_io", None)
    if isinstance(provenance, dict):
        result._gwexpy_io = copy.deepcopy(provenance)
    return result


def _validate_gwf_parallel_source(source: Any, gwf_kwargs: dict[str, Any]) -> bool:
    """Validate parallel source shape before any optional channel discovery."""
    requested_parallel, _ = _normalize_gwf_parallel_kwargs(
        dict(gwf_kwargs),
        number_of_spans=len(source) if isinstance(source, (list, tuple)) else 1,
    )
    if isinstance(source, (list, tuple)) and not source:
        raise ValueError("GWF source list/tuple must be non-empty")
    if requested_parallel:
        if not isinstance(source, (list, tuple)) or not all(
            _is_filesystem_path(item) for item in source
        ):
            raise _GWFParallelContractError(
                "Parallel GWF reads require a list or tuple of filesystem paths"
            )
    return requested_parallel


def read_gwf_timeseriesdict(
    source: Any,
    channels: list[str],
    *,
    start: Any | None = None,
    end: Any | None = None,
    backend: str | None = None,
    dict_class: type[Any],
    series_class: type[Any],
    **gwf_kwargs: Any,
) -> Any:
    """Read GWF source(s) into a TimeSeriesDict-like class with GWpy merge semantics."""
    read_kwargs = dict(gwf_kwargs)
    number_of_spans = len(source) if isinstance(source, (list, tuple)) else 1
    requested_parallel, workers = _normalize_gwf_parallel_kwargs(
        read_kwargs, number_of_spans=number_of_spans
    )

    if isinstance(source, (list, tuple)) and not source:
        raise ValueError("GWF source list/tuple must be non-empty")

    from gwpy.timeseries.io.gwf.core import read_timeseriesdict

    pad = read_kwargs.pop("pad", None)
    gap = read_kwargs.pop("gap", None)
    start = _normalize_gwf_read_limit(start)
    end = _normalize_gwf_read_limit(end)
    merge_gap, merge_pad = _normalize_gwf_gap_options(pad, gap)

    def read_one(item: Any) -> Any:
        return _coerce_gwf_timeseriesdict(
            read_timeseriesdict(
                item,
                channels,
                start=start,
                end=end,
                backend=backend,
                series_class=series_class,
                **read_kwargs,
            ),
            dict_class,
            series_class,
        )

    if requested_parallel and not isinstance(source, (list, tuple)):
        raise _GWFParallelContractError(
            "Parallel GWF reads require a list or tuple of filesystem paths"
        )

    parallel_spans: list[tuple[Any, Any]] | None = None
    if isinstance(source, (list, tuple)):
        sources = list(source)
        if requested_parallel:
            if not all(_is_filesystem_path(item) for item in sources):
                raise _GWFParallelContractError(
                    "Parallel GWF reads require a list or tuple of filesystem paths"
                )

            parallel_spans = [
                _resolve_gwf_path_span(item, channels, backend) for item in sources
            ]
            if workers > 1:
                executor = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )
                futures = []
                try:
                    for item in sources:
                        futures.append(
                            executor.submit(
                                _read_gwf_timeseriesdict_worker,
                                item,
                                channels,
                                start,
                                end,
                                backend,
                                dict_class,
                                series_class,
                                read_kwargs,
                            )
                        )
                    completed_parts = {}
                    for future in as_completed(futures):
                        completed_parts[future] = future.result()
                except BaseException:
                    for future in futures:
                        try:
                            future.cancel()
                        except BaseException:
                            pass
                    try:
                        executor.shutdown(wait=True, cancel_futures=True)
                    except BaseException:
                        pass
                    raise
                else:
                    executor.shutdown(wait=True)

                parts = [
                    _coerce_gwf_timeseriesdict(
                        completed_parts[future], dict_class, series_class
                    )
                    for future in futures
                ]
            else:
                parts = [read_one(item) for item in sources]
        else:
            parts = [read_one(item) for item in sources]
        non_empty_parts = [part for part in parts if len(part) > 0]
        if not non_empty_parts:
            raise ValueError("No data found in any provided GWF source")

        if requested_parallel:
            assert parallel_spans is not None
            ordered_parts = [
                part
                for _, part in sorted(
                    zip(
                        (
                            _gwf_span_sort_key(span, index)
                            for index, span in enumerate(parallel_spans)
                        ),
                        parts,
                        strict=True,
                    ),
                    key=lambda item: item[0],
                )
                if len(part) > 0
            ]
        else:
            ordered_parts = sorted(non_empty_parts, key=lambda item: item.span)

        out = dict_class()
        prev_ends: dict[str, float] = {}
        merge_parts = [part for part in ordered_parts if len(part) > 0]
        for part in merge_parts:
            if not hasattr(out, "_gwexpy_io"):
                provenance = getattr(part, "_gwexpy_io", None)
                if isinstance(provenance, dict):
                    out._gwexpy_io = copy.deepcopy(provenance)
            if merge_gap == "pad":
                for key, series in part.items():
                    prev_end = prev_ends.get(key)
                    if (
                        prev_end is not None
                        and (series.span[0] - prev_end) >= _GWF_GAP_TOL
                    ):
                        # A real per-channel gap exists — only then does
                        # gwpy's Series.append actually materialize
                        # `merge_pad` values for *this* channel, so only
                        # then can non-float dtype corruption occur.
                        check_pad_dtype_compatible(series.dtype, merge_pad)
            out.append(part, gap=merge_gap, pad=merge_pad)
            for key, series in part.items():
                prev_ends[key] = series.span[1]
                gps_ns = getattr(series, "_gwex_t0_gps_ns", None)
                if (
                    gps_ns is not None
                    and getattr(out[key], "_gwex_t0_gps_ns", None) is None
                ):
                    out[key]._gwex_t0_gps_ns = copy.deepcopy(gps_ns)
                    out[key]._gwex_t0_gps_precision = copy.deepcopy(
                        getattr(series, "_gwex_t0_gps_precision", None)
                    )
                provenance = getattr(series, "_gwexpy_io", None)
                if (
                    isinstance(provenance, dict)
                    and getattr(out[key], "_gwexpy_io", None) is None
                ):
                    out[key]._gwexpy_io = copy.deepcopy(provenance)
        result = out
    else:
        result = read_one(source)

    result = _coerce_gwf_timeseriesdict(result, dict_class, series_class)

    if merge_gap in ("pad", "raise") and (start is not None or end is not None):
        for key in result:
            result[key] = _pad_gwf_series_to_span(
                result[key],
                merge_pad,
                start,
                end,
                error=(merge_gap == "raise"),
            )
    return result


def _normalize_gwf_channels(channels: Any) -> list[str] | None:
    """Normalize channel selector(s) for GWF readers to list form."""
    if channels is None:
        return None
    if isinstance(channels, (list, tuple, set)):
        return [str(channel) for channel in channels]
    return [str(channels)]


def _extract_gwf_read_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    allow_multiple_channels: bool = True,
) -> tuple[list[str] | None, Any | None, Any | None, dict[str, Any]]:
    """Normalize positional and keyword GWF read arguments for TimeSeries and matrix."""
    gwf_kwargs = dict(kwargs)
    gwf_kwargs.pop("format", None)

    if len(args) > 3:
        raise TypeError(
            "TimeSeries-like .read() accepts at most three positional arguments "
            "for GWF readers: channels, start, end."
        )

    has_start_kw = "start" in gwf_kwargs
    has_end_kw = "end" in gwf_kwargs
    has_channel_alias_kw = any(
        key in gwf_kwargs for key in ("channels", "names", "channel", "name")
    )

    if args:
        channel_arg = args[0]
        gwf_kwargs.pop("channels", None)
        gwf_kwargs.pop("names", None)
        gwf_kwargs.pop("channel", None)
        gwf_kwargs.pop("name", None)
    else:
        channel_arg = gwf_kwargs.pop(
            "channels",
            gwf_kwargs.pop(
                "names",
                gwf_kwargs.pop("channel", gwf_kwargs.pop("name", None)),
            ),
        )
    start = args[1] if len(args) > 1 else None
    end = args[2] if len(args) > 2 else None

    if len(args) > 1:
        if has_start_kw and not has_channel_alias_kw:
            raise TypeError(
                "Cannot specify both positional and keyword 'start' for GWF read."
            )
        start = args[1]
    else:
        start = gwf_kwargs.pop("start", None)

    if len(args) > 2:
        if has_end_kw and not has_channel_alias_kw:
            raise TypeError(
                "Cannot specify both positional and keyword 'end' for GWF read."
            )
        end = args[2]
    else:
        end = gwf_kwargs.pop("end", None)

    # When positional start/end override keyword start/end (allowed when a channel alias
    # keyword is also present), the keyword values are still in gwf_kwargs.  Remove them
    # so callers that pass start= and end= explicitly don't get "multiple values" errors.
    gwf_kwargs.pop("start", None)
    gwf_kwargs.pop("end", None)

    channels = _normalize_gwf_channels(channel_arg)
    if channels is not None and len(channels) == 0:
        raise ValueError("No channels selected for GWF read.")

    if not allow_multiple_channels and channels is not None and len(channels) > 1:
        raise ValueError("Single-channel GWF read accepts exactly one channel.")

    return channels, start, end, gwf_kwargs


__all__ = [
    "_extract_gwf_read_args",
    "_pad_gwf_series_to_span",
    "_normalize_gwf_channels",
    "_normalize_gwf_format",
    "_resolve_gwf_format",
    "_source_for_gwf_channel_listing",
    "_sync_gwf_registry_aliases",
    "_GWF_BACKENDS",
    "_GWF_FORMATS",
    "_format_gwf_import_error",
    "read_gwf_timeseriesdict",
]

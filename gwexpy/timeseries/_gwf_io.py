from __future__ import annotations

from pathlib import Path
from typing import Any

from gwpy.io.registry import default_registry as io_registry
from gwpy.time import to_gps

from gwexpy.io.utils import _pad_gwf_series_to_span

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
    except Exception:
        return None


def _safe_get_writer(format_name: str, cls: type[Any]) -> Any | None:
    try:
        return io_registry.get_writer(format_name, cls)
    except Exception:
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
    except Exception:
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
    merge_pad = 0.0 if merge_gap == "pad" and pad is None else pad
    return merge_gap, merge_pad


def _consume_gwf_parallel_kwargs(gwf_kwargs: dict[str, Any]) -> int | None:
    """Remove GWpy high-level parallel kwargs before low-level GWF reads."""
    parallel = gwf_kwargs.pop("parallel", None)
    nproc = gwf_kwargs.pop("nproc", None)
    if parallel is None:
        parallel = nproc
    if parallel is None:
        return None
    try:
        return max(int(parallel), 1)
    except (TypeError, ValueError):
        return None


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
    if isinstance(source, (list, tuple)) and not source:
        raise ValueError("GWF source list/tuple must be non-empty")

    from gwpy.timeseries.io.gwf.core import read_timeseriesdict

    read_kwargs = dict(gwf_kwargs)
    pad = read_kwargs.pop("pad", None)
    gap = read_kwargs.pop("gap", None)
    _consume_gwf_parallel_kwargs(read_kwargs)
    start = _normalize_gwf_read_limit(start)
    end = _normalize_gwf_read_limit(end)
    merge_gap, merge_pad = _normalize_gwf_gap_options(pad, gap)

    def read_one(item: Any) -> Any:
        return dict_class(
            read_timeseriesdict(
                item,
                channels,
                start=start,
                end=end,
                backend=backend,
                series_class=series_class,
                **read_kwargs,
            )
        )

    if isinstance(source, (list, tuple)):
        sources = list(source)
        parts = [read_one(item) for item in sources]
        non_empty_parts = [part for part in parts if len(part) > 0]
        if not non_empty_parts:
            raise ValueError("No data found in any provided GWF source")

        out = dict_class()
        for part in sorted(non_empty_parts, key=lambda item: item.span):
            out.append(part, gap=merge_gap, pad=merge_pad)
        result = out
    else:
        result = read_one(source)

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

from __future__ import annotations

from pathlib import Path
from typing import Any

from gwpy.io.registry import default_registry as io_registry

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
        io_registry.register_reader(fmt, TimeSeriesMatrix, _read_timeseriesmatrix_gwf, force=True)

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
            io_registry.register_writer(alias, TimeSeriesMatrix, canonical_matrix_writer, force=True)

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

    try:
        path = Path(source)
    except TypeError:
        return None

    if path.suffix.lower() == ".gwf":
        return "gwf"
    return None


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
            raise TypeError("Cannot specify both positional and keyword 'start' for GWF read.")
        start = args[1]
    else:
        start = gwf_kwargs.pop("start", None)

    if len(args) > 2:
        if has_end_kw and not has_channel_alias_kw:
            raise TypeError("Cannot specify both positional and keyword 'end' for GWF read.")
        end = args[2]
    else:
        end = gwf_kwargs.pop("end", None)

    channels = _normalize_gwf_channels(channel_arg)
    if channels is not None and len(channels) == 0:
        raise ValueError("No channels selected for GWF read.")

    if not allow_multiple_channels and channels is not None and len(channels) > 1:
        raise ValueError("Single-channel GWF read accepts exactly one channel.")

    if has_start_kw and "start" in gwf_kwargs:
        gwf_kwargs.pop("start")
    if has_end_kw and "end" in gwf_kwargs:
        gwf_kwargs.pop("end")

    return channels, start, end, gwf_kwargs


__all__ = [
    "_extract_gwf_read_args",
    "_normalize_gwf_channels",
    "_normalize_gwf_format",
    "_resolve_gwf_format",
    "_sync_gwf_registry_aliases",
    "_GWF_BACKENDS",
    "_GWF_FORMATS",
    "_format_gwf_import_error",
]

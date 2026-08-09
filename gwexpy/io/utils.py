from __future__ import annotations

import contextlib
import datetime as _dt
import importlib
import math
import re
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
from astropy import units as u
from gwpy.time import to_gps

_NUMERIC_TYPES = (int, float, np.integer, np.floating)
_UTC_OFFSET_RE = re.compile(r"^([+-])(\d{2})(?::?(\d{2}))?$")
_TIMEZONE_ROUTING_SENTINEL = object()
_WARNING_STATE_SENTINEL = object()


def _is_numeric_epoch(value: Any) -> bool:
    """Return whether *value* is a supported scalar numeric epoch."""
    return isinstance(value, _NUMERIC_TYPES) and not isinstance(value, (bool, np.bool_))


def _coerce_numeric_epoch(value: Any) -> float:
    """Coerce a supported Python or NumPy scalar epoch to ``float``."""
    if not _is_numeric_epoch(value):
        raise TypeError(f"epoch must be numeric, got {type(value)}")
    return float(value)


def _make_timezone_routing_state(
    epoch_timezone: _dt.tzinfo | None,
) -> tuple[object, _dt.tzinfo | None]:
    """Create trusted state for one reader's recursive dispatch."""
    return (_TIMEZONE_ROUTING_SENTINEL, epoch_timezone)


def _consume_timezone_routing_state(
    kwargs: dict[str, Any],
) -> tuple[bool, _dt.tzinfo | None]:
    """Consume trusted recursive state while discarding legacy caller markers."""
    state = kwargs.pop("_timezone_routing_state", None)
    kwargs.pop("_timezone_checked", None)
    kwargs.pop("_epoch_timezone", None)
    if (
        isinstance(state, tuple)
        and len(state) == 2
        and state[0] is _TIMEZONE_ROUTING_SENTINEL
        and (state[1] is None or isinstance(state[1], _dt.tzinfo))
    ):
        return True, state[1]
    return False, None


def _make_warning_state(marker: list[bool]) -> tuple[object, list[bool]]:
    """Create trusted state for coalescing recursive reader warnings."""
    return (_WARNING_STATE_SENTINEL, marker)


def _consume_warning_state(
    kwargs: dict[str, Any],
    key: str,
    *legacy_keys: str,
) -> list[bool] | None:
    """Consume trusted warning state and discard caller-forgeable markers."""
    state = kwargs.pop(key, None)
    for legacy_key in legacy_keys:
        kwargs.pop(legacy_key, None)
    if (
        isinstance(state, tuple)
        and len(state) == 2
        and state[0] is _WARNING_STATE_SENTINEL
        and isinstance(state[1], list)
        and len(state[1]) == 1
        and isinstance(state[1][0], bool)
    ):
        return state[1]
    return None


def parse_timezone(tz: Any) -> _dt.tzinfo:
    r"""Convert a timezone specifier into a tzinfo.

    Accepts IANA zone names (\"Asia/Tokyo\") or numeric offsets like \"+09:00\".
    """
    if tz is None:
        raise ValueError("timezone must be specified for this format")
    if isinstance(tz, _dt.tzinfo):
        return tz
    if _is_numeric_epoch(tz):
        offset = float(tz)
        if not math.isfinite(offset) or abs(offset) >= 24:
            raise ValueError(f"Could not parse timezone {tz!r}")
        try:
            return _dt.timezone(_dt.timedelta(hours=offset))
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Could not parse timezone {tz!r}") from exc
    if isinstance(tz, str):
        cleaned = tz.strip()
        with contextlib.suppress(ZoneInfoNotFoundError):
            return ZoneInfo(cleaned)
        if cleaned.lower() in {"utc", "gmt"}:
            return _dt.UTC
        match = _UTC_OFFSET_RE.fullmatch(cleaned)
        if match is None:
            raise ValueError(f"Could not parse timezone {tz!r}")
        sign_text, hours_text, minutes_text = match.groups()
        hours = int(hours_text)
        minutes = int(minutes_text or "0")
        if hours > 23 or minutes > 59:
            raise ValueError(f"Could not parse timezone {tz!r}")
        sign = -1 if sign_text == "-" else 1
        try:
            delta = _dt.timedelta(hours=sign * hours, minutes=sign * minutes)
            return _dt.timezone(delta)
        except (
            ValueError,
            TypeError,
            OverflowError,
        ) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Could not parse timezone {tz!r}") from exc
    raise ValueError(f"Unsupported timezone specifier: {tz!r}")


def _parse_timezone_for_format(format_name: str, timezone: Any) -> _dt.tzinfo:
    """Parse *timezone* and add reader context to invalid input errors."""
    try:
        return parse_timezone(timezone)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse timezone {timezone!r} for format '{format_name}'"
        ) from exc


def _reject_timezone_reinterpretation(
    format_name: str,
    timezone: Any,
    epoch: Any,
) -> _dt.tzinfo | None:
    """Validate timezone use without reinterpreting an absolute source time.

    The returned timezone is only for localizing a naive explicit ``epoch``.
    Absolute epoch values preserve their value and report that ``timezone`` is
    ignored.  Parsing happens before either branch so a dummy epoch can never
    hide an invalid timezone specification.
    """
    if timezone is None:
        return None

    tzinfo = _parse_timezone_for_format(format_name, timezone)
    if epoch is None:
        raise ValueError(
            f"timezone must not be specified for format '{format_name}'; "
            "the format defines its time semantics"
        )

    aware_iso_epoch = False
    if isinstance(epoch, str):
        with contextlib.suppress(ValueError):
            parsed_epoch = _dt.datetime.fromisoformat(
                epoch.strip().replace("Z", "+00:00")
            )
            aware_iso_epoch = parsed_epoch.tzinfo is not None

    if (
        _is_numeric_epoch(epoch)
        or (isinstance(epoch, _dt.datetime) and epoch.tzinfo is not None)
        or aware_iso_epoch
    ):
        warnings.warn(
            f"timezone is ignored for format '{format_name}' because epoch "
            "already defines an absolute time",
            UserWarning,
            stacklevel=2,
        )
        return None

    return tzinfo


def datetime_to_gps(dt: _dt.datetime) -> float:
    """Convert an aware datetime or date into a LIGO GPS float."""
    if isinstance(dt, _dt.date) and not isinstance(dt, _dt.datetime):
        dt = _dt.datetime.combine(dt, _dt.time(0, 0), tzinfo=_dt.UTC)
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware to convert to GPS")
    return float(to_gps(dt))


def _apply_tzinfo(dt: _dt.datetime, tz: _dt.tzinfo) -> _dt.datetime:
    """Attach tzinfo to naive datetime, rejecting ambiguous or nonexistent times."""
    if tz is _dt.UTC or tz == _dt.UTC:
        return dt.replace(tzinfo=tz)
    
    dt0 = dt.replace(tzinfo=tz, fold=0)
    dt1 = dt.replace(tzinfo=tz, fold=1)
    
    try:
        utc0 = dt0.astimezone(_dt.UTC)
        utc1 = dt1.astimezone(_dt.UTC)
    except Exception:
        # Fallback if tzinfo doesn't support astimezone (e.g. some mock objects)
        return dt0
        
    if utc0 != utc1:
        raise ValueError(
            f"Local time {dt.strftime('%Y-%m-%d %H:%M:%S')} is ambiguous or nonexistent "
            f"in timezone {tz} due to DST transitions."
        )
        
    return dt0


def ensure_datetime(value: Any, tzinfo: _dt.tzinfo | None = None) -> _dt.datetime:
    """Parse a timestamp into a timezone-aware datetime.

    Tries common formats like ``YYYY/MM/DD HH:MM:SS(.fff)``.
    """
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None and tzinfo is not None:
            return _apply_tzinfo(value, tzinfo)
        if value.tzinfo is None:
            raise ValueError("Naive datetime requires timezone")
        return value
    if _is_numeric_epoch(value):
        return _dt.datetime.fromtimestamp(_coerce_numeric_epoch(value), tz=_dt.UTC)
    if isinstance(value, str):
        text = value.strip()
        with contextlib.suppress(ValueError):
            dt = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None and tzinfo is not None:
                dt = dt.replace(tzinfo=tzinfo)
            return dt
        formats = [
            "%Y/%m/%d %H:%M:%S.%f",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d,%H:%M:%S",
        ]
        for fmt in formats:
            with contextlib.suppress(ValueError):
                dt = _dt.datetime.strptime(text, fmt)
                if dt.tzinfo is None and tzinfo is not None:
                    dt = _apply_tzinfo(dt, tzinfo)
                return dt
    raise ValueError(f"Unrecognised time value: {value!r}")


def apply_unit(series: Any, unit: Any | None) -> Any:
    """Override the unit on a series-like object if requested."""
    if unit is None:
        return series
    if unit == "":
        return series
    try:
        from gwexpy.interop._registry import ConverterRegistry

        SeriesMatrix = ConverterRegistry.get_constructor("SeriesMatrix")
        series_matrix_types: tuple[type, ...] = (SeriesMatrix,)
    except (ImportError, KeyError, AttributeError, TypeError):
        series_matrix_types = ()
    if isinstance(series, series_matrix_types):
        series_obj = cast(Any, series)
        try:
            for i in range(series_obj.meta.shape[0]):
                for j in range(series_obj.meta.shape[1]):
                    series_obj.meta[i, j]["unit"] = u.Unit(unit)
            return series_obj
        except (KeyError, IndexError, AttributeError):
            pass
    # GWpy series objects keep `.unit` immutable after construction, but
    # provide `override_unit()` for metadata-only changes.
    with contextlib.suppress(AttributeError, TypeError, ValueError):
        if hasattr(series, "override_unit"):
            series.override_unit(unit)
            return series
    try:
        series.unit = u.Unit(unit)
        return series
    except (AttributeError, TypeError):
        # fallback to constructor
        try:
            return series.__class__(
                series.value,
                times=getattr(series, "times", None),
                dt=getattr(series, "dt", None),
                t0=getattr(series, "t0", None),
                frequencies=getattr(series, "frequencies", None),
                df=getattr(series, "df", None),
                f0=getattr(series, "f0", None),
                unit=unit,
                channel=getattr(series, "channel", None),
                name=getattr(series, "name", None),
                epoch=getattr(series, "epoch", None),
            )
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError(f"Could not apply unit {unit!r}") from exc


def set_provenance(obj: Any, info: dict[str, Any]) -> None:
    """Attach provenance metadata to a gwexpy or gwpy object."""
    try:
        if hasattr(obj, "attrs") and isinstance(obj.attrs, dict):
            obj.attrs.update(info)
            return
    except (TypeError, AttributeError):
        pass
    with contextlib.suppress(AttributeError, TypeError):
        setattr(obj, "_gwexpy_io", {**info})


def filter_by_channels(mapping: dict[str, Any], channels: Iterable[str] | None):
    """Return a mapping filtered by selected channel names."""
    if channels is None:
        return mapping
    wanted = set(channels)
    return {k: v for k, v in mapping.items() if k in wanted}


def _ceil_nonnegative_sample_count(value: Any) -> int:
    """Return a conservative non-negative sample count for float spans."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    if numeric <= 0:
        return 0
    return int(math.ceil(numeric - 1e-12))


def check_pad_dtype_compatible(dtype: Any, pad: Any) -> None:
    """Raise ValueError if a NaN pad value cannot be represented by dtype."""
    # Detect NaN for any float scalar type (Python ``float``, ``np.float64``,
    # ``np.float32``, ...), not just ``isinstance(pad, float)`` which misses the
    # numpy 32-bit scalar. ``np.isnan`` raises on non-numeric pads (None, str),
    # so treat those as "not NaN" and let the actual pad path handle them.
    try:
        pad_is_nan = bool(np.isnan(pad))
    except (TypeError, ValueError):
        pad_is_nan = False
    if (
        pad_is_nan
        and not np.issubdtype(dtype, np.floating)
        and not np.issubdtype(dtype, np.complexfloating)
    ):
        raise ValueError(
            f"Cannot pad a gap with NaN for non-floating dtype {dtype!r}; "
            f"pass an explicit ``pad=`` value that the dtype can represent."
        )


def _pad_gwf_series_to_span(
    ts: Any,
    pad: Any,
    start: Any | None = None,
    end: Any | None = None,
    *,
    error: bool = False,
) -> Any:
    """Pad or reject a GWF series that does not cover the requested interval."""
    span = ts.span
    if start is None:
        start = span[0]
    if end is None:
        end = span[1]

    rate = ts.sample_rate.value
    pada = _ceil_nonnegative_sample_count((span[0] - start) * rate)
    padb = _ceil_nonnegative_sample_count((end - span[1]) * rate)
    if not (pada or padb):
        return ts
    if error:
        msg = (
            f"{type(ts).__name__} with span {span} does not cover "
            f"requested interval {type(span)(start, end)}"
        )
        raise ValueError(msg)
    # Outer start/end padding materializes `pad` values just like the
    # multi-file gap-merge path, so it needs the same non-floating dtype
    # guard: a NaN pad on an int channel would otherwise surface NumPy's
    # opaque "cannot convert float NaN to integer" instead of the clear
    # gwexpy-level ValueError promised for GWF reads (#481).
    check_pad_dtype_compatible(ts.dtype, pad)
    return ts.pad((pada, padb), mode="constant", constant_values=(pad,))


def maybe_pad_timeseries(ts, pad_value=np.nan, start=None, end=None, gap="pad"):
    """Pad gaps or raise using gwpy join semantics."""
    if gap not in ("pad", "raise"):
        return ts

    return _pad_gwf_series_to_span(
        ts,
        pad_value,
        start=start,
        end=end,
        error=(gap == "raise"),
    )


def ensure_dependency(
    package_name: str,
    *,
    extra: str | None = None,
    import_name: str | None = None,
) -> Any:
    """Import a package or raise a standardized ImportError.

    Parameters
    ----------
    package_name : str
        PyPI package name (for pip install instructions).
    extra : str, optional
        Optional extras specifier (e.g., "gui", "analysis").
    import_name : str, optional
        Import name if different from package_name.

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    ImportError
        With standardized installation instructions.

    Examples
    --------
    >>> xarray = ensure_dependency("xarray")  # doctest: +SKIP
    >>> nptdms = ensure_dependency("nptdms", import_name="nptdms")  # doctest: +SKIP

    """
    try:
        name = import_name or package_name
        return importlib.import_module(name)
    except ImportError as exc:
        if extra:
            install_cmd = f"pip install 'gwexpy[{extra}]'"
        else:
            install_cmd = f"pip install {package_name}"
        msg = f"{package_name} is required. Install with: {install_cmd}"
        raise ImportError(msg) from exc


def extract_audio_metadata(source: str | Path) -> dict[str, Any]:
    """Extract audio metadata using tinytag.

    Attempts to read common audio metadata fields (title, artist, album, genre,
    duration, bitrate, etc.) from audio files. Returns an empty dictionary if
    tinytag is not installed or if metadata extraction fails.

    Parameters
    ----------
    source : str or Path
        Path to the audio file.

    Returns
    -------
    dict
        Dictionary of metadata fields. Only non-None values are included.
        Possible keys: title, artist, album, genre, duration, bitrate,
        comment, track, year.

    Notes
    -----
    Requires the optional ``tinytag`` package. Install it via a GWexpy extra::

        pip install 'gwexpy[audio]'

    Or install the bundled optional dependencies used by GWexpy tutorials::

        pip install "gwexpy[all]"

    Examples
    --------
    >>> metadata = extract_audio_metadata("song.mp3")  # doctest: +SKIP
    >>> print(metadata.get("title"))  # doctest: +SKIP
    'Song Title'

    """
    try:
        from tinytag import TinyTag

        tag = TinyTag.get(str(source))
        metadata = {
            "title": tag.title,
            "artist": tag.artist,
            "album": tag.album,
            "genre": tag.genre,
            "duration": tag.duration,
            "bitrate": tag.bitrate,
            "comment": tag.comment,
            "track": tag.track,
            "year": tag.year,
        }
        # Filter out None values
        return {k: v for k, v in metadata.items() if v is not None}
    except ImportError:
        warnings.warn(
            "tinytag is required for metadata extraction. "
            "Install with: pip install 'gwexpy[audio]' "
            'or pip install "gwexpy[all]"',
            UserWarning,
            stacklevel=3,
        )
        return {}
    except Exception as e:
        # Metadata extraction failure should not break file reading
        warnings.warn(
            f"Failed to extract metadata: {e}",
            UserWarning,
            stacklevel=3,
        )
        return {}
